import argparse
import copy
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
import sys

import numpy as np
import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from morphobase.assays.sequential_rules import SequentialRulesAssay
from morphobase.config.validate import config_to_dict, load_config
from morphobase.diagnostics.alerts import classify_run, hard_fail_alerts
from morphobase.diagnostics.logger import JsonlLogger
from morphobase.diagnostics.plots import plot_scalar_history, plot_stage_occupancy
from morphobase.diagnostics.summaries import build_markdown_summary, write_summary
from morphobase.registry import append_run_row
from morphobase.seeds import set_seed
from morphobase.training.trainer import SequentialLinearTrainer
from scripts.run_sequential_rules_seed_robustness import biomarker_score, primary_score


DEFAULT_CONDITIONS = tuple(SequentialRulesAssay.CONDITION_SPECS.keys())


def _task_labels_by_split(assay: SequentialRulesAssay, labels: np.ndarray) -> list[np.ndarray]:
    return [np.asarray(labels[np.isin(labels, task_classes)], dtype=int) for task_classes in assay.TASK_SPLITS]


def _collect_embeddings_by_split(
    assay: SequentialRulesAssay,
    cfg,
    *,
    sequences: np.ndarray,
    labels: np.ndarray,
    condition_name: str,
) -> tuple[list[np.ndarray], list[dict[str, float]]]:
    embeddings_by_task: list[np.ndarray] = []
    metric_summaries: list[dict[str, float]] = []
    for task_classes in assay.TASK_SPLITS:
        task_mask = np.isin(labels, task_classes)
        task_embeddings = []
        task_metrics = []
        for sequence in sequences[task_mask]:
            rollout = assay._rollout_sequence(cfg, sequence, condition_name=condition_name)
            task_embeddings.append(rollout["embedding"])
            task_metrics.append(rollout["final_metrics"])
        embeddings_by_task.append(np.stack(task_embeddings, axis=0))
        metric_summaries.append(
            {
                "mean_energy": float(np.mean([item.get("mean_energy", 0.0) for item in task_metrics])) if task_metrics else 0.0,
                "mean_stress": float(np.mean([item.get("mean_stress", 0.0) for item in task_metrics])) if task_metrics else 0.0,
                "mean_z_alignment": float(np.mean([item.get("mean_z_alignment", 0.0) for item in task_metrics])) if task_metrics else 0.0,
                "mean_growth_pressure": float(np.mean([item.get("mean_growth_pressure", 0.0) for item in task_metrics])) if task_metrics else 0.0,
            }
        )
    return embeddings_by_task, metric_summaries


def _build_baseline_training_payload(assay: SequentialRulesAssay, cfg) -> dict:
    support_sequences, support_labels = assay._build_dataset(
        classes=assay.CLASS_IDS,
        per_class=assay.SUPPORT_PER_CLASS,
        seed=cfg.run.seed,
    )
    eval_sequences, eval_labels = assay._build_dataset(
        classes=assay.CLASS_IDS,
        per_class=assay.EVAL_PER_CLASS,
        seed=cfg.run.seed + 17,
    )
    support_embeddings_by_task, _ = _collect_embeddings_by_split(
        assay,
        cfg,
        sequences=support_sequences,
        labels=support_labels,
        condition_name="baseline",
    )
    return {
        "support_embeddings_by_task": support_embeddings_by_task,
        "support_labels_by_task": _task_labels_by_split(assay, support_labels),
        "eval_sequences": eval_sequences,
        "eval_labels": eval_labels,
    }


def _retained_competence_metrics(
    assay: SequentialRulesAssay,
    cfg,
    *,
    condition_name: str,
    baseline_payload: dict,
) -> dict[str, float]:
    eval_embeddings_by_task, eval_metric_summaries = _collect_embeddings_by_split(
        assay,
        cfg,
        sequences=baseline_payload["eval_sequences"],
        labels=baseline_payload["eval_labels"],
        condition_name=condition_name,
    )
    eval_labels_by_task = _task_labels_by_split(assay, baseline_payload["eval_labels"])

    trainer = SequentialLinearTrainer(
        np.array(assay.CLASS_IDS, dtype=int),
        baseline_payload["support_embeddings_by_task"][0].shape[1],
        seed=cfg.run.seed + 503,
    )
    seen_classes: list[int] = []
    seen_train_embeddings: list[np.ndarray] = []
    seen_train_labels: list[np.ndarray] = []
    task_peak_accuracies: dict[int, float] = {}
    task_initial_accuracies: dict[int, float] = {}
    task_final_accuracies: dict[int, float] = {}
    final_model = None

    for task_index, task_classes in enumerate(assay.TASK_SPLITS):
        seen_train_embeddings.append(baseline_payload["support_embeddings_by_task"][task_index])
        seen_train_labels.append(baseline_payload["support_labels_by_task"][task_index])
        train_embeddings = np.concatenate(seen_train_embeddings, axis=0)
        train_labels = np.concatenate(seen_train_labels, axis=0)
        final_model = trainer.train_task(
            train_embeddings,
            train_labels,
            epochs=88,
            learning_rate=0.16,
            l2=2e-4,
        )

        seen_classes.extend(int(label) for label in task_classes)
        allowed_classes = np.array(sorted(set(seen_classes)), dtype=int)
        for seen_task_index in range(task_index + 1):
            accuracy = final_model.score(
                eval_embeddings_by_task[seen_task_index],
                eval_labels_by_task[seen_task_index],
                allowed_classes=allowed_classes,
            )
            task_peak_accuracies[seen_task_index] = max(task_peak_accuracies.get(seen_task_index, 0.0), accuracy)
            if seen_task_index == task_index:
                task_initial_accuracies[seen_task_index] = accuracy
            task_final_accuracies[seen_task_index] = accuracy

    forgetting_values = [
        task_peak_accuracies[index] - task_final_accuracies.get(index, 0.0)
        for index in range(len(assay.TASK_SPLITS) - 1)
    ]
    bwt_values = [
        task_final_accuracies.get(index, 0.0) - task_initial_accuracies.get(index, 0.0)
        for index in range(len(assay.TASK_SPLITS) - 1)
    ]
    final_accuracy_mean = float(np.mean(list(task_final_accuracies.values()))) if task_final_accuracies else 0.0
    peak_accuracy_mean = float(np.mean(list(task_peak_accuracies.values()))) if task_peak_accuracies else 0.0
    mean_margin = (
        final_model.mean_margin(np.concatenate(eval_embeddings_by_task, axis=0), allowed_classes=np.array(assay.CLASS_IDS, dtype=int))
        if final_model is not None
        else 0.0
    )
    peak_growth_pressure = max((item["mean_growth_pressure"] for item in eval_metric_summaries), default=0.0)
    growth_crossed_fraction = float(
        np.mean(
            [
                1.0 if item["mean_growth_pressure"] >= assay.GROWTH_TRIGGER_THRESHOLD else 0.0
                for item in eval_metric_summaries
            ]
        )
    ) if eval_metric_summaries else 0.0

    return {
        "sequential_rules_retained_final_accuracy_mean": final_accuracy_mean,
        "sequential_rules_retained_peak_accuracy_mean": peak_accuracy_mean,
        "sequential_rules_retained_mean_forgetting": float(np.mean(forgetting_values)) if forgetting_values else 0.0,
        "sequential_rules_retained_bwt": float(np.mean(bwt_values)) if bwt_values else 0.0,
        "sequential_rules_retained_mean_margin": float(mean_margin),
        "sequential_rules_retained_mean_energy": float(np.mean([item["mean_energy"] for item in eval_metric_summaries])) if eval_metric_summaries else 0.0,
        "sequential_rules_retained_mean_stress": float(np.mean([item["mean_stress"] for item in eval_metric_summaries])) if eval_metric_summaries else 0.0,
        "sequential_rules_retained_mean_z_alignment": float(np.mean([item["mean_z_alignment"] for item in eval_metric_summaries])) if eval_metric_summaries else 0.0,
        "sequential_rules_retained_peak_growth_pressure_mean": float(peak_growth_pressure),
        "sequential_rules_retained_growth_trigger_crossed_fraction": growth_crossed_fraction,
    }


def _persist_condition_run(cfg, result, *, condition_name: str) -> dict:
    out_dir = Path(cfg.run.output_dir) / cfg.run.name
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_dict = config_to_dict(cfg)
    cfg_dict["sequential_rules_condition"] = condition_name
    (out_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg_dict, sort_keys=False), encoding="utf-8")

    logger = JsonlLogger(out_dir / cfg.logging.event_log_name)
    for item in result.history:
        logger.log(item)

    if cfg.run.save_plots and result.history:
        plot_scalar_history(result.history, "mean_energy", out_dir / "mean_energy.png")
        plot_scalar_history(result.history, "mean_stress", out_dir / "mean_stress.png")
        plot_scalar_history(result.history, "mean_z_alignment", out_dir / "mean_z_alignment.png")
        plot_stage_occupancy(result.history, out_dir / "stage_occupancy.png")

    alerts = hard_fail_alerts(result.final_metrics)
    notes = result.notes
    triggered = [alert.name for alert in alerts if alert.triggered]
    if triggered:
        notes = f"{notes} Alerts triggered: {', '.join(triggered)}.".strip()

    write_summary(out_dir / cfg.run.summary_name, build_markdown_summary(cfg_dict, result.final_metrics, notes))
    (out_dir / "final_metrics.json").write_text(json.dumps(result.final_metrics, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "alerts.json").write_text(json.dumps([asdict(alert) for alert in alerts], indent=2), encoding="utf-8")

    verdict = classify_run(result.final_metrics).value
    append_run_row(
        Path(cfg.run.output_dir) / cfg.run.registry_name,
        {
            "date": datetime.now(UTC).isoformat(),
            "run_name": cfg.run.name,
            "assay": f"{cfg.assay.name}:{condition_name}",
            "seed": cfg.run.seed,
            "verdict": verdict,
            "mean_energy": result.final_metrics.get("mean_energy", ""),
            "mean_stress": result.final_metrics.get("mean_stress", ""),
            "mean_z_alignment": result.final_metrics.get("mean_z_alignment", ""),
            "notes": notes,
        },
    )

    return {
        "run_name": cfg.run.name,
        "condition": condition_name,
        "artifacts_dir": str(out_dir),
        "metrics": result.final_metrics,
        "verdict": verdict,
        "notes": notes,
        "primary_score": primary_score(result.final_metrics),
        "biomarker_score": biomarker_score(result.final_metrics),
    }


def _delta_metrics(baseline_metrics: dict, ablated_metrics: dict) -> dict[str, float]:
    baseline_accuracy = float(baseline_metrics.get("sequential_rules_retained_final_accuracy_mean", baseline_metrics.get("sequential_rules_final_accuracy_mean", 0.0)))
    baseline_peak = float(baseline_metrics.get("sequential_rules_retained_peak_accuracy_mean", baseline_metrics.get("sequential_rules_peak_accuracy_mean", 0.0)))
    baseline_forgetting = float(baseline_metrics.get("sequential_rules_retained_mean_forgetting", baseline_metrics.get("sequential_rules_mean_forgetting", 0.0)))
    baseline_bwt = float(baseline_metrics.get("sequential_rules_retained_bwt", baseline_metrics.get("sequential_rules_bwt", 0.0)))
    baseline_margin = float(baseline_metrics.get("sequential_rules_retained_mean_margin", baseline_metrics.get("sequential_rules_mean_margin", 0.0)))
    baseline_energy = float(baseline_metrics.get("mean_energy", 0.0))
    baseline_stress = float(baseline_metrics.get("mean_stress", 0.0))
    baseline_z = float(abs(baseline_metrics.get("mean_z_alignment", 0.0)))
    baseline_growth_pressure = float(baseline_metrics.get("mean_growth_pressure", 0.0))
    baseline_growth_transfer = float(baseline_metrics.get("recent_growth_energy_transferred", 0.0))

    accuracy_drop = baseline_accuracy - float(ablated_metrics.get("sequential_rules_retained_final_accuracy_mean", ablated_metrics.get("sequential_rules_final_accuracy_mean", 0.0)))
    peak_drop = baseline_peak - float(ablated_metrics.get("sequential_rules_retained_peak_accuracy_mean", ablated_metrics.get("sequential_rules_peak_accuracy_mean", 0.0)))
    forgetting_increase = float(ablated_metrics.get("sequential_rules_retained_mean_forgetting", ablated_metrics.get("sequential_rules_mean_forgetting", 0.0))) - baseline_forgetting
    bwt_drop = baseline_bwt - float(ablated_metrics.get("sequential_rules_retained_bwt", ablated_metrics.get("sequential_rules_bwt", 0.0)))
    margin_drop = baseline_margin - float(ablated_metrics.get("sequential_rules_retained_mean_margin", ablated_metrics.get("sequential_rules_mean_margin", 0.0)))
    energy_drop = baseline_energy - float(ablated_metrics.get("mean_energy", 0.0))
    stress_rise = float(ablated_metrics.get("mean_stress", 0.0)) - baseline_stress
    z_drop = baseline_z - float(abs(ablated_metrics.get("mean_z_alignment", 0.0)))
    growth_pressure_drop = baseline_growth_pressure - float(ablated_metrics.get("mean_growth_pressure", 0.0))
    growth_transfer_drop = baseline_growth_transfer - float(ablated_metrics.get("recent_growth_energy_transferred", 0.0))

    support_score = float(
        (1 if accuracy_drop >= 0.02 else 0)
        + (1 if peak_drop >= 0.02 else 0)
        + (1 if forgetting_increase >= 0.015 else 0)
        + (1 if bwt_drop >= 0.015 else 0)
        + (1 if margin_drop >= 0.01 else 0)
        + (1 if energy_drop >= 0.01 else 0)
        + (1 if stress_rise >= 0.01 else 0)
        + (1 if z_drop >= 0.04 else 0)
        + (1 if growth_pressure_drop >= 0.05 else 0)
        + (1 if growth_transfer_drop >= 0.002 else 0)
    )
    return {
        "final_accuracy_drop": float(accuracy_drop),
        "peak_accuracy_drop": float(peak_drop),
        "forgetting_increase": float(forgetting_increase),
        "bwt_drop": float(bwt_drop),
        "margin_drop": float(margin_drop),
        "energy_drop": float(energy_drop),
        "stress_rise": float(stress_rise),
        "z_alignment_magnitude_drop": float(z_drop),
        "growth_pressure_drop": float(growth_pressure_drop),
        "growth_energy_transfer_drop": float(growth_transfer_drop),
        "dependency_support_score": support_score,
        "dependency_supported": bool(support_score >= 2.0),
    }


def summarize_ablation_runs(condition_runs: list[dict]) -> dict:
    runs_by_condition = {item["condition"]: item for item in condition_runs}
    baseline = runs_by_condition["baseline"]
    baseline_metrics = baseline["metrics"]

    summary = {
        "baseline_run": baseline["run_name"],
        "condition_count": float(len(condition_runs)),
        "baseline_final_accuracy_mean": float(baseline_metrics.get("sequential_rules_retained_final_accuracy_mean", baseline_metrics.get("sequential_rules_final_accuracy_mean", 0.0))),
        "baseline_mean_forgetting": float(baseline_metrics.get("sequential_rules_retained_mean_forgetting", baseline_metrics.get("sequential_rules_mean_forgetting", 0.0))),
        "baseline_bwt": float(baseline_metrics.get("sequential_rules_retained_bwt", baseline_metrics.get("sequential_rules_bwt", 0.0))),
        "baseline_mean_margin": float(baseline_metrics.get("sequential_rules_retained_mean_margin", baseline_metrics.get("sequential_rules_mean_margin", 0.0))),
        "baseline_mean_energy": float(baseline_metrics.get("mean_energy", 0.0)),
        "baseline_mean_stress": float(baseline_metrics.get("mean_stress", 0.0)),
    }

    supported_count = 0
    for condition_name, run in runs_by_condition.items():
        metrics = run["metrics"]
        prefix = condition_name
        summary[f"{prefix}_final_accuracy_mean"] = float(metrics.get("sequential_rules_retained_final_accuracy_mean", metrics.get("sequential_rules_final_accuracy_mean", 0.0)))
        summary[f"{prefix}_mean_forgetting"] = float(metrics.get("sequential_rules_retained_mean_forgetting", metrics.get("sequential_rules_mean_forgetting", 0.0)))
        summary[f"{prefix}_bwt"] = float(metrics.get("sequential_rules_retained_bwt", metrics.get("sequential_rules_bwt", 0.0)))
        summary[f"{prefix}_mean_margin"] = float(metrics.get("sequential_rules_retained_mean_margin", metrics.get("sequential_rules_mean_margin", 0.0)))
        summary[f"{prefix}_mean_growth_pressure"] = float(metrics.get("mean_growth_pressure", 0.0))
        summary[f"{prefix}_recent_growth_energy_transferred"] = float(metrics.get("recent_growth_energy_transferred", 0.0))
        summary[f"{prefix}_primary_score"] = float(run["primary_score"])
        summary[f"{prefix}_biomarker_score"] = float(run["biomarker_score"])
        if condition_name == "baseline":
            continue
        deltas = _delta_metrics(baseline_metrics, metrics)
        for key, value in deltas.items():
            summary[f"{prefix}_{key}"] = value
        if deltas["dependency_supported"]:
            supported_count += 1

    summary["mechanism_dependency_supported_count"] = float(supported_count)
    summary["mechanism_dependency_supported_fraction"] = float(supported_count / max(len(condition_runs) - 1, 1))
    return summary


def build_ablation_markdown(condition_runs: list[dict], summary: dict) -> str:
    lines = [
        "# Sequential Rules Matched Ablations",
        "",
        f"- baseline_run: {summary['baseline_run']}",
        f"- baseline_final_accuracy_mean: {summary['baseline_final_accuracy_mean']:.4f}",
        f"- baseline_mean_forgetting: {summary['baseline_mean_forgetting']:.4f}",
        f"- baseline_bwt: {summary['baseline_bwt']:.4f}",
        f"- mechanism_dependency_supported_count: {int(summary['mechanism_dependency_supported_count'])}",
        f"- mechanism_dependency_supported_fraction: {summary['mechanism_dependency_supported_fraction']:.4f}",
        "",
        "## Conditions",
        "",
    ]
    for run in condition_runs:
        metrics = run["metrics"]
        lines.extend(
            [
                f"### {run['condition']}",
                "",
                f"- run_name: {run['run_name']}",
                f"- verdict: {run['verdict']}",
                f"- retained_final_accuracy_mean: {metrics.get('sequential_rules_retained_final_accuracy_mean', metrics.get('sequential_rules_final_accuracy_mean', 0.0)):.4f}",
                f"- retained_peak_accuracy_mean: {metrics.get('sequential_rules_retained_peak_accuracy_mean', metrics.get('sequential_rules_peak_accuracy_mean', 0.0)):.4f}",
                f"- retained_mean_forgetting: {metrics.get('sequential_rules_retained_mean_forgetting', metrics.get('sequential_rules_mean_forgetting', 0.0)):.4f}",
                f"- retained_bwt: {metrics.get('sequential_rules_retained_bwt', metrics.get('sequential_rules_bwt', 0.0)):.4f}",
                f"- retained_mean_margin: {metrics.get('sequential_rules_retained_mean_margin', metrics.get('sequential_rules_mean_margin', 0.0)):.4f}",
                f"- mean_energy: {metrics.get('mean_energy', 0.0):.4f}",
                f"- mean_stress: {metrics.get('mean_stress', 0.0):.4f}",
                f"- mean_z_alignment: {metrics.get('mean_z_alignment', 0.0):.4f}",
                f"- mean_growth_pressure: {metrics.get('mean_growth_pressure', 0.0):.4f}",
                f"- recent_growth_energy_transferred: {metrics.get('recent_growth_energy_transferred', 0.0):.4f}",
            ]
        )
        if run["condition"] != "baseline":
            prefix = run["condition"]
            lines.extend(
                [
                    f"- final_accuracy_drop_vs_baseline: {summary[f'{prefix}_final_accuracy_drop']:.4f}",
                    f"- forgetting_increase_vs_baseline: {summary[f'{prefix}_forgetting_increase']:.4f}",
                    f"- bwt_drop_vs_baseline: {summary[f'{prefix}_bwt_drop']:.4f}",
                    f"- z_alignment_magnitude_drop_vs_baseline: {summary[f'{prefix}_z_alignment_magnitude_drop']:.4f}",
                    f"- growth_pressure_drop_vs_baseline: {summary[f'{prefix}_growth_pressure_drop']:.4f}",
                    f"- growth_energy_transfer_drop_vs_baseline: {summary[f'{prefix}_growth_energy_transfer_drop']:.4f}",
                    f"- dependency_support_score: {summary[f'{prefix}_dependency_support_score']:.1f}",
                    f"- dependency_supported: {summary[f'{prefix}_dependency_supported']}",
                ]
            )
        lines.append("")
    return "\n".join(lines)


def run_ablation_suite(config_path: Path, conditions: tuple[str, ...], *, challenge_variant: str = "repair_probe") -> dict:
    base_cfg = load_config(config_path)
    assay = SequentialRulesAssay()
    assay.challenge_variant = challenge_variant
    baseline_payload = _build_baseline_training_payload(assay, base_cfg)
    condition_runs = []
    base_name = base_cfg.run.name

    for condition_name in conditions:
        cfg = copy.deepcopy(base_cfg)
        cfg.run.name = f"{base_name}_{condition_name}"
        set_seed(cfg.run.seed)
        result = assay.run_condition(cfg, condition_name)
        result.final_metrics.update(
            _retained_competence_metrics(
                assay,
                cfg,
                condition_name=condition_name,
                baseline_payload=baseline_payload,
            )
        )
        condition_runs.append(_persist_condition_run(cfg, result, condition_name=condition_name))

    ordering = {name: idx for idx, name in enumerate(conditions)}
    condition_runs.sort(key=lambda item: ordering[item["condition"]])
    return {"runs": condition_runs, "summary": summarize_ablation_runs(condition_runs)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/assay/sequential_rules.yaml")
    parser.add_argument("--conditions", nargs="*", default=list(DEFAULT_CONDITIONS))
    parser.add_argument("--challenge-variant", default="repair_probe")
    parser.add_argument("--json-out", default="artifacts/sequential_rules_ablations.json")
    parser.add_argument("--markdown-out", default="artifacts/sequential_rules_ablations.md")
    args = parser.parse_args()

    report = run_ablation_suite(Path(args.config), tuple(args.conditions), challenge_variant=args.challenge_variant)
    json_path = Path(args.json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    markdown_path = Path(args.markdown_out)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(build_ablation_markdown(report["runs"], report["summary"]), encoding="utf-8")

    print(json_path)
    print(markdown_path)


if __name__ == "__main__":
    main()
