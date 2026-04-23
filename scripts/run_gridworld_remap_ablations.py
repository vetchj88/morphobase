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

from morphobase.assays.gridworld_remap import GridworldRemapAssay
from morphobase.config.validate import config_to_dict, load_config
from morphobase.diagnostics.alerts import classify_run, hard_fail_alerts
from morphobase.diagnostics.logger import JsonlLogger
from morphobase.diagnostics.plots import plot_scalar_history, plot_stage_occupancy
from morphobase.diagnostics.summaries import build_markdown_summary, write_summary
from morphobase.registry import append_run_row
from morphobase.seeds import set_seed
from morphobase.training.trainer import SequentialLinearTrainer
from scripts.run_gridworld_remap_seed_robustness import biomarker_score, primary_score


DEFAULT_CONDITIONS = tuple(GridworldRemapAssay.CONDITION_SPECS.keys())


def _build_baseline_training_payload(assay: GridworldRemapAssay, cfg) -> dict:
    support_embeddings_by_task: list[np.ndarray] = []
    support_labels_by_task: list[np.ndarray] = []
    for task_index, task_spec in enumerate(assay.TASK_SPECS):
        support_obs, support_labels = assay._support_samples(task_spec, seed=assay.SUPPORT_CURRICULUM_SEED + 43 * task_index)
        task_embeddings = []
        for obs, label in zip(support_obs, support_labels, strict=True):
            rollout = assay._rollout_observation(cfg, task_spec, obs, int(label), condition_name="baseline")
            task_embeddings.append(rollout["embedding"])
        support_embeddings_by_task.append(np.stack(task_embeddings, axis=0))
        support_labels_by_task.append(np.asarray(support_labels, dtype=int))
    return {
        "support_embeddings_by_task": support_embeddings_by_task,
        "support_labels_by_task": support_labels_by_task,
    }


def _evaluate_episode_fixed(
    assay: GridworldRemapAssay,
    cfg,
    task_spec: dict,
    final_model,
    *,
    episode_seed: int,
    condition_name: str,
) -> tuple[float, float, list[dict[str, float]]]:
    position = assay._initial_position(task_spec, episode_seed=episode_seed)
    prev_action = 2
    reward_zone = assay._positions(task_spec, "reward_zone")
    rollout_metrics: list[dict[str, float]] = []
    for step in range(assay.HORIZON):
        observation = assay._observation(task_spec, position, prev_action)
        rollout = assay._rollout_observation(
            cfg,
            task_spec,
            observation,
            assay._expert_action(task_spec, position, prev_action),
            condition_name=condition_name,
        )
        rollout_metrics.append(rollout["final_metrics"])
        action = int(final_model.predict(rollout["embedding"][None, :])[0])
        position = assay._transition(task_spec, position, action)
        prev_action = action
        if position in reward_zone:
            efficiency = 1.0 - (step / max(assay.HORIZON - 1, 1))
            return 1.0, float(np.clip(efficiency, 0.0, 1.0)), rollout_metrics
    return 0.0, 0.0, rollout_metrics


def _retained_competence_metrics(
    assay: GridworldRemapAssay,
    cfg,
    *,
    condition_name: str,
    baseline_payload: dict,
) -> dict[str, float]:
    trainer = SequentialLinearTrainer(
        assay.ACTION_LABELS.copy(),
        baseline_payload["support_embeddings_by_task"][0].shape[1],
        seed=cfg.run.seed + 719,
    )
    seen_train_embeddings: list[np.ndarray] = []
    seen_train_labels: list[np.ndarray] = []
    task_peak_scores: dict[int, float] = {}
    task_initial_scores: dict[int, float] = {}
    task_final_scores: dict[int, float] = {}
    task_efficiency_scores: dict[int, float] = {}
    all_rollout_metrics: list[dict[str, float]] = []
    final_model = None

    for task_index, task_embeddings in enumerate(baseline_payload["support_embeddings_by_task"]):
        seen_train_embeddings.append(task_embeddings)
        seen_train_labels.append(baseline_payload["support_labels_by_task"][task_index])
        train_embeddings = np.concatenate(seen_train_embeddings, axis=0)
        train_labels = np.concatenate(seen_train_labels, axis=0)
        final_model = trainer.train_task(train_embeddings, train_labels, epochs=80, learning_rate=0.12, l2=2e-4)

        for seen_task_index in range(task_index + 1):
            successes = []
            efficiencies = []
            for episode_idx in range(assay.EVAL_EPISODES):
                success, efficiency, rollout_metrics = _evaluate_episode_fixed(
                    assay,
                    cfg,
                    assay.TASK_SPECS[seen_task_index],
                    final_model,
                    episode_seed=assay.EVAL_CURRICULUM_SEED + 89 * seen_task_index + episode_idx,
                    condition_name=condition_name,
                )
                successes.append(success)
                efficiencies.append(efficiency)
                all_rollout_metrics.extend(rollout_metrics)
            success_rate = float(np.mean(successes))
            efficiency_rate = float(np.mean(efficiencies))
            task_peak_scores[seen_task_index] = max(task_peak_scores.get(seen_task_index, 0.0), success_rate)
            if seen_task_index == task_index:
                task_initial_scores[seen_task_index] = success_rate
            task_final_scores[seen_task_index] = success_rate
            task_efficiency_scores[seen_task_index] = efficiency_rate

    forgetting_values = [
        task_peak_scores[index] - task_final_scores.get(index, 0.0)
        for index in range(len(assay.TASK_SPECS) - 1)
    ]
    bwt_values = [
        task_final_scores.get(index, 0.0) - task_initial_scores.get(index, 0.0)
        for index in range(len(assay.TASK_SPECS) - 1)
    ]
    mean_margin = (
        final_model.mean_margin(np.concatenate(baseline_payload["support_embeddings_by_task"], axis=0))
        if final_model is not None
        else 0.0
    )
    peak_growth_pressure = max((item.get("mean_growth_pressure", 0.0) for item in all_rollout_metrics), default=0.0)
    growth_crossed_fraction = float(
        np.mean(
            [
                1.0 if item.get("mean_growth_pressure", 0.0) >= assay.GROWTH_TRIGGER_THRESHOLD else 0.0
                for item in all_rollout_metrics
            ]
        )
    ) if all_rollout_metrics else 0.0

    return {
        "gridworld_remap_retained_final_success_mean": float(np.mean(list(task_final_scores.values()))) if task_final_scores else 0.0,
        "gridworld_remap_retained_peak_success_mean": float(np.mean(list(task_peak_scores.values()))) if task_peak_scores else 0.0,
        "gridworld_remap_retained_mean_forgetting": float(np.mean(forgetting_values)) if forgetting_values else 0.0,
        "gridworld_remap_retained_bwt": float(np.mean(bwt_values)) if bwt_values else 0.0,
        "gridworld_remap_retained_efficiency_mean": float(np.mean(list(task_efficiency_scores.values()))) if task_efficiency_scores else 0.0,
        "gridworld_remap_retained_mean_margin": float(mean_margin),
        "gridworld_remap_retained_mean_energy": float(np.mean([item.get("mean_energy", 0.0) for item in all_rollout_metrics])) if all_rollout_metrics else 0.0,
        "gridworld_remap_retained_mean_stress": float(np.mean([item.get("mean_stress", 0.0) for item in all_rollout_metrics])) if all_rollout_metrics else 0.0,
        "gridworld_remap_retained_mean_z_alignment": float(np.mean([item.get("mean_z_alignment", 0.0) for item in all_rollout_metrics])) if all_rollout_metrics else 0.0,
        "gridworld_remap_retained_peak_growth_pressure_mean": float(peak_growth_pressure),
        "gridworld_remap_retained_growth_trigger_crossed_fraction": growth_crossed_fraction,
    }


def _persist_condition_run(cfg, result, *, condition_name: str) -> dict:
    out_dir = Path(cfg.run.output_dir) / cfg.run.name
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_dict = config_to_dict(cfg)
    cfg_dict["gridworld_remap_condition"] = condition_name
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
    baseline_success = float(baseline_metrics.get("gridworld_remap_retained_final_success_mean", baseline_metrics.get("gridworld_remap_final_success_mean", 0.0)))
    baseline_peak = float(baseline_metrics.get("gridworld_remap_retained_peak_success_mean", baseline_metrics.get("gridworld_remap_peak_success_mean", 0.0)))
    baseline_forgetting = float(baseline_metrics.get("gridworld_remap_retained_mean_forgetting", baseline_metrics.get("gridworld_remap_mean_forgetting", 0.0)))
    baseline_bwt = float(baseline_metrics.get("gridworld_remap_retained_bwt", baseline_metrics.get("gridworld_remap_bwt", 0.0)))
    baseline_efficiency = float(baseline_metrics.get("gridworld_remap_retained_efficiency_mean", baseline_metrics.get("gridworld_remap_efficiency_mean", 0.0)))
    baseline_margin = float(baseline_metrics.get("gridworld_remap_retained_mean_margin", baseline_metrics.get("gridworld_remap_mean_margin", 0.0)))
    baseline_energy = float(baseline_metrics.get("mean_energy", 0.0))
    baseline_stress = float(baseline_metrics.get("mean_stress", 0.0))
    baseline_z = float(abs(baseline_metrics.get("mean_z_alignment", 0.0)))
    baseline_growth_pressure = float(baseline_metrics.get("mean_growth_pressure", 0.0))
    baseline_growth_transfer = float(baseline_metrics.get("recent_growth_energy_transferred", 0.0))

    success_drop = baseline_success - float(ablated_metrics.get("gridworld_remap_retained_final_success_mean", ablated_metrics.get("gridworld_remap_final_success_mean", 0.0)))
    peak_drop = baseline_peak - float(ablated_metrics.get("gridworld_remap_retained_peak_success_mean", ablated_metrics.get("gridworld_remap_peak_success_mean", 0.0)))
    forgetting_increase = float(ablated_metrics.get("gridworld_remap_retained_mean_forgetting", ablated_metrics.get("gridworld_remap_mean_forgetting", 0.0))) - baseline_forgetting
    bwt_drop = baseline_bwt - float(ablated_metrics.get("gridworld_remap_retained_bwt", ablated_metrics.get("gridworld_remap_bwt", 0.0)))
    efficiency_drop = baseline_efficiency - float(ablated_metrics.get("gridworld_remap_retained_efficiency_mean", ablated_metrics.get("gridworld_remap_efficiency_mean", 0.0)))
    margin_drop = baseline_margin - float(ablated_metrics.get("gridworld_remap_retained_mean_margin", ablated_metrics.get("gridworld_remap_mean_margin", 0.0)))
    energy_drop = baseline_energy - float(ablated_metrics.get("mean_energy", 0.0))
    stress_rise = float(ablated_metrics.get("mean_stress", 0.0)) - baseline_stress
    z_drop = baseline_z - float(abs(ablated_metrics.get("mean_z_alignment", 0.0)))
    growth_pressure_drop = baseline_growth_pressure - float(ablated_metrics.get("mean_growth_pressure", 0.0))
    growth_transfer_drop = baseline_growth_transfer - float(ablated_metrics.get("recent_growth_energy_transferred", 0.0))

    support_score = float(
        (1 if success_drop >= 0.03 else 0)
        + (1 if peak_drop >= 0.03 else 0)
        + (1 if forgetting_increase >= 0.02 else 0)
        + (1 if bwt_drop >= 0.02 else 0)
        + (1 if efficiency_drop >= 0.03 else 0)
        + (1 if margin_drop >= 0.01 else 0)
        + (1 if energy_drop >= 0.01 else 0)
        + (1 if stress_rise >= 0.01 else 0)
        + (1 if z_drop >= 0.05 else 0)
        + (1 if growth_pressure_drop >= 0.05 else 0)
        + (1 if growth_transfer_drop >= 0.002 else 0)
    )
    return {
        "final_success_drop": float(success_drop),
        "peak_success_drop": float(peak_drop),
        "forgetting_increase": float(forgetting_increase),
        "bwt_drop": float(bwt_drop),
        "efficiency_drop": float(efficiency_drop),
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
        "baseline_final_success_mean": float(baseline_metrics.get("gridworld_remap_retained_final_success_mean", baseline_metrics.get("gridworld_remap_final_success_mean", 0.0))),
        "baseline_mean_forgetting": float(baseline_metrics.get("gridworld_remap_retained_mean_forgetting", baseline_metrics.get("gridworld_remap_mean_forgetting", 0.0))),
        "baseline_bwt": float(baseline_metrics.get("gridworld_remap_retained_bwt", baseline_metrics.get("gridworld_remap_bwt", 0.0))),
        "baseline_efficiency_mean": float(baseline_metrics.get("gridworld_remap_retained_efficiency_mean", baseline_metrics.get("gridworld_remap_efficiency_mean", 0.0))),
        "baseline_mean_margin": float(baseline_metrics.get("gridworld_remap_retained_mean_margin", baseline_metrics.get("gridworld_remap_mean_margin", 0.0))),
    }

    supported_count = 0
    for condition_name, run in runs_by_condition.items():
        metrics = run["metrics"]
        prefix = condition_name
        summary[f"{prefix}_final_success_mean"] = float(metrics.get("gridworld_remap_retained_final_success_mean", metrics.get("gridworld_remap_final_success_mean", 0.0)))
        summary[f"{prefix}_mean_forgetting"] = float(metrics.get("gridworld_remap_retained_mean_forgetting", metrics.get("gridworld_remap_mean_forgetting", 0.0)))
        summary[f"{prefix}_bwt"] = float(metrics.get("gridworld_remap_retained_bwt", metrics.get("gridworld_remap_bwt", 0.0)))
        summary[f"{prefix}_efficiency_mean"] = float(metrics.get("gridworld_remap_retained_efficiency_mean", metrics.get("gridworld_remap_efficiency_mean", 0.0)))
        summary[f"{prefix}_mean_margin"] = float(metrics.get("gridworld_remap_retained_mean_margin", metrics.get("gridworld_remap_mean_margin", 0.0)))
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
        "# Gridworld Remap Matched Ablations",
        "",
        f"- baseline_run: {summary['baseline_run']}",
        f"- baseline_final_success_mean: {summary['baseline_final_success_mean']:.4f}",
        f"- baseline_mean_forgetting: {summary['baseline_mean_forgetting']:.4f}",
        f"- baseline_bwt: {summary['baseline_bwt']:.4f}",
        f"- baseline_efficiency_mean: {summary['baseline_efficiency_mean']:.4f}",
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
                f"- retained_final_success_mean: {metrics.get('gridworld_remap_retained_final_success_mean', metrics.get('gridworld_remap_final_success_mean', 0.0)):.4f}",
                f"- retained_peak_success_mean: {metrics.get('gridworld_remap_retained_peak_success_mean', metrics.get('gridworld_remap_peak_success_mean', 0.0)):.4f}",
                f"- retained_mean_forgetting: {metrics.get('gridworld_remap_retained_mean_forgetting', metrics.get('gridworld_remap_mean_forgetting', 0.0)):.4f}",
                f"- retained_bwt: {metrics.get('gridworld_remap_retained_bwt', metrics.get('gridworld_remap_bwt', 0.0)):.4f}",
                f"- retained_efficiency_mean: {metrics.get('gridworld_remap_retained_efficiency_mean', metrics.get('gridworld_remap_efficiency_mean', 0.0)):.4f}",
                f"- retained_mean_margin: {metrics.get('gridworld_remap_retained_mean_margin', metrics.get('gridworld_remap_mean_margin', 0.0)):.4f}",
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
                    f"- final_success_drop_vs_baseline: {summary[f'{prefix}_final_success_drop']:.4f}",
                    f"- forgetting_increase_vs_baseline: {summary[f'{prefix}_forgetting_increase']:.4f}",
                    f"- bwt_drop_vs_baseline: {summary[f'{prefix}_bwt_drop']:.4f}",
                    f"- efficiency_drop_vs_baseline: {summary[f'{prefix}_efficiency_drop']:.4f}",
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
    assay = GridworldRemapAssay()
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
    parser.add_argument("--config", default="configs/assay/gridworld_remap.yaml")
    parser.add_argument("--conditions", nargs="*", default=list(DEFAULT_CONDITIONS))
    parser.add_argument("--challenge-variant", default="repair_probe")
    parser.add_argument("--json-out", default="artifacts/gridworld_remap_ablations.json")
    parser.add_argument("--markdown-out", default="artifacts/gridworld_remap_ablations.md")
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
