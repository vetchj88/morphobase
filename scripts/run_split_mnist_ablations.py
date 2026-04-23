import argparse
import copy
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
import sys

import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from morphobase.assays.split_mnist import SplitMNISTAssay
from morphobase.config.validate import config_to_dict, load_config
from morphobase.diagnostics.alerts import classify_run, hard_fail_alerts
from morphobase.diagnostics.logger import JsonlLogger
from morphobase.diagnostics.plots import plot_scalar_history, plot_stage_occupancy
from morphobase.diagnostics.summaries import build_markdown_summary, write_summary
from morphobase.registry import append_run_row
from morphobase.seeds import set_seed
from scripts.rank_split_mnist_sweeps import biomarker_score, primary_score


DEFAULT_CONDITIONS = SplitMNISTAssay.ABLATION_CONDITIONS


def _persist_condition_run(cfg, result, *, condition_name: str) -> dict:
    out_dir = Path(cfg.run.output_dir) / cfg.run.name
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_dict = config_to_dict(cfg)
    cfg_dict["split_mnist_condition"] = condition_name
    (out_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(cfg_dict, sort_keys=False),
        encoding="utf-8",
    )

    logger = JsonlLogger(out_dir / cfg.logging.event_log_name)
    for item in result.history:
        logger.log(item)

    if cfg.run.save_plots and result.history:
        plot_scalar_history(result.history, "mean_energy", out_dir / "mean_energy.png")
        plot_scalar_history(result.history, "mean_stress", out_dir / "mean_stress.png")
        plot_scalar_history(result.history, "mean_z_alignment", out_dir / "mean_z_alignment.png")
        plot_scalar_history(result.history, "dormant_fraction", out_dir / "plasticity_dormant_fraction.png")
        plot_stage_occupancy(result.history, out_dir / "stage_occupancy.png")

    alerts = hard_fail_alerts(result.final_metrics)
    triggered = [alert.name for alert in alerts if alert.triggered]
    notes = result.notes
    if triggered:
        notes = f"{notes} Alerts triggered: {', '.join(triggered)}.".strip()

    write_summary(
        out_dir / cfg.run.summary_name,
        build_markdown_summary(cfg_dict, result.final_metrics, notes),
    )
    (out_dir / "final_metrics.json").write_text(
        json.dumps(result.final_metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out_dir / "alerts.json").write_text(
        json.dumps([asdict(alert) for alert in alerts], indent=2),
        encoding="utf-8",
    )

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
    baseline_accuracy = float(baseline_metrics.get("split_mnist_final_accuracy_mean", 0.0))
    baseline_peak = float(baseline_metrics.get("split_mnist_peak_accuracy_mean", 0.0))
    baseline_forgetting = float(baseline_metrics.get("split_mnist_mean_forgetting", 0.0))
    baseline_bwt = float(baseline_metrics.get("split_mnist_bwt", 0.0))
    baseline_margin = float(baseline_metrics.get("split_mnist_mean_margin", 0.0))
    baseline_energy = float(baseline_metrics.get("mean_energy", 0.0))
    baseline_stress = float(baseline_metrics.get("mean_stress", 0.0))
    baseline_z = float(baseline_metrics.get("mean_z_alignment", 0.0))
    baseline_growth_peak = float(baseline_metrics.get("split_mnist_peak_growth_pressure_mean", 0.0))
    baseline_growth_trigger = float(baseline_metrics.get("split_mnist_growth_trigger_crossed_fraction", 0.0))

    accuracy_drop = baseline_accuracy - float(ablated_metrics.get("split_mnist_final_accuracy_mean", 0.0))
    peak_drop = baseline_peak - float(ablated_metrics.get("split_mnist_peak_accuracy_mean", 0.0))
    forgetting_increase = float(ablated_metrics.get("split_mnist_mean_forgetting", 0.0)) - baseline_forgetting
    bwt_drop = float(baseline_bwt - float(ablated_metrics.get("split_mnist_bwt", 0.0)))
    margin_drop = baseline_margin - float(ablated_metrics.get("split_mnist_mean_margin", 0.0))
    energy_drop = baseline_energy - float(ablated_metrics.get("mean_energy", 0.0))
    stress_rise = float(ablated_metrics.get("mean_stress", 0.0)) - baseline_stress
    z_drop = baseline_z - float(ablated_metrics.get("mean_z_alignment", 0.0))
    growth_peak_drop = baseline_growth_peak - float(ablated_metrics.get("split_mnist_peak_growth_pressure_mean", 0.0))
    growth_trigger_drop = baseline_growth_trigger - float(
        ablated_metrics.get("split_mnist_growth_trigger_crossed_fraction", 0.0)
    )

    support_score = float(
        (1 if accuracy_drop >= 0.02 else 0)
        + (1 if peak_drop >= 0.02 else 0)
        + (1 if forgetting_increase >= 0.02 else 0)
        + (1 if bwt_drop >= 0.02 else 0)
        + (1 if margin_drop >= 0.01 else 0)
        + (1 if energy_drop >= 0.02 else 0)
        + (1 if stress_rise >= 0.01 else 0)
        + (1 if z_drop >= 0.05 else 0)
        + (1 if growth_peak_drop >= 0.05 else 0)
        + (1 if growth_trigger_drop >= 0.005 else 0)
    )
    return {
        "final_accuracy_drop": float(accuracy_drop),
        "peak_accuracy_drop": float(peak_drop),
        "forgetting_increase": float(forgetting_increase),
        "bwt_drop": float(bwt_drop),
        "margin_drop": float(margin_drop),
        "energy_drop": float(energy_drop),
        "stress_rise": float(stress_rise),
        "z_alignment_drop": float(z_drop),
        "growth_peak_pressure_drop": float(growth_peak_drop),
        "growth_trigger_crossing_drop": float(growth_trigger_drop),
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
        "baseline_final_accuracy_mean": float(baseline_metrics.get("split_mnist_final_accuracy_mean", 0.0)),
        "baseline_mean_forgetting": float(baseline_metrics.get("split_mnist_mean_forgetting", 0.0)),
        "baseline_bwt": float(baseline_metrics.get("split_mnist_bwt", 0.0)),
        "baseline_mean_margin": float(baseline_metrics.get("split_mnist_mean_margin", 0.0)),
        "baseline_mean_energy": float(baseline_metrics.get("mean_energy", 0.0)),
        "baseline_mean_stress": float(baseline_metrics.get("mean_stress", 0.0)),
    }

    supported_count = 0
    for condition_name, run in runs_by_condition.items():
        metrics = run["metrics"]
        prefix = condition_name
        summary[f"{prefix}_final_accuracy_mean"] = float(metrics.get("split_mnist_final_accuracy_mean", 0.0))
        summary[f"{prefix}_mean_forgetting"] = float(metrics.get("split_mnist_mean_forgetting", 0.0))
        summary[f"{prefix}_bwt"] = float(metrics.get("split_mnist_bwt", 0.0))
        summary[f"{prefix}_mean_margin"] = float(metrics.get("split_mnist_mean_margin", 0.0))
        summary[f"{prefix}_peak_growth_pressure_mean"] = float(metrics.get("split_mnist_peak_growth_pressure_mean", 0.0))
        summary[f"{prefix}_growth_trigger_crossed_fraction"] = float(
            metrics.get("split_mnist_growth_trigger_crossed_fraction", 0.0)
        )
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
    summary["mechanism_dependency_supported_fraction"] = float(
        supported_count / max(len(condition_runs) - 1, 1)
    )
    return summary


def build_ablation_markdown(condition_runs: list[dict], summary: dict) -> str:
    lines = [
        "# Split-MNIST Matched Ablations",
        "",
        f"- baseline_run: {summary['baseline_run']}",
        f"- baseline_final_accuracy_mean: {summary['baseline_final_accuracy_mean']:.4f}",
        f"- baseline_mean_forgetting: {summary['baseline_mean_forgetting']:.4f}",
        f"- baseline_bwt: {summary['baseline_bwt']:.4f}",
        f"- baseline_mean_margin: {summary['baseline_mean_margin']:.4f}",
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
                f"- final_accuracy_mean: {metrics.get('split_mnist_final_accuracy_mean', 0.0):.4f}",
                f"- peak_accuracy_mean: {metrics.get('split_mnist_peak_accuracy_mean', 0.0):.4f}",
                f"- mean_forgetting: {metrics.get('split_mnist_mean_forgetting', 0.0):.4f}",
                f"- bwt: {metrics.get('split_mnist_bwt', 0.0):.4f}",
                f"- mean_margin: {metrics.get('split_mnist_mean_margin', 0.0):.4f}",
                f"- mean_energy: {metrics.get('mean_energy', 0.0):.4f}",
                f"- mean_stress: {metrics.get('mean_stress', 0.0):.4f}",
                f"- mean_z_alignment: {metrics.get('mean_z_alignment', 0.0):.4f}",
                f"- peak_growth_pressure_mean: {metrics.get('split_mnist_peak_growth_pressure_mean', 0.0):.4f}",
                f"- growth_trigger_crossed_fraction: {metrics.get('split_mnist_growth_trigger_crossed_fraction', 0.0):.4f}",
            ]
        )
        if run["condition"] != "baseline":
            prefix = run["condition"]
            lines.extend(
                [
                    f"- final_accuracy_drop_vs_baseline: {summary[f'{prefix}_final_accuracy_drop']:.4f}",
                    f"- forgetting_increase_vs_baseline: {summary[f'{prefix}_forgetting_increase']:.4f}",
                    f"- bwt_drop_vs_baseline: {summary[f'{prefix}_bwt_drop']:.4f}",
                    f"- margin_drop_vs_baseline: {summary[f'{prefix}_margin_drop']:.4f}",
                    f"- growth_peak_pressure_drop_vs_baseline: {summary[f'{prefix}_growth_peak_pressure_drop']:.4f}",
                    f"- growth_trigger_crossing_drop_vs_baseline: {summary[f'{prefix}_growth_trigger_crossing_drop']:.4f}",
                    f"- dependency_support_score: {summary[f'{prefix}_dependency_support_score']:.1f}",
                    f"- dependency_supported: {summary[f'{prefix}_dependency_supported']}",
                ]
            )
        lines.append("")

    return "\n".join(lines)


def run_ablation_suite(config_path: Path, conditions: tuple[str, ...], *, challenge_variant: str = "growth_probe") -> dict:
    base_cfg = load_config(config_path)
    assay = SplitMNISTAssay()
    assay.challenge_variant = challenge_variant
    condition_runs = []
    base_name = base_cfg.run.name

    for condition_name in conditions:
        cfg = copy.deepcopy(base_cfg)
        cfg.run.name = f"{base_name}_{condition_name}"
        set_seed(cfg.run.seed)
        result = assay.run_condition(cfg, condition_name)
        condition_runs.append(_persist_condition_run(cfg, result, condition_name=condition_name))

    ordering = {name: idx for idx, name in enumerate(conditions)}
    condition_runs.sort(key=lambda item: ordering[item["condition"]])
    summary = summarize_ablation_runs(condition_runs)
    return {"runs": condition_runs, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/assay/split_mnist.yaml")
    parser.add_argument("--conditions", nargs="*", default=list(DEFAULT_CONDITIONS))
    parser.add_argument("--challenge-variant", default="growth_probe")
    parser.add_argument("--json-out", default="artifacts/split_mnist_ablations.json")
    parser.add_argument("--markdown-out", default="artifacts/split_mnist_ablations.md")
    args = parser.parse_args()

    report = run_ablation_suite(
        Path(args.config),
        tuple(args.conditions),
        challenge_variant=args.challenge_variant,
    )

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
