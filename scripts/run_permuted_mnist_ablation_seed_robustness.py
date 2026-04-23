import argparse
import copy
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from morphobase.config.validate import load_config
from scripts.run_permuted_mnist_ablations import DEFAULT_CONDITIONS, run_ablation_suite


DEFAULT_SEEDS = (42, 123, 321, 777, 999)


def _stats(series: list[float]) -> dict[str, float]:
    if not series:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    mean = sum(series) / len(series)
    variance = sum((value - mean) ** 2 for value in series) / len(series)
    return {
        "mean": float(mean),
        "std": float(variance ** 0.5),
        "min": float(min(series)),
        "max": float(max(series)),
    }


def summarize_seed_reports(seed_reports: list[dict]) -> dict:
    def summary_values(key: str) -> list[float]:
        return [float(item["summary"].get(key, 0.0)) for item in seed_reports]

    def supported_fraction(condition_name: str) -> float:
        if not seed_reports:
            return 0.0
        supported = sum(1 for item in seed_reports if item["summary"].get(f"{condition_name}_dependency_supported", False))
        return float(supported / len(seed_reports))

    supported_all = sum(
        1
        for item in seed_reports
        if float(item["summary"].get("mechanism_dependency_supported_count", 0.0)) >= 3.0
    )

    summary = {
        "seed_count": float(len(seed_reports)),
        "baseline_final_accuracy": _stats(summary_values("baseline_final_accuracy_mean")),
        "baseline_mean_forgetting": _stats(summary_values("baseline_mean_forgetting")),
        "baseline_bwt": _stats(summary_values("baseline_bwt")),
        "baseline_peak_growth_pressure": _stats(summary_values("baseline_peak_growth_pressure_mean")),
        "baseline_growth_trigger_crossed_fraction": _stats(summary_values("baseline_growth_trigger_crossed_fraction")),
        "mechanism_dependency_supported_count": _stats(summary_values("mechanism_dependency_supported_count")),
        "no_growth_supported_fraction": supported_fraction("no_growth"),
        "no_stress_supported_fraction": supported_fraction("no_stress"),
        "no_z_field_supported_fraction": supported_fraction("no_z_field"),
        "all_mechanisms_supported_fraction": float(supported_all / len(seed_reports)) if seed_reports else 0.0,
    }
    summary["stable_across_seeds"] = bool(
        summary["baseline_growth_trigger_crossed_fraction"]["min"] >= 0.10
        and summary["mechanism_dependency_supported_count"]["min"] >= 2.0
        and summary["all_mechanisms_supported_fraction"] >= 0.60
    )
    return summary


def build_markdown(seed_reports: list[dict], summary: dict) -> str:
    lines = [
        "# Permuted-MNIST Growth-Probe Ablation Seed Robustness",
        "",
        f"- seed_count: {int(summary['seed_count'])}",
        f"- baseline_final_accuracy_mean: {summary['baseline_final_accuracy']['mean']:.4f}",
        f"- baseline_final_accuracy_std: {summary['baseline_final_accuracy']['std']:.4f}",
        f"- baseline_growth_trigger_crossed_fraction_mean: {summary['baseline_growth_trigger_crossed_fraction']['mean']:.4f}",
        f"- baseline_growth_trigger_crossed_fraction_min: {summary['baseline_growth_trigger_crossed_fraction']['min']:.4f}",
        f"- mechanism_dependency_supported_count_mean: {summary['mechanism_dependency_supported_count']['mean']:.4f}",
        f"- mechanism_dependency_supported_count_min: {summary['mechanism_dependency_supported_count']['min']:.4f}",
        f"- no_growth_supported_fraction: {summary['no_growth_supported_fraction']:.4f}",
        f"- no_stress_supported_fraction: {summary['no_stress_supported_fraction']:.4f}",
        f"- no_z_field_supported_fraction: {summary['no_z_field_supported_fraction']:.4f}",
        f"- all_mechanisms_supported_fraction: {summary['all_mechanisms_supported_fraction']:.4f}",
        f"- stable_across_seeds: {summary['stable_across_seeds']}",
        "",
        "## Runs",
        "",
    ]

    for report in seed_reports:
        seed_summary = report["summary"]
        lines.extend(
            [
                f"### {report['run_name']}",
                "",
                f"- seed: {report['seed']}",
                f"- baseline_final_accuracy_mean: {seed_summary.get('baseline_final_accuracy_mean', 0.0):.4f}",
                f"- baseline_growth_trigger_crossed_fraction: {seed_summary.get('baseline_growth_trigger_crossed_fraction', 0.0):.4f}",
                f"- mechanism_dependency_supported_count: {seed_summary.get('mechanism_dependency_supported_count', 0.0):.1f}",
                f"- no_growth_dependency_supported: {seed_summary.get('no_growth_dependency_supported', False)}",
                f"- no_stress_dependency_supported: {seed_summary.get('no_stress_dependency_supported', False)}",
                f"- no_z_field_dependency_supported: {seed_summary.get('no_z_field_dependency_supported', False)}",
                "",
            ]
        )

    return "\n".join(lines)


def run_seed_robustness(
    config_path: Path,
    seeds: tuple[int, ...],
    *,
    conditions: tuple[str, ...],
    challenge_variant: str = "growth_probe",
) -> dict:
    base_cfg = load_config(config_path)
    seed_reports = []

    for seed in seeds:
        cfg = copy.deepcopy(base_cfg)
        cfg.run.seed = int(seed)
        cfg.run.name = f"permuted_mnist_growth_probe_seed_{seed}"
        report = run_ablation_suite(
            config_path,
            conditions,
            challenge_variant=challenge_variant,
            base_cfg=cfg,
        )
        report["run_name"] = cfg.run.name
        report["seed"] = seed
        seed_reports.append(report)

    seed_reports.sort(
        key=lambda item: (
            float(item["summary"].get("mechanism_dependency_supported_count", 0.0)),
            float(item["summary"].get("baseline_growth_trigger_crossed_fraction", 0.0)),
            float(item["summary"].get("baseline_final_accuracy_mean", 0.0)),
        ),
        reverse=True,
    )
    summary = summarize_seed_reports(seed_reports)
    return {"runs": seed_reports, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/assay/permuted_mnist.yaml")
    parser.add_argument("--seeds", nargs="*", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--conditions", nargs="*", default=list(DEFAULT_CONDITIONS))
    parser.add_argument("--challenge-variant", default="growth_probe")
    parser.add_argument("--json-out", default="artifacts/permuted_mnist_growth_probe_seed_robustness.json")
    parser.add_argument("--markdown-out", default="artifacts/permuted_mnist_growth_probe_seed_robustness.md")
    args = parser.parse_args()

    report = run_seed_robustness(
        Path(args.config),
        tuple(args.seeds),
        conditions=tuple(args.conditions),
        challenge_variant=args.challenge_variant,
    )

    json_path = Path(args.json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    markdown_path = Path(args.markdown_out)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(build_markdown(report["runs"], report["summary"]), encoding="utf-8")

    print(json_path)
    print(markdown_path)


if __name__ == "__main__":
    main()
