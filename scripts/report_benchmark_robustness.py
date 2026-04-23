import argparse
import json
from pathlib import Path


STACK_C_BENCHMARKS = (
    {
        "assay": "split_mnist",
        "phase": "phase_5_split_mnist_bridge",
        "label": "Split-MNIST",
        "seed_report": "split_mnist_seed_robustness.json",
        "ablation_report": "split_mnist_ablations.json",
        "chamber_seed_report": None,
    },
    {
        "assay": "permuted_mnist",
        "phase": "phase_6_permuted_mnist_stressor",
        "label": "Permuted-MNIST",
        "seed_report": "permuted_mnist_seed_robustness.json",
        "ablation_report": "permuted_mnist_ablations.json",
        "chamber_seed_report": "permuted_mnist_growth_probe_seed_robustness.json",
    },
    {
        "assay": "split_fashion_mnist",
        "phase": "phase_7_split_fashion_mnist_diversity",
        "label": "Split-FashionMNIST",
        "seed_report": "split_fashion_mnist_seed_robustness.json",
        "ablation_report": "split_fashion_mnist_ablations.json",
        "chamber_seed_report": None,
    },
    {
        "assay": "permuted_fashion_mnist",
        "phase": "phase_8_permuted_fashion_mnist_variant",
        "label": "Permuted-FashionMNIST",
        "seed_report": "permuted_fashion_mnist_seed_robustness.json",
        "ablation_report": "permuted_fashion_mnist_ablations.json",
        "chamber_seed_report": None,
    },
)


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _status_for_benchmark(seed_stable: bool, mechanism_full: bool, chamber_seed_stable: bool | None) -> str:
    if not seed_stable and chamber_seed_stable:
        return "causal_chamber_only"
    if not seed_stable:
        return "seed_unstable"
    if not mechanism_full:
        return "mechanism_incomplete"
    if chamber_seed_stable is False:
        return "mechanism_chamber_unstable"
    return "ready"


def _benchmark_entry(artifacts_root: Path, spec: dict) -> dict:
    seed_path = artifacts_root / spec["seed_report"]
    ablation_path = artifacts_root / spec["ablation_report"]
    chamber_seed_path = artifacts_root / spec["chamber_seed_report"] if spec["chamber_seed_report"] else None

    seed_payload = _load_json(seed_path)
    ablation_payload = _load_json(ablation_path)
    chamber_seed_payload = _load_json(chamber_seed_path) if chamber_seed_path else None

    seed_summary = (seed_payload or {}).get("summary", {})
    ablation_summary = (ablation_payload or {}).get("summary", {})
    chamber_seed_summary = (chamber_seed_payload or {}).get("summary", {})

    seed_stable = bool(seed_summary.get("stable_across_seeds", False))
    mechanism_count = float(ablation_summary.get("mechanism_dependency_supported_count", 0.0))
    mechanism_fraction = float(ablation_summary.get("mechanism_dependency_supported_fraction", 0.0))
    mechanism_full = mechanism_count >= 3.0 or mechanism_fraction >= 1.0
    chamber_seed_stable = None if chamber_seed_path is None else bool(chamber_seed_summary.get("stable_across_seeds", False))

    final_accuracy_stats = seed_summary.get("final_accuracy", {})
    forgetting_stats = seed_summary.get("forgetting", {})
    bwt_stats = seed_summary.get("bwt", {})
    primary_score_stats = seed_summary.get("primary_score", {})

    entry = {
        "assay": spec["assay"],
        "label": spec["label"],
        "phase": spec["phase"],
        "seed_report_present": seed_payload is not None,
        "ablation_report_present": ablation_payload is not None,
        "chamber_seed_report_present": chamber_seed_payload is not None,
        "seed_report_path": str(seed_path),
        "ablation_report_path": str(ablation_path),
        "chamber_seed_report_path": str(chamber_seed_path) if chamber_seed_path else "",
        "seed_count": float(seed_summary.get("seed_count", 0.0)),
        "final_accuracy_mean": float(final_accuracy_stats.get("mean", 0.0)),
        "final_accuracy_std": float(final_accuracy_stats.get("std", 0.0)),
        "final_accuracy_min": float(final_accuracy_stats.get("min", 0.0)),
        "forgetting_max": float(forgetting_stats.get("max", 0.0)),
        "bwt_mean": float(bwt_stats.get("mean", 0.0)),
        "primary_score_mean": float(primary_score_stats.get("mean", 0.0)),
        "seed_stable_across_seeds": seed_stable,
        "mechanism_dependency_supported_count": mechanism_count,
        "mechanism_dependency_supported_fraction": mechanism_fraction,
        "mechanism_full_support": mechanism_full,
        "mechanism_chamber_seed_stable": chamber_seed_stable,
        "mechanism_chamber_all_mechanisms_supported_fraction": float(
            chamber_seed_summary.get("all_mechanisms_supported_fraction", 0.0)
        )
        if chamber_seed_summary
        else 0.0,
    }
    entry["status"] = _status_for_benchmark(seed_stable, mechanism_full, chamber_seed_stable)
    entry["attention_required"] = entry["status"] != "ready"
    return entry


def build_benchmark_robustness_report(artifacts_root: Path) -> dict:
    benchmarks = [_benchmark_entry(artifacts_root, spec) for spec in STACK_C_BENCHMARKS]
    status_counts: dict[str, int] = {}
    for item in benchmarks:
        status_counts[item["status"]] = status_counts.get(item["status"], 0) + 1

    summary = {
        "stack": "Stack C",
        "benchmark_count": float(len(benchmarks)),
        "seed_report_count": float(sum(1 for item in benchmarks if item["seed_report_present"])),
        "stable_seed_benchmark_count": float(sum(1 for item in benchmarks if item["seed_stable_across_seeds"])),
        "full_mechanism_support_count": float(sum(1 for item in benchmarks if item["mechanism_full_support"])),
        "mechanism_chamber_seed_stable_count": float(
            sum(1 for item in benchmarks if item["mechanism_chamber_seed_stable"] is True)
        ),
        "ready_benchmark_count": float(sum(1 for item in benchmarks if item["status"] == "ready")),
        "attention_required_count": float(sum(1 for item in benchmarks if item["attention_required"])),
        "status_counts": status_counts,
        "attention_required_benchmarks": [
            {"assay": item["assay"], "label": item["label"], "status": item["status"]}
            for item in benchmarks
            if item["attention_required"]
        ],
    }
    return {"summary": summary, "benchmarks": benchmarks}


def build_benchmark_robustness_markdown(report: dict) -> str:
    summary = report["summary"]
    lines = [
        "# Stack C Benchmark Robustness",
        "",
        f"- benchmark_count: {int(summary['benchmark_count'])}",
        f"- stable_seed_benchmark_count: {int(summary['stable_seed_benchmark_count'])}",
        f"- full_mechanism_support_count: {int(summary['full_mechanism_support_count'])}",
        f"- mechanism_chamber_seed_stable_count: {int(summary['mechanism_chamber_seed_stable_count'])}",
        f"- ready_benchmark_count: {int(summary['ready_benchmark_count'])}",
        f"- attention_required_count: {int(summary['attention_required_count'])}",
        "",
        "| Benchmark | Phase | Seed Stable | Final Acc | Std | Forget Max | Mechanisms | Status |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in report["benchmarks"]:
        mechanism_text = f"{item['mechanism_dependency_supported_count']:.0f}/3"
        if item["mechanism_chamber_seed_stable"] is True:
            mechanism_text += " + chamber"
        lines.append(
            "| {label} | {phase} | {seed_stable} | {acc:.3f} | {std:.3f} | {forget:.3f} | {mechanisms} | {status} |".format(
                label=item["label"],
                phase=item["phase"],
                seed_stable="yes" if item["seed_stable_across_seeds"] else "no",
                acc=item["final_accuracy_mean"],
                std=item["final_accuracy_std"],
                forget=item["forgetting_max"],
                mechanisms=mechanism_text,
                status=item["status"],
            )
        )
    lines.append("")
    if summary["attention_required_benchmarks"]:
        lines.append("## Attention")
        lines.append("")
        for item in summary["attention_required_benchmarks"]:
            lines.append(f"- {item['label']}: {item['status']}")
        lines.append("")
    return "\n".join(lines)


def write_benchmark_robustness_outputs(report: dict, artifacts_root: Path) -> tuple[Path, Path]:
    artifacts_root.mkdir(parents=True, exist_ok=True)
    json_path = artifacts_root / "benchmark_phase_robustness.json"
    markdown_path = artifacts_root / "benchmark_phase_robustness.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path.write_text(build_benchmark_robustness_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()

    artifacts_root = Path(args.artifacts)
    report = build_benchmark_robustness_report(artifacts_root)
    json_path, markdown_path = write_benchmark_robustness_outputs(report, artifacts_root)
    print(json_path)
    print(markdown_path)


if __name__ == "__main__":
    main()
