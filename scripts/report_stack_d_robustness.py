import argparse
import json
from pathlib import Path


STACK_D_BENCHMARKS = (
    {
        "assay": "gridworld_remap",
        "phase": "phase_9_gridworld_control_bridge",
        "label": "Gridworld Remap",
        "seed_report": "gridworld_remap_seed_robustness.json",
        "ablation_report": "gridworld_remap_ablations.json",
        "primary_summary_key": "final_success",
    },
    {
        "assay": "sequential_rules",
        "phase": "phase_10_symbolic_rules_bridge",
        "label": "Sequential Rules",
        "seed_report": "sequential_rules_seed_robustness.json",
        "ablation_report": "sequential_rules_ablations.json",
        "primary_summary_key": "final_accuracy",
    },
)


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _status_for_assay(seed_stable: bool, mechanism_full: bool) -> str:
    if not seed_stable:
        return "seed_unstable"
    if not mechanism_full:
        return "mechanism_incomplete"
    return "ready"


def _entry(artifacts_root: Path, spec: dict) -> dict:
    seed_path = artifacts_root / spec["seed_report"]
    ablation_path = artifacts_root / spec["ablation_report"]
    seed_payload = _load_json(seed_path)
    ablation_payload = _load_json(ablation_path)
    seed_summary = (seed_payload or {}).get("summary", {})
    ablation_summary = (ablation_payload or {}).get("summary", {})
    primary_stats = seed_summary.get(spec["primary_summary_key"], {})
    forgetting_stats = seed_summary.get("forgetting", {})
    primary_mean_key = "final_success_mean" if spec["primary_summary_key"] == "final_success" else "final_accuracy_mean"

    seed_stable = bool(seed_summary.get("stable_across_seeds", False))
    mechanism_count = float(ablation_summary.get("mechanism_dependency_supported_count", 0.0))
    mechanism_fraction = float(ablation_summary.get("mechanism_dependency_supported_fraction", 0.0))
    mechanism_full = mechanism_count >= 2.0 or mechanism_fraction >= 1.0

    entry = {
        "assay": spec["assay"],
        "label": spec["label"],
        "phase": spec["phase"],
        "seed_report_present": seed_payload is not None,
        "ablation_report_present": ablation_payload is not None,
        "seed_report_path": str(seed_path),
        "ablation_report_path": str(ablation_path),
        "seed_count": float(seed_summary.get("seed_count", 0.0)),
        primary_mean_key: float(primary_stats.get("mean", 0.0)),
        f"{primary_mean_key.replace('_mean', '_std')}": float(primary_stats.get("std", 0.0)),
        f"{primary_mean_key.replace('_mean', '_min')}": float(primary_stats.get("min", 0.0)),
        "forgetting_max": float(forgetting_stats.get("max", 0.0)),
        "mechanism_dependency_supported_count": mechanism_count,
        "mechanism_dependency_supported_fraction": mechanism_fraction,
        "seed_stable_across_seeds": seed_stable,
        "mechanism_full_support": mechanism_full,
    }
    entry["status"] = _status_for_assay(seed_stable, mechanism_full)
    entry["attention_required"] = entry["status"] != "ready"
    return entry


def build_stack_d_robustness_report(artifacts_root: Path) -> dict:
    assays = [_entry(artifacts_root, spec) for spec in STACK_D_BENCHMARKS]
    status_counts: dict[str, int] = {}
    for item in assays:
        status_counts[item["status"]] = status_counts.get(item["status"], 0) + 1
    summary = {
        "stack": "Stack D",
        "assay_count": float(len(assays)),
        "stable_seed_assay_count": float(sum(1 for item in assays if item["seed_stable_across_seeds"])),
        "full_mechanism_support_count": float(sum(1 for item in assays if item["mechanism_full_support"])),
        "ready_assay_count": float(sum(1 for item in assays if item["status"] == "ready")),
        "attention_required_count": float(sum(1 for item in assays if item["attention_required"])),
        "status_counts": status_counts,
        "attention_required_assays": [
            {"assay": item["assay"], "label": item["label"], "status": item["status"]}
            for item in assays
            if item["attention_required"]
        ],
    }
    return {"summary": summary, "assays": assays}


def build_stack_d_robustness_markdown(report: dict) -> str:
    summary = report["summary"]
    lines = [
        "# Stack D Robustness",
        "",
        f"- assay_count: {int(summary['assay_count'])}",
        f"- stable_seed_assay_count: {int(summary['stable_seed_assay_count'])}",
        f"- full_mechanism_support_count: {int(summary['full_mechanism_support_count'])}",
        f"- ready_assay_count: {int(summary['ready_assay_count'])}",
        f"- attention_required_count: {int(summary['attention_required_count'])}",
        "",
        "| Assay | Phase | Seed Stable | Primary Mean | Std | Forget Max | Mechanisms | Status |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in report["assays"]:
        primary_mean = item.get("final_accuracy_mean", item.get("final_success_mean", 0.0))
        primary_std = item.get("final_accuracy_std", item.get("final_success_std", 0.0))
        lines.append(
            "| {label} | {phase} | {seed_stable} | {primary:.3f} | {std:.3f} | {forget:.3f} | {mechanisms:.0f}/2 | {status} |".format(
                label=item["label"],
                phase=item["phase"],
                seed_stable="yes" if item["seed_stable_across_seeds"] else "no",
                primary=primary_mean,
                std=primary_std,
                forget=item["forgetting_max"],
                mechanisms=item["mechanism_dependency_supported_count"],
                status=item["status"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_stack_d_robustness_outputs(report: dict, artifacts_root: Path) -> tuple[Path, Path]:
    artifacts_root.mkdir(parents=True, exist_ok=True)
    json_path = artifacts_root / "stack_d_phase_robustness.json"
    markdown_path = artifacts_root / "stack_d_phase_robustness.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path.write_text(build_stack_d_robustness_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()

    artifacts_root = Path(args.artifacts)
    report = build_stack_d_robustness_report(artifacts_root)
    json_path, markdown_path = write_stack_d_robustness_outputs(report, artifacts_root)
    print(json_path)
    print(markdown_path)


if __name__ == "__main__":
    main()
