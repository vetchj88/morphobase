import argparse
import copy
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from morphobase.config.validate import load_config
from scripts.run_assay import execute_assay


DEFAULT_SEEDS = (42, 123, 321, 777, 999)


def primary_score(metrics: dict) -> float:
    return float(
        metrics.get("gridworld_remap_final_success_mean", 0.0)
        - 0.40 * metrics.get("gridworld_remap_mean_forgetting", 0.0)
        + 0.25 * metrics.get("gridworld_remap_bwt", 0.0)
        + 0.20 * metrics.get("gridworld_remap_efficiency_mean", 0.0)
        + 0.08 * metrics.get("gridworld_remap_mean_margin", 0.0)
    )


def biomarker_score(metrics: dict) -> float:
    return float(
        0.55 * metrics.get("mean_energy", 0.0)
        - 0.35 * metrics.get("mean_stress", 0.0)
        + 0.10 * abs(metrics.get("mean_z_alignment", 0.0))
    )


def summarize_seed_runs(seed_runs: list[dict]) -> dict:
    def values(key: str) -> list[float]:
        return [float(item["metrics"].get(key, 0.0)) for item in seed_runs]

    final_success = values("gridworld_remap_final_success_mean")
    forgetting = values("gridworld_remap_mean_forgetting")
    bwt = values("gridworld_remap_bwt")
    efficiency = values("gridworld_remap_efficiency_mean")
    margins = values("gridworld_remap_mean_margin")
    primary_scores = [float(item["primary_score"]) for item in seed_runs]
    biomarker_scores = [float(item["biomarker_score"]) for item in seed_runs]

    def stats(series: list[float]) -> dict[str, float]:
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

    summary = {
        "seed_count": float(len(seed_runs)),
        "final_success": stats(final_success),
        "forgetting": stats(forgetting),
        "bwt": stats(bwt),
        "efficiency": stats(efficiency),
        "margin": stats(margins),
        "primary_score": stats(primary_scores),
        "biomarker_score": stats(biomarker_scores),
    }
    summary["stable_across_seeds"] = bool(
        summary["final_success"]["min"] >= 0.75
        and summary["forgetting"]["max"] <= 0.12
        and summary["final_success"]["std"] <= 0.10
    )
    return summary


def build_seed_markdown(seed_runs: list[dict], summary: dict) -> str:
    lines = [
        "# Gridworld Remap Seed Robustness",
        "",
        f"- seed_count: {int(summary['seed_count'])}",
        f"- final_success_mean: {summary['final_success']['mean']:.4f}",
        f"- final_success_std: {summary['final_success']['std']:.4f}",
        f"- final_success_min: {summary['final_success']['min']:.4f}",
        f"- efficiency_mean: {summary['efficiency']['mean']:.4f}",
        f"- forgetting_max: {summary['forgetting']['max']:.4f}",
        f"- primary_score_mean: {summary['primary_score']['mean']:.4f}",
        f"- stable_across_seeds: {summary['stable_across_seeds']}",
        "",
        "## Runs",
        "",
    ]
    for item in seed_runs:
        metrics = item["metrics"]
        lines.extend(
            [
                f"### {item['run_name']}",
                "",
                f"- seed: {item['seed']}",
                f"- primary_score: {item['primary_score']:.4f}",
                f"- final_success_mean: {metrics.get('gridworld_remap_final_success_mean', 0.0):.4f}",
                f"- mean_forgetting: {metrics.get('gridworld_remap_mean_forgetting', 0.0):.4f}",
                f"- bwt: {metrics.get('gridworld_remap_bwt', 0.0):.4f}",
                f"- efficiency_mean: {metrics.get('gridworld_remap_efficiency_mean', 0.0):.4f}",
                "",
            ]
        )
    return "\n".join(lines)


def run_seed_robustness(config_path: Path, seeds: tuple[int, ...]) -> dict:
    base_cfg = load_config(config_path)
    seed_runs = []
    for seed in seeds:
        cfg = copy.deepcopy(base_cfg)
        cfg.run.seed = int(seed)
        cfg.run.name = f"gridworld_remap_seed_{seed}"
        execution = execute_assay(cfg)
        metrics = execution["result"].final_metrics
        seed_runs.append(
            {
                "run_name": cfg.run.name,
                "seed": seed,
                "artifacts_dir": str(execution["artifacts_dir"]),
                "metrics": metrics,
                "primary_score": primary_score(metrics),
                "biomarker_score": biomarker_score(metrics),
            }
        )

    seed_runs.sort(key=lambda item: (item["primary_score"], item["biomarker_score"]), reverse=True)
    summary = summarize_seed_runs(seed_runs)
    return {"runs": seed_runs, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/assay/gridworld_remap.yaml")
    parser.add_argument("--seeds", nargs="*", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--json-out", default="artifacts/gridworld_remap_seed_robustness.json")
    parser.add_argument("--markdown-out", default="artifacts/gridworld_remap_seed_robustness.md")
    args = parser.parse_args()

    report = run_seed_robustness(Path(args.config), tuple(args.seeds))
    json_path = Path(args.json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    markdown_path = Path(args.markdown_out)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(build_seed_markdown(report["runs"], report["summary"]), encoding="utf-8")

    print(json_path)
    print(markdown_path)


if __name__ == "__main__":
    main()
