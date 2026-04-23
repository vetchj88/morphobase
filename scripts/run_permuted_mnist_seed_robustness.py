import argparse
import copy
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from morphobase.config.validate import load_config
from scripts.rank_permuted_mnist_sweeps import biomarker_score, build_markdown_report, primary_score
from scripts.run_assay import execute_assay


DEFAULT_SEEDS = (42, 123, 321, 777, 999)


def summarize_seed_runs(seed_runs: list[dict]) -> dict:
    def values(key: str) -> list[float]:
        return [float(item["metrics"].get(key, 0.0)) for item in seed_runs]

    final_acc = values("permuted_mnist_final_accuracy_mean")
    forgetting = values("permuted_mnist_mean_forgetting")
    bwt = values("permuted_mnist_bwt")
    margins = values("permuted_mnist_mean_margin")
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
        "final_accuracy": stats(final_acc),
        "forgetting": stats(forgetting),
        "bwt": stats(bwt),
        "margin": stats(margins),
        "primary_score": stats(primary_scores),
        "biomarker_score": stats(biomarker_scores),
    }
    summary["stable_across_seeds"] = bool(
        summary["final_accuracy"]["min"] >= 0.24
        and summary["forgetting"]["max"] <= 0.12
        and summary["final_accuracy"]["std"] <= 0.08
    )
    return summary


def build_seed_markdown(seed_runs: list[dict], summary: dict) -> str:
    lines = [
        "# Permuted-MNIST Seed Robustness",
        "",
        f"- seed_count: {int(summary['seed_count'])}",
        f"- final_accuracy_mean: {summary['final_accuracy']['mean']:.4f}",
        f"- final_accuracy_std: {summary['final_accuracy']['std']:.4f}",
        f"- final_accuracy_min: {summary['final_accuracy']['min']:.4f}",
        f"- final_accuracy_max: {summary['final_accuracy']['max']:.4f}",
        f"- forgetting_mean: {summary['forgetting']['mean']:.4f}",
        f"- forgetting_max: {summary['forgetting']['max']:.4f}",
        f"- bwt_mean: {summary['bwt']['mean']:.4f}",
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
                f"- final_accuracy_mean: {metrics.get('permuted_mnist_final_accuracy_mean', 0.0):.4f}",
                f"- mean_forgetting: {metrics.get('permuted_mnist_mean_forgetting', 0.0):.4f}",
                f"- bwt: {metrics.get('permuted_mnist_bwt', 0.0):.4f}",
                f"- mean_margin: {metrics.get('permuted_mnist_mean_margin', 0.0):.4f}",
                "",
            ]
        )
    lines.extend(["## Ranking", "", build_markdown_report(seed_runs).strip(), ""])
    return "\n".join(lines)


def run_seed_robustness(config_path: Path, seeds: tuple[int, ...]) -> dict:
    base_cfg = load_config(config_path)
    seed_runs = []
    for seed in seeds:
        cfg = copy.deepcopy(base_cfg)
        cfg.run.seed = int(seed)
        cfg.run.name = f"permuted_mnist_seed_{seed}"
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
    parser.add_argument("--config", default="configs/assay/permuted_mnist.yaml")
    parser.add_argument("--seeds", nargs="*", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--json-out", default="artifacts/permuted_mnist_seed_robustness.json")
    parser.add_argument("--markdown-out", default="artifacts/permuted_mnist_seed_robustness.md")
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
