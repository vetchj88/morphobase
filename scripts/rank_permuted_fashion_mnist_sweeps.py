import argparse
import json
from pathlib import Path

import yaml


def _load_run_payload(run_dir: Path) -> dict | None:
    metrics_path = run_dir / "final_metrics.json"
    config_path = run_dir / "resolved_config.yaml"
    if not metrics_path.exists() or not config_path.exists():
        return None

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    assay_name = config.get("assay", {}).get("name")
    if assay_name != "permuted_fashion_mnist":
        return None

    run_name = config.get("run", {}).get("name", run_dir.name)
    seed = config.get("run", {}).get("seed", "")
    return {
        "run_dir": str(run_dir),
        "run_name": run_name,
        "seed": seed,
        "metrics": metrics,
    }


def primary_score(metrics: dict) -> float:
    return float(
        1.00 * metrics.get("permuted_fashion_mnist_final_accuracy_mean", 0.0)
        + 0.35 * metrics.get("permuted_fashion_mnist_peak_accuracy_mean", 0.0)
        - 0.85 * metrics.get("permuted_fashion_mnist_mean_forgetting", 0.0)
        + 0.35 * metrics.get("permuted_fashion_mnist_bwt", 0.0)
        + 0.10 * metrics.get("permuted_fashion_mnist_mean_margin", 0.0)
    )


def biomarker_score(metrics: dict) -> float:
    return float(
        0.40 * metrics.get("mean_energy", 0.0)
        - 0.25 * metrics.get("mean_stress", 0.0)
        + 0.15 * metrics.get("mean_plasticity", 0.0)
        + 0.10 * metrics.get("mean_z_alignment", 0.0)
    )


def rank_runs(artifacts_root: Path) -> list[dict]:
    candidates = []
    for run_dir in sorted(path for path in artifacts_root.iterdir() if path.is_dir()):
        payload = _load_run_payload(run_dir)
        if payload is None:
            continue
        metrics = payload["metrics"]
        payload["primary_score"] = primary_score(metrics)
        payload["biomarker_score"] = biomarker_score(metrics)
        candidates.append(payload)

    candidates.sort(key=lambda item: (item["primary_score"], item["biomarker_score"]), reverse=True)
    return candidates


def build_markdown_report(ranked_runs: list[dict]) -> str:
    lines = [
        "# Permuted-FashionMNIST Sweep Ranking",
        "",
        "Primary metrics are ranked first; biomarkers are only secondary tie-breakers.",
        "",
    ]
    if not ranked_runs:
        lines.append("No Permuted-FashionMNIST runs were found.")
        return "\n".join(lines) + "\n"

    for index, item in enumerate(ranked_runs, start=1):
        metrics = item["metrics"]
        lines.extend(
            [
                f"## {index}. {item['run_name']}",
                "",
                f"- seed: {item['seed']}",
                f"- primary_score: {item['primary_score']:.4f}",
                f"- biomarker_score: {item['biomarker_score']:.4f}",
                f"- final_accuracy_mean: {metrics.get('permuted_fashion_mnist_final_accuracy_mean', 0.0):.4f}",
                f"- mean_forgetting: {metrics.get('permuted_fashion_mnist_mean_forgetting', 0.0):.4f}",
                f"- bwt: {metrics.get('permuted_fashion_mnist_bwt', 0.0):.4f}",
                f"- mean_margin: {metrics.get('permuted_fashion_mnist_mean_margin', 0.0):.4f}",
                f"- mean_energy: {metrics.get('mean_energy', 0.0):.4f}",
                f"- mean_stress: {metrics.get('mean_stress', 0.0):.4f}",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--json-out", default="artifacts/permuted_fashion_mnist_sweep_ranking.json")
    parser.add_argument("--markdown-out", default="artifacts/permuted_fashion_mnist_sweep_ranking.md")
    args = parser.parse_args()

    ranked = rank_runs(Path(args.artifacts))
    json_path = Path(args.json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(ranked, indent=2), encoding="utf-8")

    markdown_path = Path(args.markdown_out)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(build_markdown_report(ranked), encoding="utf-8")

    print(json_path)
    print(markdown_path)


if __name__ == "__main__":
    main()
