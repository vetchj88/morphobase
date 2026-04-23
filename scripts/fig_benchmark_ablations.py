#!/usr/bin/env python3
"""Figure 3: Benchmark Performance with Ablation Controls.

Grouped bar chart for all 6 benchmarks (Stack C + Stack D) showing
baseline, no_growth, no_stress, and no_z_field accuracy with error bars
from seed robustness reports.
"""

import json
import pathlib
import numpy as np
import matplotlib.pyplot as plt

ARTIFACTS = pathlib.Path(__file__).resolve().parent.parent / "artifacts"
OUT = ARTIFACTS / "figures"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # ── Gather data from seed robustness + ablation JSONs ───────────
    benchmarks = []

    # Stack C visual benchmarks
    for name, prefix, metric_key in [
        ("Split-MNIST",           "split_mnist",           "split_mnist_final_accuracy_mean"),
        ("Permuted-MNIST",        "permuted_mnist",        "permuted_mnist_final_accuracy_mean"),
        ("Split-FashionMNIST",    "split_fashion_mnist",   "split_fashion_mnist_final_accuracy_mean"),
        ("Permuted-FashionMNIST", "permuted_fashion_mnist","permuted_fashion_mnist_final_accuracy_mean"),
    ]:
        seed_path = ARTIFACTS / f"{prefix}_seed_robustness.json"
        abl_path = ARTIFACTS / f"{prefix}_ablations.json"
        if not seed_path.exists() or not abl_path.exists():
            continue
        seed = load_json(seed_path)
        abl = load_json(abl_path)

        baseline_mean = seed["summary"]["final_accuracy"]["mean"] * 100
        baseline_std = seed["summary"]["final_accuracy"]["std"] * 100

        # Ablation conditions (single-seed from ablation report)
        conditions = {}
        for run in abl["runs"]:
            cond = run["condition"]
            acc = run["metrics"].get(metric_key, 0) * 100
            conditions[cond] = acc

        benchmarks.append({
            "name": name,
            "baseline_mean": baseline_mean,
            "baseline_std": baseline_std,
            "no_growth": conditions.get("no_growth", 0),
            "no_stress": conditions.get("no_stress", 0),
            "no_z_field": conditions.get("no_z_field", 0),
        })

    # Stack D non-visual benchmarks
    for name, prefix, metric_key in [
        ("Gridworld\nRemap", "gridworld_remap", "gridworld_remap_final_success_mean"),
        ("Sequential\nRules", "sequential_rules", "sequential_rules_final_accuracy_mean"),
    ]:
        seed_path = ARTIFACTS / f"{prefix}_seed_robustness.json"
        abl_path = ARTIFACTS / f"{prefix}_ablations.json"
        if not seed_path.exists() or not abl_path.exists():
            continue
        seed = load_json(seed_path)
        abl = load_json(abl_path)

        # Stack D seed reports use different key names
        summary = seed["summary"]
        if "final_accuracy" in summary:
            baseline_mean = summary["final_accuracy"]["mean"] * 100
            baseline_std = summary["final_accuracy"]["std"] * 100
        elif "final_success" in summary:
            baseline_mean = summary["final_success"]["mean"] * 100
            baseline_std = summary["final_success"]["std"] * 100
        else:
            # Try to compute from runs
            vals = [r["metrics"].get(metric_key, 0) for r in seed["runs"]]
            baseline_mean = np.mean(vals) * 100
            baseline_std = np.std(vals) * 100

        conditions = {}
        for run in abl["runs"]:
            cond = run["condition"]
            acc = run["metrics"].get(metric_key, 0) * 100
            conditions[cond] = acc

        benchmarks.append({
            "name": name,
            "baseline_mean": baseline_mean,
            "baseline_std": baseline_std,
            "no_growth": conditions.get("no_growth", 0),
            "no_stress": conditions.get("no_stress", 0),
            "no_z_field": conditions.get("no_z_field", 0),
        })

    # ── Plot ────────────────────────────────────────────────────────
    n = len(benchmarks)
    x = np.arange(n)
    w = 0.19

    fig, ax = plt.subplots(figsize=(13, 5.5))

    colors = {
        "Baseline":   "#2196F3",
        "No Growth":  "#FF9800",
        "No Stress":  "#F44336",
        "No Z-Field": "#9C27B0",
    }

    ax.bar(x - 1.5 * w,
           [b["baseline_mean"] for b in benchmarks], w,
           yerr=[b["baseline_std"] for b in benchmarks],
           label="Baseline (5 seeds)", color=colors["Baseline"],
           edgecolor="white", linewidth=0.8, capsize=3, zorder=3)
    ax.bar(x - 0.5 * w,
           [b["no_growth"] for b in benchmarks], w,
           label="No Growth", color=colors["No Growth"],
           edgecolor="white", linewidth=0.8, zorder=3)
    ax.bar(x + 0.5 * w,
           [b["no_stress"] for b in benchmarks], w,
           label="No Stress", color=colors["No Stress"],
           edgecolor="white", linewidth=0.8, zorder=3)
    ax.bar(x + 1.5 * w,
           [b["no_z_field"] for b in benchmarks], w,
           label="No Z-Field", color=colors["No Z-Field"],
           edgecolor="white", linewidth=0.8, zorder=3)

    # Chance line for 10-class tasks
    ax.axhline(10, color="#aaa", ls=":", lw=0.8, zorder=1, label="Chance (10%)")

    # Separator between Stack C and Stack D
    if n > 4:
        ax.axvline(3.5, color="#ccc", ls="--", lw=1.0, zorder=1)
        ax.text(1.5, ax.get_ylim()[1] * 0.95, "Stack C (Visual)",
                ha="center", fontsize=9, color="#666")
        ax.text(4.5 if n > 5 else 4, ax.get_ylim()[1] * 0.95,
                "Stack D (Non-Visual)",
                ha="center", fontsize=9, color="#666")

    ax.set_xticks(x)
    ax.set_xticklabels([b["name"] for b in benchmarks], fontsize=9)
    ax.set_ylabel("Accuracy / Success Rate (%)", fontsize=11)
    ax.set_title("Benchmark Performance with Ablation Controls",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=8.5, ncol=5, loc="upper center",
              bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(OUT / "fig3_benchmark_ablations.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "fig3_benchmark_ablations.pdf", bbox_inches="tight")
    print(f"Saved → {OUT / 'fig3_benchmark_ablations.png'}")
    plt.close(fig)


if __name__ == "__main__":
    main()
