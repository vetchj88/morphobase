#!/usr/bin/env python3
"""Figure 6: Organism vs. Baseline Comparisons.

Grouped bar chart comparing organism (baseline + ablation conditions)
against transformer and MLP baselines across lesion-aware and standard
task bridges.
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

    ovt = load_json(ARTIFACTS / "single_organism_vs_transformer.json")

    task_labels = []
    org_scores = []
    org_nog = []
    org_noz = []
    trans_scores = []
    mlp_scores = []

    for task in ovt["tasks"]:
        task_labels.append(task["label"])
        org_scores.append(task["competitors"]["organism"]["primary_metric"] * 100)
        org_nog.append(task["competitors"]["organism_no_growth"]["primary_metric"] * 100)
        org_noz.append(task["competitors"]["organism_no_z_field"]["primary_metric"] * 100)
        trans_scores.append(task["competitors"]["transformer"]["primary_metric"] * 100)
        mlp_scores.append(task["competitors"]["mlp"]["primary_metric"] * 100)

    # Also include standalone seq rules comparison if available
    sr_path = ARTIFACTS / "single_organism_vs_transformer_seqrules.json"
    if sr_path.exists():
        sr = load_json(sr_path)
        for task in sr["tasks"]:
            task_labels.append(task["label"])
            org_scores.append(task["organism"]["primary_metric"] * 100)
            org_nog.append(0)  # not available
            org_noz.append(0)
            trans_scores.append(task["transformer"]["primary_metric"] * 100)
            mlp_scores.append(0)

    n = len(task_labels)
    x = np.arange(n)
    w = 0.15

    fig, ax = plt.subplots(figsize=(13, 5.5))

    ax.bar(x - 2 * w, org_scores, w, label="Organism",
           color="#2196F3", edgecolor="white", linewidth=0.8, zorder=3)
    ax.bar(x - w, org_nog, w, label="Organism (No Growth)",
           color="#90CAF9", edgecolor="white", linewidth=0.8, zorder=3)
    ax.bar(x, org_noz, w, label="Organism (No Z-Field)",
           color="#BBDEFB", edgecolor="white", linewidth=0.8, zorder=3)
    ax.bar(x + w, trans_scores, w, label="Transformer",
           color="#FF9800", edgecolor="white", linewidth=0.8, zorder=3)
    ax.bar(x + 2 * w, mlp_scores, w, label="MLP",
           color="#F44336", edgecolor="white", linewidth=0.8, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, fontsize=8.5, rotation=15, ha="right")
    ax.set_ylabel("Accuracy / Success Rate (%)", fontsize=11)
    ax.set_title("Organism vs. Baseline Comparisons (incl. Ablation Conditions)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=8, ncol=5, loc="upper center",
              bbox_to_anchor=(0.5, -0.15))
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotation for context
    ax.text(0.98, 0.95,
            "Lesion-aware tasks test competence\n"
            "retention after organismal injury.\n"
            "Organism built for repair, not raw score.",
            transform=ax.transAxes, fontsize=8, ha="right", va="top",
            bbox=dict(facecolor="white", edgecolor="#ccc", boxstyle="round,pad=0.3"))

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    fig.savefig(OUT / "fig6_organism_vs_baselines.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "fig6_organism_vs_baselines.pdf", bbox_inches="tight")
    print(f"Saved → {OUT / 'fig6_organism_vs_baselines.png'}")
    plt.close(fig)


if __name__ == "__main__":
    main()
