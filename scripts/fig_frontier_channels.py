#!/usr/bin/env python3
"""Figure 7: Frontier Communication Channel Localization.

Horizontal bar chart showing localization ratios for the five exploratory
communication channels, plus the light cone ablation-supported mechanisms.
"""

import json
import pathlib
import numpy as np
import matplotlib.pyplot as plt

ARTIFACTS = pathlib.Path(__file__).resolve().parent.parent / "artifacts"
OUT = ARTIFACTS / "figures"


# Localization ratios from whitepaper Section 5
FRONTIER_DATA = {
    "Tissue Fields":        3.07,
    "Oscillatory Coupling": 22.57,
    "Reaction-Diffusion":   10.10,
    "Stigmergic Highways":  12.64,
    "Predictive Coding":    21.45,
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    lc = load_json(ARTIFACTS / "lightcone_assay" / "final_metrics.json")

    # ── Panel A: Frontier channel localization ──────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                                    gridspec_kw={"width_ratios": [1.2, 1]})

    channels = list(FRONTIER_DATA.keys())
    ratios = list(FRONTIER_DATA.values())
    y = np.arange(len(channels))

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(channels)))
    ax1.barh(y, ratios, color=colors, edgecolor="white", linewidth=1.0,
             height=0.6, zorder=3)

    for i, v in enumerate(ratios):
        ax1.text(v + 0.3, i, f"{v:.2f}x", va="center", fontsize=10,
                 fontweight="bold")

    ax1.set_yticks(y)
    ax1.set_yticklabels(channels, fontsize=10)
    ax1.set_xlabel("Localization Ratio", fontsize=11)
    ax1.set_title("Frontier Channel Localization", fontsize=12,
                  fontweight="bold")
    ax1.axvline(1.0, color="#888", ls="--", lw=0.8, zorder=1)
    ax1.grid(axis="x", alpha=0.3, zorder=0)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ── Panel B: Light cone ablation support ────────────────────────
    ablation_names = ["Stress Sharing", "Conductance", "Z-Memory"]
    ablation_scores = [
        lc["stress_sharing_off_ablation_support_score"],
        lc["conductance_ablated_ablation_support_score"],
        lc["z_memory_ablated_ablation_support_score"],
    ]
    effect_deltas = [
        lc["stress_sharing_off_effect_total_delta"],
        lc["conductance_ablated_effect_total_delta"],
        lc["z_memory_ablated_effect_total_delta"],
    ]

    y2 = np.arange(len(ablation_names))
    ax2.barh(y2, effect_deltas, color=["#2196F3", "#FF9800", "#9C27B0"],
             edgecolor="white", linewidth=1.0, height=0.5, zorder=3)

    for i, (v, s) in enumerate(zip(effect_deltas, ablation_scores)):
        ax2.text(v + 10, i, f"{v:.0f} (score: {s:.0f})",
                 va="center", fontsize=10, fontweight="bold")

    ax2.set_yticks(y2)
    ax2.set_yticklabels(ablation_names, fontsize=10)
    ax2.set_xlabel("Effect Total Delta (ablated vs. intact)", fontsize=10)
    ax2.set_title("Light Cone Ablation Support", fontsize=12, fontweight="bold")
    ax2.grid(axis="x", alpha=0.3, zorder=0)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("MorphoBASE v1.3 — Communication Architecture Evidence",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig7_frontier_channels.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "fig7_frontier_channels.pdf", bbox_inches="tight")
    print(f"Saved → {OUT / 'fig7_frontier_channels.png'}")
    plt.close(fig)


if __name__ == "__main__":
    main()
