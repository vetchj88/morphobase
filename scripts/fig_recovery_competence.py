#!/usr/bin/env python3
"""Figure 1: Recovery and Competence Metrics (Grouped Bar Chart).

Plots the headline organismal properties from the lesion_battery and
lesion_preserves_competence assays.
"""

import json
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ARTIFACTS = pathlib.Path(__file__).resolve().parent.parent / "artifacts"
OUT = ARTIFACTS / "figures"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    lb = load_json(ARTIFACTS / "lesion_battery_assay" / "final_metrics.json")
    lpc = load_json(ARTIFACTS / "lesion_preserves_competence_assay" / "final_metrics.json")

    # ── Data ────────────────────────────────────────────────────────
    labels = [
        "Morphological\nRecovery",
        "Recovery vs.\nRetraining",
        "Competence\nRetention",
        "Gradient-Free\nRecovery Ratio",
        "Hidden-Memory\nGap Advantage",
    ]
    values = [
        lb["lesion_recovery_mean"],                        # 0.8495
        lb["organismal_recovery_vs_retraining_ratio"],     # 0.7664
        lpc["competence_retention_ratio"],                 # 0.9894
        lb["recovery_retention_without_gradients"],        # 1.1684
        lpc.get("organismal_competence_advantage_vs_no_gradient",
                load_json(ARTIFACTS / "setpoint_rewrite_assay" / "final_metrics.json")
                ["hidden_z_memory_gap_advantage"]),        # 0.40
    ]

    # Scale: first three are percentages, last two are ratios
    display = [v * 100 if i < 3 else v for i, v in enumerate(values)]
    units = ["%", "%", "%", "x", ""]

    # ── Plot ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5.5))

    colors = ["#2196F3", "#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    bars = ax.bar(range(len(labels)), display, color=colors, edgecolor="white",
                  linewidth=1.2, width=0.65, zorder=3)

    # Value labels on top of each bar
    for bar, val, unit in zip(bars, display, units):
        y = bar.get_height()
        fmt = f"{y:.1f}{unit}" if unit == "%" else f"{y:.2f}{unit}"
        ax.text(bar.get_x() + bar.get_width() / 2, y + 1.5, fmt,
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Reference lines
    ax.axhline(100, color="#888", ls="--", lw=0.8, zorder=1)
    ax.axhline(1.0, color="#888", ls=":", lw=0.6, zorder=1)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("MorphoBASE v1.3 — Organismal Recovery & Competence",
                 fontsize=14, fontweight="bold", pad=12)

    # Y-axis formatting: dual meaning (% and ratio share same axis visually)
    ax.set_ylim(0, max(display) * 1.18)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(8))
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotation explaining the dual scale
    ax.text(0.98, 0.95,
            "Blue/Green bars: percentage scale\nOrange/Purple bars: ratio scale",
            transform=ax.transAxes, fontsize=8, ha="right", va="top",
            bbox=dict(facecolor="white", edgecolor="#ccc", boxstyle="round,pad=0.3"))

    fig.tight_layout()
    fig.savefig(OUT / "fig1_recovery_competence.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "fig1_recovery_competence.pdf", bbox_inches="tight")
    print(f"Saved → {OUT / 'fig1_recovery_competence.png'}")
    plt.close(fig)


if __name__ == "__main__":
    main()
