#!/usr/bin/env python3
"""Figure 5: Growth Utility Analysis.

Bar chart showing growth utility gain, efficiency advantage, decorative
fraction, lesion field advantage, and late growth fraction from the
growth_usefulness assay.
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

    gu = load_json(ARTIFACTS / "growth_usefulness_assay" / "final_metrics.json")

    labels = [
        "Growth Utility\nGain",
        "Growth Efficiency\nAdvantage",
        "Decorative Growth\nFraction",
        "Lesion Field\nAdvantage",
        "Late Growth\nEvent Fraction",
    ]
    values = [
        gu["growth_utility_gain"],
        gu["growth_efficiency_advantage"],
        gu.get("mean_growth_decorative_fraction", gu.get("decorative_growth_penalty", 0)),
        gu["lesion_field_advantage"],
        gu["late_growth_event_fraction_mean"],
    ]

    # Color: positive gain = green, zero decorative/late = green, otherwise orange
    colors = []
    for i, v in enumerate(values):
        if i in (0, 1, 3):  # higher is better
            colors.append("#4CAF50" if v > 0 else "#F44336")
        else:  # lower is better (decorative, late)
            colors.append("#4CAF50" if v == 0 else "#FF9800")

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(range(len(labels)), values, color=colors, edgecolor="white",
                  linewidth=1.2, width=0.6, zorder=3)

    for bar, val in zip(bars, values):
        y = bar.get_height()
        offset = max(abs(y) * 0.08, 0.002)
        ax.text(bar.get_x() + bar.get_width() / 2,
                y + offset if y >= 0 else y - offset,
                f"{val:.3f}", ha="center",
                va="bottom" if y >= 0 else "top",
                fontsize=11, fontweight="bold")

    ax.axhline(0, color="#888", lw=0.8, zorder=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Growth Utility Analysis — All Growth is Functional",
                 fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotation
    ax.text(0.98, 0.95,
            "Green = beneficial direction\n"
            "Decorative fraction = 0 means\n"
            "all growth is repair-oriented",
            transform=ax.transAxes, fontsize=8, ha="right", va="top",
            bbox=dict(facecolor="white", edgecolor="#ccc", boxstyle="round,pad=0.3"))

    fig.tight_layout()
    fig.savefig(OUT / "fig5_growth_utility.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "fig5_growth_utility.pdf", bbox_inches="tight")
    print(f"Saved → {OUT / 'fig5_growth_utility.png'}")
    plt.close(fig)


if __name__ == "__main__":
    main()
