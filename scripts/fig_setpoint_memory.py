#!/usr/bin/env python3
"""Figure 8 (bonus): Setpoint Memory & Cryptic Rewrite Evidence.

Shows rewrite persistence and hidden-memory gap across three rewrite
modes (stress_bias, z_bias, conductance_bias), plus cycle-by-cycle
persistence decay.
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

    sp = load_json(ARTIFACTS / "setpoint_rewrite_assay" / "final_metrics.json")

    modes = ["stress_bias", "z_bias", "conductance_bias"]
    mode_labels = ["Stress Bias", "Z-Field Bias", "Conductance Bias"]

    # ── Panel A: Per-mode persistence & hidden-memory gap ───────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    persistence_means = [sp[f"{m}_rewrite_persistence_mean"] for m in modes]
    hmg = [sp[f"{m}_hidden_z_memory_gap_advantage"] for m in modes]

    x = np.arange(len(modes))
    w = 0.35

    ax1.bar(x - w / 2, persistence_means, w, label="Rewrite Persistence",
            color="#2196F3", edgecolor="white", linewidth=1.0, zorder=3)
    ax1.bar(x + w / 2, hmg, w, label="Hidden-Memory Gap",
            color="#9C27B0", edgecolor="white", linewidth=1.0, zorder=3)

    for i, (p, h) in enumerate(zip(persistence_means, hmg)):
        ax1.text(i - w / 2, p + 0.01, f"{p:.3f}", ha="center", fontsize=9,
                 fontweight="bold")
        ax1.text(i + w / 2, h + 0.01, f"{h:.3f}", ha="center", fontsize=9,
                 fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(mode_labels, fontsize=10)
    ax1.set_ylabel("Score", fontsize=11)
    ax1.set_title("Rewrite Modes: Persistence & Hidden Memory",
                  fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3, zorder=0)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ── Panel B: Cycle-by-cycle persistence decay ───────────────────
    cycles = [1, 2, 3]
    for mode, label, color in zip(modes, mode_labels,
                                   ["#2196F3", "#FF9800", "#9C27B0"]):
        persis = [sp[f"{mode}_cycle_{c}_persistence"] for c in cycles]
        falsif = [sp[f"{mode}_cycle_{c}_falsification_margin"] for c in cycles]
        ax2.plot(cycles, persis, "o-", color=color, label=f"{label} (persistence)",
                 linewidth=2, markersize=7, zorder=3)
        ax2.plot(cycles, falsif, "s--", color=color, alpha=0.5,
                 label=f"{label} (falsif. margin)", linewidth=1.5,
                 markersize=5, zorder=3)

    ax2.set_xlabel("Rewrite Cycle", fontsize=11)
    ax2.set_ylabel("Score", fontsize=11)
    ax2.set_title("Cycle-by-Cycle Persistence Decay",
                  fontsize=12, fontweight="bold")
    ax2.set_xticks(cycles)
    ax2.legend(fontsize=7.5, ncol=2, loc="upper right")
    ax2.grid(alpha=0.3, zorder=0)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("Setpoint Memory — Cryptic Rewrite Evidence",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig8_setpoint_memory.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "fig8_setpoint_memory.pdf", bbox_inches="tight")
    print(f"Saved → {OUT / 'fig8_setpoint_memory.png'}")
    plt.close(fig)


if __name__ == "__main__":
    main()
