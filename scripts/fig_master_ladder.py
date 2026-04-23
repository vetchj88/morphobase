#!/usr/bin/env python3
"""Figure 4: Master Ladder Phase Completion Heatmap.

Displays each build phase as a pass/fail heatmap plus the Stack C/D
readiness summary.
"""

import json
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ARTIFACTS = pathlib.Path(__file__).resolve().parent.parent / "artifacts"
OUT = ARTIFACTS / "figures"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    mbr = load_json(ARTIFACTS / "master_build_report.json")
    bpr = load_json(ARTIFACTS / "benchmark_phase_robustness.json")
    sdr = load_json(ARTIFACTS / "stack_d_phase_robustness.json")

    # ── Organism-first phases from master build ─────────────────────
    phase_names = []
    phase_passed = []
    for phase in mbr["phases"]:
        short = phase["name"].replace("phase_", "P").split("_", 1)
        label = short[0] + ": " + short[1].replace("_", " ").title() if len(short) > 1 else short[0]
        phase_names.append(label)
        phase_passed.append(1 if phase["passed"] else 0)

    # ── Stack C benchmarks ──────────────────────────────────────────
    for bm in bpr["benchmarks"]:
        phase_names.append(f"C: {bm['label']}")
        phase_passed.append(1 if bm["status"] == "ready" else 0)

    # ── Stack D benchmarks ──────────────────────────────────────────
    for a in sdr["assays"]:
        phase_names.append(f"D: {a['label']}")
        phase_passed.append(1 if a["status"] == "ready" else 0)

    # ── Plot as horizontal heatmap-style figure ─────────────────────
    n = len(phase_names)
    fig, ax = plt.subplots(figsize=(8, max(5, n * 0.35)))

    colors = ["#F44336" if v == 0 else "#4CAF50" for v in phase_passed]
    y_pos = np.arange(n)[::-1]

    ax.barh(y_pos, [1] * n, color=colors, edgecolor="white",
            linewidth=1.5, height=0.7, zorder=3)

    # Phase labels
    for i, (name, passed) in enumerate(zip(phase_names, phase_passed)):
        status = "PASS" if passed else "FAIL"
        ax.text(-0.02, y_pos[i], name, ha="right", va="center",
                fontsize=9, transform=ax.get_yaxis_transform())
        ax.text(0.5, y_pos[i], status, ha="center", va="center",
                fontsize=10, fontweight="bold", color="white", zorder=4)

    # Separators
    org_count = len(mbr["phases"])
    c_count = len(bpr["benchmarks"])
    if org_count < n:
        sep1 = n - org_count - 0.5
        ax.axhline(sep1, color="#888", ls="--", lw=1)
    if org_count + c_count < n:
        sep2 = n - org_count - c_count - 0.5
        ax.axhline(sep2, color="#888", ls="--", lw=1)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("MorphoBASE v1.3 — Master Build Ladder Status",
                 fontsize=13, fontweight="bold", pad=12)

    green_patch = mpatches.Patch(color="#4CAF50", label="PASS / Ready")
    red_patch = mpatches.Patch(color="#F44336", label="FAIL / Attention")
    ax.legend(handles=[green_patch, red_patch], fontsize=9, loc="lower right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / "fig4_master_ladder.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "fig4_master_ladder.pdf", bbox_inches="tight")
    print(f"Saved → {OUT / 'fig4_master_ladder.png'}")
    plt.close(fig)


if __name__ == "__main__":
    main()
