#!/usr/bin/env python3
"""Figure 2: Per-Lesion-Type Recovery Comparison.

Grouped bar chart showing morphological recovery, no-gradient recovery,
and retraining recovery for each lesion type from the lesion_battery assay.
"""

import json
import pathlib
import numpy as np
import matplotlib.pyplot as plt

ARTIFACTS = pathlib.Path(__file__).resolve().parent.parent / "artifacts"
OUT = ARTIFACTS / "figures"

LESION_TYPES = [
    ("cell_ablation",            "Cell\nAblation"),
    ("conductance_severance",    "Conductance\nSeverance"),
    ("field_corruption",         "Field\nCorruption"),
    ("parameter_corruption",     "Parameter\nCorruption"),
    ("z_field_corruption",       "Z-Field\nCorruption"),
    ("targeted_tissue_ablation", "Targeted\nTissue Abl."),
    ("port_disruption",          "Port\nDisruption"),
    ("whole_body_port_disruption", "Whole-Body\nPort Disr."),
]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lb = load_json(ARTIFACTS / "lesion_battery_assay" / "final_metrics.json")

    labels = []
    org_recovery = []
    nogr_recovery = []
    retr_recovery = []

    for key, label in LESION_TYPES:
        rec = lb.get(f"{key}_recovery")
        if rec is None:
            continue
        labels.append(label)
        org_recovery.append(rec * 100)
        # no-gradient recovery may not exist for all types
        ngr = lb.get(f"{key}_no_gradient_recovery")
        nogr_recovery.append(ngr * 100 if ngr is not None else 0)
        rtr = lb.get(f"{key}_retraining_recovery")
        retr_recovery.append(rtr * 100 if rtr is not None else 0)

    x = np.arange(len(labels))
    w = 0.25

    fig, ax = plt.subplots(figsize=(12, 5.5))

    b1 = ax.bar(x - w, org_recovery, w, label="Organismal Recovery",
                color="#2196F3", edgecolor="white", linewidth=0.8, zorder=3)
    b2 = ax.bar(x, nogr_recovery, w, label="No-Gradient Recovery",
                color="#FF9800", edgecolor="white", linewidth=0.8, zorder=3)
    b3 = ax.bar(x + w, retr_recovery, w, label="Retraining Recovery",
                color="#4CAF50", edgecolor="white", linewidth=0.8, zorder=3)

    # Value labels
    for bars in [b1, b2, b3]:
        for bar in bars:
            y = bar.get_height()
            if y > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, y + 1,
                        f"{y:.0f}%", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Recovery (%)", fontsize=12)
    ax.set_title("Lesion Recovery by Type — Organism vs. Gradient-Free vs. Retraining",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(0, 115)
    ax.axhline(100, color="#888", ls="--", lw=0.7, zorder=1)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / "fig2_lesion_recovery_by_type.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "fig2_lesion_recovery_by_type.pdf", bbox_inches="tight")
    print(f"Saved → {OUT / 'fig2_lesion_recovery_by_type.png'}")
    plt.close(fig)


if __name__ == "__main__":
    main()
