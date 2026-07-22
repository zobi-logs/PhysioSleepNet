# =========================================================================
# figure_06_confusion.py
#
# Figure 1 — V8 confusion matrices (pooled across folds, row-normalized).
# Two side-by-side panels:
#     left:  MESA, summed CM across the 5 seed-42 V8 folds
#     right: CFS,  same
# Each row sums to 100% (per-true-class normalization), so the diagonal
# is per-class recall.
#
# Only seed 42 is pooled, matching the single-seed protocol used for
# every baseline in the paper.
#
# Output: _paper_results/figures/figure_06_confusion.pdf (300 DPI)
# =========================================================================

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from figstyle import apply_style, STAGE_ORDER, PALETTE, save

apply_style()

ROOT    = Path("/data2/Akbar1/PPG_Stages/benchmark_results")
OUT_DIR = ROOT / "_paper_results" / "figures"
SEED    = 42          # single-seed protocol; see Section III-E


def pooled_cm(cohort, seed=SEED):
    """Sum confusion matrices across the 5 folds of one seed."""
    cm = np.zeros((4, 4), dtype=np.int64)
    n_folds = 0
    for d in sorted((ROOT / cohort / "v8").glob(f"seed{seed}_fold*")):
        if not (d / "DONE").exists():
            continue
        with open(d / "metrics.json") as f:
            m = json.load(f)
        if "cm" in m and m["cm"] is not None:
            cm += np.array(m["cm"], dtype=np.int64)
            n_folds += 1
    if n_folds != 5:
        print(f"  WARNING {cohort}: pooled {n_folds} folds, expected 5")
    return cm, n_folds


# Sequential colormap white -> navy for academic look
cmap = LinearSegmentedColormap.from_list(
    "navy_seq", ["#ffffff", PALETTE["teal"], PALETTE["navy"]])


fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
im = None

for ax, cohort in zip(axes, ["mesa", "cfs"]):
    cm, n_folds = pooled_cm(cohort)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = (cm / np.maximum(row_sums, 1)) * 100.0

    im = ax.imshow(cm_pct, cmap=cmap, vmin=0, vmax=100, aspect="equal")

    for i in range(4):
        for j in range(4):
            v = cm_pct[i, j]
            color = "white" if v > 55 else PALETTE["ink"]
            weight = "bold" if i == j else "normal"
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    fontsize=8, color=color, fontweight=weight)

    ax.set_xticks(range(4)); ax.set_xticklabels(STAGE_ORDER, fontsize=8)
    ax.set_yticks(range(4)); ax.set_yticklabels(STAGE_ORDER, fontsize=8)
    ax.set_xlabel("Predicted stage")
    ax.set_ylabel("True stage")
    ax.set_title(f"{cohort.upper()}  (V8, seed {SEED}, "
                 f"pooled across {n_folds} folds)",
                 fontsize=9, pad=6)
    ax.grid(False)
    ax.set_aspect("equal")

    for sp in ax.spines.values():
        sp.set_visible(False)

    print(f"{cohort.upper()}: {n_folds} folds, {cm.sum():,} epochs, "
          f"diag recall {np.round(np.diag(cm_pct), 1)}")

fig.subplots_adjust(right=0.86, wspace=0.30)
cax = fig.add_axes([0.89, 0.18, 0.018, 0.65])
cb  = fig.colorbar(im, cax=cax)
cb.set_label("Row %  (= per-true-class recall)", fontsize=8)
cb.ax.tick_params(labelsize=7)
cb.outline.set_linewidth(0.4)

OUT_DIR.mkdir(parents=True, exist_ok=True)
save(fig, OUT_DIR / "figure_06_confusion")
plt.close(fig)
print(f"Figure saved to {OUT_DIR / 'figure_06_confusion.pdf'}")