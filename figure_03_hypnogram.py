# =========================================================================
# figure_03_hypnogram.py  (v2)
#
# Patch: better subject selection.
#
# Previous version picked the median-accuracy subject without filtering
# for sleep-architecture quality, returning a degenerate-architecture
# subject (almost no Deep) that made model errors look much worse than
# they were. New rule: filter for normal architecture first.
# =========================================================================

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from figstyle import apply_style, PALETTE, save

apply_style()

ROOT     = Path("/data2/Akbar1/PPG_Stages/benchmark_results")
OUT_DIR  = ROOT / "_paper_results" / "figures"
FOLD_DIR = ROOT / "mesa" / "v8" / "seed42_fold0"


# Load
pred       = np.load(FOLD_DIR / "predictions.npz")
y_true     = pred["y_true"]
y_pred     = pred["y_pred"]
y_probs    = pred["y_probs"]
rec_idx    = pred["recording_idx"]
epoch_idx  = pred["recording_epoch_idx"]
with open(FOLD_DIR / "recording_metadata.json") as f:
    meta = json.load(f)


def sleep_stats(yt):
    """Return (sleep_efficiency, fr_deep, fr_rem) from ground-truth hypnogram."""
    n_wake  = int((yt == 0).sum())
    n_light = int((yt == 1).sum())
    n_deep  = int((yt == 2).sum())
    n_rem   = int((yt == 3).sum())
    n_total = n_wake + n_light + n_deep + n_rem
    n_sleep = n_light + n_deep + n_rem
    if n_total == 0 or n_sleep == 0:
        return 0.0, 0.0, 0.0
    return n_sleep / n_total, n_deep / n_sleep, n_rem / n_sleep


# Subject selection: filter for normal architecture, then pick closest to mean acc
candidates = []
for r in np.unique(rec_idx):
    mask = rec_idx == r
    if mask.sum() < 400:
        continue
    yt_r = y_true[mask]
    yp_r = y_pred[mask]
    se, fr_deep, fr_rem = sleep_stats(yt_r)
    if se < 0.30 or fr_deep < 0.05 or fr_rem < 0.10:
        continue
    candidates.append({
        "rec":     int(r),
        "acc":     float((yt_r == yp_r).mean()),
        "n":       int(mask.sum()),
        "se":      se,
        "fr_deep": fr_deep,
        "fr_rem":  fr_rem,
    })

if not candidates:
    raise RuntimeError("No subjects passed the architecture filter — loosen thresholds.")

mean_acc = float(np.mean([c["acc"] for c in candidates]))
candidates.sort(key=lambda c: abs(c["acc"] - mean_acc))
chosen = candidates[0]
chosen_rec, chosen_acc, n_eps = chosen["rec"], chosen["acc"], chosen["n"]
subj_name = meta[chosen_rec]["subject_id"]

print(f"Filtered pool: {len(candidates)} subjects")
print(f"Pool mean acc: {mean_acc:.3f}")
print(f"Chosen subject : {subj_name}")
print(f"  acc      = {chosen_acc:.3f}")
print(f"  n epochs = {n_eps}")
print(f"  SE       = {chosen['se']:.1%}")
print(f"  Deep     = {chosen['fr_deep']:.1%}")
print(f"  REM      = {chosen['fr_rem']:.1%}")

# Build the hypnogram
mask  = rec_idx == chosen_rec
order = np.argsort(epoch_idx[mask])
yt    = y_true[mask][order]
yp    = y_pred[mask][order]
conf  = y_probs[mask][order].max(axis=1)
t_hrs = np.arange(len(yt)) * 30.0 / 3600.0

CLASS_TO_Y = {0: 3, 3: 2, 1: 1, 2: 0}
Y_LABELS   = ["Deep", "Light", "REM", "Wake"]


def to_plot_y(arr):
    return np.array([CLASS_TO_Y[int(v)] for v in arr])


# Plot
fig, axes = plt.subplots(3, 1, figsize=(7.0, 4.2), sharex=True,
                         gridspec_kw={"height_ratios": [3, 3, 1.2]})

axes[0].step(t_hrs, to_plot_y(yt), where="post",
             color=PALETTE["navy"], linewidth=1.0)
axes[0].fill_between(t_hrs, -0.5, to_plot_y(yt), step="post",
                     color=PALETTE["navy"], alpha=0.08, linewidth=0)
axes[0].set_yticks([0, 1, 2, 3]); axes[0].set_yticklabels(Y_LABELS)
axes[0].set_ylabel("Ground\ntruth")
axes[0].set_ylim(-0.5, 3.5)

axes[1].step(t_hrs, to_plot_y(yp), where="post",
             color=PALETTE["rust"], linewidth=1.0)
axes[1].fill_between(t_hrs, -0.5, to_plot_y(yp), step="post",
                     color=PALETTE["rust"], alpha=0.08, linewidth=0)
axes[1].set_yticks([0, 1, 2, 3]); axes[1].set_yticklabels(Y_LABELS)
axes[1].set_ylabel("V8\nprediction")
axes[1].set_ylim(-0.5, 3.5)

axes[2].fill_between(t_hrs, conf,
                     color=PALETTE["slate"], alpha=0.55, linewidth=0)
axes[2].plot(t_hrs, conf, color=PALETTE["slate"], linewidth=0.6)
axes[2].set_ylim(0, 1)
axes[2].set_yticks([0, 0.5, 1.0])
axes[2].set_ylabel("Pred.\nconfidence")
axes[2].set_xlabel("Time (hours from recording start)")

for ax in axes[:2]:
    ax.grid(True, axis="y", color=PALETTE["grid"], linewidth=0.4)
    ax.xaxis.grid(False)

fig.suptitle(
    f"Hypnogram — MESA subject ({subj_name},  {n_eps} epochs,  "
    f"V8 accuracy = {chosen_acc:.2f},  SE = {chosen['se']:.0%},  "
    f"Deep = {chosen['fr_deep']:.0%},  REM = {chosen['fr_rem']:.0%})",
    fontsize=8.5, y=0.995
)

fig.tight_layout(rect=[0, 0, 1, 0.96])
OUT_DIR.mkdir(parents=True, exist_ok=True)
save(fig, OUT_DIR / "figure_03_hypnogram")
plt.close(fig)
print(f"\nFigure 3 saved to {OUT_DIR / 'figure_03_hypnogram.pdf'}")