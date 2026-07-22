# =========================================================================
# figure_01_linear_probe.py
#
# Figure 1 — Linear probe interpretability.
# Bar plot of R^2 (mean +- std across 15 folds) for each PPG-derived
# autonomic target, MESA vs CFS side by side. Reads from the already-
# generated per-fold CSV in _paper_results/.
#
# Output:
#   _paper_results/figures/figure_01_linear_probe.pdf  (300 DPI)
#   _paper_results/figures/figure_01_linear_probe.png  (preview)
# =========================================================================

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

from figstyle import apply_style, COHORT_COLORS, PALETTE, save

apply_style()

RESULTS = Path("/data2/Akbar1/PPG_Stages/benchmark_results/_paper_results")
PER_FOLD_CSV = RESULTS / "linear_probe_per_fold.csv"
OUT_DIR      = RESULTS / "figures"


# -------------------------------------------------------------------------
# Load per-fold R^2 values from the probe CSV
# -------------------------------------------------------------------------
TARGETS = ["HR_mean", "HR_std", "RMSSD", "HF_power", "resp_rate"]
TARGET_LABELS = {
    "HR_mean":   "HR\n(mean)",
    "HR_std":    "HR std\n(SDNN)",
    "RMSSD":     "RMSSD",
    "HF_power":  "HF power",
    "resp_rate": "Resp.\nrate",
}

# rows: (cohort, fold, ... metric_r2 columns)
rows = []
with open(PER_FOLD_CSV) as f:
    for r in csv.DictReader(f):
        rows.append(r)

def collect(cohort, target):
    """Return numpy array of R^2 values across folds for one cohort+target."""
    out = []
    key = f"{target}_r2"
    for r in rows:
        if r["cohort"] != cohort:
            continue
        try:
            v = float(r[key])
            if np.isfinite(v):
                out.append(v)
        except (KeyError, ValueError):
            continue
    return np.array(out)


means_mesa = np.array([collect("mesa", t).mean() for t in TARGETS])
stds_mesa  = np.array([collect("mesa", t).std()  for t in TARGETS])
means_cfs  = np.array([collect("cfs",  t).mean() for t in TARGETS])
stds_cfs   = np.array([collect("cfs",  t).std()  for t in TARGETS])
n_mesa     = len(collect("mesa", TARGETS[0]))
n_cfs      = len(collect("cfs",  TARGETS[0]))


# -------------------------------------------------------------------------
# Plot
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.6, 3.0))

x = np.arange(len(TARGETS))
bar_w = 0.36

# Bars
b1 = ax.bar(x - bar_w / 2, means_mesa, bar_w,
            yerr=stds_mesa, capsize=2.5,
            color=COHORT_COLORS["mesa"], label=f"MESA  (n = {n_mesa} folds)",
            ecolor=PALETTE["ink"], error_kw={"linewidth": 0.7})

b2 = ax.bar(x + bar_w / 2, means_cfs, bar_w,
            yerr=stds_cfs, capsize=2.5,
            color=COHORT_COLORS["cfs"], label=f"CFS    (n = {n_cfs} folds)",
            ecolor=PALETTE["ink"], error_kw={"linewidth": 0.7})

# Reference line at R^2 = 0
ax.axhline(0, color=PALETTE["ink"], linewidth=0.6, linestyle="-", zorder=1)

# Axis cosmetics
ax.set_xticks(x)
ax.set_xticklabels([TARGET_LABELS[t] for t in TARGETS])
ax.set_ylabel(r"Linear probe $R^2$")
ymin = min(0.0, float(means_cfs.min() - stds_cfs.max()) - 0.05)
ymax = max(0.75, float(means_mesa.max() + stds_mesa.max()) + 0.05)
ax.set_ylim(ymin, ymax)
ax.set_axisbelow(True)
ax.yaxis.grid(True, color=PALETTE["grid"], linewidth=0.5)
ax.xaxis.grid(False)

# Mean-value annotations on top of each MESA bar (the headline numbers)
for xi, m, s in zip(x - bar_w / 2, means_mesa, stds_mesa):
    if m > 0.05:
        ax.text(xi, m + s + 0.015, f"{m:.2f}",
                ha="center", va="bottom", fontsize=7,
                color=PALETTE["ink"])

# Legend
ax.legend(loc="upper right", frameon=False, fontsize=8,
          handlelength=1.2, borderaxespad=0.4)

# Title (small, paper-style)
ax.set_title("Linear decodability of PPG-derived autonomic features\n"
             "from V8's unsupervised 32-dim latent",
             fontsize=9, pad=8)

# Save
OUT_DIR.mkdir(parents=True, exist_ok=True)
save(fig, OUT_DIR / "figure_01_linear_probe")
plt.close(fig)


# -------------------------------------------------------------------------
# Console summary so you can verify quickly
# -------------------------------------------------------------------------
print("Figure 1 saved to:")
print(f"  {OUT_DIR / 'figure_01_linear_probe.pdf'}")
print(f"  {OUT_DIR / 'figure_01_linear_probe.png'}\n")
print(f"{'target':<12} {'MESA mean ± std':<22} {'CFS mean ± std':<22}")
for t, mm, ms, cm, cs in zip(TARGETS, means_mesa, stds_mesa, means_cfs, stds_cfs):
    print(f"  {t:<10} {mm:+.3f} ± {ms:.3f}        {cm:+.3f} ± {cs:.3f}")