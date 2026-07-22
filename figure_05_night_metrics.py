# =========================================================================
# figure_05_night_metrics.py  (v2)
#
# Figure 5 — Night-level sleep parameters.
# Cleaner story: only the clinically meaningful aggregate metrics that
# hospitals actually use to evaluate sleep quality:
#     TST   total sleep time (minutes)
#     SE    sleep efficiency (%)
# Per-stage fractions (FR Light / FR Deep / FR REM) are intentionally
# omitted because their per-subject r^2 is dominated by within-cohort
# variance shrinkage and creates a misleading visual impression.
#
# Annotations are placed outside the data region (below each panel)
# so they never overlap the scatter or the identity line.
#
# Output: _paper_results/figures/figure_05_night_metrics.pdf (300 DPI)
# =========================================================================

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from figstyle import apply_style, COHORT_COLORS, PALETTE, save

apply_style()

ROOT     = Path("/data2/Akbar1/PPG_Stages/benchmark_results")
OUT_DIR  = ROOT / "_paper_results" / "figures"

EPOCH_MIN = 0.5   # 30 s epoch -> 0.5 min


# -------------------------------------------------------------------------
# Per-subject metric computation (seed42 only -> one entry per subject)
# -------------------------------------------------------------------------
def night_summary(arr):
    """arr: per-epoch class indices (0=Wake, 1=Light, 2=Deep, 3=REM).
       Returns (TST_min, sleep_efficiency_pct)."""
    n_wake  = int((arr == 0).sum())
    n_light = int((arr == 1).sum())
    n_deep  = int((arr == 2).sum())
    n_rem   = int((arr == 3).sum())
    n_total = n_wake + n_light + n_deep + n_rem
    n_sleep = n_light + n_deep + n_rem
    if n_total == 0:
        return None
    tst = n_sleep * EPOCH_MIN
    se  = (n_sleep / n_total) * 100.0
    return tst, se


def per_subject_for_cohort(cohort):
    out = []
    for fold_dir in sorted((ROOT / cohort / "v8").glob("seed42_fold*")):
        if not (fold_dir / "DONE").exists():
            continue
        pred   = np.load(fold_dir / "predictions.npz")
        y_true = pred["y_true"]
        y_pred = pred["y_pred"]
        subj   = pred["subject_id"]
        for s in np.unique(subj):
            mask = subj == s
            mt = night_summary(y_true[mask])
            mp = night_summary(y_pred[mask])
            if mt is None or mp is None:
                continue
            out.append({
                "subject":   str(s),
                "TST_true":  mt[0], "TST_pred": mp[0],
                "SE_true":   mt[1], "SE_pred":  mp[1],
            })
    return out


print("Computing per-subject TST/SE for MESA ...")
mesa = per_subject_for_cohort("mesa")
print(f"   {len(mesa)} subjects")
print("Computing per-subject TST/SE for CFS ...")
cfs = per_subject_for_cohort("cfs")
print(f"   {len(cfs)} subjects")


# -------------------------------------------------------------------------
# Stats helpers
# -------------------------------------------------------------------------
def r_squared(t, p):
    t, p = np.asarray(t), np.asarray(p)
    ss_res = float(np.sum((t - p) ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-9)


def pearson_r(t, p):
    t, p = np.asarray(t), np.asarray(p)
    if t.std() < 1e-9 or p.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(t, p)[0, 1])


def mae(t, p):
    return float(np.mean(np.abs(np.asarray(t) - np.asarray(p))))


# -------------------------------------------------------------------------
# Plot — two panels side by side, clean layout
# -------------------------------------------------------------------------
PARAMS = [
    ("Total sleep time",  "Total sleep time (min)",  "TST_true", "TST_pred", "min"),
    ("Sleep efficiency",  "Sleep efficiency (%)",    "SE_true",  "SE_pred",  "%"),
]

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.8))

for ax, (name, label, kt, kp, unit) in zip(axes, PARAMS):

    for rows, cname, color in [(mesa, "MESA", COHORT_COLORS["mesa"]),
                               (cfs,  "CFS",  COHORT_COLORS["cfs"])]:
        ts = [r[kt] for r in rows]
        ps = [r[kp] for r in rows]
        ax.scatter(ts, ps, s=10, alpha=0.40, c=color,
                   edgecolors="none", label=cname)

    # Identity line spanning the data range
    all_vals = ([r[kt] for r in mesa] + [r[kt] for r in cfs] +
                [r[kp] for r in mesa] + [r[kp] for r in cfs])
    lo, hi = float(min(all_vals)), float(max(all_vals))
    pad    = (hi - lo) * 0.05
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
            color=PALETTE["ink"], linewidth=0.7,
            linestyle="--", alpha=0.85, zorder=0)

    # Per-cohort stats
    r2_m = r_squared([r[kt] for r in mesa], [r[kp] for r in mesa])
    r2_c = r_squared([r[kt] for r in cfs],  [r[kp] for r in cfs])
    rp_m = pearson_r([r[kt] for r in mesa], [r[kp] for r in mesa])
    rp_c = pearson_r([r[kt] for r in cfs],  [r[kp] for r in cfs])
    mae_m = mae([r[kt] for r in mesa], [r[kp] for r in mesa])
    mae_c = mae([r[kt] for r in cfs],  [r[kp] for r in cfs])

    # Annotation placed OUTSIDE data region — below the x-axis label
    ax.text(0.5, -0.30,
            f"MESA:  $r$ = {rp_m:.2f},  $r^{{2}}$ = {r2_m:.2f},  MAE = {mae_m:.1f} {unit}\n"
            f"CFS:    $r$ = {rp_c:.2f},  $r^{{2}}$ = {r2_c:.2f},  MAE = {mae_c:.1f} {unit}",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=8, color=PALETTE["ink"])

    ax.set_xlabel(f"True {label}")
    ax.set_ylabel(f"Predicted {label}")
    ax.set_title(name, fontsize=10, pad=6)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")

# Shared legend at top
handles = [
    plt.Line2D([0], [0], marker="o", color="w",
               markerfacecolor=COHORT_COLORS["mesa"],
               markersize=7, label=f"MESA  (n = {len(mesa)} subjects)",
               markeredgecolor="none"),
    plt.Line2D([0], [0], marker="o", color="w",
               markerfacecolor=COHORT_COLORS["cfs"],
               markersize=7, label=f"CFS   (n = {len(cfs)} subjects)",
               markeredgecolor="none"),
    plt.Line2D([0], [0], linestyle="--", color=PALETTE["ink"],
               linewidth=1.0, label="y = x (identity)"),
]
fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, 1.02), fontsize=8.5)

# Leave room for legend above and stats below
fig.tight_layout(rect=[0, 0.13, 1, 0.93])

OUT_DIR.mkdir(parents=True, exist_ok=True)
save(fig, OUT_DIR / "figure_05_night_metrics")
plt.close(fig)
print("Figure 5 saved.")