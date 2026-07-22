# =========================================================================
# figure_07_hrv_dynamics.py
#
# Figure 7 — Time-resolved HRV tracking.
#
# What this figure shows:
#   For one representative subject (same as Figure 3 hypnogram), we plot
#   the ground-truth RMSSD trajectory over the whole night versus the
#   RMSSD predicted by linearly projecting V8's 32-dim latent onto its
#   HRV-decoding axis (Ridge from Figure 1's probe). If V8's latent
#   actually tracks HRV, the two traces should follow the same shape.
#
# Why this matters scientifically:
#   Figure 1 already showed V8's latent linearly decodes RMSSD in
#   aggregate (R^2 = 0.44 on MESA). But that's a static, per-epoch
#   correlation. THIS figure shows the time-resolved structure is
#   preserved -- the model's representation tracks moment-to-moment HRV
#   dynamics across the entire night, not just per-epoch correlation.
#
# =========================================================================
# DESIGN DECISIONS — documented for future reference
# =========================================================================
#
# (1) Subject selection
#     -----------------
#     Same logic as figure_03_hypnogram.py so the two figures show the
#     same subject. Filter for normal sleep architecture (SE >= 30%,
#     %Deep >= 5%, %REM >= 10%), then pick the one closest to mean
#     accuracy of the filtered pool.
#
# (2) Ridge training — subject held out
#     ---------------------------------
#     To predict the target subject's RMSSD trace without data leakage,
#     we train the Ridge regression on every OTHER subject in the same
#     test fold (seed42_fold0). The target subject's epochs never appear
#     in Ridge training. This makes the figure honest: the Ridge has
#     literally never seen this subject's PPG.
#
# (3) Target standardisation and inverse transform
#     ---------------------------------------------
#     For Ridge stability, the training target (RMSSD in ms) is
#     standardised: y_std = (y - mu_train) / sd_train. After prediction
#     we invert back to original ms units so both traces are on the same
#     scale and the figure reads as "RMSSD (ms)".
#
# (4) Smoothing
#     ---------
#     Per-epoch RMSSD from PPG is noisy due to peak-detection errors.
#     A 5-epoch rolling median (2.5-min window) cleans this without
#     distorting the slow autonomic structure that interests us. We
#     apply the SAME smoothing to both true and predicted traces so the
#     comparison is fair.
#
# (5) Hypnogram strip
#     ---------------
#     We include the ground-truth hypnogram as a thin panel above the
#     HRV traces so the reader can see how RMSSD changes line up with
#     stage transitions (RMSSD typically high in REM, low in Wake).
#
# (6) Correlation annotation
#     ----------------------
#     We annotate the Pearson r between true and predicted SMOOTHED
#     traces for this subject. This quantifies tracking quality on the
#     time series and is a separate test from Figure 1's per-epoch R^2.
#
# =========================================================================

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from scipy.signal import butter, filtfilt, find_peaks

from figstyle import apply_style, PALETTE, STAGE_COLORS, save

apply_style()


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
ROOT     = Path("/data2/Akbar1/PPG_Stages/benchmark_results")
OUT_DIR  = ROOT / "_paper_results" / "figures"
FOLD_DIR = ROOT / "mesa" / "v8" / "seed42_fold0"

FS                 = 125         # sampling rate of PPG
EPOCH_SAMPLES      = 30 * FS     # 3750
SMOOTH_WINDOW      = 5           # epochs (= 2.5 min)
RIDGE_ALPHA        = 1.0
RANDOM_SEED        = 0


# -------------------------------------------------------------------------
# PPG → per-epoch RMSSD  (same algorithm as feature_extractor_v2.py)
# -------------------------------------------------------------------------
def _bandpass(x, lo, hi, fs=FS, order=4):
    ny = fs * 0.5
    b, a = butter(order, [lo / ny, hi / ny], btype="band")
    return filtfilt(b, a, x)


def _ppg_peaks(x, fs=FS):
    try:
        xs = _bandpass(x, 0.5, 5.0, fs=fs)
    except Exception:
        return np.array([], dtype=int)
    if np.std(xs) < 1e-8:
        return np.array([], dtype=int)
    xs = (xs - np.mean(xs)) / (np.std(xs) + 1e-8)
    peaks, _ = find_peaks(xs, distance=int(0.25 * fs), prominence=0.4)
    return peaks


def compute_rmssd(epoch_ppg, fs=FS):
    """Per-epoch RMSSD in ms (or NaN if signal too poor)."""
    x = np.asarray(epoch_ppg, dtype=np.float32).ravel()
    if x.size < fs * 10:
        return np.nan
    peaks = _ppg_peaks(x, fs=fs)
    if len(peaks) < 5:
        return np.nan
    ibi_ms = np.diff(peaks) / fs * 1000.0
    valid  = (ibi_ms > 300) & (ibi_ms < 2000)
    ibi_ms = ibi_ms[valid]
    if len(ibi_ms) < 4:
        return np.nan
    diffs = np.diff(ibi_ms)
    if len(diffs) < 1:
        return np.nan
    return float(np.sqrt(np.mean(diffs ** 2)))


# -------------------------------------------------------------------------
# Subject selection — must match figure_03_hypnogram.py
# -------------------------------------------------------------------------
def sleep_stats(yt):
    n_wake  = int((yt == 0).sum())
    n_light = int((yt == 1).sum())
    n_deep  = int((yt == 2).sum())
    n_rem   = int((yt == 3).sum())
    n_total = n_wake + n_light + n_deep + n_rem
    n_sleep = n_light + n_deep + n_rem
    if n_total == 0 or n_sleep == 0:
        return 0.0, 0.0, 0.0
    return n_sleep / n_total, n_deep / n_sleep, n_rem / n_sleep


# -------------------------------------------------------------------------
# Load fold predictions, latents, metadata
# -------------------------------------------------------------------------
print("Loading fold predictions and latents ...")
pred      = np.load(FOLD_DIR / "predictions.npz")
lat       = np.load(FOLD_DIR / "latents.npz")
y_true    = pred["y_true"]
y_pred    = pred["y_pred"]
rec_idx   = pred["recording_idx"]
epoch_idx = pred["recording_epoch_idx"]
Z         = lat["z"]                                      # (N_epochs, 32)

with open(FOLD_DIR / "recording_metadata.json") as f:
    meta = json.load(f)
rec_meta = {entry["recording_idx"]: entry for entry in meta}

assert Z.shape[0] == y_true.shape[0], "latent/prediction length mismatch"


# -------------------------------------------------------------------------
# Pick subject — same logic as Figure 3
# -------------------------------------------------------------------------
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
    raise RuntimeError("No subjects passed the architecture filter.")

mean_acc = float(np.mean([c["acc"] for c in candidates]))
candidates.sort(key=lambda c: abs(c["acc"] - mean_acc))
chosen = candidates[0]
chosen_rec, chosen_acc, n_eps = chosen["rec"], chosen["acc"], chosen["n"]
subj_name = rec_meta[chosen_rec]["subject_id"]
npz_path  = rec_meta[chosen_rec]["npz_path"]

print(f"\nChosen subject (same as Figure 3): {subj_name}")
print(f"  V8 accuracy = {chosen_acc:.3f}")
print(f"  n epochs    = {n_eps}")
print(f"  SE          = {chosen['se']:.1%}")
print(f"  %Deep       = {chosen['fr_deep']:.1%}")
print(f"  %REM        = {chosen['fr_rem']:.1%}")


# -------------------------------------------------------------------------
# Compute ground-truth RMSSD per epoch for EVERY subject in this fold.
# We need this for all subjects: target + every Ridge training subject.
# Caches per-recording results via the rec_meta npz paths.
# -------------------------------------------------------------------------
print("\nComputing ground-truth RMSSD per epoch ...")
rmssd_all = np.full(Z.shape[0], np.nan, dtype=np.float32)

unique_recs = np.unique(rec_idx)
for k, r in enumerate(unique_recs):
    if k % 25 == 0:
        print(f"   {k}/{len(unique_recs)} recordings")
    info = rec_meta.get(int(r))
    if info is None:
        continue
    try:
        arr = np.load(info["npz_path"])
        x_rec = arr["x"]                                  # (E, 3750)
    except Exception as e:
        print(f"   skip {info['npz_path']}: {e}")
        continue
    # for each epoch index assigned to this recording, compute RMSSD
    idx_in_pred = np.where(rec_idx == r)[0]
    for i in idx_in_pred:
        e_idx = int(epoch_idx[i])
        if e_idx < 0 or e_idx >= x_rec.shape[0]:
            continue
        rmssd_all[i] = compute_rmssd(x_rec[e_idx])

print(f"   Got RMSSD for {(~np.isnan(rmssd_all)).sum()} / {len(rmssd_all)} epochs")


# -------------------------------------------------------------------------
# Train Ridge: latent -> RMSSD, EXCLUDING the target subject
# -------------------------------------------------------------------------
print("\nTraining Ridge on held-out training subjects (target excluded) ...")
target_mask = (rec_idx == chosen_rec)
train_mask  = (~target_mask) & np.isfinite(rmssd_all)
print(f"   training epochs: {train_mask.sum()} from {len(np.unique(rec_idx[train_mask]))} subjects")

# Standardise RMSSD target on the training set, store mu/sd to invert later
y_train     = rmssd_all[train_mask]
mu, sd      = float(np.mean(y_train)), float(np.std(y_train) + 1e-8)
y_train_std = (y_train - mu) / sd

reg = Ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_SEED)
reg.fit(Z[train_mask], y_train_std)


# -------------------------------------------------------------------------
# Predict RMSSD trace for the target subject, invert standardisation
# -------------------------------------------------------------------------
target_idx = np.where(target_mask)[0]
# order by epoch_idx so the trace is chronological
order      = np.argsort(epoch_idx[target_idx])
target_idx = target_idx[order]

Z_target          = Z[target_idx]
rmssd_true_subj   = rmssd_all[target_idx]
rmssd_pred_std    = reg.predict(Z_target)
rmssd_pred_subj   = rmssd_pred_std * sd + mu              # back to ms
stages_true_subj  = y_true[target_idx]
t_hrs             = np.arange(len(target_idx)) * 30.0 / 3600.0


# -------------------------------------------------------------------------
# Smooth both traces with the SAME rolling median for fair comparison
# -------------------------------------------------------------------------
def rolling_median(x, w):
    s = pd.Series(x)
    return s.rolling(w, center=True, min_periods=1).median().to_numpy()


rmssd_true_smooth = rolling_median(rmssd_true_subj, SMOOTH_WINDOW)
rmssd_pred_smooth = rolling_median(rmssd_pred_subj, SMOOTH_WINDOW)

# Pearson correlation on the smoothed traces, where both are finite
finite = np.isfinite(rmssd_true_smooth) & np.isfinite(rmssd_pred_smooth)
if finite.sum() > 10:
    pearson_r = float(np.corrcoef(rmssd_true_smooth[finite],
                                   rmssd_pred_smooth[finite])[0, 1])
else:
    pearson_r = float("nan")
print(f"\nSmoothed-trace Pearson r (true vs predicted) = {pearson_r:.3f}")



# (1) Trim trailing wake tail (> 30 contiguous minutes of wake at the end)
# -------------------------------------------------------------------------
def trim_wake_tail(stages, max_wake_min=30, epoch_min=0.5):
    """Return slice index up to which we keep data."""
    max_wake_epochs = int(max_wake_min / epoch_min)
    n = len(stages)
    # walk backwards from the end; count consecutive Wake (class 0)
    last_nonwake = n - 1
    consec_wake = 0
    for i in range(n - 1, -1, -1):
        if stages[i] == 0:
            consec_wake += 1
        else:
            break
    if consec_wake > max_wake_epochs:
        # keep epochs up to and including the last non-wake plus 5 minutes
        last_nonwake = n - consec_wake
        keep_end = min(n, last_nonwake + int(5 / epoch_min))
    else:
        keep_end = n
    return keep_end
 
 
keep_end = trim_wake_tail(stages_true_subj)
print(f"Trimming display to first {keep_end} of {len(t_hrs)} epochs "
      f"({t_hrs[keep_end - 1]:.1f} hrs of {t_hrs[-1]:.1f} hrs)")
 
t_hrs_p           = t_hrs[:keep_end]
stages_p          = stages_true_subj[:keep_end]
rmssd_true_p      = rmssd_true_smooth[:keep_end]
rmssd_pred_p      = rmssd_pred_smooth[:keep_end]
 
# Re-compute Pearson r on the trimmed traces (so the annotation reflects
# what is displayed, not the artifact-corrupted tail)
finite_p = np.isfinite(rmssd_true_p) & np.isfinite(rmssd_pred_p)
if finite_p.sum() > 10:
    pearson_r_disp = float(np.corrcoef(rmssd_true_p[finite_p],
                                        rmssd_pred_p[finite_p])[0, 1])
else:
    pearson_r_disp = float("nan")
print(f"Pearson r on displayed (trimmed) traces = {pearson_r_disp:.3f}")
 
 
# -------------------------------------------------------------------------
# (2) Plot — hypnogram strip + HRV overlay with stage-coloured background
# -------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.0), sharex=True,
                         gridspec_kw={"height_ratios": [1, 3]})
 
CLASS_TO_Y = {0: 3, 3: 2, 1: 1, 2: 0}
Y_LABELS   = ["Deep", "Light", "REM", "Wake"]
y_strip = np.array([CLASS_TO_Y[int(v)] for v in stages_p])
 
# Hypnogram strip (top)
axes[0].step(t_hrs_p, y_strip, where="post",
             color=PALETTE["navy"], linewidth=0.8)
axes[0].fill_between(t_hrs_p, -0.5, y_strip, step="post",
                     color=PALETTE["navy"], alpha=0.07, linewidth=0)
axes[0].set_yticks([0, 1, 2, 3]); axes[0].set_yticklabels(Y_LABELS)
axes[0].set_ylabel("Stage")
axes[0].set_ylim(-0.5, 3.5)
axes[0].grid(True, axis="y", color=PALETTE["grid"], linewidth=0.4)
axes[0].xaxis.grid(False)
 
# Background bands for REM and Deep in the HRV panel
ax = axes[1]
ymax = np.nanmax(rmssd_true_p) * 1.05 if np.isfinite(np.nanmax(rmssd_true_p)) else 600
# Find contiguous spans of each stage
def contig_spans(arr, target_val):
    spans = []
    in_run = False
    start = None
    for i, v in enumerate(arr):
        if int(v) == target_val and not in_run:
            in_run, start = True, i
        elif int(v) != target_val and in_run:
            spans.append((start, i))
            in_run = False
    if in_run:
        spans.append((start, len(arr)))
    return spans
 
for s, e in contig_spans(stages_p, 3):    # REM = 3
    ax.axvspan(t_hrs_p[s], t_hrs_p[e - 1],
               color=STAGE_COLORS["REM"], alpha=0.10, linewidth=0, zorder=0)
for s, e in contig_spans(stages_p, 2):    # Deep = 2
    ax.axvspan(t_hrs_p[s], t_hrs_p[e - 1],
               color=STAGE_COLORS["Deep"], alpha=0.10, linewidth=0, zorder=0)
 
# Overlaid traces
ax.plot(t_hrs_p, rmssd_true_p, color=PALETTE["navy"], linewidth=1.2,
        label="Ground truth (PPG-derived)", zorder=3)
ax.plot(t_hrs_p, rmssd_pred_p, color=PALETTE["rust"], linewidth=1.2,
        linestyle="--", label="Predicted from V8 latent (linear probe)",
        zorder=3)
 
ax.set_ylabel("RMSSD  (ms,  5-epoch smoothed)")
ax.set_xlabel("Time (hours from recording start)")
ax.set_ylim(0, ymax)
 
# Pearson r — use the value computed on what we display
ax.text(0.985, 0.97,
        f"Pearson r = {pearson_r_disp:.2f}",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=8, color=PALETTE["ink"])
 
# Legend explicitly notes that shaded bands are stages
import matplotlib.patches as mpatches
band_rem  = mpatches.Patch(facecolor=STAGE_COLORS["REM"],  alpha=0.18,
                            label="REM (background)", linewidth=0)
band_deep = mpatches.Patch(facecolor=STAGE_COLORS["Deep"], alpha=0.18,
                            label="Deep (background)", linewidth=0)
ax.legend(handles=[ax.lines[0], ax.lines[1], band_rem, band_deep],
          loc="upper left", frameon=False, fontsize=7.5, ncol=2)
 
fig.suptitle(
    f"V8's latent tracks HRV dynamics  —  MESA subject {subj_name}  "
    f"(V8 acc = {chosen_acc:.2f},  SE = {chosen['se']:.0%})",
    fontsize=9, y=0.995
)
fig.tight_layout(rect=[0, 0, 1, 0.96])
 
OUT_DIR.mkdir(parents=True, exist_ok=True)
save(fig, OUT_DIR / "figure_07_hrv_dynamics")
plt.close(fig)
 
print(f"Figure 7 saved (with v2 trim + stage bands).")