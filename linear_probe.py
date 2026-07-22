# =========================================================================
# linear_probe.py
#
# Tier-1 interpretability analysis for PhysioSleepNet (V8).
#
# Question: did the model's unsupervised 32-dim autonomic latent rediscover
# classical autonomic features (HR, HRV, respiration) end-to-end, despite
# never being trained to predict them?
#
# Method:
#   1. For each V8 test fold, load the saved 32-dim latent (latents.npz)
#      and the per-epoch metadata (predictions.npz, recording_metadata.json).
#   2. For each test epoch, compute PPG-derived target features from the
#      raw 30-s PPG epoch in the source npz file:
#          - HR_mean      mean heart rate (bpm)
#          - HR_std       SDNN — std of IBI intervals (ms)
#          - RMSSD        root mean square of successive IBI differences (ms)
#          - HF_power     0.15-0.40 Hz power band of IBI series (autonomic)
#          - resp_rate    respiration rate estimated from PPG envelope (bpm)
#   3. Train Ridge regression per fold per target:
#          latent (32) -> target scalar
#      Train on a within-fold 80/20 random subject split, report R^2 on
#      held-out subjects. Higher R^2 means more of the target is decodable
#      from the latent linearly.
#   4. Aggregate R^2 across the 15 V8 folds per cohort, report mean +- std.
#
# Outputs (written to /data2/Akbar1/PPG_Stages/benchmark_results/_paper_results/):
#   linear_probe_per_fold.csv    every (cohort, fold, target) row with R^2
#   linear_probe_summary.csv     mean +- std per cohort per target
#   linear_probe_SUMMARY.md      paper-ready markdown table
#
# Notes:
#   - We use only the latent at the harness's output, so the probe results
#     reflect what V8's bottleneck represents, not random features.
#   - We do NOT use any sleep-stage labels in the probe. Target features
#     are computed purely from raw PPG.
#   - PPG-derived (not ECG-derived) features keep the pipeline pure-PPG.
#     Accuracy is slightly lower than ECG-derived HR/HRV but methodologically
#     clean: "model rediscovers PPG-derivable autonomic features."
# =========================================================================

import csv
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from scipy.signal import butter, filtfilt, welch, find_peaks

ROOT = Path("/data2/Akbar1/PPG_Stages/benchmark_results")
OUT_DIR = ROOT / "_paper_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FS = 125              # PPG sampling rate (Hz)
EPOCH_SAMPLES = 30 * FS  # 3750
COHORTS = ["mesa", "cfs"]
MODEL = "v8"          # latents come from V8 only

# Probe controls
ALPHA = 1.0           # Ridge regularisation
TEST_FRAC = 0.2       # held-out subject fraction within a fold
MAX_SUBJECTS_PER_FOLD = 30   # set to e.g. 50 to subsample for speed; None = all
RANDOM_SEED = 0


# =========================================================================
# Patched PPG feature extractor for linear_probe.py
#
# Replace the existing `_features_from_epoch` function in linear_probe.py
# with this version. The only change is the respiration-rate estimator:
# old version used PPG envelope peak-finding (broken — produced near-constant
# values from filter artifacts). This version derives respiration from the
# HR-tachogram spectral peak in 0.15-0.40 Hz, the standard cardiology
# approach (Respiratory Sinus Arrhythmia method).
#
# Drop this function into linear_probe.py and rerun.
# =========================================================================

import numpy as np
from scipy.signal import butter, filtfilt, welch, find_peaks

FS = 125


# =========================================================================
# PPG feature extractor v2 — robust respiration estimator
#
# Drop-in replacement for _features_from_epoch (and helpers) in
# linear_probe.py.
#
# Changes from v1:
#   * Welch PSD: nperseg increased (better spectral resolution)
#   * Parabolic interpolation around the peak bin for sub-bin precision
#   * Reject epochs where the peak power is not meaningfully above the
#     band median — those epochs return NaN for resp_rate instead of
#     locking onto a default frequency
#
# HR/HRV computation unchanged.
# =========================================================================

import numpy as np
from scipy.signal import butter, filtfilt, welch, find_peaks

FS = 125


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


def _parabolic_interp(f, p, idx):
    """
    Quadratic interpolation around a spectral peak at bin idx.
    Returns sub-bin frequency estimate. Falls back to f[idx] at edges.
    """
    if idx <= 0 or idx >= len(p) - 1:
        return float(f[idx])
    y0, y1, y2 = p[idx - 1], p[idx], p[idx + 1]
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-12:
        return float(f[idx])
    # peak offset (in bins) from the central sample
    delta = 0.5 * (y0 - y2) / denom
    df = f[1] - f[0]
    return float(f[idx] + delta * df)


def _features_from_epoch(epoch_ppg, fs=FS):
    """
    Compute PPG-derived autonomic features for one 30-s epoch.
    Returns dict or None if the signal is too degraded for HR/HRV.
    For respiration, returns NaN if the spectrum has no clean peak in the
    respiratory band (RSA approach).
    """
    x = np.asarray(epoch_ppg, dtype=np.float32).ravel()
    if x.size < fs * 10:
        return None

    peaks = _ppg_peaks(x, fs=fs)
    if len(peaks) < 5:
        return None

    # IBI series (ms)
    ibi_ms = np.diff(peaks) / fs * 1000.0
    valid = (ibi_ms > 300) & (ibi_ms < 2000)
    ibi_ms = ibi_ms[valid]
    if len(ibi_ms) < 4:
        return None

    # ----- HR + HRV (unchanged) -----
    hr_mean = 60000.0 / np.mean(ibi_ms)
    hr_std  = float(np.std(ibi_ms))
    diffs   = np.diff(ibi_ms)
    rmssd   = float(np.sqrt(np.mean(diffs ** 2))) if len(diffs) >= 1 else 0.0

    # ----- HF power + RSA respiration via uniform 8 Hz tachogram -----
    hf_power = 0.0
    resp_rate = np.nan
    try:
        ibi_times = np.cumsum(ibi_ms) / 1000.0
        if len(ibi_times) >= 4 and ibi_times[-1] > 6:
            # Uniform 8 Hz tachogram (was 4 Hz in v1)
            t_uniform = np.arange(0, ibi_times[-1], 1.0 / 8.0)
            if len(t_uniform) < 8:
                return _pack_features(hr_mean, hr_std, rmssd, hf_power, resp_rate)
            ibi_uniform = np.interp(t_uniform, ibi_times, ibi_ms)
            ibi_demean = ibi_uniform - np.mean(ibi_uniform)

            # Welch PSD with larger window for better frequency resolution
            nperseg = min(128, len(ibi_demean))
            f, p = welch(ibi_demean, fs=8.0, nperseg=nperseg)

            # ---- HF power (0.15-0.40 Hz integrated) ----
            hf_band = (f >= 0.15) & (f <= 0.40)
            if hf_band.any():
                hf_power = float(np.trapz(p[hf_band], f[hf_band]))

            # ---- Respiration via spectral peak, with quality gating ----
            resp_band = (f >= 0.15) & (f <= 0.40)
            if resp_band.any():
                band_idx = np.where(resp_band)[0]
                p_band = p[band_idx]
                if len(p_band) >= 3 and p_band.max() > 0:
                    # quality gate: peak power must dominate the band
                    peak_local = int(np.argmax(p_band))
                    peak_global_idx = band_idx[peak_local]
                    peak_power = p[peak_global_idx]
                    band_median = float(np.median(p_band))
                    # require peak >= 1.5x band median (some signal-to-noise)
                    if band_median > 0 and peak_power >= 1.5 * band_median:
                        peak_freq_hz = _parabolic_interp(f, p, peak_global_idx)
                        # clamp to physiologic respiration range, otherwise NaN
                        candidate_bpm = peak_freq_hz * 60.0
                        if 6.0 <= candidate_bpm <= 30.0:
                            resp_rate = float(candidate_bpm)
    except Exception:
        pass

    return _pack_features(hr_mean, hr_std, rmssd, hf_power, resp_rate)


def _pack_features(hr_mean, hr_std, rmssd, hf_power, resp_rate):
    return {
        "HR_mean":   float(hr_mean),
        "HR_std":    float(hr_std),
        "RMSSD":     float(rmssd),
        "HF_power":  float(hf_power),
        "resp_rate": float(resp_rate) if np.isfinite(resp_rate) else np.nan,
    }

TARGETS = ["HR_mean", "HR_std", "RMSSD", "HF_power", "resp_rate"]


# =========================================================================
# Per-fold processing
# =========================================================================
def process_fold(cohort, fold_dir):
    """
    Loads a single fold's latents and predictions, extracts target features
    from raw PPG, trains Ridge per target, returns dict of R^2 values.
    """
    fold_name = fold_dir.name
    print(f"\n[{cohort} / {fold_name}]")

    # Load latent and metadata
    lat = np.load(fold_dir / "latents.npz")
    pred = np.load(fold_dir / "predictions.npz")
    with open(fold_dir / "recording_metadata.json") as f:
        meta = json.load(f)
    rec_meta = {entry["recording_idx"]: entry for entry in meta}

    Z = lat["z"]                                  # (N_epochs, 32)
    subj_arr = pred["subject_id"]                 # (N_epochs,)
    rec_arr = pred["recording_idx"]               # (N_epochs,)
    epoch_arr = pred["recording_epoch_idx"]       # (N_epochs,)
    assert Z.shape[0] == subj_arr.shape[0], "latent/prediction length mismatch"

    # Group epoch indices by (recording_idx, npz_path) so we load each
    # source npz exactly once
    epochs_per_rec = defaultdict(list)
    for i in range(len(subj_arr)):
        epochs_per_rec[int(rec_arr[i])].append(i)

    # Optional subsample
    rec_ids = sorted(epochs_per_rec.keys())
    if MAX_SUBJECTS_PER_FOLD is not None and len(rec_ids) > MAX_SUBJECTS_PER_FOLD:
        rng = np.random.default_rng(RANDOM_SEED)
        rec_ids = sorted(rng.choice(rec_ids, MAX_SUBJECTS_PER_FOLD, replace=False).tolist())

    # ---- extract targets per epoch ----
    targets = {t: np.full(Z.shape[0], np.nan, dtype=np.float32) for t in TARGETS}
    n_recs = len(rec_ids)
    for k, ridx in enumerate(rec_ids):
        if k % 25 == 0:
            print(f"   features... {k}/{n_recs} recordings", flush=True)
        info = rec_meta.get(ridx)
        if info is None:
            continue
        npz_path = info["npz_path"]
        try:
            arr = np.load(npz_path)
            x_all = arr["x"]                       # (E, 3750)
        except Exception as e:
            print(f"   skip {npz_path}: {e}")
            continue
        for i in epochs_per_rec[ridx]:
            e_idx = int(epoch_arr[i])
            if e_idx < 0 or e_idx >= x_all.shape[0]:
                continue
            feats = _features_from_epoch(x_all[e_idx])
            if feats is None:
                continue
            for t in TARGETS:
                targets[t][i] = feats[t]

    # ---- per-target Ridge regression on subject-level held-out split ----
    # Split by subject (unique subject_id), within this fold's test pool
    unique_subj = np.array(sorted(set(subj_arr.tolist())))
    rng = np.random.default_rng(RANDOM_SEED)
    subj_train, subj_test = train_test_split(
        unique_subj, test_size=TEST_FRAC, random_state=RANDOM_SEED)
    train_mask = np.isin(subj_arr, subj_train)
    test_mask  = np.isin(subj_arr, subj_test)

    out = {"fold": fold_name, "n_train_subjects": int(len(subj_train)),
           "n_test_subjects": int(len(subj_test)),
           "n_total_epochs": int(Z.shape[0])}

    for t in TARGETS:
        y = targets[t]
        finite = np.isfinite(y)
        tr = train_mask & finite
        te = test_mask  & finite
        out[f"{t}_n_train"] = int(tr.sum())
        out[f"{t}_n_test"]  = int(te.sum())
        if tr.sum() < 100 or te.sum() < 50:
            out[f"{t}_r2"] = float("nan")
            print(f"   {t:<10}  insufficient samples (tr={tr.sum()}, te={te.sum()})")
            continue
        # Standardise target on train fold for numeric stability
        y_train = y[tr]; y_test = y[te]
        mu, sd = float(np.mean(y_train)), float(np.std(y_train) + 1e-8)
        yt_train = (y_train - mu) / sd
        yt_test  = (y_test - mu) / sd

        reg = Ridge(alpha=ALPHA, random_state=RANDOM_SEED)
        reg.fit(Z[tr], yt_train)
        pred_yt = reg.predict(Z[te])
        r2 = r2_score(yt_test, pred_yt)
        out[f"{t}_r2"] = float(r2)
        print(f"   {t:<10}  R^2 = {r2:.3f}  (train n={tr.sum()}, test n={te.sum()})")

    return out


# =========================================================================
# Driver
# =========================================================================
def main():
    per_fold_rows = []
    for cohort in COHORTS:
        cohort_root = ROOT / cohort / MODEL
        if not cohort_root.exists():
            print(f"Skip cohort {cohort}: no V8 results directory")
            continue
        fold_dirs = sorted(cohort_root.glob("seed*_fold*"))
        for fd in fold_dirs:
            if not (fd / "DONE").exists():
                continue
            if not (fd / "latents.npz").exists():
                print(f"Skip {fd.name}: no latents.npz")
                continue
            row = process_fold(cohort, fd)
            row["cohort"] = cohort
            per_fold_rows.append(row)

    # ---- write per-fold CSV ----
    if not per_fold_rows:
        print("\nNo folds processed; aborting.")
        return

    per_fold_path = OUT_DIR / "linear_probe_per_fold.csv"
    fields = ["cohort", "fold", "n_train_subjects", "n_test_subjects",
              "n_total_epochs"]
    for t in TARGETS:
        fields += [f"{t}_r2", f"{t}_n_train", f"{t}_n_test"]
    with open(per_fold_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in per_fold_rows:
            w.writerow(r)
    print(f"\nWROTE  {per_fold_path}  ({len(per_fold_rows)} rows)")

    # ---- summary per cohort × target ----
    summary_path = OUT_DIR / "linear_probe_summary.csv"
    summary_rows = []
    for cohort in COHORTS:
        crows = [r for r in per_fold_rows if r["cohort"] == cohort]
        if not crows:
            continue
        for t in TARGETS:
            vals = np.array([r[f"{t}_r2"] for r in crows if np.isfinite(r.get(f"{t}_r2", np.nan))])
            if len(vals) == 0:
                summary_rows.append({"cohort": cohort, "target": t,
                                     "n_folds": 0, "r2_mean": float("nan"),
                                     "r2_std": float("nan"),
                                     "r2_min": float("nan"), "r2_max": float("nan")})
                continue
            summary_rows.append({
                "cohort":   cohort,
                "target":   t,
                "n_folds":  int(len(vals)),
                "r2_mean":  float(np.mean(vals)),
                "r2_std":   float(np.std(vals)),
                "r2_min":   float(np.min(vals)),
                "r2_max":   float(np.max(vals)),
            })
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cohort", "target", "n_folds",
                                          "r2_mean", "r2_std", "r2_min", "r2_max"])
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)
    print(f"WROTE  {summary_path}")

    # ---- paper-ready markdown ----
    md_path = OUT_DIR / "linear_probe_SUMMARY.md"
    with open(md_path, "w") as f:
        f.write("# Linear Probe — V8 autonomic latent -> PPG-derived features\n\n")
        f.write("Ridge regression from V8's 32-dim autonomic latent to "
                "classical PPG-derived autonomic features. Higher R^2 means more "
                "of the target is decodable linearly from the latent.\n\n")
        for cohort in COHORTS:
            f.write(f"## {cohort.upper()}\n\n")
            f.write("| Target | folds | R^2 (mean +- std) | min | max |\n")
            f.write("|---|---:|---|---|---|\n")
            for r in summary_rows:
                if r["cohort"] != cohort:
                    continue
                f.write(f"| {r['target']} | {r['n_folds']} | "
                        f"{r['r2_mean']:.3f} +- {r['r2_std']:.3f} | "
                        f"{r['r2_min']:.3f} | {r['r2_max']:.3f} |\n")
            f.write("\n")
    print(f"WROTE  {md_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()