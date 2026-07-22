# =========================================================================
# harness.py — universal training & 5-fold evaluation harness
#              for the PPG sleep-staging benchmark.
#
# WHAT IT DOES
#   * Loads MESA or CFS manifest, generates FROZEN subject-level 5-fold
#     splits (seed=42, written once to disk, reused by every run).
#   * For (model, cohort, seed, fold), runs full training with early
#     stopping and saves per-fold artifacts:
#         predictions.npz   y_true, y_pred, y_probs, subject_id,
#                           recording_epoch_idx, recording_idx
#         latents.npz       z  (only when model has latent — V8)
#         metrics.json      per-fold metrics
#         checkpoint.pt     best val checkpoint
#         DONE              marker written last (resume safety)
#   * Resume-safe — folds already DONE are skipped.
#   * Two-GPU coordination via --gpu_slot N --total_slots M:
#       slot 0 runs experiments at index 0, 2, 4, ...
#       slot 1 runs                       1, 3, 5, ...
#     No shared file writes (each (seed,fold) has its own directory).
#
# USAGE
#   # smoke test (5 epochs, 1 fold, one seed)
#   python harness.py --model v8 --cohort cfs --seeds 42 --folds 0 --quick
#
#   # full V8 run on CFS (5 folds x 3 seeds)
#   python harness.py --model v8 --cohort cfs --seeds 42 1337 2024 --folds 0 1 2 3 4
#
#   # two-GPU split across one experiment grid
#   # terminal 1
#   CUDA_VISIBLE_DEVICES=4 python harness.py --model v8 --cohort mesa \
#       --seeds 42 1337 2024 --folds 0 1 2 3 4 --gpu_slot 0 --total_slots 2
#   # terminal 2
#   CUDA_VISIBLE_DEVICES=5 python harness.py --model v8 --cohort mesa \
#       --seeds 42 1337 2024 --folds 0 1 2 3 4 --gpu_slot 1 --total_slots 2
#
# OUTPUT STRUCTURE
#   /data2/Akbar1/PPG_Stages/benchmark_results/
#       folds/
#           mesa_folds.json
#           cfs_folds.json
#       {cohort}/
#           {model_name}/
#               seed{S}_fold{F}/
#                   predictions.npz   latents.npz   metrics.json
#                   checkpoint.pt     DONE
# =========================================================================

import os
import sys
import json
import time
import argparse
import random
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import butter, sosfiltfilt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score, confusion_matrix,
    roc_auc_score, average_precision_score,
)
from sklearn.preprocessing import label_binarize

# local imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import build_model, has_latent, MODEL_REGISTRY


# =========================================================================
# CONFIG (the things that are global to every experiment)
# =========================================================================
PROJECT_ROOT = Path("/data2/Akbar1/PPG_Stages")
RESULTS_ROOT = PROJECT_ROOT / "benchmark_results"

# cohort -> (manifest_path, npz_dir hint for sanity)
COHORT_PATHS = {
    "mesa": dict(
        manifest=PROJECT_ROOT / "npz_mesa_ppg" / "manifest_mesa_ppg.csv",
        npz_dir =PROJECT_ROOT / "npz_mesa_ppg",
    ),
    "cfs": dict(
        manifest=PROJECT_ROOT / "npz_cfs_ppg" / "manifest_CFS_ppg.csv",
        npz_dir =PROJECT_ROOT / "npz_cfs_ppg",
    ),
}

# class config — 4-class is the paper's primary task
NUM_CLASSES = 4
LABELS = {0: "Wake", 1: "Light", 2: "Deep", 3: "REM"}
LABEL_MAP = np.array([0, 1, 1, 2, 3], dtype=np.int64)   # W,N1,N2,N3,REM -> W,L,L,D,R

# signal
FS = 125
EPOCH_SEC = 30
T_EPOCH = FS * EPOCH_SEC                     # 3750

# k-fold
N_FOLDS = 5
FOLD_SEED = 42                                # FIXED, NEVER VARIES

# training
EPOCHS_CAP = 50
PATIENCE = 12
BATCH_SUBJ = 4
MAX_HOURS_TRAIN = 5.0
GRAD_CLIP = 1.0
BASE_LR = 2e-4
MAX_LR = 5e-4
WEIGHT_DECAY = 0.05

# loss
LOSS_WEIGHTS = dict(ce=1.0, focal=0.5, smooth=0.05)


# =========================================================================
# Device & AMP
# =========================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = (device.type == "cuda")


def _bf16_ok():
    try:
        return torch.cuda.is_bf16_supported()
    except Exception:
        return False


AMP_DTYPE = torch.bfloat16 if (AMP_ENABLED and _bf16_ok()) else torch.float32


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================================
# Signal preprocessing
# =========================================================================
_SOS = butter(2, [0.05 / (FS * 0.5), 10.0 / (FS * 0.5)], btype="bandpass", output="sos")


def bandpass(x):
    try:
        return sosfiltfilt(_SOS, x, axis=-1).astype(np.float32)
    except ValueError:
        return sosfiltfilt(_SOS, x, axis=-1, padlen=0).astype(np.float32)


def preprocess_recording(x, clip=10.0):
    x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    x = bandpass(x)
    flat = x.reshape(-1)
    med = float(np.median(flat))
    mad = float(np.median(np.abs(flat - med)) + 1e-6)
    x = (x - med) / (1.4826 * mad)
    x = np.clip(x, -clip, clip)
    return np.nan_to_num(x, nan=0.0, posinf=clip, neginf=-clip).astype(np.float32)


# =========================================================================
# Augmentation (only during training)
# =========================================================================
class PPGAugmentSafe:
    def __init__(self, p_noise=0.5, noise_std=0.015, p_gain=0.5, gain_range=(0.9, 1.1),
                 p_drift=0.4, drift_amp=0.3, drift_freq_max=0.05,
                 p_mask=0.3, mask_max_ms=200, mask_n_max=2, p_chan_dropout=0.05):
        self.p_noise = p_noise; self.noise_std = noise_std
        self.p_gain = p_gain; self.gain_range = gain_range
        self.p_drift = p_drift; self.drift_amp = drift_amp; self.drift_freq_max = drift_freq_max
        self.p_mask = p_mask; self.mask_max_ms = mask_max_ms; self.mask_n_max = mask_n_max
        self.p_chan_dropout = p_chan_dropout

    def __call__(self, x):
        x = x.copy()
        E, T = x.shape
        if np.random.rand() < self.p_gain:
            x = x * np.random.uniform(*self.gain_range, size=(E, 1)).astype(np.float32)
        if np.random.rand() < self.p_noise:
            std = x.std(axis=1, keepdims=True) + 1e-6
            x = x + np.random.randn(*x.shape).astype(np.float32) * (self.noise_std * std)
        if np.random.rand() < self.p_drift:
            freq = np.random.uniform(0.005, self.drift_freq_max)
            phase = np.random.uniform(0, 2 * np.pi)
            t = np.arange(T) / FS
            x = x + (self.drift_amp * np.sin(2 * np.pi * freq * t + phase)).astype(np.float32)[None, :]
        if np.random.rand() < self.p_mask:
            max_len = int(self.mask_max_ms * FS / 1000)
            if max_len > 4:
                for i in range(E):
                    for _ in range(np.random.randint(1, self.mask_n_max + 1)):
                        seg = np.random.randint(4, max_len)
                        s = np.random.randint(0, T - seg)
                        taper = 0.5 * (1 - np.cos(np.linspace(0, 2 * np.pi, seg))).astype(np.float32)
                        x[i, s:s + seg] = x[i, s:s + seg] * (1 - taper)
        if np.random.rand() < self.p_chan_dropout:
            x[np.random.randint(0, E)] = 0.0
        return x.astype(np.float32)


# =========================================================================
# Dataset — raw PPG only, with epoch_idx for ultradian PE
# =========================================================================
class SleepDataset(Dataset):
    """
    Yields per-recording:
        x       : (L, 1, T_EPOCH)   raw PPG
        y       : (L,)              4-class labels (-1 epochs already removed)
        idx     : (L,)              epoch index within ORIGINAL recording
                                    (preserves absolute night-time across the random crop)
        mask    : (L,)              valid mask (all True before collate padding)
        subj_id : str               for metadata
        rec_idx : int               position in the test-set list (set by caller)
        path    : str               npz_path for downstream HR extraction
    """
    def __init__(self, df, mode="train", max_hours=MAX_HOURS_TRAIN, augmentor=None):
        self.paths = df["npz_path"].tolist()
        self.subj_ids = df["subject_id"].astype(str).tolist()
        self.mode = mode
        self.max_hours = max_hours if mode == "train" else None
        self.augmentor = augmentor if mode == "train" else None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        sid = self.subj_ids[i]
        d = np.load(p, allow_pickle=True)
        x = d["x"].astype(np.float32)
        y = d["y"].astype(np.int64)
        keep = y >= 0
        x = x[keep]; y = y[keep]
        if y.size == 0:
            x = np.zeros((1, T_EPOCH), dtype=np.float32); y = np.zeros((1,), dtype=np.int64)
        y = LABEL_MAP[y]
        x = preprocess_recording(x)

        E = len(y)
        offset = 0
        if self.max_hours is not None:
            L = min(int((self.max_hours * 3600) / EPOCH_SEC), E)
            if E > L:
                offset = np.random.randint(0, E - L + 1)
                x = x[offset:offset + L]; y = y[offset:offset + L]; E = L
        epoch_idx = np.arange(offset, offset + E, dtype=np.int64)

        if self.augmentor is not None:
            x = self.augmentor(x)
        x = np.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)

        return dict(
            x=torch.from_numpy(x).unsqueeze(1),
            y=torch.from_numpy(y).long(),
            idx=torch.from_numpy(epoch_idx).long(),
            mask=torch.ones(E, dtype=torch.bool),
            subj_id=sid, path=str(p),
        )


def collate(batch):
    Ls = [b["x"].shape[0] for b in batch]
    Lmax = max(Ls)
    xs, ys, eis, ms = [], [], [], []
    subj_ids, paths = [], []
    for b in batch:
        x, y, ei, m = b["x"], b["y"], b["idx"], b["mask"]
        L = x.shape[0]; pad = Lmax - L
        if pad > 0:
            x  = torch.cat([x,  torch.zeros((pad, 1, T_EPOCH), dtype=x.dtype)], 0)
            y  = torch.cat([y,  torch.zeros((pad,), dtype=y.dtype)], 0)
            ei = torch.cat([ei, torch.zeros((pad,), dtype=ei.dtype)], 0)
            m  = torch.cat([m,  torch.zeros((pad,), dtype=torch.bool)], 0)
        xs.append(x); ys.append(y); eis.append(ei); ms.append(m)
        subj_ids.append(b["subj_id"]); paths.append(b["path"])
    return (torch.stack(xs), torch.stack(ys), torch.stack(eis), torch.stack(ms),
            subj_ids, paths)


# =========================================================================
# Frozen subject-level 5-fold splits
# =========================================================================
def get_or_make_folds(cohort: str) -> dict:
    """One frozen split per cohort, reused by every model/seed/fold."""
    fold_path = RESULTS_ROOT / "folds" / f"{cohort}_folds.json"
    fold_path.parent.mkdir(parents=True, exist_ok=True)
    if fold_path.exists():
        with open(fold_path) as f:
            return json.load(f)

    manifest_path = COHORT_PATHS[cohort]["manifest"]
    df = pd.read_csv(manifest_path)
    subjects = sorted(df["subject_id"].astype(str).unique().tolist())
    rng = np.random.default_rng(FOLD_SEED)
    shuffled = list(subjects)
    rng.shuffle(shuffled)
    folds = [list(map(str, f)) for f in np.array_split(shuffled, N_FOLDS)]

    info = dict(cohort=cohort, n_folds=N_FOLDS, fold_seed=FOLD_SEED,
                n_subjects=len(subjects), folds=folds)
    with open(fold_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"  frozen folds written -> {fold_path}  ({len(subjects)} subjects)")
    return info


def split_dfs_for_fold(manifest_df, fold_info, fold_idx):
    """
    Convention: test = folds[k], val = folds[(k+1) % n], train = the rest.
    Each subject appears in test exactly once across the 5 folds.
    """
    n = fold_info["n_folds"]
    folds = fold_info["folds"]
    test_subj = set(folds[fold_idx])
    val_subj  = set(folds[(fold_idx + 1) % n])
    train_subj = set()
    for i, f in enumerate(folds):
        if i != fold_idx and i != (fold_idx + 1) % n:
            train_subj.update(f)
    sid = manifest_df["subject_id"].astype(str)
    tr = manifest_df[sid.isin(train_subj)].copy()
    va = manifest_df[sid.isin(val_subj)].copy()
    te = manifest_df[sid.isin(test_subj)].copy()
    return tr, va, te


# =========================================================================
# Loss
# =========================================================================
def _masked_weighted_ce(logits, y, mask, weights):
    B, L, C = logits.shape
    lf = logits.reshape(-1, C); yf = y.reshape(-1); mf = mask.reshape(-1)
    if mf.sum() == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(lf[mf], yf[mf], weight=weights)


def _masked_focal(logits, y, mask, gamma=1.5, weights=None):
    B, L, C = logits.shape
    lf = logits.reshape(-1, C); yf = y.reshape(-1); mf = mask.reshape(-1)
    if mf.sum() == 0:
        return logits.sum() * 0.0
    lf = lf[mf]; yf = yf[mf]
    logp = F.log_softmax(lf, -1)
    py = logp.exp().gather(1, yf.unsqueeze(1)).squeeze(1).clamp(0, 1)
    lp = logp.gather(1, yf.unsqueeze(1)).squeeze(1)
    loss = -((1 - py) ** gamma) * lp
    if weights is not None:
        loss = loss * weights[yf]
    return loss.mean()


def _autonomic_smoothness(latent, mask):
    if latent is None:
        return torch.zeros((), device=mask.device)
    B, L, K = latent.shape
    diff = latent[:, 1:, :] - latent[:, :-1, :]
    pair_valid = (mask[:, 1:] & mask[:, :-1]).unsqueeze(-1)
    sq = (diff ** 2) * pair_valid
    denom = pair_valid.sum().clamp(min=1) * K
    return sq.sum() / denom


def combined_loss(out, y, mask, class_weights):
    L_ce    = _masked_weighted_ce(out["main"], y, mask, class_weights)
    L_focal = _masked_focal(out["main"], y, mask, 1.5, class_weights)
    L_sm    = _autonomic_smoothness(out.get("latent"), mask)
    return (LOSS_WEIGHTS["ce"] * L_ce
            + LOSS_WEIGHTS["focal"] * L_focal
            + LOSS_WEIGHTS["smooth"] * L_sm)


# =========================================================================
# Class weights (computed from training fold only)
# =========================================================================
def compute_class_weights(train_df):
    cnt = np.zeros(NUM_CLASSES, dtype=np.int64)
    for p in tqdm(train_df["npz_path"].tolist(), desc="class-weights", leave=False):
        d = np.load(p, allow_pickle=True)
        y = d["y"]
        y = y[y >= 0]
        y = LABEL_MAP[y]
        for c in range(NUM_CLASSES):
            cnt[c] += int((y == c).sum())
    inv = 1.0 / np.sqrt(cnt + 1e-6)
    w = np.clip(inv / inv.mean(), 0.5, 3.0)
    return torch.tensor(w, dtype=torch.float32, device=device), cnt.tolist()


# =========================================================================
# Train / eval
# =========================================================================
def train_epoch(model, loader, optimizer, scheduler, class_weights):
    model.train()
    running = 0.0; seen = 0; skipped = 0
    for raw, y, ei, mask, _sids, _paths in loader:
        raw = raw.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        ei  = ei.to(device, non_blocking=True);  mask = mask.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=AMP_ENABLED, dtype=AMP_DTYPE):
            out = model(raw, ei, mask)
            loss = combined_loss(out, y, mask, class_weights)
        if not torch.isfinite(loss):
            skipped += 1
            optimizer.zero_grad(set_to_none=True)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step(); scheduler.step()
        n = int(mask.sum().item())
        running += float(loss.item()) * n; seen += n
    return running / max(seen, 1), skipped


def _multiclass_auc(y_true, probs, C):
    Y = label_binarize(y_true, classes=list(range(C)))
    a, p = [], []
    for c in range(C):
        if Y[:, c].sum() == 0:
            continue
        try:
            a.append(roc_auc_score(Y[:, c], probs[:, c]))
            p.append(average_precision_score(Y[:, c], probs[:, c]))
        except Exception:
            pass
    return (float(np.mean(a)) if a else float("nan"),
            float(np.mean(p)) if p else float("nan"))


def expected_calibration_error(y_true, probs, n_bins=15):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == y_true).astype(np.float32)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() > 0:
            ece += (m.sum() / len(conf)) * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def metrics_from_predictions(y_true, y_pred, probs):
    auroc, auprc = _multiclass_auc(y_true, probs, NUM_CLASSES)
    return {
        "n_epochs": int(len(y_true)),
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "f1_per_class": {LABELS[i]: float(f1_score((y_true == i).astype(int),
                                                   (y_pred == i).astype(int)))
                         for i in range(NUM_CLASSES)},
        "cm": confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES))).tolist(),
        "AUROC": auroc, "AUPRC": auprc,
        "ECE": expected_calibration_error(y_true, probs),
    }


@torch.no_grad()
def eval_for_selection(model, loader):
    """Quick val/test metrics for early-stopping decisions."""
    model.eval()
    ys, preds, probs_l = [], [], []
    for raw, y, ei, mask, _sids, _paths in loader:
        raw = raw.to(device); y = y.to(device); ei = ei.to(device); mask = mask.to(device)
        with torch.cuda.amp.autocast(enabled=AMP_ENABLED, dtype=AMP_DTYPE):
            out = model(raw, ei, mask)
        probs = torch.softmax(out["main"].float(), -1)
        pred = probs.argmax(-1)
        yv = y[mask].cpu().numpy(); pv = pred[mask].cpu().numpy(); pr = probs[mask].cpu().numpy()
        if yv.size == 0:
            continue
        ys.append(yv); preds.append(pv); probs_l.append(pr)
    y_true = np.concatenate(ys); y_pred = np.concatenate(preds); probs = np.concatenate(probs_l)
    return metrics_from_predictions(y_true, y_pred, probs)


@torch.no_grad()
def eval_and_dump(model, loader, model_has_latent: bool, out_dir: Path):
    """
    Final test-fold evaluation. Saves:
        predictions.npz   y_true, y_pred, y_probs, subject_id, epoch_idx, recording_idx
        latents.npz       z (only if model has latent)
        metrics.json
    Returns the metrics dict.
    """
    model.eval()
    y_all, p_all, pr_all = [], [], []
    sid_all, ei_all, ri_all = [], [], []
    lat_all = []
    rec_meta = []                              # parallel list of (rec_idx, subj_id, path)
    rec_idx = 0

    for raw, y, ei, mask, sids, paths in loader:
        raw = raw.to(device); y = y.to(device); ei = ei.to(device); mask = mask.to(device)
        with torch.cuda.amp.autocast(enabled=AMP_ENABLED, dtype=AMP_DTYPE):
            out = model(raw, ei, mask)
        probs = torch.softmax(out["main"].float(), -1)
        pred = probs.argmax(-1)
        B = raw.shape[0]
        for b in range(B):
            mb = mask[b]
            if int(mb.sum().item()) == 0:
                continue
            y_all.append(y[b, mb].cpu().numpy())
            p_all.append(pred[b, mb].cpu().numpy())
            pr_all.append(probs[b, mb].cpu().numpy())
            sid_all.append(np.array([sids[b]] * int(mb.sum().item()), dtype=object))
            ei_all.append(ei[b, mb].cpu().numpy())
            ri_all.append(np.full(int(mb.sum().item()), rec_idx, dtype=np.int64))
            if model_has_latent and out.get("latent") is not None:
                lat_all.append(out["latent"][b, mb].float().cpu().numpy())
            rec_meta.append(dict(recording_idx=int(rec_idx),
                                 subject_id=str(sids[b]),
                                 npz_path=str(paths[b]),
                                 n_epochs=int(mb.sum().item())))
            rec_idx += 1

    y_true = np.concatenate(y_all)
    y_pred = np.concatenate(p_all)
    probs  = np.concatenate(pr_all)
    sids   = np.concatenate(sid_all).astype(str)
    eis    = np.concatenate(ei_all)
    ris    = np.concatenate(ri_all)

    np.savez(out_dir / "predictions.npz",
             y_true=y_true.astype(np.int64),
             y_pred=y_pred.astype(np.int64),
             y_probs=probs.astype(np.float32),
             subject_id=sids,
             recording_epoch_idx=eis.astype(np.int64),
             recording_idx=ris.astype(np.int64))
    if lat_all:
        z = np.concatenate(lat_all, axis=0).astype(np.float32)
        np.savez(out_dir / "latents.npz", z=z)
    with open(out_dir / "recording_metadata.json", "w") as f:
        json.dump(rec_meta, f, indent=2)

    metrics = metrics_from_predictions(y_true, y_pred, probs)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


# =========================================================================
# One full training run for a single (model, cohort, seed, fold)
# =========================================================================
def get_param_groups(model, wd=WEIGHT_DECAY):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or n.endswith(".bias") or "norm" in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0}]


def run_one(model_name, cohort, seed, fold, fold_info, manifest_df, args):
    out_dir = RESULTS_ROOT / cohort / model_name / f"seed{seed}_fold{fold}"
    if args.quick:
        out_dir = RESULTS_ROOT / cohort / model_name / f"SMOKE_seed{seed}_fold{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if (out_dir / "DONE").exists() and not args.quick:
        print(f"[skip] {out_dir.relative_to(RESULTS_ROOT)}  (DONE exists)")
        return None

    print(f"\n{'='*78}\n[run] {model_name} | {cohort} | seed {seed} | fold {fold}"
          f"\n      out: {out_dir}\n{'='*78}")
    t0 = time.time()
    set_seed(seed)

    tr_df, va_df, te_df = split_dfs_for_fold(manifest_df, fold_info, fold)
    print(f"      subjects: train={tr_df['subject_id'].nunique()} "
          f"val={va_df['subject_id'].nunique()} test={te_df['subject_id'].nunique()}")
    print(f"      recordings: train={len(tr_df)} val={len(va_df)} test={len(te_df)}")

    cw, cnt = compute_class_weights(tr_df)
    print(f"      class counts (train): {dict(zip(LABELS.values(), cnt))}")
    print(f"      class weights        : "
          f"{ {LABELS[i]: round(float(cw[i].item()), 3) for i in range(NUM_CLASSES)} }")

    augmentor = PPGAugmentSafe()
    tr_ds = SleepDataset(tr_df, mode="train", augmentor=augmentor)
    va_ds = SleepDataset(va_df, mode="eval")
    te_ds = SleepDataset(te_df, mode="eval")
    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SUBJ, shuffle=True,
                           num_workers=6, pin_memory=True, collate_fn=collate,
                           persistent_workers=True)
    va_loader = DataLoader(va_ds, batch_size=1, shuffle=False,
                           num_workers=2, pin_memory=True, collate_fn=collate)
    te_loader = DataLoader(te_ds, batch_size=1, shuffle=False,
                           num_workers=2, pin_memory=True, collate_fn=collate)

    model = build_model(model_name, num_classes=NUM_CLASSES).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"      model: {model_name}  params: {n_params:.2f} M  "
          f"latent: {has_latent(model_name)}")

    epochs = 5 if args.quick else EPOCHS_CAP
    optimizer = torch.optim.AdamW(get_param_groups(model, WEIGHT_DECAY),
                                  lr=BASE_LR, betas=(0.9, 0.98))
    steps = max(1, len(tr_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=MAX_LR, epochs=epochs, steps_per_epoch=steps,
        pct_start=0.10, div_factor=20.0, final_div_factor=200.0)

    best_val_mf1 = -1.0; best_epoch = -1; since_improve = 0
    ckpt_path = out_dir / "checkpoint.pt"
    history = []

    for ep in range(1, epochs + 1):
        tr_loss, n_skip = train_epoch(model, tr_loader, optimizer, scheduler, cw)
        val_m = eval_for_selection(model, va_loader)
        history.append(dict(epoch=ep, train_loss=tr_loss, val=val_m))

        v_mf1 = val_m["macro_f1"]; v_k = val_m["kappa"]
        v_deep = val_m["f1_per_class"]["Deep"]; v_rem = val_m["f1_per_class"]["REM"]
        flag = ""
        if v_mf1 > best_val_mf1:
            best_val_mf1 = v_mf1; best_epoch = ep; since_improve = 0
            torch.save({"model_state": model.state_dict(),
                        "model_name": model_name, "seed": seed, "fold": fold,
                        "epoch": ep, "val_metrics": val_m,
                        "n_params_M": n_params}, ckpt_path)
            flag = "  <-- best"
        else:
            since_improve += 1
        print(f"  ep{ep:02d} | tr={tr_loss:.4f} | val mF1={v_mf1:.4f} k={v_k:.4f} "
              f"Deep={v_deep:.4f} REM={v_rem:.4f} | skip={n_skip}{flag}")
        if (not args.quick) and since_improve >= PATIENCE:
            print(f"      early stop at epoch {ep} (best {best_epoch})")
            break

    # final test eval with the best checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = eval_and_dump(model, te_loader, has_latent(model_name), out_dir)

    summary = dict(
        model=model_name, cohort=cohort, seed=seed, fold=fold,
        best_epoch=best_epoch, best_val_macroF1=best_val_mf1,
        test_metrics=test_metrics, n_params_M=n_params,
        wall_time_min=round((time.time() - t0) / 60.0, 2),
    )
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2, default=str)

    # write DONE last (resume safety)
    if not args.quick:
        (out_dir / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"\n[done] {model_name} | {cohort} | seed{seed} fold{fold}"
          f"  test mF1={test_metrics['macro_f1']:.4f}  "
          f"Deep={test_metrics['f1_per_class']['Deep']:.4f}  "
          f"REM={test_metrics['f1_per_class']['REM']:.4f}"
          f"  ({summary['wall_time_min']} min)")

    del model, optimizer, scheduler, tr_loader, va_loader, te_loader
    torch.cuda.empty_cache()
    return summary


# =========================================================================
# Driver
# =========================================================================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True,
                    help=f"one of: {list(MODEL_REGISTRY)}")
    ap.add_argument("--cohort", type=str, required=True, choices=list(COHORT_PATHS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(N_FOLDS)))
    ap.add_argument("--gpu_slot", type=int, default=0,
                    help="this process's slot index (0-based) for two-GPU split")
    ap.add_argument("--total_slots", type=int, default=1,
                    help="total parallel processes; this one runs every Mth experiment")
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: 5 epochs, no early stopping, separate output dir")
    return ap.parse_args()


def main():
    args = parse_args()

    print(f"device: {device} | AMP dtype: {AMP_DTYPE}")
    print(f"results root: {RESULTS_ROOT}")
    print(f"model: {args.model}  cohort: {args.cohort}")
    print(f"seeds: {args.seeds}  folds: {args.folds}")
    print(f"gpu slot {args.gpu_slot} of {args.total_slots}  quick={args.quick}")

    if args.model not in MODEL_REGISTRY:
        raise SystemExit(f"unknown model '{args.model}'. choose from {list(MODEL_REGISTRY)}")

    manifest_df = pd.read_csv(COHORT_PATHS[args.cohort]["manifest"])
    fold_info = get_or_make_folds(args.cohort)

    # build the linear list of (seed, fold) jobs, then take this slot's slice
    jobs = [(s, f) for s in args.seeds for f in args.folds]
    my_jobs = [j for i, j in enumerate(jobs) if (i % args.total_slots) == args.gpu_slot]
    print(f"this slot will run {len(my_jobs)} / {len(jobs)} jobs:")
    for s, f in my_jobs:
        print(f"  - seed {s}  fold {f}")

    for s, f in my_jobs:
        try:
            run_one(args.model, args.cohort, s, f, fold_info, manifest_df, args)
        except Exception as e:
            print(f"\n!! FAILED: {args.model} {args.cohort} seed{s} fold{f}: {e}")
            import traceback; traceback.print_exc()

    print("\nall requested jobs in this slot are complete.")


if __name__ == "__main__":
    main()