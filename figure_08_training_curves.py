# =========================================================================
# figure_08_training_curves.py  (v2 — corrected for nested val format)
#
# Two-panel training-dynamics figure for V8.
#
# Harness format (confirmed):
#   history.json is a LIST of per-epoch dicts. Each entry looks like:
#     {
#       "epoch":       int,
#       "train_loss":  float,
#       "val": {
#         "kappa":     float,
#         "macro_f1":  float,
#         "acc":       float,
#         ...
#       }
#     }
#
#   No val_loss is saved (harness only tracks train loss during training and
#   validation is used for metric-based early stopping via kappa). So the
#   loss panel shows train loss only; the right panel shows val kappa.
#
# DESIGN
#   Left  — Training loss, mean +/- std across all completed folds per cohort
#   Right — Validation Cohen's kappa, mean +/- std across folds
#   Dotted vertical line on kappa panel = mean epoch of peak val kappa
#     (useful for reviewers to see when early-stopping typically fires)
# =========================================================================

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from figstyle import apply_style, COHORT_COLORS, PALETTE, save

apply_style()

ROOT     = Path("/data2/Akbar1/PPG_Stages/benchmark_results")
OUT_DIR  = ROOT / "_paper_results" / "figures"


# -------------------------------------------------------------------------
# History loader — knows about the nested "val" dict format
# -------------------------------------------------------------------------
def load_one_history(path):
    """
    Returns dict mapping canonical_key -> 1D numpy array.
    Canonical keys: train_loss, val_kappa, val_macrof1, epoch.
    Missing keys are simply absent from the returned dict.
    """
    with open(path) as f:
        h = json.load(f)
    if not isinstance(h, list) or not h or not isinstance(h[0], dict):
        return {}

    out = {}
    n = len(h)

    # Top-level scalar per-epoch fields
    if "train_loss" in h[0]:
        out["train_loss"] = np.array(
            [r.get("train_loss", np.nan) for r in h], dtype=np.float64)
    if "epoch" in h[0]:
        out["epoch"] = np.array(
            [r.get("epoch", np.nan) for r in h], dtype=np.float64)

    # Nested validation metrics (under key "val" or "valid")
    val_key = "val" if "val" in h[0] else ("valid" if "valid" in h[0] else None)
    if val_key:
        val_dicts = [r.get(val_key, {}) if isinstance(r.get(val_key), dict)
                     else {} for r in h]
        if any("kappa" in v for v in val_dicts):
            out["val_kappa"] = np.array(
                [v.get("kappa", np.nan) for v in val_dicts], dtype=np.float64)
        if any("macro_f1" in v for v in val_dicts):
            out["val_macrof1"] = np.array(
                [v.get("macro_f1", np.nan) for v in val_dicts], dtype=np.float64)

    return out


def load_v8_histories(cohort):
    out = []
    for fold_dir in sorted((ROOT / cohort / "v8").glob("seed*_fold*")):
        if not (fold_dir / "DONE").exists():
            continue
        hist_path = fold_dir / "history.json"
        if not hist_path.exists():
            continue
        try:
            norm = load_one_history(hist_path)
        except Exception as e:
            print(f"  [skip] {fold_dir.name}: {e}")
            continue
        if not norm:
            continue
        out.append(norm)
    return out


def aggregate(histories, key):
    """Stack across runs padding shorter with NaN; return (epochs, mean, std)."""
    arrs = [h[key] for h in histories if key in h]
    if not arrs:
        return None, None, None
    max_len = max(len(a) for a in arrs)
    M = np.full((len(arrs), max_len), np.nan, dtype=np.float64)
    for i, a in enumerate(arrs):
        M[i, :len(a)] = a
    return (np.arange(1, max_len + 1),
            np.nanmean(M, axis=0),
            np.nanstd(M, axis=0))


# -------------------------------------------------------------------------
# Load
# -------------------------------------------------------------------------
print("Loading V8 training histories ...")
data = {}
for cohort in ["mesa", "cfs"]:
    print(f"\n[{cohort.upper()}]")
    hs = load_v8_histories(cohort)
    print(f"  loaded {len(hs)} fold histories")
    if hs:
        print(f"  keys in first fold: {sorted(hs[0].keys())}")
        for k in ["train_loss", "val_kappa"]:
            if k in hs[0]:
                a = hs[0][k]
                print(f"     {k}: n_epochs={len(a)}, "
                      f"first 3 = {a[:3].round(3).tolist()}, "
                      f"last 3 = {a[-3:].round(3).tolist()}")
    data[cohort] = hs

if not any(data[c] for c in ("mesa", "cfs")):
    raise SystemExit("No V8 histories loaded — nothing to plot.")


# -------------------------------------------------------------------------
# Plot
# -------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))

# ---- Left: training loss ----
ax = axes[0]
for cohort in ("mesa", "cfs"):
    if not data[cohort]:
        continue
    color = COHORT_COLORS[cohort]
    e, m, s = aggregate(data[cohort], "train_loss")
    if e is None:
        continue
    ax.plot(e, m, color=color, linewidth=1.4,
            label=f"{cohort.upper()}  (n = {len(data[cohort])} folds)")
    ax.fill_between(e, m - s, m + s, color=color, alpha=0.15, linewidth=0)
ax.set_xlabel("Epoch")
ax.set_ylabel("Training loss")
ax.set_title("Training loss", fontsize=10, pad=6)
ax.legend(loc="upper right", frameon=False, fontsize=7.5)

# ---- Right: val kappa + best-epoch marker ----
ax = axes[1]
for cohort in ("mesa", "cfs"):
    if not data[cohort]:
        continue
    color = COHORT_COLORS[cohort]
    e, m, s = aggregate(data[cohort], "val_kappa")
    if e is None:
        continue
    ax.plot(e, m, color=color, linewidth=1.4,
            label=f"{cohort.upper()}  (n = {len(data[cohort])} folds)")
    ax.fill_between(e, m - s, m + s, color=color, alpha=0.15, linewidth=0)

    # mean best epoch per cohort
    best_eps = []
    for h in data[cohort]:
        if "val_kappa" in h and np.any(np.isfinite(h["val_kappa"])):
            best_eps.append(int(np.nanargmax(h["val_kappa"])) + 1)
    if best_eps:
        mean_best = float(np.mean(best_eps))
        ax.axvline(mean_best, color=color, linewidth=0.7,
                   linestyle=":", alpha=0.7)
        # small label near bottom
        ymin = ax.get_ylim()[0] if ax.get_ylim()[1] > ax.get_ylim()[0] else 0
        ax.text(mean_best + 0.4, ymin + 0.02,
                f"best ~ epoch {mean_best:.0f}",
                color=color, fontsize=6.5, va="bottom", ha="left")

ax.set_xlabel("Epoch")
ax.set_ylabel(r"Validation Cohen's $\kappa$")
ax.set_title(r"Validation $\kappa$ across training", fontsize=10, pad=6)
ax.legend(loc="lower right", frameon=False, fontsize=7.5)

fig.tight_layout()
OUT_DIR.mkdir(parents=True, exist_ok=True)
save(fig, OUT_DIR / "figure_08_training_curves")
plt.close(fig)
print(f"\nFigure 8 saved to {OUT_DIR / 'figure_08_training_curves.pdf'}")