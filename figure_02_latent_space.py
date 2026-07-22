# =========================================================================
# figure_02_latent_space.py
#
# Figure 2 — Latent-space structure.
# Visualize V8's 32-dim autonomic latent via PCA (linear) and t-SNE
# (non-linear), colored by sleep stage. 2 rows × 2 cols:
#     row 0: PCA      (MESA, CFS)
#     row 1: t-SNE    (MESA, CFS)
#
# Subsamples to N_PER_STAGE epochs per stage from one V8 fold per cohort
# (balanced sampling so visual class density is comparable).
#
# Output: _paper_results/figures/figure_02_latent_space.pdf  (300 DPI)
# =========================================================================

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from figstyle import apply_style, STAGE_ORDER, STAGE_COLORS, PALETTE, save

apply_style()

ROOT    = Path("/data2/Akbar1/PPG_Stages/benchmark_results")
OUT_DIR = ROOT / "_paper_results" / "figures"

N_PER_STAGE   = 1500      # 1500 × 4 = 6 000 points per cohort
RANDOM_SEED   = 0
LABEL_MAP     = {0: "Wake", 1: "Light", 2: "Deep", 3: "REM"}


def load_fold_latents(fold_dir):
    lat  = np.load(fold_dir / "latents.npz")
    pred = np.load(fold_dir / "predictions.npz")
    return lat["z"], pred["y_true"]


fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.6))

for col, cohort in enumerate(["mesa", "cfs"]):
    fold_dir = ROOT / cohort / "v8" / "seed42_fold0"
    Z, y = load_fold_latents(fold_dir)

    # Balanced subsample
    rng = np.random.default_rng(RANDOM_SEED)
    keep = []
    for stage_val in [0, 1, 2, 3]:
        idx = np.where(y == stage_val)[0]
        n   = min(N_PER_STAGE, len(idx))
        keep.append(rng.choice(idx, n, replace=False))
    keep = np.concatenate(keep)
    rng.shuffle(keep)

    Zs = Z[keep].astype(np.float32)
    ys = y[keep]

    print(f"[{cohort.upper()}] PCA on {Zs.shape[0]} points ...")
    pca = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(Zs)

    print(f"[{cohort.upper()}] t-SNE on {Zs.shape[0]} points ...")
    try:
        tsne = TSNE(n_components=2, perplexity=30, random_state=RANDOM_SEED,
                    init="pca", learning_rate="auto").fit_transform(Zs)
    except TypeError:
        # older sklearn doesn't support init="pca" / learning_rate="auto"
        tsne = TSNE(n_components=2, perplexity=30,
                    random_state=RANDOM_SEED).fit_transform(Zs)

    for stage_val in [0, 1, 2, 3]:
        mask  = ys == stage_val
        name  = LABEL_MAP[stage_val]
        color = STAGE_COLORS[name]
        axes[0, col].scatter(pca[mask, 0],  pca[mask, 1],
                              s=2, alpha=0.45, c=color, edgecolors="none")
        axes[1, col].scatter(tsne[mask, 0], tsne[mask, 1],
                              s=2, alpha=0.45, c=color, edgecolors="none")

    axes[0, col].set_title(f"{cohort.upper()}  —  PCA",  fontsize=9, pad=4)
    axes[1, col].set_title(f"{cohort.upper()}  —  t-SNE", fontsize=9, pad=4)
    axes[0, col].set_xlabel("PC 1"); axes[0, col].set_ylabel("PC 2")
    axes[1, col].set_xlabel("t-SNE 1"); axes[1, col].set_ylabel("t-SNE 2")
    for ax in (axes[0, col], axes[1, col]):
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(False)

# Single shared legend across the top
handles = [plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=STAGE_COLORS[s],
                       markersize=7, label=s, markeredgecolor="none")
           for s in STAGE_ORDER]
fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
           bbox_to_anchor=(0.5, 1.01))

fig.suptitle("V8 autonomic latent (32-dim) — stage structure by cohort",
             fontsize=10, y=1.05)
fig.tight_layout(rect=[0, 0, 1, 0.96])

OUT_DIR.mkdir(parents=True, exist_ok=True)
save(fig, OUT_DIR / "figure_02_latent_space")
plt.close(fig)
print(f"\nFigure 2 saved to {OUT_DIR / 'figure_02_latent_space.pdf'}")