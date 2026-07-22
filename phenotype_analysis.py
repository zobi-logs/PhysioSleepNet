# =========================================================================
# phenotype_analysis.py
#
# Figure 4 — Phenotype breakdown of V8 sleep-staging performance.
#
# Generates two figures:
#   Figure 4A (main paper, simpler):
#     Two panels — V8 kappa by AHI severity (both cohorts) +
#     V8 kappa by diabetes status (CFS only).
#
#   Figure 4B (supplementary, comprehensive):
#     Four panels — AHI severity, diabetes status, age decade, BMI category.
#
# Also writes intermediate CSVs to _paper_results/ so the numbers backing
# the figures can be audited and reused.
#
# =========================================================================
# WHY THIS ANALYSIS EXISTS — the scientific story it supports
# =========================================================================
#
# Step 1 (Fig 1, the linear probe): V8's unsupervised 32-dim latent
#   linearly decodes HRV (RMSSD R^2 = 0.44, SDNN R^2 = 0.50 on MESA).
#   So V8 works by reading autonomic signal from PPG.
#
# Step 2 (Fig 4 = this script): if the model reads autonomic signal, then
#   anything that disrupts autonomic signal should hurt performance.
#   Two natural disruptors:
#     a) Sleep apnea (AHI):  apneas cause autonomic arousals that
#        confound HRV. Predict V8 kappa drops as AHI increases.
#     b) Diabetes:           type-2 diabetes causes autonomic neuropathy
#        which degrades HRV at the source. Predict V8 kappa lower in
#        diabetic subjects.
#
# If both predictions hold, the mechanism story is locked: V8's strengths
# AND weaknesses both stem from the autonomic-decoding mechanism. This is
# the central interpretability claim of the paper.
#
# =========================================================================
# DESIGN DECISIONS — DOCUMENTED HERE SO WE NEVER FORGET WHY
# =========================================================================
#
# (1) AHI column choice
#     -----------------
#     We use `nsrr_ahi_hp3u` from the harmonized datasets in BOTH cohorts.
#     This is NSRR's cross-study standardized definition: hypopneas counted
#     when accompanied by either >=3% desaturation OR an arousal,
#     unrestricted. We chose this over `nsrr_ahi_hp4u_aasm15` (strict 4%
#     AASM) because:
#       - It's the most inclusive definition (catches more mild OSA),
#         which gives better separation across our bins.
#       - It's the most commonly cited AHI in sleep epidemiology.
#       - Both MESA and CFS harmonized datasets provide it.
#     CFS also has a raw `ahi` column which is computed slightly
#     differently per-cohort; we intentionally avoid it for cross-cohort
#     comparability.
#
# (2) AHI severity bins
#     -----------------
#     Standard AASM 2017 clinical severity categories:
#         None      AHI < 5
#         Mild      5  <= AHI < 15
#         Moderate  15 <= AHI < 30
#         Severe    AHI >= 30
#     These are clinical convention. Using anything else would invite
#     reviewer pushback.
#
# (3) Diabetes — why CFS only, not MESA
#     ---------------------------------
#     MESA Sleep Dataset 0.8.0 (the one we have at
#     /data1/Zubair_ECG_PPG_P5/mesa/datasets/) does NOT include any
#     metabolic markers — no diabetes flag, no glucose, no HbA1c, no
#     diabetes medications. We confirmed this by:
#       - Searching the data dictionary description text for
#         "diabet|insulin|hba1c|hypergly|glycemic" -> 0 hits
#       - Direct column-name search for "diab|dm|insul|gluc|hba1c" -> 0
#         hits (the apparent dm/insul hits were false positives like
#         `sdmainsleep5` matching "dm" in the middle)
#     The full MESA Exam 5 clinical dataset has diabetes data but it's
#     in a separate NSRR distribution that we don't have here.
#
#     CFS has `diabetesdx` (clinical diagnosis flag, 0/1) directly in
#     the visit5 dataset, so the diabetes analysis is restricted to CFS.
#     This is methodologically clean and reviewer-defensible: the paper
#     states that diabetes analysis is restricted to CFS because of data
#     availability.
#
# (4) Per-subject metric definition
#     -----------------------------
#     We compute Cohen's kappa per subject from V8's predictions on that
#     subject's epochs. Per-subject kappa is the standard sleep-staging
#     metric. macro-F1 is reported too as a secondary metric.
#
# (5) Why seed42 only (not all 3 seeds)
#     ---------------------------------
#     We have 3 random seeds (42, 1337, 2024) for V8, each producing
#     independent predictions for the same subjects. For phenotype
#     analysis we want ONE estimate per subject, not three. Two ways:
#       (a) Use seed42 only -> simpler, one prediction per subject
#       (b) Average per-subject kappa across the 3 seeds -> noisier
#           per-subject but more stable
#     We pick (a) for consistency with Figure 5 (night-level metrics),
#     which also uses seed42 only. The phenotype trend should be visible
#     regardless of seed since differences across seeds are small
#     (visible in the seed-stratified comparison in the main results).
#
# (6) Subject ID mapping
#     ------------------
#     CFS:  predictions use `800002` and metadata uses `nsrrid` = '800002'.
#           Direct string match, no transformation needed.
#     MESA: predictions use `mesa-sleep-0002_v1` and metadata uses integer
#           `mesaid` = 2. We extract the integer from the subject_id
#           with regex and join on it.
#
# (7) Statistical tests
#     -----------------
#     For AHI bins (4 groups): Kruskal-Wallis test (non-parametric,
#       robust to non-normality). Tests whether per-subject kappa differs
#       across the 4 severity groups.
#     For diabetes (2 groups): Mann-Whitney U test (non-parametric).
#       Tests whether per-subject kappa is lower in diabetics.
#     For age decade and BMI categories: same Kruskal-Wallis.
#     We report p-values descriptively (not corrected for multiplicity)
#     because phenotype analysis is exploratory / supportive of the main
#     story, not the headline claim.
#
# (8) Cohort-specific quirks
#     ----------------------
#     CFS has young subjects (age range 21-90) and includes children. We
#     exclude subjects with age < 18 from CFS phenotype analysis because
#     sleep architecture is qualitatively different in children. This is
#     standard practice in adult sleep research. (CFS also has families
#     with parents and offspring, so excluding minors is biologically
#     and statistically the right call.)
#
# =========================================================================

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, f1_score
from scipy.stats import kruskal, mannwhitneyu

from figstyle import apply_style, COHORT_COLORS, PALETTE, save

apply_style()


# =========================================================================
# CONFIGURATION — all paths and constants in one place
# =========================================================================
ROOT     = Path("/data2/Akbar1/PPG_Stages/benchmark_results")
OUT_DIR  = ROOT / "_paper_results"
FIG_DIR  = OUT_DIR / "figures"

# Metadata sources
MESA_HARMONIZED = Path("/data1/Zubair_ECG_PPG_P5/mesa/datasets/"
                       "mesa-sleep-harmonized-dataset-0.8.0.csv")
MESA_FULL       = Path("/data1/Zubair_ECG_PPG_P5/mesa/datasets/"
                       "mesa-sleep-dataset-0.8.0.csv")
CFS_HARMONIZED  = Path("/data1/Zubair_ECG_PPG_P5/cfs/datasets/"
                       "cfs-visit5-harmonized-dataset-0.7.0.csv")
CFS_FULL        = Path("/data1/Zubair_ECG_PPG_P5/cfs/datasets/"
                       "cfs-visit5-dataset-0.7.0.csv")

# AHI severity bins — AASM 2017 standard
AHI_EDGES  = [-0.01, 5.0, 15.0, 30.0, 1000.0]
AHI_LABELS = ["None\n(<5)", "Mild\n(5-15)", "Moderate\n(15-30)", "Severe\n(\u226530)"]

# Age decade bins for supplementary analysis
AGE_EDGES  = [0, 39.99, 49.99, 59.99, 69.99, 79.99, 200.0]
AGE_LABELS = ["<40", "40-49", "50-59", "60-69", "70-79", "\u226580"]

# BMI categories — WHO standard
BMI_EDGES  = [0, 18.5, 25.0, 30.0, 200.0]
BMI_LABELS = ["Under\n(<18.5)", "Normal\n(18.5-25)", "Over\n(25-30)", "Obese\n(\u226530)"]

# Minimum group size for plotting — bins with fewer subjects than this
# are still computed but flagged in the figure
MIN_BIN_SIZE = 10


# =========================================================================
# (A) METADATA LOADERS
# =========================================================================
def load_mesa_metadata():
    """
    Returns a DataFrame indexed by subject_id (in our 'mesa-sleep-XXXX_v1'
    format) with columns: AHI, age, sex, bmi, diabetes (all NaN -- not
    available in MESA Sleep dataset).

    AHI comes from the HARMONIZED file (nsrr_ahi_hp3u, cross-cohort
    comparable). Age/BMI/sex come from the FULL file because the
    harmonized file only has the gt89 censored age flag.
    """
    harm = pd.read_csv(MESA_HARMONIZED, low_memory=False)
    full = pd.read_csv(MESA_FULL,       low_memory=False)

    # Harmonized file uses 'nsrrid', full file uses 'mesaid' -- they are
    # the same integer despite the different column name.
    harm["mesaid"] = pd.to_numeric(harm["nsrrid"], errors="coerce").astype("Int64")
    full["mesaid"] = pd.to_numeric(full["mesaid"], errors="coerce").astype("Int64")

    merged = full.merge(
        harm[["mesaid", "nsrr_ahi_hp3u"]], on="mesaid", how="left"
    )

    out = pd.DataFrame({
        "mesaid":   merged["mesaid"],
        "AHI":      pd.to_numeric(merged["nsrr_ahi_hp3u"], errors="coerce"),
        "age":      pd.to_numeric(merged["sleepage5c"],     errors="coerce"),
        "sex":      pd.to_numeric(merged["gender1"],        errors="coerce"),
        "bmi":      pd.to_numeric(merged["bmi5c"],          errors="coerce"),
        "diabetes": np.nan,         # explicitly NOT available in MESA Sleep
    })
    out = out.dropna(subset=["mesaid"]).copy()
    # Subject id mapping: mesaid=2 -> 'mesa-sleep-0002_v1'
    out["subject_id"] = out["mesaid"].apply(lambda x: f"mesa-sleep-{int(x):04d}_v1")
    out["cohort"] = "MESA"
    return out.set_index("subject_id")


def load_cfs_metadata():
    """
    Returns a DataFrame indexed by subject_id (CFS nsrrid string) with
    columns: AHI, age, sex, bmi, diabetes.

    AHI comes from the HARMONIZED file for cross-cohort comparability.
    Everything else comes from the FULL file.

    Diabetes uses the clinical diagnosis column `diabetesdx` (0/1).
    Subjects < 18 years old are excluded (children -- different sleep
    architecture; standard adult-sleep practice).
    """
    harm = pd.read_csv(CFS_HARMONIZED, low_memory=False)
    full = pd.read_csv(CFS_FULL,       low_memory=False)

    harm["nsrrid"] = harm["nsrrid"].astype(str)
    full["nsrrid"] = full["nsrrid"].astype(str)

    merged = full.merge(
        harm[["nsrrid", "nsrr_ahi_hp3u"]], on="nsrrid", how="left"
    )

    out = pd.DataFrame({
        "nsrrid":   merged["nsrrid"].astype(str),
        "AHI":      pd.to_numeric(merged["nsrr_ahi_hp3u"], errors="coerce"),
        "age":      pd.to_numeric(merged["age"],            errors="coerce"),
        "sex":      pd.to_numeric(merged["sex"],            errors="coerce"),
        "bmi":      pd.to_numeric(merged["bmi"],            errors="coerce"),
        "diabetes": pd.to_numeric(merged["diabetesdx"],     errors="coerce"),
    })

    # Drop children
    out = out[out["age"] >= 18].copy()

    out["subject_id"] = out["nsrrid"]
    out["cohort"]     = "CFS"
    return out.drop(columns="nsrrid").set_index("subject_id")


# =========================================================================
# (B) PER-SUBJECT PERFORMANCE FROM V8 PREDICTIONS
# =========================================================================
def per_subject_performance(cohort):
    """
    Aggregates V8 predictions across all 5 folds (seed42 only -- see design
    note (5) above) and returns one row per test subject with kappa,
    macro-F1, and epoch count.

    Each subject appears in exactly one fold's test set, so there is no
    overlap across folds.
    """
    rows = []
    for fold_dir in sorted((ROOT / cohort / "v8").glob("seed42_fold*")):
        if not (fold_dir / "DONE").exists():
            continue
        pred   = np.load(fold_dir / "predictions.npz")
        y_true = pred["y_true"]
        y_pred = pred["y_pred"]
        subj   = pred["subject_id"]
        for s in np.unique(subj):
            mask = subj == s
            if mask.sum() < 100:    # exclude very short recordings
                continue
            yt = y_true[mask]
            yp = y_pred[mask]
            rows.append({
                "subject_id": str(s),
                "kappa":      float(cohen_kappa_score(yt, yp)),
                "macro_f1":   float(f1_score(yt, yp, average="macro", zero_division=0)),
                "n_epochs":   int(mask.sum()),
            })
    return pd.DataFrame(rows).set_index("subject_id")


# =========================================================================
# (C) JOIN METADATA + PERFORMANCE INTO ONE TABLE
# =========================================================================
def build_phenotype_table():
    print("Loading metadata ...")
    mesa_meta = load_mesa_metadata()
    cfs_meta  = load_cfs_metadata()
    print(f"  MESA metadata: {len(mesa_meta)} subjects")
    print(f"  CFS  metadata: {len(cfs_meta)} subjects (after age>=18 filter)")

    print("Loading per-subject V8 performance (seed42) ...")
    mesa_perf = per_subject_performance("mesa")
    cfs_perf  = per_subject_performance("cfs")
    print(f"  MESA V8 test subjects: {len(mesa_perf)}")
    print(f"  CFS  V8 test subjects: {len(cfs_perf)}")

    mesa_join = mesa_meta.join(mesa_perf, how="inner")
    cfs_join  = cfs_meta.join(cfs_perf,   how="inner")
    print(f"  MESA after join: {len(mesa_join)} subjects "
          f"(meta missing: {len(mesa_perf) - len(mesa_join)})")
    print(f"  CFS  after join: {len(cfs_join)} subjects "
          f"(meta missing: {len(cfs_perf) - len(cfs_join)})")

    # Bin categorical variables
    for df in (mesa_join, cfs_join):
        df["AHI_bin"] = pd.cut(df["AHI"], bins=AHI_EDGES,
                               labels=AHI_LABELS, right=False, ordered=True)
        df["age_bin"] = pd.cut(df["age"], bins=AGE_EDGES,
                               labels=AGE_LABELS, right=False, ordered=True)
        df["bmi_bin"] = pd.cut(df["bmi"], bins=BMI_EDGES,
                               labels=BMI_LABELS, right=False, ordered=True)

    combined = pd.concat([mesa_join, cfs_join]).reset_index()
    return mesa_join.reset_index(), cfs_join.reset_index(), combined


# =========================================================================
# (D) HELPERS for the figures
# =========================================================================
def grouped_stats(df, bin_col, metric="kappa"):
    """
    Returns mean / SEM / count per bin, preserving the categorical order
    of the bin column.
 
    No filtering happens here -- the count is preserved so the plotting
    function can decide what to do with small bins.
    """
    g = df.groupby(bin_col, observed=False)[metric]
    return pd.DataFrame({
        "mean":  g.mean(),
        "sem":   g.std() / np.sqrt(g.count().clip(lower=1)),
        "count": g.count(),
    })

def kruskal_p(df, bin_col, metric="kappa"):
    """Kruskal-Wallis across all non-empty bins."""
    groups = [g[metric].dropna().values
              for _, g in df.groupby(bin_col, observed=False) if len(g) > 0]
    groups = [g for g in groups if len(g) >= 3]
    if len(groups) < 2:
        return np.nan
    return kruskal(*groups).pvalue


def mwu_p(df, bin_col, metric="kappa"):
    """Mann-Whitney U for a binary categorical (e.g. diabetes 0/1)."""
    levels = sorted(df[bin_col].dropna().unique())
    if len(levels) != 2:
        return np.nan
    a = df[df[bin_col] == levels[0]][metric].dropna().values
    b = df[df[bin_col] == levels[1]][metric].dropna().values
    if len(a) < 3 or len(b) < 3:
        return np.nan
    return mannwhitneyu(a, b, alternative="two-sided").pvalue


def grouped_bar(ax, stats, color, label=None, offset=0.0, bar_w=0.36,
                show_n=True, min_n=MIN_BIN_SIZE):
    """
    Plot one set of bars (one cohort) with mean +/- SEM.
 
    Behaviour for n counts:
      - bins with n >= min_n  -> plotted as a normal bar, n label sits
                                 just above the top of the error bar
      - bins with n < min_n   -> NO bar drawn; an italic '(n=X)' label is
                                 placed near the x-axis so the reader
                                 sees the bin is sparse
 
    The y-axis is auto-padded to leave room for the n annotation above
    the tallest error bar.
    """
    n_arr = stats["count"].values.astype(int)
    means = stats["mean"].values.astype(float)
    sems  = stats["sem"].values.astype(float)
    x = np.arange(len(stats)) + offset
 
    valid = n_arr >= min_n
 
    # Plot only the valid bars
    if valid.any():
        ax.bar(x[valid], means[valid], bar_w,
               yerr=sems[valid], capsize=2.5,
               color=color, label=label,
               ecolor=PALETTE["ink"], error_kw={"linewidth": 0.7})
 
    if show_n:
        for xi, m, s, n, v in zip(x, means, sems, n_arr, valid):
            if v:
                # n above the top of the error bar
                ytop = m + (s if np.isfinite(s) else 0.0)
                ax.text(xi, ytop + 0.018, f"n={n}",
                        ha="center", va="bottom", fontsize=6.5,
                        color=PALETTE["slate"])
            else:
                # excluded bin: italic note near the x-axis
                ax.text(xi, 0.02, f"(n={n})",
                        ha="center", va="bottom", fontsize=6,
                        color=PALETTE["stone"], style="italic")
 
    # Ensure the y-axis leaves headroom for the n annotation
    current_top = ax.get_ylim()[1]
    needed_top = (np.nanmax(means + sems) if valid.any() else 0.7) + 0.10
    ax.set_ylim(0, max(current_top, needed_top, 0.85))


# =========================================================================
# (E) FIGURE 4A — main paper figure (two panels)
# =========================================================================
def make_figure_4A(mesa, cfs):
    """
    Two side-by-side panels:
      left:  V8 kappa by AHI severity (MESA + CFS bars side by side)
      right: V8 kappa by diabetes status (CFS only -- see decision (3))
    """
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4),
                             gridspec_kw={"width_ratios": [1.6, 1.0]})

    # ---- Panel 1: AHI severity, both cohorts ----
    ax = axes[0]
    mesa_ahi = grouped_stats(mesa, "AHI_bin")
    cfs_ahi  = grouped_stats(cfs,  "AHI_bin")
    bar_w = 0.36
    grouped_bar(ax, mesa_ahi, COHORT_COLORS["mesa"], "MESA", -bar_w/2, bar_w)
    grouped_bar(ax, cfs_ahi,  COHORT_COLORS["cfs"],  "CFS",  +bar_w/2, bar_w)
    ax.set_xticks(np.arange(len(AHI_LABELS)))
    ax.set_xticklabels(AHI_LABELS)
    ax.set_ylabel(r"V8 Cohen's $\kappa$")
    ax.set_title("Performance by AHI severity", fontsize=10, pad=6)
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    p_mesa = kruskal_p(mesa, "AHI_bin")
    p_cfs  = kruskal_p(cfs,  "AHI_bin")
    ax.text(0.98, 0.98,
            f"Kruskal-Wallis  MESA p = {p_mesa:.3f}\n"
            f"                 CFS p = {p_cfs:.3f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7, color=PALETTE["ink"])
    ax.set_ylim(0, max(1.0, ax.get_ylim()[1]))

    # ---- Panel 2: diabetes (CFS only) ----
    ax = axes[1]
    cfs_dm = cfs.dropna(subset=["diabetes"]).copy()
    cfs_dm["diabetes_label"] = cfs_dm["diabetes"].map(
        {0: "No diabetes", 1: "Diabetes"})
    cfs_dm["diabetes_label"] = pd.Categorical(
        cfs_dm["diabetes_label"], categories=["No diabetes", "Diabetes"],
        ordered=True)
    dm_stats = grouped_stats(cfs_dm, "diabetes_label")
    bar_w = 0.5
    grouped_bar(ax, dm_stats, COHORT_COLORS["cfs"], None, 0.0, bar_w)
    ax.set_xticks(np.arange(len(dm_stats)))
    ax.set_xticklabels(list(dm_stats.index))
    ax.set_ylabel(r"V8 Cohen's $\kappa$")
    ax.set_title("Performance by diabetes (CFS)", fontsize=10, pad=6)
    p_dm = mwu_p(cfs_dm, "diabetes_label")
    ax.text(0.98, 0.98,
            f"Mann-Whitney U  p = {p_dm:.3f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7, color=PALETTE["ink"])
    ax.set_ylim(0, max(1.0, ax.get_ylim()[1]))

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    save(fig, FIG_DIR / "figure_04A_phenotype_main")
    plt.close(fig)
    print(f"Figure 4A saved.")
    return {"AHI_mesa": p_mesa, "AHI_cfs": p_cfs, "diabetes_cfs": p_dm}


# =========================================================================
# (F) FIGURE 4B — supplementary figure (four panels)
# =========================================================================
def make_figure_4B(mesa, cfs):
    """
    2x2 grid of phenotype analyses:
      (a) AHI severity            both cohorts
      (b) diabetes status         CFS only
      (c) age decade              both cohorts
      (d) BMI category            both cohorts
    """
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.4))

    # (a) AHI - same as Figure 4A panel 1
    ax = axes[0, 0]
    mesa_ahi = grouped_stats(mesa, "AHI_bin")
    cfs_ahi  = grouped_stats(cfs,  "AHI_bin")
    bar_w = 0.36
    grouped_bar(ax, mesa_ahi, COHORT_COLORS["mesa"], "MESA", -bar_w/2, bar_w)
    grouped_bar(ax, cfs_ahi,  COHORT_COLORS["cfs"],  "CFS",  +bar_w/2, bar_w)
    ax.set_xticks(np.arange(len(AHI_LABELS))); ax.set_xticklabels(AHI_LABELS, fontsize=7)
    ax.set_ylabel(r"V8 $\kappa$")
    ax.set_title("(a) AHI severity", fontsize=9, pad=4)
    ax.legend(loc="lower left", frameon=False, fontsize=7)
    ax.set_ylim(0, max(1.0, ax.get_ylim()[1]))

    # (b) diabetes (CFS only)
    ax = axes[0, 1]
    cfs_dm = cfs.dropna(subset=["diabetes"]).copy()
    cfs_dm["diabetes_label"] = cfs_dm["diabetes"].map(
        {0: "No diabetes", 1: "Diabetes"})
    cfs_dm["diabetes_label"] = pd.Categorical(
        cfs_dm["diabetes_label"], categories=["No diabetes", "Diabetes"],
        ordered=True)
    dm_stats = grouped_stats(cfs_dm, "diabetes_label")
    grouped_bar(ax, dm_stats, COHORT_COLORS["cfs"], None, 0.0, 0.5)
    ax.set_xticks(np.arange(len(dm_stats)))
    ax.set_xticklabels(list(dm_stats.index), fontsize=8)
    ax.set_ylabel(r"V8 $\kappa$")
    ax.set_title("(b) Diabetes status (CFS)", fontsize=9, pad=4)
    ax.set_ylim(0, max(1.0, ax.get_ylim()[1]))

    # (c) age decade, both cohorts
    ax = axes[1, 0]
    mesa_age = grouped_stats(mesa, "age_bin")
    cfs_age  = grouped_stats(cfs,  "age_bin")
    grouped_bar(ax, mesa_age, COHORT_COLORS["mesa"], "MESA", -bar_w/2, bar_w)
    grouped_bar(ax, cfs_age,  COHORT_COLORS["cfs"],  "CFS",  +bar_w/2, bar_w)
    ax.set_xticks(np.arange(len(AGE_LABELS))); ax.set_xticklabels(AGE_LABELS, fontsize=7)
    ax.set_ylabel(r"V8 $\kappa$")
    ax.set_title("(c) Age decade", fontsize=9, pad=4)
    ax.legend(loc="lower left", frameon=False, fontsize=7)
    ax.set_ylim(0, max(1.0, ax.get_ylim()[1]))

    # (d) BMI category, both cohorts
    ax = axes[1, 1]
    mesa_bmi = grouped_stats(mesa, "bmi_bin")
    cfs_bmi  = grouped_stats(cfs,  "bmi_bin")
    grouped_bar(ax, mesa_bmi, COHORT_COLORS["mesa"], "MESA", -bar_w/2, bar_w)
    grouped_bar(ax, cfs_bmi,  COHORT_COLORS["cfs"],  "CFS",  +bar_w/2, bar_w)
    ax.set_xticks(np.arange(len(BMI_LABELS))); ax.set_xticklabels(BMI_LABELS, fontsize=7)
    ax.set_ylabel(r"V8 $\kappa$")
    ax.set_title("(d) BMI category (WHO)", fontsize=9, pad=4)
    ax.legend(loc="lower left", frameon=False, fontsize=7)
    ax.set_ylim(0, max(1.0, ax.get_ylim()[1]))

    fig.tight_layout()
    save(fig, FIG_DIR / "figure_04B_phenotype_supplement")
    plt.close(fig)
    print(f"Figure 4B saved.")


# =========================================================================
# (G) MAIN DRIVER
# =========================================================================
def main():
    mesa, cfs, combined = build_phenotype_table()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save the joined tables so the numbers backing the figures are auditable
    mesa.to_csv(OUT_DIR / "phenotype_mesa_subjects.csv", index=False)
    cfs.to_csv(OUT_DIR  / "phenotype_cfs_subjects.csv",  index=False)

    # Per-bin summary CSVs (these are what the figures plot)
    summary_rows = []
    for cohort_name, df in [("MESA", mesa), ("CFS", cfs)]:
        for var, bin_col in [("AHI", "AHI_bin"),
                             ("age", "age_bin"),
                             ("bmi", "bmi_bin")]:
            stats = grouped_stats(df, bin_col)
            for bin_label, row in stats.iterrows():
                summary_rows.append({
                    "cohort":     cohort_name,
                    "phenotype":  var,
                    "bin":        bin_label,
                    "n":          int(row["count"]),
                    "kappa_mean": float(row["mean"]),
                    "kappa_sem":  float(row["sem"]),
                })
    # Diabetes (CFS only)
    cfs_dm = cfs.dropna(subset=["diabetes"]).copy()
    cfs_dm["diabetes_label"] = cfs_dm["diabetes"].map({0: "No diabetes", 1: "Diabetes"})
    for bin_label, row in grouped_stats(cfs_dm, "diabetes_label").iterrows():
        summary_rows.append({
            "cohort":     "CFS",
            "phenotype":  "diabetes",
            "bin":        bin_label,
            "n":          int(row["count"]),
            "kappa_mean": float(row["mean"]),
            "kappa_sem":  float(row["sem"]),
        })
    pd.DataFrame(summary_rows).to_csv(
        OUT_DIR / "phenotype_summary.csv", index=False)

    print(f"\nSaved subject-level tables and bin summary to {OUT_DIR}/")

    # Generate the two figures
    p_main = make_figure_4A(mesa, cfs)
    make_figure_4B(mesa, cfs)

    # Console summary of main statistical tests
    print("\n" + "=" * 60)
    print("PHENOTYPE-ANALYSIS STATISTICAL TESTS")
    print("=" * 60)
    print(f"  AHI severity      MESA  Kruskal-Wallis p = {p_main['AHI_mesa']:.4f}")
    print(f"  AHI severity      CFS   Kruskal-Wallis p = {p_main['AHI_cfs']:.4f}")
    print(f"  Diabetes (CFS)          Mann-Whitney U p = {p_main['diabetes_cfs']:.4f}")


if __name__ == "__main__":
    main()