# =========================================================================
# figstyle.py
#
# Shared visual style for every paper figure. Import this at the top of
# any figure script:
#
#     from figstyle import apply_style, PALETTE, COHORT_COLORS, STAGE_COLORS
#     apply_style()
#
# All figures inherit:
#   - Serif fonts sized for a two-column journal layout
#   - A muted academic palette (ColorBrewer-derived)
#   - PDF output at 300 DPI, vector where possible
#   - Consistent grid, spine, legend, and tick styling
# =========================================================================

import matplotlib as mpl
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------
# PALETTE — academic, muted; no rainbow, no defaults
# -------------------------------------------------------------------------
# Primary qualitative palette (ColorBrewer "Set2"-ish, hand-tuned)
PALETTE = {
    "navy":    "#1f3a5f",   # primary dark
    "teal":    "#3d6b7d",   # secondary
    "olive":   "#6c8e68",   # tertiary
    "rust":    "#a85751",   # accent (highlight)
    "amber":   "#c9a14a",   # warm accent
    "slate":   "#5e6472",   # neutral
    "stone":   "#8a8676",   # neutral light
    "ink":     "#2b2b2b",   # text / axes
    "bg":      "#ffffff",   # background
    "grid":    "#e1e1e1",   # subtle grid lines
}

# Cohort colors — used everywhere MESA vs CFS appears
COHORT_COLORS = {
    "mesa": PALETTE["navy"],
    "MESA": PALETTE["navy"],
    "cfs":  PALETTE["rust"],
    "CFS":  PALETTE["rust"],
}

# Sleep-stage colors — used in hypnogram, PCA/t-SNE, confusion matrices
STAGE_COLORS = {
    "Wake":  PALETTE["amber"],
    "Light": PALETTE["teal"],
    "Deep":  PALETTE["navy"],
    "REM":   PALETTE["rust"],
}
STAGE_ORDER = ["Wake", "Light", "Deep", "REM"]

# Model colors — for the baseline comparison figures
MODEL_COLORS = {
    "v8":              PALETTE["navy"],
    "PhysioSleepNet":  PALETTE["navy"],
    "deepsleepnet":    PALETTE["stone"],
    "DeepSleepNet":    PALETTE["stone"],
    "sleeppgnet":      PALETTE["olive"],
    "SleepPPG-Net":    PALETTE["olive"],
    "insightsleepnet": PALETTE["amber"],
    "InsightSleepNet": PALETTE["amber"],
}


# -------------------------------------------------------------------------
# RC settings — applied once via apply_style()
# -------------------------------------------------------------------------
def apply_style():
    """Set matplotlib rcParams for consistent paper-grade figures."""
    mpl.rcParams.update({
        # Fonts: serif, sized for 2-column journal
        "font.family":       "serif",
        "font.serif":        ["DejaVu Serif", "Times New Roman", "Times",
                              "Computer Modern Roman"],
        "mathtext.fontset":  "dejavuserif",
        "font.size":         9,
        "axes.titlesize":    10,
        "axes.labelsize":    9,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "legend.fontsize":   8,
        "figure.titlesize":  11,

        # Axes
        "axes.edgecolor":     PALETTE["ink"],
        "axes.labelcolor":    PALETTE["ink"],
        "axes.linewidth":     0.8,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "axes.axisbelow":     True,

        # Grid
        "grid.color":         PALETTE["grid"],
        "grid.linewidth":     0.5,
        "grid.linestyle":     "-",

        # Ticks
        "xtick.color":        PALETTE["ink"],
        "ytick.color":        PALETTE["ink"],
        "xtick.major.size":   3,
        "ytick.major.size":   3,
        "xtick.major.width":  0.8,
        "ytick.major.width":  0.8,

        # Legend
        "legend.frameon":       False,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.2,

        # Lines & markers
        "lines.linewidth":   1.4,
        "lines.markersize":  4,
        "patch.linewidth":   0.6,
        "patch.edgecolor":   PALETTE["ink"],

        # Saving
        "savefig.dpi":         300,
        "savefig.format":      "pdf",
        "savefig.bbox":        "tight",
        "savefig.pad_inches":  0.05,
        "pdf.fonttype":        42,   # TrueType, editable in Illustrator
        "ps.fonttype":         42,

        # Figure background
        "figure.facecolor":   PALETTE["bg"],
        "axes.facecolor":     PALETTE["bg"],
    })


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
def save(fig, out_path, also_png=True):
    """Save figure as PDF (and optionally a PNG preview alongside)."""
    from pathlib import Path
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"))
    if also_png:
        fig.savefig(out_path.with_suffix(".png"), dpi=300)


def make_axes(width_in=3.4, height_in=2.4):
    """One-column journal axis sized for two-column layout."""
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    return fig, ax


def two_panel(width_in=7.0, height_in=2.8):
    """Two-column-wide figure with two side-by-side panels."""
    fig, axes = plt.subplots(1, 2, figsize=(width_in, height_in))
    return fig, axes


def grid_panel(rows, cols, width_in=7.0, height_in=None):
    """Multi-panel grid."""
    if height_in is None:
        height_in = 2.4 * rows
    fig, axes = plt.subplots(rows, cols, figsize=(width_in, height_in))
    return fig, axes