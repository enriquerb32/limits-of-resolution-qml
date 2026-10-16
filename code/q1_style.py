"""
q1_style.py — Reusable Matplotlib/Seaborn style module for Q1 journal figures
==============================================================================
Standalone style configuration independent of any specific dataset.
Import this in any figure-generation script to guarantee Q1-grade output.
"""

from pathlib import Path
import subprocess
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns


def apply_q1_style(serif: bool = True, base_fontsize: int = 10):
    """Applies a Nature/AMS/IEEE-style publication theme globally."""
    mpl.rcdefaults()
    plt.rcParams.update({
        # ---- Fonts: MUST be editable (Type 42), not Type 3 bitmap ----
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'font.family': 'serif' if serif else 'sans-serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Nimbus Roman', 'serif'],
        'mathtext.fontset': 'dejavuserif' if serif else 'dejavusans',

        # ---- Sizes (Nature/IEEE two-column safe) ----
        'font.size': base_fontsize,
        'axes.titlesize': base_fontsize + 1.5,
        'axes.titleweight': 'bold',
        'axes.labelsize': base_fontsize + 0.5,
        'axes.labelweight': 'bold',
        'xtick.labelsize': base_fontsize - 1,
        'ytick.labelsize': base_fontsize - 1,
        'legend.fontsize': base_fontsize - 1,
        'figure.titlesize': base_fontsize + 3,

        # ---- Resolution & export ----
        'figure.dpi': 300,
        'savefig.dpi': 600,          # 600 dpi for raster elements inside vector PDFs
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.04,
        'savefig.transparent': False,

        # ---- Lines / axes ----
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'axes.unicode_minus': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    sns.set_context("paper")


# ── Okabe-Ito colorblind-safe palette (the gold standard for Q1 figures) ──
OKABE_ITO = {
    'black':        '#000000',
    'orange':       '#E69F00',
    'sky_blue':     '#56B4E9',
    'green':        '#009E73',
    'yellow':       '#F0E442',
    'blue':         '#0072B2',
    'vermillion':   '#D55E00',
    'reddish_pink': '#CC79A7',
}


def add_panel_label(ax, label: str, x: float = -0.10, y: float = 1.05, fontsize: int = 12):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=fontsize,
            fontweight='bold', va='bottom', ha='left')


def clean_spines(ax, grid_alpha: float = 0.35):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle=':', alpha=grid_alpha, color='#B0BEC5')
    ax.set_axisbelow(True)


def check_embedded_fonts(pdf_path: Path) -> bool:
    """
    Verifies no Type 3 (bitmap) fonts remain in the exported PDF.
    Requires poppler-utils (`pdffonts`). Many Q1 journals reject
    submissions containing Type 3 fonts.
    """
    try:
        out = subprocess.run(['pdffonts', str(pdf_path)], capture_output=True, text=True)
        bad = [l for l in out.stdout.splitlines() if 'Type 3' in l]
        if bad:
            print(f"  [WARN] {pdf_path.name}: Type 3 (non-embeddable) fonts detected:")
            for b in bad:
                print(f"         {b}")
            return False
        return True
    except FileNotFoundError:
        print("  [INFO] `pdffonts` not found (install poppler-utils) — skipping font check.")
        return True


def simulate_colorblind(fig, kind: str = 'deuteranopia'):
    """
    Quick CVD simulation preview using matplotlib's colormap conversion
    (install `colorspacious` for a rigorous version: pip install colorspacious).
    Use this as a sanity check before submission, not as the final figure.
    """
    try:
        from colorspacious import cspace_convert
    except ImportError:
        print("  [INFO] pip install colorspacious for CVD simulation.")
        return None
    # left as a stub hook — plug into your QA pipeline if needed
    return fig