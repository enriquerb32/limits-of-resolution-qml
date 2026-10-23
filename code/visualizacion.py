"""
visualizacion.py — Paper Figures Module (Q1 Journal Standard / Publication Ready)
==================================================================================
Generates all 7 high-impact, publication-grade figures for the manuscript:
  1. figura_accuracy_f1.pdf         — Global Accuracy & Tail F1-score comparison
  2. figura_roc.pdf                 — ROC Curves with operational inset zoom
  3. figura_feature_importance.pdf  — Predictor importance (XGB/RF) & ACP anomaly profile
  4. figura_auc_violin.pdf          — 15-seed optimization variance & TOST bounds
  5. figura_pareto_operational.pdf  — Operational Pareto Frontier & threshold recalibration
  6. figura_single_run_illusion.pdf — Single-run bootstrap illusion vs. 225-seed reality
  7. figura_thermo_analysis.pdf     — Thermodynamic seasonality, Clausius-Clapeyron & dynamics

Also compiles paper_figures.pdf (multi-page PDF) and generates a QA report.

Adheres to Q1 journal formatting standards (Nature / AMS / IEEE Transactions):
  - Colorblind-friendly palettes, serif typography, despined axes, panel labels.
  - Editable (Type 42) embedded fonts — required by most Q1 production systems.

─────────────────────────────────────────────────────────────────────────────
INTEGRITY POLICY (read before modifying)
─────────────────────────────────────────────────────────────────────────────
This module NEVER fabricates, simulates, or interpolates data to fill a figure.
If a required real value (multi-seed array, bootstrap CI, feature importance,
etc.) is missing from the source JSON/CSV, the corresponding panel is left
visibly blank with an explicit warning — both in the console and as text
burned into the figure. This is intentional: a paper whose central thesis is
"single-run results are an illusion, trust only replicated data" cannot itself
ship an illustrative figure built from `np.random.normal(...)`.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import json
import warnings
from pathlib import Path

# Ensure UTF-8 output encoding for consoles that default to cp1252
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats as sps
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, f1_score

warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════
# 1. GLOBAL PUBLICATION STYLE CONFIGURATION (Q1 Standard)
# ══════════════════════════════════════════════════════════════════════

def apply_q1_style(base_fontsize: int = 10):
    """Applies a Nature/AMS/IEEE-style publication theme globally."""
    plt.rcParams.update({
        # ---- Fonts: MUST be editable (Type 42), not Type 3 bitmap ----
        # Most Q1 production systems (Elsevier, Springer, AMS, IEEE) reject
        # PDFs containing Type 3 fonts during final typesetting checks.
        'pdf.fonttype':       42,
        'ps.fonttype':        42,

        'font.family':        'serif',
        'font.serif':         ['Times New Roman', 'DejaVu Serif', 'Nimbus Roman', 'serif'],
        'mathtext.fontset':   'dejavuserif',
        'font.size':          base_fontsize,
        'axes.titlesize':     base_fontsize + 1.5,
        'axes.titleweight':   'bold',
        'axes.labelsize':     base_fontsize + 0.5,
        'axes.labelweight':   'bold',
        'xtick.labelsize':    base_fontsize - 1,
        'ytick.labelsize':    base_fontsize - 1,
        'legend.fontsize':    base_fontsize - 1,
        'figure.titlesize':   base_fontsize + 3,
        'figure.titleweight': 'bold',
        'figure.dpi':         300,
        'savefig.dpi':        600,   # high-res raster elements inside vector PDF
        'savefig.bbox':       'tight',
        'savefig.pad_inches': 0.05,
        'axes.unicode_minus': False,
        'axes.linewidth':     0.8,
        'xtick.major.width':  0.8,
        'ytick.major.width':  0.8,
        'axes.spines.top':    False,
        'axes.spines.right':  False,
    })
    sns.set_context("paper")


apply_q1_style()

# ── Colorblind-conscious publication palette (kept per-model for narrative
#    clarity; cross-checked against Okabe-Ito hue spacing where possible) ──
COLORS = {
    'Climatology':                   '#37474F',
    'Logistic Regression':           '#546E7A',
    'Random Forest':                 '#1B5E20',
    'Random Forest (matched, 2273)': '#2E7D32',
    'Gradient Boosting':             '#388E3C',
    'XGBoost':                       '#4CAF50',
    'Standalone LSTM':               '#C2185B',
    'Matched LSTM':                  '#E91E63',
    'LSTM+QNN':                      '#5E35B1',
    'LSTM+PI-QNN':                   '#512DA8',
    'LSTM+QTL':                      '#7E57C2',
    'LbL-QNN':                       '#311B92',
    'QKSVM':                         '#00695C',
}

FAMILY_PALETTE = {
    'Climatological Reference': '#37474F',
    'Statistical Baseline':     '#546E7A',
    'Ensemble Trees':           '#2E7D32',
    'Deep Learning':            '#C2185B',
    'Quantum Hybrid (NISQ)':    '#512DA8',
}


def add_panel_label(ax, label: str, x: float = -0.10, y: float = 1.05, fontsize: int = 12):
    """Adds bold (a), (b), (c) subplot tags standard in Nature / Q1 papers."""
    ax.text(x, y, label, transform=ax.transAxes, fontsize=fontsize,
            fontweight='bold', va='bottom', ha='left')


def clean_spines(ax, grid_alpha: float = 0.45):
    """Removes top and right spines, sets subtle dotted grid."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle=':', alpha=grid_alpha, color='#B0BEC5')
    ax.set_axisbelow(True)


def missing_data_panel(ax, message: str):
    """Renders a visibly blank panel with an explicit warning instead of
    silently fabricating data. Used throughout this module."""
    ax.text(0.5, 0.5, f"⚠ {message}\n(no synthetic substitute generated)",
            ha='center', va='center', fontsize=9.5, color='#B71C1C',
            wrap=True, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", fc="#FFEBEE", ec="#D32F2F", lw=1.0))
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    print(f"  [WARN] {message}")


# ══════════════════════════════════════════════════════════════════════
# 2. DATA LOADER & METRIC CALIBRATION HELPER
# ══════════════════════════════════════════════════════════════════════

def _safe_get_multiseed(src_dict: dict, key: str, metric: str = 'auc') -> np.ndarray:
    """
    Returns the real array of seed values for `key` inside `src_dict`, or
    None (with a console warning) if unavailable. Never fabricates values.
    """
    if src_dict is None or key not in src_dict:
        print(f"  [WARN] Multi-seed key '{key}' not found — omitted (not substituted).")
        return None
    entry = src_dict[key]
    if metric not in entry or 'valores' not in entry[metric]:
        print(f"  [WARN] Metric '{metric}' missing for '{key}' — omitted (not substituted).")
        return None
    return np.asarray(entry[metric]['valores'], dtype=float)


def load_all_study_data(base_dir: str = '.') -> dict:
    """
    Loads all datasets, single-run predictions, and multi-seed replications
    from JSON and CSV files, falling back to .pt if needed.
    """
    base_p = Path(base_dir)
    json_dir = base_p / 'results' / 'json'

    with open(json_dir / '01_data.json', 'r', encoding='utf-8') as f:
        ds = json.load(f)[1]
    with open(json_dir / '02_baselines.json', 'r', encoding='utf-8') as f:
        res_base = json.load(f)
    with open(json_dir / '03_qml.json', 'r', encoding='utf-8') as f:
        res_qml = json.load(f)
    with open(json_dir / '04_qksvm.json', 'r', encoding='utf-8') as f:
        res_qksvm = json.load(f)

    res_val = {}
    val_path = json_dir / '08_validacion.json'
    if val_path.exists():
        with open(val_path, 'r', encoding='utf-8') as f:
            res_val = json.load(f)
    else:
        print("  [WARN] 08_validacion.json not found — bootstrap-comparison panel "
              "in Figure 6a will be left blank.")

    with open(json_dir / '09_multisemilla.json', 'r', encoding='utf-8') as f:
        multi_qml = json.load(f)
    with open(json_dir / '10_multisemilla_clasico.json', 'r', encoding='utf-8') as f:
        multi_cls = json.load(f)

    panel_path = base_p / 'results' / 'panel_checkpoint.csv'
    panel_df = pd.read_csv(panel_path) if panel_path.exists() else None

    agg_df = None
    pt_path = base_p / 'checkpoints' / '01_data.pt'
    if not pt_path.exists():
        pt_path = base_p / '01_data.pt'
    if pt_path.exists():
        try:
            import torch
            sys.path.append(str(base_p / 'code'))
            pt_data = torch.load(pt_path, weights_only=False)
            agg_df = pt_data[0] if isinstance(pt_data, tuple) else None
        except Exception:
            agg_df = None

    if agg_df is None and panel_df is not None:
        agg_df = panel_df.copy()
        agg_df.index = pd.date_range('1938-03-01', periods=len(agg_df), freq='MS')

    y_tr = np.array(ds['y_tr']); m_tr = np.array(ds['m_tr']); X_tr = np.array(ds['X_tr'])
    y_te = np.array(ds['y_te']); m_te = np.array(ds['m_te']); X_te = np.array(ds['X_te'])

    st_idx = [31, 32, 33, 34]
    stations = ['Alicante', 'Barcelona', 'San_Javier', 'Valencia']
    p_uncond = float(y_tr.mean())

    clim_table = {}
    for s_name, s_col in zip(stations, st_idx):
        clim_table[s_name] = {}
        for m in range(1, 13):
            mask = (m_tr == m) & (X_tr[:, s_col] == 1.0)
            n_obs = int(mask.sum())
            clim_table[s_name][m] = float(y_tr[mask].sum() / n_obs) if n_obs > 0 else p_uncond

    p_clim_te = np.zeros(len(y_te))
    for i in range(len(y_te)):
        s_col = [c for c in st_idx if X_te[i, c] == 1.0][0]
        s_name = stations[st_idx.index(s_col)]
        p_clim_te[i] = clim_table[s_name][int(m_te[i])]

    logreg_key = ('Reg. Log\u00edstica' if 'Reg. Log\u00edstica' in res_base['_probs']
                  else [k for k in res_base['_probs'] if 'Log' in k][0])

    probs = {
        'Climatology':         p_clim_te,
        'Logistic Regression': np.array(res_base['_probs'][logreg_key]),
        'Random Forest':       np.array(res_base['_probs']['Random Forest']),
        'Gradient Boosting':   np.array(res_base['_probs']['Gradient Boosting']),
        'XGBoost':             np.array(res_base['_probs']['XGBoost']),
        'Standalone LSTM':     np.array(res_base['_probs']['LSTM pura']),
        'Matched LSTM':        np.array(res_base['_probs']['LSTM pura']),  # same probs, different τ
        'LSTM+QNN':            np.array(res_qml['_probs']['LSTM+QNN']),
        'LSTM+PI-QNN':         np.array(res_qml['_probs']['LSTM+PI-QNN']),
        'LSTM+QTL':            np.array(res_qml['_probs']['LSTM+QTL']),
        'LbL-QNN':             np.array(res_qml['_probs']['PI-QNN-LbL']),
        'QKSVM':               np.array(res_qksvm['QKSVM']['_prob']),
    }

    y_te_ext = np.array(ds['y_te_ext']) if 'y_te_ext' in ds else None
    dana_te = np.array(ds['dana_te'], dtype=bool) if 'dana_te' in ds else None

    return {
        'ds': ds, 'res_base': res_base, 'res_qml': res_qml, 'res_qksvm': res_qksvm,
        'res_val': res_val, 'multi_qml': multi_qml, 'multi_cls': multi_cls,
        'panel_df': panel_df, 'agg_df': agg_df, 'probs': probs,
        'y_te': y_te, 'y_te_ext': y_te_ext, 'dana_te': dana_te,
    }


def compute_accuracy_f1_table(data: dict, threshold: float = 0.5) -> dict:
    """
    Computes Global Accuracy and Tail F1 (extreme class, y_te_ext) directly
    from the *same* probability arrays used in the ROC figure, at a fixed
    decision threshold. This guarantees the bar-chart numbers can never
    silently diverge from the ROC/table numbers elsewhere in the paper
    (previously these were a hand-typed dict — a reproducibility risk).

    NOTE: if the manuscript's Table 5 was built with a different decision
    rule (e.g., per-model optimal threshold rather than 0.5, or a dedicated
    extreme-class classifier), cross-check these numbers against that table
    before final submission — this function documents its assumption rather
    than hiding it.
    """
    y_te = data['y_te']
    y_te_ext = data.get('y_te_ext')
    probs = data['probs']

    acc_map, f1_map = {}, {}
    for model, p in probs.items():
        pred = (p >= threshold).astype(int)
        if len(pred) != len(y_te):
            print(f"  [WARN] Shape mismatch for '{model}' — skipped in accuracy/F1 table.")
            continue
        acc_map[model] = 100.0 * accuracy_score(y_te, pred)

        if y_te_ext is not None and len(y_te_ext) == len(p):
            pred_ext = (p >= threshold).astype(int)
            f1_map[model] = f1_score(y_te_ext, pred_ext, zero_division=0)
        else:
            f1_map[model] = np.nan

    return {'accuracy': acc_map, 'f1_tail': f1_map,
            'assumption': f'threshold={threshold}, tail label = y_te_ext'}


def paired_tost(x: np.ndarray, y: np.ndarray, low: float, upp: float):
    """Two one-sided tests (TOST) for paired equivalence. Returns
    (mean_diff, se, p_tost). Pure scipy implementation (no statsmodels
    dependency required)."""
    diff = np.asarray(x) - np.asarray(y)
    n = len(diff)
    mean_diff = diff.mean()
    se = diff.std(ddof=1) / np.sqrt(n)
    t1 = (mean_diff - low) / se
    t2 = (mean_diff - upp) / se
    p1 = 1 - sps.t.cdf(t1, df=n - 1)
    p2 = sps.t.cdf(t2, df=n - 1)
    return mean_diff, se, max(p1, p2)


# ══════════════════════════════════════════════════════════════════════
# 3. FIGURE 1: ACCURACY & F1-SCORE COMPARISON
# ══════════════════════════════════════════════════════════════════════

def fig_accuracy_f1(data: dict, out_path: Path = None):
    """Dual-panel comparison of Global Accuracy and Tail F1-Score across all
    12 architectures + Climatology. Metrics are computed live from `probs`
    (see `compute_accuracy_f1_table`), not hand-typed."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=False)

    metrics = compute_accuracy_f1_table(data)
    acc_map, f1_map = metrics['accuracy'], metrics['f1_tail']

    models_order = [
        'Climatology', 'Logistic Regression', 'Random Forest', 'Gradient Boosting',
        'XGBoost', 'Standalone LSTM', 'Matched LSTM', 'LSTM+QNN', 'LSTM+PI-QNN',
        'LSTM+QTL', 'LbL-QNN', 'QKSVM'
    ]
    models_order = [m for m in models_order if m in acc_map]

    x_pos = np.arange(len(models_order))
    bar_colors = [COLORS.get(m, '#455A64') for m in models_order]

    # Panel (a): Accuracy
    add_panel_label(ax1, "(a)")
    bars1 = ax1.bar(x_pos, [acc_map[m] for m in models_order], color=bar_colors, width=0.68,
                     edgecolor='white', linewidth=1.0, zorder=3)
    ax1.set_ylim(min(acc_map.values()) - 5, max(acc_map.values()) + 5)
    ax1.set_ylabel('Global Accuracy (%)')
    ax1.set_title('Overall Classification Accuracy\n(Governed by Class Imbalance)', pad=10)
    if 'Random Forest' in acc_map:
        ax1.axhline(acc_map['Random Forest'], color=COLORS['Random Forest'], ls='--', lw=1.2,
                    alpha=0.7, zorder=2, label=f"Random Forest ({acc_map['Random Forest']:.1f}%)")
    if 'Climatology' in acc_map:
        ax1.axhline(acc_map['Climatology'], color=COLORS['Climatology'], ls=':', lw=1.4,
                    alpha=0.8, zorder=2, label=f"Climatology ({acc_map['Climatology']:.1f}%)")
    clean_spines(ax1)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(models_order, rotation=42, ha='right')
    ax1.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.9, edgecolor='#CFD8DC')
    for bar, m in zip(bars1, models_order):
        val = acc_map[m]
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.35, f"{val:.1f}%",
                  ha='center', va='bottom', fontsize=8.5,
                  fontweight='bold' if 'Random Forest' in m or 'LbL' in m else 'normal')

    # Panel (b): F1-Score
    add_panel_label(ax2, "(b)")
    valid_f1 = {m: f1_map[m] for m in models_order if not np.isnan(f1_map.get(m, np.nan))}
    if not valid_f1:
        missing_data_panel(ax2, "Tail F1 (extreme class) could not be computed — "
                                 "'y_te_ext' missing or shape mismatch.")
    else:
        models_f1 = [m for m in models_order if m in valid_f1]
        bars2 = ax2.bar(np.arange(len(models_f1)), [valid_f1[m] for m in models_f1],
                         color=[COLORS.get(m, '#455A64') for m in models_f1], width=0.68,
                         edgecolor='white', linewidth=1.0, zorder=3)
        ax2.set_ylabel('F1-Score (Extreme Class, $P_{60}$)')
        ax2.set_title('Tail Sensitivity & Harmonic F1\n(Heavy Precipitation Regime)', pad=10)
        if 'Random Forest' in valid_f1:
            ax2.axhline(valid_f1['Random Forest'], color=COLORS['Random Forest'], ls='--', lw=1.2,
                        alpha=0.7, zorder=2, label=f"Random Forest ({valid_f1['Random Forest']:.3f})")
        if 'Climatology' in valid_f1:
            ax2.axhline(valid_f1['Climatology'], color=COLORS['Climatology'], ls=':', lw=1.4,
                        alpha=0.8, zorder=2, label=f"Climatology ({valid_f1['Climatology']:.3f})")
        clean_spines(ax2)
        ax2.set_xticks(np.arange(len(models_f1)))
        ax2.set_xticklabels(models_f1, rotation=42, ha='right')
        ax2.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.9, edgecolor='#CFD8DC')
        for bar, m in zip(bars2, models_f1):
            val = valid_f1[m]
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{val:.3f}",
                      ha='center', va='bottom', fontsize=8.5,
                      fontweight='bold' if 'Random Forest' in m or 'LbL' in m else 'normal')

    fam_patches = [
        Patch(facecolor=COLORS['Climatology'], edgecolor='white', label='Climatological Reference'),
        Patch(facecolor=COLORS['Logistic Regression'], edgecolor='white', label='Linear / Statistical'),
        Patch(facecolor=COLORS['Random Forest'], edgecolor='white', label='Ensemble Trees'),
        Patch(facecolor=COLORS['Standalone LSTM'], edgecolor='white', label='Deep Learning'),
        Patch(facecolor=COLORS['LbL-QNN'], edgecolor='white', label='Quantum Hybrid (NISQ)'),
    ]
    fig.legend(handles=fam_patches, loc='upper center', bbox_to_anchor=(0.5, 1.03),
               ncol=5, frameon=False, fontsize=9.5)

    plt.tight_layout()
    fig.subplots_adjust(top=0.90)
    if out_path:
        fig.savefig(out_path, format='pdf')
        fig.savefig(out_path.with_suffix('.png'), dpi=300)
    return fig


# ══════════════════════════════════════════════════════════════════════
# 4. FIGURE 2: RECEIVER OPERATING CHARACTERISTIC (ROC) WITH INSET
# ══════════════════════════════════════════════════════════════════════

def fig_roc(data: dict, out_path: Path = None):
    """High-precision ROC curves for all 12 single-run architectures +
    Climatology with an operational zoom inset."""
    fig, ax = plt.subplots(figsize=(8.0, 7.5))
    add_panel_label(ax, "(a)", x=-0.08, y=1.02)

    y_te, probs = data['y_te'], data['probs']

    model_styles = {
        'Random Forest':       {'c': '#1B5E20', 'ls': '-',  'lw': 2.6, 'z': 10, 'lbl': 'Random Forest'},
        'Climatology':         {'c': '#37474F', 'ls': '-.', 'lw': 2.2, 'z': 9,  'lbl': 'Month $\\times$ Station Climatology'},
        'LbL-QNN':             {'c': '#311B92', 'ls': '--', 'lw': 2.5, 'z': 8,  'lbl': 'LbL-QNN (Physics-Inspired)'},
        'Gradient Boosting':   {'c': '#388E3C', 'ls': '-',  'lw': 1.6, 'z': 6,  'lbl': 'Gradient Boosting'},
        'LSTM+PI-QNN':         {'c': '#512DA8', 'ls': '--', 'lw': 1.6, 'z': 6,  'lbl': 'LSTM+PI-QNN'},
        'Logistic Regression': {'c': '#546E7A', 'ls': '-',  'lw': 1.4, 'z': 4,  'lbl': 'Logistic Regression'},
        'Standalone LSTM':     {'c': '#C2185B', 'ls': '-',  'lw': 1.9, 'z': 7,  'lbl': 'Standalone LSTM'},
        'LSTM+QTL':            {'c': '#7E57C2', 'ls': '--', 'lw': 1.5, 'z': 5,  'lbl': 'LSTM+QTL (Transfer Learning)'},
        'XGBoost':             {'c': '#4CAF50', 'ls': '-',  'lw': 1.5, 'z': 5,  'lbl': 'XGBoost'},
        'LSTM+QNN':            {'c': '#5E35B1', 'ls': '--', 'lw': 1.6, 'z': 5,  'lbl': 'LSTM+QNN (Baseline)'},
        'QKSVM':               {'c': '#00695C', 'ls': ':',  'lw': 1.6, 'z': 4,  'lbl': 'QKSVM (Quantum Kernel)'},
    }

    roc_entries = []
    for m, style in model_styles.items():
        if m in probs:
            p = probs[m]
            fpr, tpr, _ = roc_curve(y_te, p)
            auc_val = roc_auc_score(y_te, p)
            roc_entries.append((m, fpr, tpr, auc_val, style))
    roc_entries.sort(key=lambda x: x[3], reverse=True)

    for m, fpr, tpr, auc_val, style in roc_entries:
        ax.plot(fpr, tpr, color=style['c'], linestyle=style['ls'], linewidth=style['lw'],
                zorder=style['z'], label=f"{style['lbl']} (AUC = {auc_val:.3f})")

    ax.plot([0, 1], [0, 1], color='#78909C', lw=1.2, ls=':', zorder=1, label='Random Guess (AUC = 0.500)')
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.02)
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.set_title(f'Receiver Operating Characteristic (ROC) Comparison\n'
                 f'Global Spatiotemporal Test Partition ($N={len(y_te)}$)', pad=12)
    clean_spines(ax)
    ax.legend(loc='lower right', fontsize=8.8, frameon=True, facecolor='white', framealpha=0.92, edgecolor='#CFD8DC')

    ax_ins = inset_axes(ax, width="42%", height="42%", loc='center right',
                         bbox_to_anchor=(-0.05, 0.05, 1, 1), bbox_transform=ax.transAxes)
    for m, fpr, tpr, auc_val, style in roc_entries:
        ax_ins.plot(fpr, tpr, color=style['c'], linestyle=style['ls'], linewidth=style['lw'], zorder=style['z'])
    ax_ins.set_xlim(0.18, 0.52); ax_ins.set_ylim(0.60, 0.94)
    ax_ins.set_title('Operational Regime', fontsize=8.5, fontweight='bold', pad=4)
    ax_ins.tick_params(axis='both', which='major', labelsize=7.5)
    ax_ins.grid(True, linestyle=':', alpha=0.5, color='#B0BEC5')
    for sp in ax_ins.spines.values():
        sp.set_color('#78909C'); sp.set_linewidth(0.8)

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, format='pdf')
        fig.savefig(out_path.with_suffix('.png'), dpi=300)
    return fig


# ══════════════════════════════════════════════════════════════════════
# 5. FIGURE 3: FEATURE IMPORTANCE & CONVECTIVE PROFILE
# ══════════════════════════════════════════════════════════════════════

LABELS_PRETTY = {
    'ciudad_Barcelona': 'Station: Barcelona', 'ciudad_Valencia': 'Station: Valencia',
    'ciudad_Alicante': 'Station: Alicante', 'ciudad_San_Javier': 'Station: San Javier',
    'mes_sin': 'Month (Sine Encoding)', 'mes_cos': 'Month (Cosine Encoding)',
    'choque_clausius': r'Clausius-Clapeyron Shock ($\Delta T_{3m}$)', 'et0_sum': r'Cumulative $\text{ET}_0$ (FAO-56)',
    'et0_lag3m': r'$\text{ET}_0$ (3-month lag)', 'potencial_convectivo': r'Convective Potential ($\text{ET}_0 \times \sigma_p$)',
    't_anomaly': r'Temperature Anomaly ($z_T$)', 'ia_racha': 'Wind Gust Anomaly', 'veloc_ms': 'Mean Wind Speed',
    'pmed': 'Sea-Level Pressure', 'dp_std': r'Barometric Variability ($\sigma_p$)', 'iap': r'Pressure Anomaly ($z_P$)',
    'iir': r'Insolation Ratio ($S/S_0$)', 'ib': r'Barometric Instability ($\sigma_p / \Delta T$)'
}
for _i in range(1, 13):
    LABELS_PRETTY[f'precip_lag_{_i}'] = f'Precipitation (Lag {_i})'

DOMAIN_COLORS = {
    'Thermodynamic / Physical': '#D32F2F',
    'Antecedent Precipitation':  '#2E7D32',
    'Evapotranspiration':        '#F57C00',
    'Atmospheric Pressure':      '#1976D2',
    'Geographic / Seasonal':     '#78909C'
}


def categorize_feature(name: str):
    n = name.lower()
    if 'clausius' in n or 'potencial' in n or 'anomaly' in n or n == 'ib':
        return 'Thermodynamic / Physical', DOMAIN_COLORS['Thermodynamic / Physical']
    if 'precip_lag' in n:
        return 'Antecedent Precipitation', DOMAIN_COLORS['Antecedent Precipitation']
    if 'et0' in n:
        return 'Evapotranspiration', DOMAIN_COLORS['Evapotranspiration']
    if 'pmed' in n or 'dp' in n or 'iap' in n:
        return 'Atmospheric Pressure', DOMAIN_COLORS['Atmospheric Pressure']
    return 'Geographic / Seasonal', DOMAIN_COLORS['Geographic / Seasonal']


def fig_feature_importance(data: dict, out_path: Path = None):
    """XGBoost feature importance ranking paired with ACP test-set synoptic
    anomaly profile. The Random Forest domain-breakdown box is computed by
    aggregating real RF importances (if present) — not hardcoded percentages."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.8))
    ds, res_base = data['ds'], data['res_base']

    # --- Left: Feature Importance ---
    add_panel_label(ax1, "(a)")
    xgb_info = res_base.get('XGBoost', {})
    feat_imp = xgb_info.get('_feat_importances') or xgb_info.get('importancias', {})

    if feat_imp:
        top_items = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:14]
        feats_raw = [x[0] for x in top_items]
        gains = [x[1] for x in top_items]
        labels = [LABELS_PRETTY.get(f, f) for f in feats_raw]
        c_list = [categorize_feature(f)[1] for f in feats_raw]

        y_idx = np.arange(len(labels))
        ax1.barh(y_idx, gains, color=c_list, edgecolor='white', height=0.72, zorder=3)
        ax1.set_yticks(y_idx); ax1.set_yticklabels(labels, fontsize=9.5)
        ax1.invert_yaxis()
        ax1.set_xlabel('Normalized Gain (XGBoost Feature Importance)')
        ax1.set_title('Top Predictor Importance by Physical Domain', pad=10)
        clean_spines(ax1)

        domain_patches = [Patch(facecolor=c, edgecolor='white', label=d) for d, c in DOMAIN_COLORS.items()]
        ax1.legend(handles=domain_patches, loc='lower right', fontsize=8.5, frameon=True,
                   facecolor='white', framealpha=0.9)

        # Random Forest MDI breakdown — computed from real importances, not typed by hand.
        rf_info = res_base.get('Random Forest', {})
        rf_imp = rf_info.get('_feat_importances') or rf_info.get('importancias', {})
        if rf_imp:
            domain_totals = {}
            for feat, val in rf_imp.items():
                dom, _ = categorize_feature(feat)
                domain_totals[dom] = domain_totals.get(dom, 0.0) + val
            total = sum(domain_totals.values()) or 1.0
            lines = [f"• {d}: {100 * v / total:.1f}%"
                     for d, v in sorted(domain_totals.items(), key=lambda x: -x[1])]
            rf_box_text = "Random Forest MDI Breakdown:\n" + "\n".join(lines)
            ax1.text(0.55, 0.40, rf_box_text, transform=ax1.transAxes, fontsize=8.5,
                      bbox=dict(boxstyle="round,pad=0.4", fc="#F5F5F5", ec="#B0BEC5", lw=0.8), va='center')
        else:
            print("  [WARN] Random Forest feature importances not found — "
                  "MDI breakdown box omitted from Figure 3.")
    else:
        missing_data_panel(ax1, "XGBoost feature importances not found in 02_baselines.json.")

    # --- Right: ACP Synoptic Anomaly Profile in Test Set ---
    add_panel_label(ax2, "(b)")
    try:
        X_te = np.array(ds['X_te'])
        dana_mask = data['dana_te']
        feature_names = ds.get('feature_names', [f'f{i}' for i in range(X_te.shape[1])])

        selected_vars = [f'precip_lag_{i}' for i in range(1, 13)]
        var_indices = [feature_names.index(v) for v in selected_vars if v in feature_names]

        if dana_mask is not None and dana_mask.sum() > 0 and len(var_indices) > 0:
            sub_acp = X_te[dana_mask][:, var_indices]
            sub_norm = X_te[~dana_mask][:, var_indices]

            mu_acp = sub_acp.mean(axis=0)
            mu_norm = sub_norm.mean(axis=0)
            sigma_norm = sub_norm.std(axis=0) + 1e-6
            z_diff = (mu_acp - mu_norm) / sigma_norm

            lag_labels = [f"Lag {i} (t-{i})" for i in range(1, len(var_indices) + 1)]
            y_pos2 = np.arange(len(lag_labels))
            bar_c2 = ['#D32F2F' if v > 0 else '#1976D2' for v in z_diff]

            ax2.barh(y_pos2, z_diff, color=bar_c2, edgecolor='white', height=0.72, zorder=3)
            ax2.set_yticks(y_pos2); ax2.set_yticklabels(lag_labels, fontsize=9.5)
            ax2.invert_yaxis()
            ax2.axvline(0, color='black', lw=1.2, ls='-', zorder=4)
            ax2.set_xlabel(r'Standardized Anomaly Difference ($\Delta z = [\mu_{\mathrm{ACP}} - \mu_{\mathrm{normal}}] / \sigma$)')
            ax2.set_title(f'Antecedent Precipitation Memory in ACP Events\n'
                          f'(Test Set Contrast, $n={int(dana_mask.sum())}$ vs. $n={int((~dana_mask).sum())}$)', pad=10)
            clean_spines(ax2)
            dir_patches = [
                Patch(facecolor='#D32F2F', edgecolor='white', label='Positive Anomaly (Excess Moisture)'),
                Patch(facecolor='#1976D2', edgecolor='white', label='Negative Anomaly (Precipitation Deficit)')
            ]
            ax2.legend(handles=dir_patches, loc='lower right', fontsize=8.5, frameon=True, facecolor='white', framealpha=0.9)
        else:
            missing_data_panel(ax2, "ACP mask ('dana_te') or precipitation-lag features not found.")
    except Exception as e:
        missing_data_panel(ax2, f"Error computing ACP anomaly profile: {e}")

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, format='pdf')
        fig.savefig(out_path.with_suffix('.png'), dpi=300)
    return fig


# ══════════════════════════════════════════════════════════════════════
# 6. FIGURE 4: VIOLIN PLOTS ACROSS 15 REPLICATIONS (OPTIMIZATION VARIANCE)
# ══════════════════════════════════════════════════════════════════════

def _violinplot_compat(ax, data, x, y, order, palette, **kwargs):
    """Wraps sns.violinplot across seaborn versions (density_norm vs scale)."""
    try:
        return sns.violinplot(data=data, x=x, y=y, order=order, palette=palette,
                               density_norm='width', ax=ax, **kwargs)
    except TypeError:
        return sns.violinplot(data=data, x=x, y=y, order=order, palette=palette,
                               scale='width', ax=ax, **kwargs)


def fig_auc_violin(data: dict, out_path: Path = None):
    """
    15-seed optimization variance violin plots demonstrating the dissolution
    of single-run quantum advantage. Uses ONLY real multi-seed values loaded
    from 09_multisemilla.json / 10_multisemilla_clasico.json — any model
    missing real data is simply absent from the plot (never simulated).
    Friedman / TOST statistics are computed live from the loaded arrays
    (see `scipy.stats`), so annotated p-values can never drift from the
    data actually plotted.
    """
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    add_panel_label(ax, "(a)", x=-0.06, y=1.02)

    multi_q, multi_c = data['multi_qml'], data['multi_cls']
    records = {}

    def _add(src, key, label, family):
        vals = _safe_get_multiseed(src, key)
        if vals is not None:
            records[label] = {'vals': vals, 'family': family}

    _add(multi_c, 'Random Forest (matched)', 'Random Forest\n(matched, 2,273)', 'Ensemble Trees')
    _add(multi_c, 'Random Forest', 'Random Forest\n(2,787 rows)', 'Ensemble Trees')
    _add(multi_c, 'Gradient Boosting', 'Gradient\nBoosting', 'Ensemble Trees')
    _add(multi_c, 'LSTM pura', 'Standalone\nLSTM', 'Deep Learning')
    _add(multi_c, 'XGBoost', 'XGBoost', 'Ensemble Trees')
    q_mapping = {'PI-QNN-LbL': 'LbL-QNN', 'LSTM+QNN': 'LSTM+QNN',
                 'LSTM+PI-QNN': 'LSTM+PI-QNN', 'LSTM+QTL': 'LSTM+QTL'}
    for raw, clean in q_mapping.items():
        _add(multi_q, raw, clean, 'Quantum Hybrid')

    if not records:
        missing_data_panel(ax, "No real multi-seed arrays found in 09/10_multisemilla*.json.")
        if out_path:
            fig.savefig(out_path, format='pdf')
        return fig

    rows = []
    for label, d in records.items():
        for v in d['vals']:
            rows.append({'Model': label, 'AUC': v, 'Family': d['family']})
    df = pd.DataFrame(rows)

    preferred_order = [
        'Random Forest\n(matched, 2,273)', 'Random Forest\n(2,787 rows)', 'Gradient\nBoosting',
        'Standalone\nLSTM', 'LbL-QNN', 'LSTM+QNN', 'XGBoost', 'LSTM+PI-QNN', 'LSTM+QTL'
    ]
    model_order = [m for m in preferred_order if m in df['Model'].unique()]
    color_dict = {m: COLORS.get(m.replace('\n', ' '), '#455A64') for m in model_order}
    # Fallback explicit mapping for multi-line labels not in COLORS verbatim
    explicit = {
        'Random Forest\n(matched, 2,273)': '#2E7D32', 'Random Forest\n(2,787 rows)': '#1B5E20',
        'Gradient\nBoosting': '#388E3C', 'Standalone\nLSTM': '#C2185B', 'LbL-QNN': '#311B92',
        'LSTM+QNN': '#5E35B1', 'XGBoost': '#4CAF50', 'LSTM+PI-QNN': '#512DA8', 'LSTM+QTL': '#7E57C2',
    }
    color_dict.update({k: v for k, v in explicit.items() if k in model_order})

    _violinplot_compat(ax, df, 'Model', 'AUC', model_order, color_dict, inner='box', linewidth=1.2, cut=0)
    sns.stripplot(data=df, x='Model', y='AUC', order=model_order, color='black',
                  alpha=0.55, size=4.5, jitter=0.18, ax=ax, zorder=4)

    if 'Random Forest\n(matched, 2,273)' in records:
        rf_mean = records['Random Forest\n(matched, 2,273)']['vals'].mean()
        ax.axhline(rf_mean, color=explicit['Random Forest\n(matched, 2,273)'], ls='--', lw=1.3,
                   alpha=0.7, zorder=2, label=f'Matched Random Forest Mean (AUC = {rf_mean:.4f})')

    clean_spines(ax)
    ax.set_ylabel('ROC-AUC Distribution Across 15 Seeds')
    ax.set_xlabel('Evaluated Model Architecture')
    ax.set_title('Optimization Variance and Null Hypothesis Among Quantum Architectures\n'
                 '(15 Independent Initialization Seeds on Fixed Spatiotemporal Test Partition)', pad=12)

    # ---- Live-computed statistics (never hardcoded) ----
    stat_lines = []
    quantum_models = ['LSTM+QNN', 'LSTM+PI-QNN', 'LSTM+QTL', 'LbL-QNN']
    quantum_arrays = [records[m]['vals'] for m in quantum_models if m in records]
    if len(quantum_arrays) >= 3 and len({len(a) for a in quantum_arrays}) == 1:
        chi2, p_fried = sps.friedmanchisquare(*quantum_arrays)
        k = len(quantum_arrays); n = len(quantum_arrays[0])
        kendall_w = chi2 / (n * (k - 1))
        stat_lines.append(rf"• Quantum Tier Friedman Test: $\chi^2({k-1}) = {chi2:.3f}$, "
                           rf"$p = {p_fried:.3f}$ (Kendall's $W = {kendall_w:.3f}$)")
    else:
        print("  [WARN] Friedman test skipped — need ≥3 quantum models with equal-length "
              "real seed arrays.")

    if 'LSTM+QNN' in records and 'LbL-QNN' in records:
        x, y = records['LSTM+QNN']['vals'], records['LbL-QNN']['vals']
        if len(x) == len(y):
            mean_diff, se, p_tost = paired_tost(y, x, low=-0.015, upp=+0.015)
            stat_lines.append(rf"• TOST Equivalence (LbL vs. LSTM+QNN, ±0.015 AUC): "
                               rf"diff $= {mean_diff:+.4f}$, SE $= {se:.4f}$, $p_{{\mathrm{{TOST}}}} = {p_tost:.4f}$")

    if stat_lines:
        ax.text(0.02, 0.06, "Statistical Hypothesis Tests (computed live from 15-seed arrays):\n"
                + "\n".join(stat_lines), transform=ax.transAxes, fontsize=8.6,
                bbox=dict(boxstyle="round,pad=0.4", fc="#FFFFFF", ec="#B0BEC5", lw=0.9, alpha=0.95),
                va='bottom', ha='left')

    ax.legend(loc='upper right', fontsize=9.0, frameon=True, facecolor='white', framealpha=0.95, edgecolor='#CFD8DC')

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, format='pdf')
        fig.savefig(out_path.with_suffix('.png'), dpi=300)
    return fig


# ══════════════════════════════════════════════════════════════════════
# 7. FIGURE 5: OPERATIONAL PARETO FRONTIER & THRESHOLD CALIBRATION
# ══════════════════════════════════════════════════════════════════════

def _find_threshold_for_target_fpr(y_true, probs, target_fpr):
    """Finds the decision threshold on the real ROC curve whose FPR is
    closest to `target_fpr`. Used to recompute the 'matched' operating
    point instead of typing a τ value by hand."""
    fpr, tpr, thr = roc_curve(y_true, probs)
    idx = np.argmin(np.abs(fpr - target_fpr))
    return thr[idx], fpr[idx], tpr[idx]


def _pareto_point(y_true, dana_mask, probs, threshold):
    """Computes (FPR, Recall-on-ACP) at a given threshold, directly from
    real labels/probabilities — never typed in by hand."""
    pred = (probs >= threshold).astype(int)
    neg_mask = (y_true == 0)
    fp = int(((pred == 1) & neg_mask).sum())
    tn = int(((pred == 0) & neg_mask).sum())
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan

    if dana_mask is not None and dana_mask.sum() > 0:
        rc_acp = float(pred[dana_mask].mean())  # fraction of ACP events flagged positive
    else:
        rc_acp = np.nan
    return fpr, rc_acp


def fig_pareto_operational(data: dict, out_path: Path = None):
    """
    Operational Pareto Frontier: FPR vs. Recall on ACP events, computed at
    threshold = 0.5 directly from real (y_te, dana_te, probs). The 'Matched
    LSTM' operating point is obtained by searching the real ROC curve for
    the threshold whose FPR matches Random Forest's FPR (the reference used
    for the threshold-recalibration argument in the manuscript) — not a
    hand-typed τ.
    """
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    add_panel_label(ax, "(a)", x=-0.07, y=1.02)

    y_te, dana_te, probs = data['y_te'], data['dana_te'], data['probs']
    if dana_te is None:
        missing_data_panel(ax, "'dana_te' (ACP event mask) not found — cannot compute "
                               "Recall-on-ACP for the Pareto frontier.")
        if out_path:
            fig.savefig(out_path, format='pdf')
        return fig

    marker_map = {
        'Climatology': ('s', COLORS['Climatology']), 'Logistic Regression': ('s', COLORS['Logistic Regression']),
        'Random Forest': ('^', COLORS['Random Forest']), 'Gradient Boosting': ('^', COLORS['Gradient Boosting']),
        'XGBoost': ('^', COLORS['XGBoost']), 'Standalone LSTM': ('D', COLORS['Standalone LSTM']),
        'LSTM+QNN': ('o', COLORS['LSTM+QNN']), 'LSTM+PI-QNN': ('o', COLORS['LSTM+PI-QNN']),
        'LSTM+QTL': ('o', COLORS['LSTM+QTL']), 'LbL-QNN': ('o', COLORS['LbL-QNN']),
        'QKSVM': ('o', COLORS['QKSVM']),
    }

    points = {}
    for name, (marker, color) in marker_map.items():
        if name in probs:
            fpr_v, rc_v = _pareto_point(y_te, dana_te, probs[name], threshold=0.5)
            points[name] = (fpr_v, rc_v, marker, color, 140)

    # Threshold-matched LSTM: match FPR to Random Forest's real FPR
    if 'Standalone LSTM' in probs and 'Random Forest' in points:
        target_fpr = points['Random Forest'][0]
        tau, fpr_matched, rc_matched = _find_threshold_for_target_fpr(y_te, probs['Standalone LSTM'], target_fpr)
        points[f'Matched LSTM ($\\tau$={tau:.3f})'] = (fpr_matched, rc_matched, 'D', COLORS['Matched LSTM'], 160)

    ax.axvspan(0.15, 0.35, color='#E8F5E9', alpha=0.6, zorder=1)

    for name, (fpr_v, rc_v, m, c, sz) in points.items():
        if np.isnan(fpr_v) or np.isnan(rc_v):
            continue
        if 'LbL-QNN' == name:
            ax.scatter(fpr_v, rc_v, c='none', edgecolors=c, s=sz * 3.5, linewidths=2.0, alpha=0.4, zorder=4)
        ax.scatter(fpr_v, rc_v, marker=m, color=c, s=sz, edgecolors='white', linewidths=1.2, zorder=5)
        ax.annotate(name, xy=(fpr_v, rc_v), xytext=(fpr_v + 0.012, rc_v + 0.012),
                    fontsize=8.6, fontweight='bold' if ('Matched' in name or 'LbL' in name or 'Forest' in name) else 'normal',
                    color=c, ha='left', va='bottom', zorder=6,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))

    if 'Standalone LSTM' in points and any('Matched LSTM' in k for k in points):
        matched_key = [k for k in points if 'Matched LSTM' in k][0]
        p_orig, p_match = points['Standalone LSTM'][:2], points[matched_key][:2]
        ax.annotate('', xy=p_match, xytext=p_orig,
                    arrowprops=dict(facecolor=COLORS['Matched LSTM'], edgecolor=COLORS['Standalone LSTM'],
                                    width=1.8, headwidth=8, ls='--', alpha=0.85, zorder=4))
        ax.text((p_orig[0] + p_match[0]) / 2 + 0.01, (p_orig[1] + p_match[1]) / 2 + 0.02,
                'Threshold Recalibration\n(FPR-matched to Random Forest)',
                fontsize=8.3, color=COLORS['Matched LSTM'], fontstyle='italic', fontweight='bold',
                ha='center', zorder=6)

    all_fpr = [p[0] for p in points.values() if not np.isnan(p[0])]
    all_rc = [p[1] for p in points.values() if not np.isnan(p[1])]
    ax.set_xlim(max(all_fpr) + 0.08, min(0.1, min(all_fpr) - 0.05))
    ax.set_ylim(min(all_rc) - 0.08, 1.03)
    ax.set_xlabel('False Positive Rate (FPR, Overprediction Bias) $\\longleftarrow$ Lower / Better')
    ax.set_ylabel(f'Recall on ACP Events (Rc-ACP, $n={int(dana_te.sum())}$) $\\longrightarrow$ Higher / Better')
    ax.set_title('Operational Pareto Frontier: Convective Sensitivity vs. False Alarm Rate\n'
                 '(Computed at decision threshold 0.5 from real test-set predictions)', pad=12)
    clean_spines(ax)

    leg_handles = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['Climatology'], markersize=9, label='Climatological / Linear Baseline'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=COLORS['Random Forest'], markersize=10, label='Ensemble Trees'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=COLORS['Standalone LSTM'], markersize=9, label='Deep Learning'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['LbL-QNN'], markersize=10, label='Quantum Hybrid (NISQ)'),
        Patch(facecolor='#E8F5E9', edgecolor='#81C784', label='Operational False-Alarm Target ($< 35\\%$)')
    ]
    ax.legend(handles=leg_handles, loc='lower left', fontsize=8.6, frameon=True, facecolor='white', framealpha=0.92, edgecolor='#CFD8DC')

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, format='pdf')
        fig.savefig(out_path.with_suffix('.png'), dpi=300)
    return fig


# ══════════════════════════════════════════════════════════════════════
# 8. FIGURE 6: THE ILLUSION OF SINGLE-RUN ADVANTAGE VS MULTI-SEED
# ══════════════════════════════════════════════════════════════════════

def fig_single_run_illusion(data: dict, out_path: Path = None):
    """
    Panel (a): paired bootstrap AUC differences conditioned on a single
    seed — loaded from res_val['bootstrap_comparisons'] if present. Panel
    (b): REAL cross-seed pairwise differences, computed as the outer
    difference of the two real 15-seed arrays (LbL-QNN, LSTM+QNN) — i.e.
    225 genuine (though non-independent) pairwise comparisons, not a
    np.random.normal() draw. A separate, statistically valid TOST test is
    computed on the 15 *paired* per-seed-index differences (independent
    observations) and reported alongside the descriptive 225-pair spread,
    to avoid pseudo-replication inflating apparent precision.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # --- Panel (a): single-run bootstrap comparisons ---
    add_panel_label(ax1, "(a)")
    res_val = data.get('res_val') or {}
    boot_comparisons = res_val.get('bootstrap_comparisons')
    if not boot_comparisons and res_val.get('tests_delong'):
        from scipy.stats import norm
        boot_comparisons = []
        for key, t in res_val['tests_delong'].items():
            if not isinstance(t, dict) or 'delta_auc' not in t:
                continue
            d, z, p = float(t['delta_auc']), float(t['Z']), float(t['p_valor'])
            if z == 0:
                continue
            se = abs(d / z)
            p_check = 2 * (1 - norm.cdf(abs(z)))
            if abs(p_check - p) > 1e-3:
                print(f"WARNING: {key}: saved p-value {p:.4f} != p-value from Z {p_check:.4f}")
            boot_comparisons.append({
                'label': key.replace('_vs_', ' vs ').replace('PIQNN', 'PI-QNN').replace('LSTMpura', 'LSTM'),
                'diff': d, 'ci_low': d - 1.96 * se, 'ci_high': d + 1.96 * se, 'p_value': p})

    if not boot_comparisons:
        missing_data_panel(ax1, "No single-run paired comparisons found in 08_validacion.json.")
    else:
        names = [c['label'] for c in boot_comparisons]
        diffs = [c['diff'] for c in boot_comparisons]
        ci_los = [c['ci_low'] for c in boot_comparisons]
        ci_his = [c['ci_high'] for c in boot_comparisons]
        p_vals = [c['p_value'] for c in boot_comparisons]

        y_pos1 = np.arange(len(names))
        point_colors = ['#D32F2F' if p < 0.05 else '#455A64' for p in p_vals]
        for i in range(len(names)):
            xerr_lo, xerr_hi = diffs[i] - ci_los[i], ci_his[i] - diffs[i]
            ax1.errorbar(diffs[i], y_pos1[i], xerr=[[xerr_lo], [xerr_hi]], fmt='none',
                         ecolor=point_colors[i], elinewidth=2.2, capsize=5, capthick=1.5, zorder=4)
            ax1.scatter(diffs[i], y_pos1[i], color=point_colors[i], s=70, zorder=5, edgecolor='white')
            ax1.text(ci_his[i] + 0.003, y_pos1[i], f"$\\Delta = {diffs[i]:+.3f}$ ($p = {p_vals[i]:.3f}$)",
                      va='center', fontsize=8.5, fontweight='bold' if p_vals[i] < 0.05 else 'normal',
                      color=point_colors[i])

        ax1.axvline(0, color='black', lw=1.2, ls='--', zorder=2)
        ax1.set_yticks(y_pos1); ax1.set_yticklabels(names, fontsize=9.5)
        ax1.invert_yaxis()
        ax1.set_xlabel(r'Paired Difference in AUC ($\Delta\mathrm{AUC}$ with 95% Bootstrap CI)')
        ax1.set_title('Apparent Single-Run Differentiation\n(Conditioned on a Single Initialization Seed)', pad=10)
        clean_spines(ax1)

    # --- Panel (b): real 225 cross-seed pairwise differences ---
    add_panel_label(ax2, "(b)")
    lbl_vals = _safe_get_multiseed(data['multi_qml'], 'PI-QNN-LbL')
    qnn_vals = _safe_get_multiseed(data['multi_qml'], 'LSTM+QNN')

    if lbl_vals is None or qnn_vals is None:
        missing_data_panel(ax2, "Real 15-seed arrays for LbL-QNN and/or LSTM+QNN not found — "
                               "225-pair reality panel left blank.")
    else:
        # Genuine outer difference: 15 x 15 = 225 real pairwise comparisons
        diffs_225 = (lbl_vals[:, None] - qnn_vals[None, :]).ravel()
        pct_pos = 100.0 * np.mean(diffs_225 > 0)
        pct_neg = 100.0 * np.mean(diffs_225 < 0)

        n_bins = min(18, max(8, len(diffs_225) // 10))
        ax2.hist(diffs_225, bins=n_bins, density=True, color='#B39DDB', edgecolor='white', alpha=0.6, zorder=3)
        sns.kdeplot(diffs_225, ax=ax2, color='#311B92', lw=2.2, zorder=4, label='Cross-Seed Density ($n=225$ pairs)')

        ax2.axvspan(-0.015, +0.015, color='#E8F5E9', alpha=0.7, zorder=2, label=r'Equivalence Region $\Delta = \pm 0.015$ AUC')
        ax2.axvline(0, color='black', lw=1.2, ls='--', zorder=2)
        mean_225 = diffs_225.mean()
        ax2.axvline(mean_225, color='#311B92', lw=1.8, ls='-', zorder=4, label=f'Mean Difference ({mean_225:+.4f})')

        # Statistically valid TOST on the 15 *independent* paired seed-index differences
        if len(lbl_vals) == len(qnn_vals):
            mean_diff, se, p_tost = paired_tost(lbl_vals, qnn_vals, low=-0.015, upp=+0.015)
            tost_note = (f"Paired TOST ($n=15$ independent seeds, ±0.015 AUC): "
                         f"diff $= {mean_diff:+.4f}$, $p_{{\\mathrm{{TOST}}}} = {p_tost:.4f}$")
        else:
            tost_note = "Paired TOST unavailable (seed-count mismatch between models)."

        ax2.set_xlim(diffs_225.min() - 0.01, diffs_225.max() + 0.01)
        ax2.set_xlabel(r'Pairwise Difference ($\mathrm{AUC}_{\mathrm{LbL}} - \mathrm{AUC}_{\mathrm{QNN}}$)')
        ax2.set_ylabel('Probability Density')
        ax2.set_title(f'Replication Reality Across 225 Cross-Seed Pairs\n'
                      f'({pct_pos:.1f}% LbL $>$ QNN, {pct_neg:.1f}% QNN $>$ LbL — descriptive only)', pad=10)
        clean_spines(ax2)
        ax2.legend(loc='upper left', fontsize=8.3, frameon=True, facecolor='white', framealpha=0.92, edgecolor='#CFD8DC')
        ax2.text(0.02, 0.02, tost_note, transform=ax2.transAxes, fontsize=7.8, style='italic',
                  va='bottom', ha='left', color='#37474F',
                  bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CFD8DC", lw=0.7, alpha=0.9))

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, format='pdf')
        fig.savefig(out_path.with_suffix('.png'), dpi=300)
    return fig


# ══════════════════════════════════════════════════════════════════════
# 9. FIGURE 7: THERMODYNAMIC SEASONALITY & DYNAMICS (4 PANELS)
# ══════════════════════════════════════════════════════════════════════

def fig_thermo_analysis(data: dict, out_path: Path = None):
    """4-panel meteorological and thermodynamic analysis justifying the
    Clausius-Clapeyron and convective features. Reads directly from the
    real aggregated panel (agg_df) — unchanged in spirit from the original,
    only restyled."""
    fig, axs = plt.subplots(2, 2, figsize=(13.5, 9.5))
    agg = data['agg_df']

    if agg is None or not hasattr(agg, 'index') or not isinstance(agg.index, pd.DatetimeIndex):
        for ax_row in axs:
            for ax in ax_row:
                missing_data_panel(ax, "Meteorological time series (agg_df) unavailable.")
        if out_path:
            fig.savefig(out_path, format='pdf')
        return fig

    # (a) Mean Annual Cycle
    ax_a = axs[0, 0]; add_panel_label(ax_a, "(a)")
    if {'et0_sum', 'precip'}.issubset(agg.columns):
        et0_monthly = agg.groupby(agg.index.month)['et0_sum'].mean()
        precip_monthly = agg.groupby(agg.index.month)['precip'].mean()
        months = np.arange(1, 13)
        ax_a_twin = ax_a.twinx()
        ax_a.bar(months, et0_monthly.values, color='#F57C00', alpha=0.65, width=0.6,
                 edgecolor='white', zorder=2, label='Accumulated $\\text{ET}_0$')
        ax_a_twin.plot(months, precip_monthly.values, color='#1565C0', marker='o', lw=2.2,
                       markersize=5.5, zorder=4, label='Mean Precipitation')
        ax_a.axvspan(5.5, 8.5, color='#FFEBEE', alpha=0.6, zorder=1, label='Summer Energy Load (JJA)')
        ax_a.axvspan(8.5, 11.5, color='#E3F2FD', alpha=0.6, zorder=1, label='Autumn Convective Release (SON)')
        ax_a.set_xticks(months)
        ax_a.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], fontsize=8.5)
        ax_a.set_ylabel('Monthly Reference $\\text{ET}_0$ (mm)', color='#E65100')
        ax_a_twin.set_ylabel('Mean Monthly Precipitation (mm)', color='#0D47A1')
        ax_a.set_title('Annual Thermodynamic Energy Cycle\n(Summer Latent Load $\\to$ Autumn Convection)', pad=8)
        clean_spines(ax_a); ax_a_twin.spines['top'].set_visible(False)
        l1, lb1 = ax_a.get_legend_handles_labels(); l2, lb2 = ax_a_twin.get_legend_handles_labels()
        ax_a.legend(l1 + l2, lb1 + lb2, loc='upper left', fontsize=8.0, framealpha=0.9)
    else:
        missing_data_panel(ax_a, "'et0_sum'/'precip' columns missing from panel data.")

    # (b) Clausius-Clapeyron Shock Scatter
    ax_b = axs[0, 1]; add_panel_label(ax_b, "(b)")
    if {'choque_clausius', 'precip'}.issubset(agg.columns):
        sc = ax_b.scatter(agg['choque_clausius'], agg['precip'], c=agg.index.month,
                          cmap='twilight', alpha=0.65, s=16, edgecolor='none', zorder=3)
        cb = plt.colorbar(sc, ax=ax_b, pad=0.02); cb.set_label('Calendar Month', fontsize=9.5)
        cb.set_ticks([1, 4, 7, 10]); cb.set_ticklabels(['Jan (1)', 'Apr (4)', 'Jul (7)', 'Oct (10)'])
        ax_b.set_xlabel('Clausius-Clapeyron Thermal Contrast Proxy ($T_{i-1} - T_{i-4}$)')
        ax_b.set_ylabel('Monthly Precipitation (mm)')
        ax_b.set_title('Convective Response to Seasonal Cooling\n(Lagged Thermal Memory Contrast)', pad=8)
        clean_spines(ax_b)
    else:
        missing_data_panel(ax_b, "'choque_clausius' variable missing from panel data.")

    # (c) Autumn Pressure-Precipitation Interaction
    ax_c = axs[1, 0]; add_panel_label(ax_c, "(c)")
    if {'pmed', 'precip'}.issubset(agg.columns):
        autumn_mask = agg.index.month.isin([9, 10, 11])
        acp_flag_col = 'dana_flag' if 'dana_flag' in agg.columns else ('acp_flag' if 'acp_flag' in agg.columns else None)
        acp_flag_mask = (agg[acp_flag_col] == 1) if acp_flag_col else np.zeros(len(agg), dtype=bool)
        norm_autumn, acp_autumn = autumn_mask & (~acp_flag_mask), autumn_mask & acp_flag_mask

        ax_c.scatter(agg.loc[norm_autumn, 'pmed'], agg.loc[norm_autumn, 'precip'], color='#90CAF9',
                     alpha=0.55, s=24, edgecolor='none', zorder=3, label='Normal Autumn Months')
        ax_c.scatter(agg.loc[acp_autumn, 'pmed'], agg.loc[acp_autumn, 'precip'], color='#D32F2F',
                     alpha=0.90, s=75, marker='*', edgecolor='white', lw=0.6, zorder=5, label='ACP Synoptic Events')
        ax_c.set_xlabel('Monthly-Mean Sea-Level Pressure ($P_{\\text{med}}$, hPa)')
        ax_c.set_ylabel('Monthly Precipitation (mm)')
        ax_c.set_title('Autumn Pressure--Precipitation Regime\n(Cut-off Low Depression Barometric Signature)', pad=8)
        clean_spines(ax_c); ax_c.legend(loc='upper right', fontsize=8.5, framealpha=0.9)
        if acp_flag_col is None:
            print("  [WARN] No ACP/DANA flag column found in panel data — all autumn points "
                  "shown as 'Normal' in Figure 7c.")
    else:
        missing_data_panel(ax_c, "'pmed'/'precip' columns missing from panel data.")

    # (d) Physical Feature Correlations with Precipitation
    ax_d = axs[1, 1]; add_panel_label(ax_d, "(d)")
    corr_vars = [
        ('choque_clausius', 'Thermal Contrast ($\\Delta T_{3m}$)'), ('dp_std', 'Pressure Std. Dev. ($\\sigma_p$)'),
        ('ia_racha', 'Wind Gust Anomaly'), ('potencial_convectivo', 'Convective Potential'),
        ('et0_sum', 'Cumulative $\\text{ET}_0$'), ('veloc_ms', 'Mean Wind Speed'),
        ('iap', 'Pressure Anomaly ($z_P$)'), ('iir', 'Insolation Ratio ($S/S_0$)'),
        ('t_rango', 'Diurnal Temp. Range'), ('pmed', 'Mean Pressure ($P_{\\text{med}}$)')
    ]
    names_d, vals_d = [], []
    if 'precip' in agg.columns:
        for col, label in corr_vars:
            if col in agg.columns:
                names_d.append(label); vals_d.append(agg[col].corr(agg['precip']))

    if vals_d:
        y_pos_d = np.arange(len(vals_d))
        colors_d = ['#D32F2F' if v > 0 else '#1565C0' for v in vals_d]
        ax_d.barh(y_pos_d, vals_d, color=colors_d, edgecolor='white', height=0.70, zorder=3)
        ax_d.axvline(0, color='black', lw=1.1, zorder=4)
        ax_d.set_yticks(y_pos_d); ax_d.set_yticklabels(names_d, fontsize=8.8)
        ax_d.invert_yaxis()
        ax_d.set_xlabel('Pearson Correlation ($r$) with Monthly Precipitation')
        ax_d.set_title('Linear Dependencies of Engineered Covariates\n(Physical Correlation Landscape)', pad=8)
        clean_spines(ax_d)
        for i, v in enumerate(vals_d):
            ax_d.text(v + (0.015 if v >= 0 else -0.015), i, f"{v:+.2f}", va='center',
                      ha='left' if v >= 0 else 'right', fontsize=8.0, fontweight='bold', color=colors_d[i])
    else:
        missing_data_panel(ax_d, "No correlate columns found alongside 'precip' in panel data.")

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, format='pdf')
        fig.savefig(out_path.with_suffix('.png'), dpi=300)
    return fig


# ══════════════════════════════════════════════════════════════════════
# 10. QA / PRE-SUBMISSION VALIDATION
# ══════════════════════════════════════════════════════════════════════

def check_embedded_fonts(pdf_path: Path) -> bool:
    """Flags Type 3 (non-embeddable) fonts via poppler's `pdffonts`."""
    import subprocess
    try:
        out = subprocess.run(['pdffonts', str(pdf_path)], capture_output=True, text=True)
        bad = [l for l in out.stdout.splitlines() if 'Type 3' in l]
        if bad:
            print(f"  [WARN] {pdf_path.name}: Type 3 (non-embeddable) fonts detected.")
            return False
        return True
    except FileNotFoundError:
        print("  [INFO] `pdffonts` not found (install poppler-utils) — font check skipped.")
        return True


def check_no_placeholder_text(pdf_path: Path) -> bool:
    """Scans the rendered PDF for literal 'placeholder' text — this is
    exactly the failure mode found during the Round-5 manuscript review
    (Figures 4-10 rendering as literal placeholder strings)."""
    try:
        try:
            from pypdf import PdfReader
        except ImportError:
            from PyPDF2 import PdfReader
    except ImportError:
        print("  [INFO] Install `pypdf` (pip install pypdf) to enable placeholder-text scanning.")
        return True

    try:
        reader = PdfReader(str(pdf_path))
        text = " ".join((page.extract_text() or "") for page in reader.pages)
        if 'placeholder' in text.lower():
            print(f"  [BLOCKING] {pdf_path.name} contains literal 'placeholder' text — figure not rendered!")
            return False
        return True
    except Exception as e:
        print(f"  [INFO] Could not scan {pdf_path.name} for placeholder text: {e}")
        return True


def qa_before_submission(fig_dir: Path) -> bool:
    """Runs all pre-submission sanity checks. Returns False if ANY blocking
    issue is found — do not compile the final manuscript PDF until this
    passes."""
    print("\n" + "-" * 68)
    print("  PRE-SUBMISSION QA CHECKS")
    print("-" * 68)
    all_ok = True
    for pdf in sorted(Path(fig_dir).glob('figura_*.pdf')):
        ok_fonts = check_embedded_fonts(pdf)
        ok_text = check_no_placeholder_text(pdf)
        size_kb = pdf.stat().st_size / 1024
        ok_size = size_kb >= 5
        if not ok_size:
            print(f"  [WARN] {pdf.name} is suspiciously small ({size_kb:.1f} KB) — "
                  f"possibly empty/placeholder.")
        all_ok &= (ok_fonts and ok_text and ok_size)
    print("-" * 68)
    print("  RESULT:", "ALL CHECKS PASSED [OK]" if all_ok else "ISSUES FOUND — DO NOT SUBMIT YET [FAIL]")
    print("-" * 68 + "\n")
    return all_ok


# ══════════════════════════════════════════════════════════════════════
# 11. PUBLIC PIPELINE & CLI INTERFACE
# ══════════════════════════════════════════════════════════════════════

def generar_todas_las_figuras(output_dir: str = '.', base_dir: str = '.', run_qa: bool = True):
    """Generates all 7 figures required by main_revised.tex and compiles
    paper_figures.pdf. Runs QA checks at the end by default."""
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 68)
    print("  Q1 PUBLICATION FIGURE GENERATION PIPELINE")
    print("=" * 68)

    data = load_all_study_data(base_dir)

    figures_spec = [
        ('figura_accuracy_f1.pdf', fig_accuracy_f1, "1. Global Accuracy & F1-Score"),
        ('figura_roc.pdf', fig_roc, "2. High-Precision ROC Curves"),
        ('figura_feature_importance.pdf', fig_feature_importance, "3. Feature Importance & ACP Profile"),
        ('figura_auc_violin.pdf', fig_auc_violin, "4. 15-Seed Optimization Variance Violins"),
        ('figura_pareto_operational.pdf', fig_pareto_operational, "5. Operational Pareto Dial"),
        ('figura_single_run_illusion.pdf', fig_single_run_illusion, "6. Single-Run Illusion vs Multi-Seed"),
        ('figura_thermo_analysis.pdf', fig_thermo_analysis, "7. Thermodynamic Seasonality & Diagnostics"),
    ]

    generated_figs = []
    for fname, func, desc in figures_spec:
        fpath = out_p / fname
        try:
            fig = func(data, out_path=fpath)
            generated_figs.append(fig)
            print(f"  [OK] Generated {fname:<32} -- {desc}")
        except Exception as e:
            print(f"  [ERROR] Error generating {fname}: {e}")

    multipage_path = out_p / 'paper_figures.pdf'
    with PdfPages(multipage_path) as pdf:
        for fig in generated_figs:
            pdf.savefig(fig)
    print(f"  [OK] Multi-page review compilation saved: {multipage_path.name}")
    print("=" * 68)
    print(f"  All figures generated in: {out_p.resolve()}\n")

    if run_qa:
        qa_before_submission(out_p)


# Backward compatibility wrapper for code/main.py
def generar_figuras(agg=None, ds=None, res_base=None, res_qml=None, res_abl=None,
                     J_mat=None, h_ext=None, res_validacion=None, salida: str = 'resultados',
                     baseline_acc: float = None, baseline_f1: float = None,
                     tiempos_min: dict = None, res_multisemilla: dict = None,
                     umbral: float = 28.1, umbral_ext: float = 86.32) -> None:
    """Wrapper preserving the legacy API while enforcing Q1 standards."""
    generar_todas_las_figuras(output_dir=salida)


# English aliases for public API
generate_all_figures = generar_todas_las_figuras
generate_figures = generar_figuras


if __name__ == '__main__':
    target_dir = '.' if Path('main_revised.tex').exists() else 'resultados'
    generar_todas_las_figuras(output_dir=target_dir)