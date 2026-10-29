"""
code/local_experiments/noise_utils.py
Shared helpers for the noise-budget analysis (exp7) and the positive control (exp8).

- Fast DeLong (Sun & Xu, 2014) for k correlated predictors scored on the same labels.
- Vectorised rank-based AUC for year-block bootstrap.
(Both delegate to the `qml_resolution` library in src/.)
- Year inference for the 528-row test partition (same rule as exp4) with a sanity check.
"""

from pathlib import Path
import json
import sys
import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))  # use the repository copy if the package is not installed

from qml_resolution import (  # noqa: E402
    cluster_bootstrap_aucs,
    delong_covariance,
    paired_delong,
    rank_auc,
)

Z975 = norm.ppf(0.975)
Z80 = norm.ppf(0.80)


def load_test_partition():
    json_path = ROOT / 'results' / 'json' / '01_data.json'
    with open(json_path, 'r', encoding='utf-8') as f:
        ds = json.load(f)[1]
    y_te = np.array(ds['y_te']).astype(int)
    m_te = np.array(ds['m_te']).astype(int)
    X_te = np.array(ds['X_te'])

    # Prefer actual calendar year array if present
    years_npy = ROOT / 'results' / 'years_te.npy'
    if not years_npy.exists():
        years_npy = ROOT / 'data' / 'years_te.npy'

    if years_npy.exists():
        years = np.load(years_npy)
        assert len(years) == len(y_te), (len(years), len(y_te))
    else:
        # Year inference (identical to exp4): new year whenever the month index decreases.
        year, years = 2008, []
        for i in range(len(m_te)):
            if i > 0 and m_te[i] < m_te[i - 1]:
                year += 1
            years.append(year)
        years = np.array(years)

    # Sanity check: with 4 stations, a calendar year can hold at most 4 rows per month.
    max_rows_per_month = max(
        int(np.max(np.bincount(m_te[years == yr], minlength=13))) for yr in np.unique(years)
    )
    return ds, y_te, m_te, X_te, years, max_rows_per_month


# Thin wrappers that keep the historical signatures used by exp7, exp8, exp16 and exp22.
# The implementations live in the tested library `qml_resolution`.

def delong_cov(y, preds):
    """Fast DeLong. preds: (k, n). Returns (aucs (k,), covariance (k, k))."""
    return delong_covariance(y, preds)


def delong_se(y, p):
    a, c = delong_covariance(y, p)
    return float(a[0]), float(np.sqrt(c[0, 0]))


def delong_paired(y, p1, p2):
    """Returns (auc1 - auc2, SE of the difference, two-sided p)."""
    test = paired_delong(y, p1, p2)
    return test.delta, test.se, test.p_value


def auc_matrix(y, P):
    """Rank-based AUC for every row of P (k, n) against labels y (n,). Handles ties/duplicates."""
    return rank_auc(y, P)


def year_block_bootstrap_aucs(y, P, years, B=2000, seed=42):
    """Returns (B_valid, k) matrix of bootstrap AUCs, resampling calendar-year blocks."""
    return cluster_bootstrap_aucs(y, P, years, n_boot=B, random_state=seed)
