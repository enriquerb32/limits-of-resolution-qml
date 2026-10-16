"""
code/local_experiments/noise_utils.py
Shared helpers for the noise-budget analysis (exp7) and the positive control (exp8).

- Fast DeLong (Sun & Xu, 2014) for k correlated predictors scored on the same labels.
- Vectorised rank-based AUC for year-block bootstrap.
- Year inference for the 528-row test partition (same rule as exp4) with a sanity check.
"""

import json
import numpy as np
from scipy.stats import rankdata, norm

Z975 = norm.ppf(0.975)
Z80 = norm.ppf(0.80)


def load_test_partition():
    with open('results/json/01_data.json', 'r') as f:
        ds = json.load(f)[1]
    y_te = np.array(ds['y_te']).astype(int)
    m_te = np.array(ds['m_te']).astype(int)
    X_te = np.array(ds['X_te'])
    # Year inference (identical to exp4): new year whenever the month index decreases.
    year, years = 0, []
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


def delong_cov(y, preds):
    """Fast DeLong. preds: (k, n). Returns (aucs (k,), covariance (k, k))."""
    y = np.asarray(y).astype(int)
    preds = np.atleast_2d(np.asarray(preds, dtype=float))
    pos, neg = preds[:, y == 1], preds[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    tx = np.apply_along_axis(rankdata, 1, pos)
    ty = np.apply_along_axis(rankdata, 1, neg)
    tz = np.apply_along_axis(rankdata, 1, np.concatenate([pos, neg], axis=1))
    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2.0) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    return aucs, sx / m + sy / n


def delong_se(y, p):
    a, c = delong_cov(y, p[None, :])
    return float(a[0]), float(np.sqrt(c[0, 0]))


def delong_paired(y, p1, p2):
    """Returns (auc1 - auc2, SE of the difference, two-sided p)."""
    a, c = delong_cov(y, np.vstack([p1, p2]))
    d = a[0] - a[1]
    se = np.sqrt(max(c[0, 0] + c[1, 1] - 2 * c[0, 1], 1e-300))
    return float(d), float(se), float(2 * norm.sf(abs(d) / se))


def auc_matrix(y, P):
    """Rank-based AUC for every row of P (k, n) against labels y (n,). Handles ties/duplicates."""
    y = np.asarray(y).astype(bool)
    n1 = y.sum()
    n0 = len(y) - n1
    R = rankdata(P, axis=1)
    return (R[:, y].sum(axis=1) - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def year_block_bootstrap_aucs(y, P, years, B=2000, seed=42):
    """Returns (B_valid, k) matrix of bootstrap AUCs, resampling calendar-year blocks."""
    rng = np.random.default_rng(seed)
    uy = np.unique(years)
    idx_by_year = [np.where(years == u)[0] for u in uy]
    out = []
    for _ in range(B):
        pick = rng.integers(0, len(uy), size=len(uy))
        idx = np.concatenate([idx_by_year[j] for j in pick])
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        out.append(auc_matrix(yb, P[:, idx]))
    return np.array(out)
