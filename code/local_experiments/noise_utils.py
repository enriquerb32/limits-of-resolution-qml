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


def _trainonly_labels():
    """Training and validation labels of the train-only dataset (threshold 28.1 mm), as used by the models.

    01_data.json stores the threshold unrounded (28.0999... mm), so a month with exactly 28.1 mm
    (Alicante, February 1991) is positive there and negative in the train-only dataset."""
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import trainonly_dataset as tds
    _, d = tds.build_dataset(str(Path(here) / 'datos_limpios_multiestacion.csv'), train_only=True,
                             fix_et0_verano=True, umbral_override=28.1)
    arr = lambda v: np.asarray(v.numpy() if hasattr(v, 'numpy') else v)  # noqa: E731
    return {k: arr(getattr(d, k)) for k in ('y_tr', 'm_tr', 'y_va', 'm_va', 'y_te')}


def load_test_partition(labels='trainonly'):
    """labels='trainonly' (default): training/validation labels of the train-only dataset (threshold 28.1 mm),
    i.e. the labels the 15-seed models were fitted on. labels='canonical': the labels stored in 01_data.json."""
    json_path = ROOT / 'results' / 'json' / '01_data.json'
    with open(json_path, 'r', encoding='utf-8') as f:
        ds = json.load(f)[1]
    if labels == 'trainonly':
        t = _trainonly_labels()
        for part in ('tr', 'va'):
            assert np.array_equal(t[f'm_{part}'].astype(int), np.asarray(ds[f'm_{part}']).astype(int))
            n_diff = int((t[f'y_{part}'].astype(int) != np.asarray(ds[f'y_{part}']).astype(int)).sum())
            assert n_diff <= 2, (part, n_diff)
            ds[f'y_{part}'] = t[f'y_{part}'].astype(float).tolist()
        assert np.array_equal(t['y_te'].astype(int), np.asarray(ds['y_te']).astype(int))
    elif labels != 'canonical':
        raise ValueError(labels)
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
