#!/usr/bin/env python3
"""
exp29_phase_intervals.py - do the cluster-robust verdicts depend on the month in which the "year" starts? (no retraining)
======================================================================================================================
exp26 shows that the ratio cluster-robust / DeLong SE of the seed-mean differences depends on where the 12-month
clusters start (0.55 for calendar years, up to 0.94 for other start months). This script asks the inferential question:
for each pair and each of the 12 possible start months, does the cluster-robust t_{K-1} interval
(Obuchowski-type sandwich on the influence values of the seed-mean AUC difference) exclude zero?

Usage (from the repository root):
  python code/local_experiments/exp29_phase_intervals.py
  python code/local_experiments/exp29_phase_intervals.py \
      --npz results/json/results_core_hybrids_controls_fixet0_p60_28.1_probs.npz \
      --pairs RF_2273-QTL_FIXED FROZEN_IDENTITY-LSTM_QNN --out results/local_exp29_controls.json
Output: results/local_exp29_phase_intervals.json
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from qml_resolution.auc import delong_components  # noqa: E402

NPZ = os.path.join(ROOT, 'results', 'json', 'results_hybrids_trainonly_fixet0_p60_28.1_probs.npz')
META = os.path.join(ROOT, 'results', 'test_meta.csv')
OUT = os.path.join(ROOT, 'results', 'local_exp29_phase_intervals.json')
PAIRS = ['RF_2273-PureLSTM', 'RF_2273-LSTM_QNN', 'RF_2273-LSTM_PIQNN', 'RF_2273-LBL_QNN',
         'CTRL_LINEAR4D-LSTM_QNN', 'PureLSTM-LSTM_QNN']
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def influence(y, pa, pb):
    ca, cb = delong_components(y, pa).seed_mean(), delong_components(y, pb).seed_mean()
    delta = float(ca.auc[0] - cb.auc[0])
    psi = np.zeros(y.size)
    psi[y] = (ca.v10[0] - cb.v10[0] - delta) / y.sum()
    psi[~y] = (ca.v01[0] - cb.v01[0] - delta) / (~y).sum()
    return delta, psi


def cluster_se(psi, groups):
    _, inv = np.unique(groups, return_inverse=True)
    s = np.bincount(inv, weights=psi)
    k = s.size
    return float(np.sqrt(k / (k - 1) * (s ** 2).sum())), k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default=NPZ)
    ap.add_argument('--pairs', nargs='+', default=PAIRS)
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()
    d = np.load(a.npz)
    meta = pd.read_csv(META, parse_dates=['fecha'])
    y = d['y_te'].astype(bool)
    assert (meta['y_te'].values.astype(bool) == y).all()
    ym = meta['fecha'].dt.year.values * 12 + meta['fecha'].dt.month.values
    out = {}
    for pair in a.pairs:
        pa, pb = pair.split('-', 1)
        delta, psi = influence(y, d[pa + '_probs'], d[pb + '_probs'])
        se_iid = float(np.sqrt((psi ** 2).sum()))
        rows = []
        for start in range(1, 13):
            se, k = cluster_se(psi, (ym - start) // 12)
            q = student_t.ppf(0.975, k - 1)
            lo, hi = delta - q * se, delta + q * se
            rows.append(dict(start=MONTHS[start - 1], K=k, se=se, ratio=se / se_iid, lo=lo, hi=hi,
                             excludes_zero=bool(lo > 0 or hi < 0)))
        n_ex = sum(r['excludes_zero'] for r in rows)
        out[pair] = dict(delta=delta, se_iid=se_iid, phases=rows, n_phases_excluding_zero=n_ex)
        print(f"{pair:28s} delta {delta:+.4f} | DeLong SE {se_iid:.4f} | cluster SE {min(r['se'] for r in rows):.4f}-"
              f"{max(r['se'] for r in rows):.4f} | t-interval excludes 0 in {n_ex}/12 start months "
              f"(lower limits {min(r['lo'] for r in rows):+.4f} to {max(r['lo'] for r in rows):+.4f})")
    with open(a.out, 'w') as f:
        json.dump(out, f, indent=2)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
