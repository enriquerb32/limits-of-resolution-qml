#!/usr/bin/env python3
"""
exp23_within_station_auc.py - Within-station AUC of the train-only 15-seed models (no retraining)
================================================================================================
The pooled AUC compares positive and negative months of different stations, so part of it rewards separating
high- from low-base-rate stations (a station-only base rate reaches AUC 0.599). The within-station AUC counts only
positive/negative pairs from the SAME station; it equals the average of the per-station AUCs weighted by
n_pos * n_neg of each station. A station-only predictor scores exactly 0.5 on it.

For every model: per-station AUC (15-seed mean), within-station AUC (15-seed mean, SD), and for selected pairs the
seed-mean difference with a calendar-year block bootstrap (B = 2,000, seed 42; the same years for both models).

Usage (from the repository root):  python code/local_experiments/exp23_within_station_auc.py
Output: results/local_exp23_within_station_auc.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from noise_utils import ROOT, load_test_partition  # noqa: E402
from exp7_noise_budget import climatology_probs  # noqa: E402

B, SEED = 2000, 42
OUT = os.path.join(ROOT, 'results', 'local_exp23_within_station_auc.json')


def auc(y, p):
    m = int(y.sum()); n = len(y) - m
    if m == 0 or n == 0:
        return np.nan
    r = rankdata(p)
    return (r[y == 1].sum() - m * (m + 1) / 2) / (m * n)


def within_auc(y, p, st):
    num = den = 0.0
    for s in np.unique(st):
        k = st == s
        m = int(y[k].sum()); n = int(k.sum()) - m
        if m and n:
            num += auc(y[k], p[k]) * m * n
            den += m * n
    return num / den


def main():
    ds, y, m_te, X_te, years, _ = load_test_partition()
    meta = pd.read_csv(os.path.join(ROOT, 'results', 'test_meta.csv'))
    assert (meta['y_te'].values == y).all()
    st = meta['ciudad'].values
    core = np.load(os.path.join(ROOT, 'results', 'json', 'results_hybrids_trainonly_fixet0_p60_28.1_probs.npz'))
    rr = np.load(os.path.join(ROOT, 'results', 'json', 'results_rerun_train_only_canonical_probs.npz'))
    P = {'Climatology': climatology_probs(ds, y, m_te, X_te)[None, :],
         'Random Forest': core['RF_2273_probs'], 'XGBoost': rr['train_only_XGB_probs'],
         'Standalone LSTM': core['PureLSTM_probs'], 'Linear-4D control': core['CTRL_LINEAR4D_probs'],
         'LSTM+QNN': core['LSTM_QNN_probs'], 'LbL-QNN': core['LBL_QNN_probs'],
         'LSTM+PI-QNN': core['LSTM_PIQNN_probs'], 'LSTM+QTL': core['LSTM_QTL_probs']}
    stations = sorted(np.unique(st))
    out = {'meta': dict(B=B, seed=SEED, stations={s: dict(n=int((st == s).sum()), pos=int(y[st == s].sum())) for s in stations}),
           'models': {}, 'pairs': {}}
    hdr = ' '.join(f'{s[:9]:>9s}' for s in stations)
    print(f"{'model':18s} {'pooled':>7s} {'within':>7s} {'SD':>6s}  {hdr}")
    for k, Pm in P.items():
        pooled = np.array([auc(y, p) for p in Pm]); wi = np.array([within_auc(y, p, st) for p in Pm])
        per = {s: float(np.mean([auc(y[st == s], p[st == s]) for p in Pm])) for s in stations}
        out['models'][k] = dict(pooled_auc=float(pooled.mean()), within_auc=float(wi.mean()),
                                within_sd=float(wi.std(ddof=1)) if len(wi) > 1 else None, per_station=per)
        print(f"{k:18s} {pooled.mean():7.4f} {wi.mean():7.4f} {(wi.std(ddof=1) if len(wi) > 1 else 0):6.4f}  "
              + ' '.join(f'{per[s]:9.4f}' for s in stations))
    # paired year-block bootstrap of within-station seed-mean differences
    uy = np.unique(years); idx_by = [np.where(years == u)[0] for u in uy]
    rng = np.random.default_rng(SEED)
    boots = [np.concatenate([idx_by[j] for j in rng.integers(0, len(uy), len(uy))]) for _ in range(B)]
    pairs = [('Random Forest', 'Climatology'), ('Random Forest', 'LSTM+QNN'), ('Random Forest', 'Standalone LSTM'),
             ('Random Forest', 'XGBoost'), ('Linear-4D control', 'LSTM+QNN'), ('Standalone LSTM', 'LSTM+QNN'),
             ('Standalone LSTM', 'Climatology'), ('LSTM+QNN', 'Climatology')]
    for a, b in pairs:
        d0 = out['models'][a]['within_auc'] - out['models'][b]['within_auc']
        ds_ = []
        for idx in boots:
            ya, sa = y[idx], st[idx]
            ds_.append(np.mean([within_auc(ya, p[idx], sa) for p in P[a]]) - np.mean([within_auc(ya, p[idx], sa) for p in P[b]]))
        lo, hi = np.percentile(ds_, [2.5, 97.5])
        out['pairs'][f'{a} - {b}'] = dict(delta=float(d0), se_block=float(np.std(ds_, ddof=1)), ci_block=[float(lo), float(hi)])
        print(f'  {a} - {b}: {d0:+.4f} [{lo:+.4f}, {hi:+.4f}]', flush=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print('saved', OUT)


if __name__ == '__main__':
    main()
