#!/usr/bin/env python3
"""
exp28b_fair_climatology.py - recalibrate the climatology too, before comparing recalibrated models with it (no retraining)
=========================================================================================================================
exp28 --analyze recalibrates every model on the validation partition (Platt, isotonic) but compares it with the RAW
month x station climatology. The climatology is also miscalibrated on the test years (ECE 0.069 with the pooled label;
training rate 0.40 against a test rate of 0.347 with the station x month label), so part of the Brier gain of a
recalibrated model can be a base-rate correction that the climatology would also get. This script applies the same
validation-fitted Platt map to the climatology and recomputes the skill of every model against it, together with a
paired year-block bootstrap of the Brier difference (15-seed mean model - Platt climatology).

Usage (from the repository root):
  python code/local_experiments/exp28b_fair_climatology.py --label orig
  python code/local_experiments/exp28b_fair_climatology.py --label stationmonth
Output: results/local_exp28b_<label>.json
"""
import argparse
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p)).reshape(-1, 1)


def platt(y_va, p_va, p_te):
    return LogisticRegression(C=1e6).fit(logit(p_va), y_va).predict_proba(logit(p_te))[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', choices=['orig', 'stationmonth'], required=True)
    ap.add_argument('--npz', default=None)
    ap.add_argument('--years', default=os.path.join(ROOT, 'results', 'years_te.npy'))
    ap.add_argument('--B', type=int, default=2000)
    a = ap.parse_args()
    npz = a.npz or os.path.join(ROOT, 'results', 'json', f'results_exp28_{a.label}_probs.npz')
    d = np.load(npz)
    y_va, y_te = d['y_va'].astype(int), d['y_te'].astype(int)
    years = np.load(a.years)
    assert years.size == y_te.size
    c_va, c_te = d['clim_val'], d['clim_te']
    c_pl = platt(y_va, c_va, c_te)
    bs = lambda p: float(np.mean((p - y_te) ** 2))  # noqa: E731
    out = {'label': a.label, 'test_rate': float(y_te.mean()), 'val_rate': float(y_va.mean()),
           'climatology': {'brier_raw': bs(c_te), 'brier_platt': bs(c_pl), 'auc': float(roc_auc_score(y_te, c_te))},
           'models': {}}
    print(f"climatology: test rate {y_te.mean():.3f} (validation {y_va.mean():.3f}) | Brier raw {bs(c_te):.4f} "
          f"Platt {bs(c_pl):.4f}")
    rng = np.random.default_rng(2028)
    U = np.unique(years)
    idx = {u: np.where(years == u)[0] for u in U}
    boots = [np.concatenate([idx[u] for u in rng.choice(U, U.size)]) for _ in range(a.B)]
    for k in [f[:-6] for f in d.files if f.endswith('_probs')]:
        P = np.array([platt(y_va, pv, pt) for pv, pt in zip(d[f'{k}_val'], d[f'{k}_probs'])])
        b = np.mean((P - y_te) ** 2, axis=1)
        bss_raw_clim = 1 - b / bs(c_te)
        bss_fair = 1 - b / bs(c_pl)
        se_row = ((P - y_te) ** 2).mean(0) - (c_pl - y_te) ** 2            # per-row difference, seed-mean model
        diffs = np.array([se_row[i].mean() for i in boots])
        out['models'][k] = dict(
            brier_platt_mean=float(b.mean()),
            bss_vs_raw_clim=dict(mean=float(bss_raw_clim.mean()), min=float(bss_raw_clim.min()),
                                 max=float(bss_raw_clim.max()), n_above=int((bss_raw_clim > 0).sum())),
            bss_vs_platt_clim=dict(mean=float(bss_fair.mean()), min=float(bss_fair.min()), max=float(bss_fair.max()),
                                   n_above=int((bss_fair > 0).sum())),
            brier_minus_platt_clim=float(se_row.mean()),
            brier_minus_platt_clim_block_ci=[float(np.quantile(diffs, .025)), float(np.quantile(diffs, .975))])
        m = out['models'][k]
        print(f"{k:14s} Brier Platt {b.mean():.4f} | BSS vs raw clim {m['bss_vs_raw_clim']['mean']:+.3f} "
              f"({m['bss_vs_raw_clim']['n_above']}/15) | BSS vs Platt clim {m['bss_vs_platt_clim']['mean']:+.3f} "
              f"[{m['bss_vs_platt_clim']['min']:+.3f},{m['bss_vs_platt_clim']['max']:+.3f}] "
              f"({m['bss_vs_platt_clim']['n_above']}/15) | Brier diff block CI "
              f"[{m['brier_minus_platt_clim_block_ci'][0]:+.4f},{m['brier_minus_platt_clim_block_ci'][1]:+.4f}]")
    path = os.path.join(ROOT, 'results', f'local_exp28b_{a.label}.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print('wrote', path)


if __name__ == '__main__':
    main()
