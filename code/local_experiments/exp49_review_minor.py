#!/usr/bin/env python3
"""
exp49_review_minor.py - within station x month AUC, SEAS5 increments, analytic population AUC (no training; archived predictions only)
=================================================================================================================
A. Within station x month discrimination with the ORIGINAL (pooled P60) label: AUC computed only over positive-negative
   pairs of the same station and calendar month (pair-weighted mean of the 48 cell AUCs; cells with one class drop
   out). The climatology is constant within a cell, so its within-cell AUC is exactly 0.5: here the contrast against
   AUC 0.5 IS the contrast against the climatology, unlike for the pooled AUC. Interval: calendar-year block bootstrap.
B. Incremental value over SEAS5 + climatology: per seed, a logistic combination fitted on the validation partition of
   logit SEAS5 (quantile-corrected), logit climatology and logit model, against the same combination without the
   model (SEAS5 + climatology). Leads 2 (operational: available before the target month) and 1. Paired AUC and
   Brier differences with calendar-year block bootstrap intervals and DeLong p.
C. Appendix A of the paper: analytic population AUC of the binormal mechanism of exp11,
   AUC_k = Phi(mu_k / sqrt(2 (1 + tau_s^2 + tau_m^2))), against the Monte Carlo references used in the paper.

  python code/local_experiments/exp49_review_minor.py
Output: results/local_exp49_review_minor.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from qml_resolution import paired_delong, rank_auc  # noqa: E402

J = os.path.join(ROOT, 'results', 'json')
MODELS = ['RF_2273', 'XGB', 'PureLSTM', 'CTRL_LINEAR4D', 'LSTM_QNN']
B, SEED = 2000, 49


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def year_boot(years, rng):
    G = np.unique(years); rows = [np.flatnonzero(years == g) for g in G]
    return [np.concatenate([rows[i] for i in rng.integers(0, len(G), len(G))]) for _ in range(B)]


def within_cell_auc(y, p, cell):
    """Pair-weighted within-cell AUC of each row of p (S, n)."""
    p = np.atleast_2d(p); num = np.zeros(p.shape[0]); den = 0.0
    for c in np.unique(cell):
        ix = cell == c; yy = y[ix]
        n1 = yy.sum(); n0 = yy.size - n1
        if n1 == 0 or n0 == 0:
            continue
        num += rank_auc(yy, p[:, ix]) * n1 * n0; den += n1 * n0
    return num / den, den


def part_a(rng):
    z = np.load(os.path.join(J, 'results_exp28_orig_probs.npz'))
    meta = pd.read_csv(os.path.join(ROOT, 'results', 'test_meta.csv'), parse_dates=['fecha'])
    y = z['y_te'].astype(int); assert np.array_equal(y, meta['y_te'].values)
    cell = (meta['ciudad'] + '_' + meta['fecha'].dt.month.astype(str)).values
    years = meta['fecha'].dt.year.values
    boots = year_boot(years, rng)
    out = {}
    P = {m: z[f'{m}_probs'] for m in MODELS}; P['CLIM'] = z['clim_te'][None]
    for m, p in P.items():
        a, pairs = within_cell_auc(y, p, cell)
        reps = [within_cell_auc(y[ix], p[:, ix], cell[ix])[0].mean() for ix in boots]
        lo, hi = np.percentile(reps, [2.5, 97.5])
        out[m] = dict(within_cell_auc=float(a.mean()), ci_block=[float(lo), float(hi)], n_pairs=int(pairs),
                      pooled_auc=float(rank_auc(y, p).mean()))
        print(f'  A {m:14s} within-cell {a.mean():.4f} [{lo:.3f}, {hi:.3f}] | pooled {out[m]["pooled_auc"]:.4f}')
    out['n_cells_with_both_classes'] = int(sum(1 for c in np.unique(cell) if 0 < y[cell == c].sum() < (cell == c).sum()))
    return out


def part_b(rng):
    z = np.load(os.path.join(J, 'results_exp33_seas5_orig_probs.npz'))
    meta = pd.read_csv(os.path.join(ROOT, 'results', 'test_meta.csv'), parse_dates=['fecha'])
    y_va, y_te = z['y_va'].astype(int), z['y_te'].astype(int)
    years = meta['fecha'].dt.year.values
    boots = year_boot(years, rng)
    cv, ct = logit(z['CLIM_val'][0]), logit(z['CLIM_probs'][0])
    out = {}
    for L in (2, 1):
        sv, st = logit(z[f'SEAS5_QM_L{L}_val'][0]), logit(z[f'SEAS5_QM_L{L}_probs'][0])
        base = LogisticRegression(C=1e6, max_iter=1000).fit(np.c_[sv, cv], y_va)
        p_base = base.predict_proba(np.c_[st, ct])[:, 1][None]
        out[f'L{L}'] = {'SEAS5+CLIM': dict(auc=float(rank_auc(y_te, p_base)[0]),
                                           brier=float(np.mean((p_base[0] - y_te) ** 2)))}
        for m in MODELS:
            pv, pt = z[f'{m}_val'], z[f'{m}_probs']
            P = []
            for s in range(pv.shape[0]):
                lr = LogisticRegression(C=1e6, max_iter=1000).fit(np.c_[sv, cv, logit(pv[s])], y_va)
                P.append(lr.predict_proba(np.c_[st, ct, logit(pt[s])])[:, 1])
            P = np.array(P)
            t = paired_delong(y_te, P, p_base)
            d_auc = [rank_auc(y_te[ix], P[:, ix]).mean() - rank_auc(y_te[ix], p_base[:, ix])[0] for ix in boots]
            bs = ((P - y_te) ** 2).mean(1).mean() - np.mean((p_base[0] - y_te) ** 2)
            d_br = [((P[:, ix] - y_te[ix]) ** 2).mean() - ((p_base[0, ix] - y_te[ix]) ** 2).mean() for ix in boots]
            out[f'L{L}'][m] = dict(auc=float(rank_auc(y_te, P).mean()), delta_auc=float(t.delta),
                                   p_delong=float(t.p_value), ci_block_auc=np.percentile(d_auc, [2.5, 97.5]).tolist(),
                                   delta_brier=float(bs), ci_block_brier=np.percentile(d_br, [2.5, 97.5]).tolist())
            r = out[f'L{L}'][m]
            print(f'  B L{L} {m:14s} dAUC {r["delta_auc"]:+.4f} (p {r["p_delong"]:.3f}) [{r["ci_block_auc"][0]:+.3f}, '
                  f'{r["ci_block_auc"][1]:+.3f}] | dBrier {r["delta_brier"]:+.4f} [{r["ci_block_brier"][0]:+.4f}, {r["ci_block_brier"][1]:+.4f}]')
    return out


def part_c():
    """exp11 design: AUC_B = 0.65 without year effects, AUC_A = AUC_B + design delta; scenarios of Table S-sim."""
    sc = {'B': (0.3, 0.0), 'C': (0.3, 0.3), 'D': (0.3, 0.3), 'E': (0.6, 0.6)}
    reported = {'B': 0.029, 'C': 0.028, 'D': 0.028, 'E': 0.023}   # population differences printed in Table S-sim
    out = {}
    for k, (ts, tm) in sc.items():
        row = {}
        for dd in (0.01, 0.02, 0.03, 0.04, 0.05):
            mu_a, mu_b = np.sqrt(2) * norm.ppf(0.65 + dd), np.sqrt(2) * norm.ppf(0.65)
            s = np.sqrt(2 * (1 + ts ** 2 + tm ** 2))
            row[f'{dd:.2f}'] = dict(auc_a=float(norm.cdf(mu_a / s)), auc_b=float(norm.cdf(mu_b / s)),
                                    delta=float(norm.cdf(mu_a / s) - norm.cdf(mu_b / s)))
        out[k] = dict(tau_s=ts, tau_m=tm, by_design_delta=row, reported_mc=reported[k])
        print(f'  C scenario {k}: analytic delta at design 0.03 = {row["0.03"]["delta"]:.4f} (MC in paper {reported[k]})')
    return out


def main():
    rng = np.random.default_rng(SEED)
    res = dict(within_cell=part_a(rng), seas5_increment=part_b(rng), analytic_auc=part_c())
    json.dump(res, open(os.path.join(ROOT, 'results', 'local_exp49_review_minor.json'), 'w'), indent=2)
    print('wrote results/local_exp49_review_minor.json')


if __name__ == '__main__':
    main()
