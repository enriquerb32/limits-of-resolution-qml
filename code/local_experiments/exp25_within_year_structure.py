#!/usr/bin/env python3
"""
exp25_within_year_structure.py - why the year-block SE is smaller than DeLong (no retraining)
=============================================================================================
For each pair of the main text, on the archived 15-seed test probabilities:
  1. influence values of the seed-mean AUC difference (DeLong structural components averaged over seeds);
  2. iid DeLong SE and the analytic cluster-robust (year) SE, sqrt(K/(K-1) * sum_years S_u^2),
     an Obuchowski (1997)-type sandwich estimator;
  3. decomposition of sum_u S_u^2 into positive-positive, negative-negative and positive-negative
     within-year terms;
  4. permutation null of the ratio cluster/iid SE: year labels permuted across rows (year sizes kept),
     so that rows carry no within-year structure;
  5. t_{K-1} interval with the cluster-robust SE;
  6. for Extractor - LSTM+QTL, the years x seeds bootstrap with seeds resampled jointly (paired by construction)
     next to the independent resampling used in exp10.
Also: test sizes at which a true difference d (smallest effect of interest) reaches 80 % power, assuming
SE proportional to N^(-1/2) from the full-test-set SE.

Usage (from the repository root):  python code/local_experiments/exp25_within_year_structure.py
Output: results/local_exp25_within_year_structure.json
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import t as student_t

_ap = argparse.ArgumentParser()
_ap.add_argument('root', nargs='?', default=None)
_ap.add_argument('--npz', default=None)
_ap.add_argument('--pairs', nargs='*', default=None, help='A-B pairs, e.g. RF_2273-QTL_FIXED')
_ap.add_argument('--out', default=None)
ARGS = _ap.parse_args()

ROOT = ARGS.root or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from qml_resolution.auc import _rank_auc_unchecked, delong_components  # noqa: E402

NPZ = ARGS.npz or os.path.join(ROOT, 'results', 'json', 'results_hybrids_trainonly_fixet0_p60_28.1_probs.npz')
YEARS = os.path.join(ROOT, 'results', 'years_te.npy')
OUT = ARGS.out or os.path.join(ROOT, 'results', 'local_exp25_within_year_structure.json')
PAIRS = [('RF_2273', 'PureLSTM'), ('RF_2273', 'LSTM_QNN'), ('CTRL_LINEAR4D', 'LSTM_QNN'), ('PureLSTM', 'LSTM_QNN'),
         ('RF_2273', 'LSTM_PIQNN'), ('RF_2273', 'LBL_QNN'), ('RF_2273', 'LSTM_QTL'),
         ('CTRL_LINEAR4D', 'LSTM_PIQNN'), ('CTRL_LINEAR4D', 'LBL_QNN'), ('CTRL_LINEAR4D', 'LSTM_QTL'),
         ('QTL_EXTRACTOR', 'LSTM_QTL')]
if ARGS.pairs:
    PAIRS = [tuple(p.split('-', 1)) for p in ARGS.pairs]
N_PERM, N_BOOT, SEED = 4000, 2000, 2025
Z = 1.959963984540054 + 0.8416212335729143


def influence(y, pa, pb):
    ca, cb = delong_components(y, pa).seed_mean(), delong_components(y, pb).seed_mean()
    delta = float(ca.auc[0] - cb.auc[0])
    psi = np.zeros(y.size)
    psi[y] = (ca.v10[0] - cb.v10[0] - delta) / y.sum()
    psi[~y] = (ca.v01[0] - cb.v01[0] - delta) / (~y).sum()
    return delta, psi


def cluster_var(psi, groups, levels):
    k = len(levels)
    return k / (k - 1) * sum(psi[groups == u].sum() ** 2 for u in levels)


def main():
    d = np.load(NPZ)
    y = d['y_te'].astype(bool)
    years = np.load(YEARS)
    levels = np.unique(years)
    k = len(levels)
    rng = np.random.default_rng(SEED)
    out = {'n': int(y.size), 'K': k, 'n_perm': N_PERM, 'pairs': {}}
    for a, b in PAIRS:
        delta, psi = influence(y, d[a + '_probs'], d[b + '_probs'])
        se_iid = float(np.sqrt((psi ** 2).sum()))
        se_cl = float(np.sqrt(cluster_var(psi, years, levels)))
        null = np.array([np.sqrt(cluster_var(psi, rng.permutation(years), levels)) / se_iid for _ in range(N_PERM)])
        pos_pos = sum(psi[(years == u) & y].sum() ** 2 for u in levels)
        neg_neg = sum(psi[(years == u) & ~y].sum() ** 2 for u in levels)
        cross = sum(2 * psi[(years == u) & y].sum() * psi[(years == u) & ~y].sum() for u in levels)
        tq = student_t.ppf(0.975, k - 1)
        out['pairs'][f'{a}-{b}'] = dict(
            delta=delta, se_iid=se_iid, se_cluster=se_cl, ratio=se_cl / se_iid,
            perm_ratio_q025=float(np.quantile(null, 0.025)), perm_ratio_median=float(np.median(null)),
            perm_ratio_q975=float(np.quantile(null, 0.975)), perm_p_lower=float((null <= se_cl / se_iid).mean()),
            within_pos_pos=float(pos_pos), within_neg_neg=float(neg_neg), within_cross=float(cross),
            iid_var=se_iid ** 2,
            ci_cluster_t=[delta - tq * se_cl, delta + tq * se_cl],
            mde_iid=Z * se_iid, mde_cluster=Z * se_cl)
    if ARGS.pairs:  # custom pairs: skip the extras tied to the default pair list
        with open(OUT, 'w') as f:
            json.dump(out, f, indent=2)
        for key, p in out['pairs'].items():
            print(f"{key:28s} d={p['delta']:+.4f} iid={p['se_iid']:.4f} cl={p['se_cluster']:.4f} ratio={p['ratio']:.3f} "
                  f"null95=[{p['perm_ratio_q025']:.2f},{p['perm_ratio_q975']:.2f}] p={p['perm_p_lower']:.4f} "
                  f"tCI=[{p['ci_cluster_t'][0]:+.3f},{p['ci_cluster_t'][1]:+.3f}]")
        print('wrote', OUT)
        return
    # joint vs independent seed resampling for the paired-by-construction QTL comparison
    A, B = d['QTL_EXTRACTOR_probs'], d['LSTM_QTL_probs']
    idx = {u: np.where(years == u)[0] for u in levels}
    ind, joint = [], []
    for _ in range(N_BOOT):
        ii = np.concatenate([idx[u] for u in rng.choice(levels, k)])
        yb = y[ii]
        if yb.all() or (~yb).all():
            continue
        sa, sb = rng.integers(0, A.shape[0], A.shape[0]), rng.integers(0, B.shape[0], B.shape[0])
        auc_a = _rank_auc_unchecked(yb, A[sa][:, ii]).mean()
        ind.append(auc_a - _rank_auc_unchecked(yb, B[sb][:, ii]).mean())
        joint.append(auc_a - _rank_auc_unchecked(yb, B[sa][:, ii]).mean())
    out['qtl_two_way'] = dict(independent=[float(np.quantile(ind, q)) for q in (0.025, 0.975)],
                              joint=[float(np.quantile(joint, q)) for q in (0.025, 0.975)])
    # test size for 80 % power at a smallest effect of interest d, SE ~ N^(-1/2)
    req = {}
    for key in ('CTRL_LINEAR4D-LSTM_QNN', 'PureLSTM-LSTM_QNN', 'RF_2273-LSTM_QNN'):
        p = out['pairs'][key]
        req[key] = {str(dd): {'iid': y.size * (p['mde_iid'] / dd) ** 2, 'cluster': y.size * (p['mde_cluster'] / dd) ** 2}
                    for dd in (0.005, 0.01, 0.02)}
    out['n_required_80pct'] = req
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    for key, p in out['pairs'].items():
        print(f"{key:28s} d={p['delta']:+.4f} iid={p['se_iid']:.4f} cl={p['se_cluster']:.4f} ratio={p['ratio']:.3f} "
              f"null95=[{p['perm_ratio_q025']:.2f},{p['perm_ratio_q975']:.2f}] p={p['perm_p_lower']:.4f} "
              f"cross={p['within_cross']:.2e} iidvar={p['iid_var']:.2e} tCI=[{p['ci_cluster_t'][0]:+.3f},{p['ci_cluster_t'][1]:+.3f}]")
    print('QTL two-way', out['qtl_two_way'])
    for key, v in req.items():
        print(key, {dd: (round(x['iid']), round(x['cluster'])) for dd, x in v.items()})
    print('wrote', OUT)


if __name__ == '__main__':
    main()
