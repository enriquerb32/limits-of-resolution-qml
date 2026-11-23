#!/usr/bin/env python3
"""
exp24_resolution_curve.py - Empirical resolution curve: how the SE and MDE of paired AUC differences shrink with
the size of the test set (no retraining)
================================================================================================================
Random subsets of K of the 19 test years (K = 5, 8, 10, 13, 16; R = 300 subsets each, plus the full set K = 19) are
drawn without replacement, all stations of a year together. On every subset, for each pair of models:
  * SE of a single-run difference: paired DeLong SE of seed s of A against seed s of B, averaged over the 15 seeds;
  * SE of the 15-seed-mean difference: paired DeLong on seed-averaged structural components (as exp10), and the
    delete-one-year jackknife SE (as exp16 C4).
MDE = 2.80 x SE. The medians over subsets are reported against the median N, a power law SE = c N^-b is fitted on
the medians, and the test size at which the MDE equals the observed full-sample gap is extrapolated
(N* = N_full (MDE_full / |gap|)^2, i.e. assuming b = 0.5). The Hanley-McNeil approximation of the main text
(AUC 0.68, correlation 0.75, prevalence 0.41) is evaluated at the same N for the single-run SE.

Usage (from the repository root):  python code/local_experiments/exp24_resolution_curve.py
Output: results/local_exp24_resolution_curve.json
"""
import json
import os
import sys

import numpy as np
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from noise_utils import ROOT, load_test_partition  # noqa: E402

KS = [5, 8, 10, 13, 16, 19]
R = 300
SEED = 7
OUT = os.path.join(ROOT, 'results', 'local_exp24_resolution_curve.json')
PAIRS = [('RF', 'XGB'), ('RF', 'LSTM'), ('RF', 'LSTM_QNN'), ('CTRL_LINEAR4D', 'LSTM_QNN'), ('LSTM', 'LSTM_QNN')]


def components(y, P):
    """DeLong structural components for every seed: returns aucs (S,), v10 (S, m), v01 (S, n)."""
    pos = y == 1; m = int(pos.sum()); n = len(y) - m
    aucs, v10, v01 = [], [], []
    for p in P:
        tz = rankdata(p); tx = rankdata(p[pos]); ty = rankdata(p[~pos])
        v10.append((tz[pos] - tx) / n); v01.append(1.0 - (tz[~pos] - ty) / m)
        aucs.append(tz[pos].sum() / (m * n) - (m + 1) / (2 * n))
    return np.array(aucs), np.array(v10), np.array(v01)


def se_paired(a10, a01, b10, b01):
    m, n = len(a10), len(a01)
    S10 = np.cov(np.vstack([a10, b10])); S01 = np.cov(np.vstack([a01, b01]))
    v = (S10[0, 0] + S10[1, 1] - 2 * S10[0, 1]) / m + (S01[0, 0] + S01[1, 1] - 2 * S01[0, 1]) / n
    return float(np.sqrt(max(v, 1e-18)))


def auc_fast(y, p):
    r = rankdata(p); m = int(y.sum()); n = len(y) - m
    return (r[y == 1].sum() - m * (m + 1) / 2) / (m * n)


def hanley_mcneil_mde(N, auc=0.68, rho=0.75, prev=0.41):
    n1, n0 = prev * N, (1 - prev) * N
    q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
    se = np.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2) + (n0 - 1) * (q2 - auc ** 2)) / (n1 * n0))
    return 2.80 * se * np.sqrt(2 * (1 - rho))


def main():
    _, y, _, _, years, _ = load_test_partition()
    core = np.load(os.path.join(ROOT, 'results', 'json', 'results_hybrids_trainonly_fixet0_p60_28.1_probs.npz'))
    rr = np.load(os.path.join(ROOT, 'results', 'json', 'results_rerun_train_only_canonical_probs.npz'))
    P = {'RF': core['RF_2273_probs'], 'XGB': rr['train_only_XGB_probs'], 'LSTM': core['PureLSTM_probs'],
         'CTRL_LINEAR4D': core['CTRL_LINEAR4D_probs'], 'LSTM_QNN': core['LSTM_QNN_probs']}
    uy = np.unique(years)
    rng = np.random.default_rng(SEED)
    res = {'meta': dict(Ks=KS, R=R, seed=SEED, pairs=[f'{a}-{b}' for a, b in PAIRS]), 'by_K': {}}
    for K in KS:
        subsets = [uy] if K == len(uy) else [np.sort(rng.choice(uy, K, replace=False)) for _ in range(R)]
        rows = []
        for ys in subsets:
            idx = np.isin(years, ys); yy = y[idx]; yr = years[idx]
            if yy.sum() in (0, len(yy)):
                continue
            C = {k: components(yy, v[:, idx]) for k, v in P.items()}
            row = dict(N=int(idx.sum()), pos=int(yy.sum()))
            for a, b in PAIRS:
                (aa, a10, a01), (ab, b10, b01) = C[a], C[b]
                single = np.mean([se_paired(a10[s], a01[s], b10[s], b01[s]) for s in range(len(aa))])
                mean_se = se_paired(a10.mean(0), a01.mean(0), b10.mean(0), b01.mean(0))
                th = []
                for g in ys:
                    k = yr != g
                    th.append(np.mean([auc_fast(yy[k], p[idx][k]) for p in P[a]]) -
                              np.mean([auc_fast(yy[k], p[idx][k]) for p in P[b]]))
                th = np.array(th); G = len(th)
                jk = float(np.sqrt((G - 1) / G * np.sum((th - th.mean()) ** 2)))
                row[f'{a}-{b}'] = dict(gap=float(aa.mean() - ab.mean()), se_single=float(single),
                                       se_mean_delong=float(mean_se), se_mean_jk=jk)
            rows.append(row)
        Nmed = float(np.median([r['N'] for r in rows]))
        summ = dict(n_subsets=len(rows), N_median=Nmed, N_range=[int(min(r['N'] for r in rows)), int(max(r['N'] for r in rows))],
                    hanley_mcneil_mde_single=float(hanley_mcneil_mde(Nmed)), pairs={})
        for a, b in PAIRS:
            k = f'{a}-{b}'
            med = {f: float(np.median([r[k][f] for r in rows])) for f in ('se_single', 'se_mean_delong', 'se_mean_jk')}
            summ['pairs'][k] = dict(**med, mde_single=2.8 * med['se_single'], mde_mean_delong=2.8 * med['se_mean_delong'],
                                    mde_mean_jk=2.8 * med['se_mean_jk'],
                                    gap_median=float(np.median([r[k]['gap'] for r in rows])))
        res['by_K'][str(K)] = summ
        print(f"K={K:2d} N~{Nmed:5.0f}  HM single MDE {summ['hanley_mcneil_mde_single']:.3f} | " +
              ' | '.join(f"{k}: single {v['mde_single']:.3f} mean {v['mde_mean_delong']:.3f}/{v['mde_mean_jk']:.3f}"
                         for k, v in summ['pairs'].items() if k in ('RF-LSTM', 'RF-LSTM_QNN')), flush=True)
    # power-law fit on the medians and extrapolation from the full sample
    Ns = np.array([res['by_K'][str(K)]['N_median'] for K in KS])
    res['fit'] = {}
    full = res['by_K'][str(KS[-1])]
    for a, b in PAIRS:
        k = f'{a}-{b}'; f = {}
        for se in ('se_single', 'se_mean_delong', 'se_mean_jk'):
            s = np.array([res['by_K'][str(K)]['pairs'][k][se] for K in KS])
            slope, icpt = np.polyfit(np.log(Ns), np.log(s), 1)
            mde_full = 2.8 * full['pairs'][k][se]; gap = abs(full['pairs'][k]['gap_median'])
            f[se] = dict(exponent=float(-slope), N_for_mde_equal_gap=float(full['N_median'] * (mde_full / gap) ** 2) if gap > 0 else None)
        res['fit'][k] = dict(gap_full=float(full['pairs'][k]['gap_median']), **f)
        print(f"{k:22s} gap {full['pairs'][k]['gap_median']:+.4f} | exponent single {f['se_single']['exponent']:.2f} "
              f"mean {f['se_mean_delong']['exponent']:.2f}/{f['se_mean_jk']['exponent']:.2f} | N* (MDE=gap) mean-DeLong "
              f"{f['se_mean_delong']['N_for_mde_equal_gap']:.0f}, mean-jk {f['se_mean_jk']['N_for_mde_equal_gap']:.0f}")
    with open(OUT, 'w', encoding='utf-8') as fh:
        json.dump(res, fh, indent=2)
    print('saved', OUT)


if __name__ == '__main__':
    main()
