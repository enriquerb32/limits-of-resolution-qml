#!/usr/bin/env python3
"""
exp35_equivalence_fewclusters.py - equivalence bounds (TOST) and few-cluster inference for the paired comparisons
=================================================================================================================
Two analyses without new training:
  1. Equivalence. For each pair, the two one-sided tests (TOST) at alpha = 0.05 declare equivalence within +-m when the
     90 % interval lies inside (-m, +m). Because no margin was fixed in advance, we report (a) the smallest margin m*
     for which each procedure (DeLong, cluster-robust t_18, wild cluster bootstrap) would declare equivalence, and
     (b) the TOST verdicts at two reference margins, 0.02 AUC (about the Random Forest - hybrid gap) and the
     DeLong MDE of the pair. These are post hoc and labelled as such.
  2. Few clusters (19 years). Besides the CR1 cluster-robust SE of exp25 (K/(K-1) sum S_u^2), the CR2 (Bell-McCaffrey)
     correction S_u / sqrt(1 - n_u/N) with t_{K-1} reference, and a studentized wild cluster bootstrap with Webb's
     6-point weights (B = 9,999), which is the recommended procedure for few clusters.
All statistics are computed from the paired DeLong influence values psi of the seed-averaged components (exp25).

Usage: python code/local_experiments/exp35_equivalence_fewclusters.py [--npz ...] [--pairs A-B ...]
Output: results/local_exp35_equivalence_fewclusters.json
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import norm, t as student_t

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from qml_resolution.auc import delong_components  # noqa: E402

NPZ = os.path.join(ROOT, 'results', 'json', 'results_core_hybrids_controls_fixet0_p60_28.1_probs.npz')
NPZ_FALLBACK = os.path.join(ROOT, 'results', 'json', 'results_hybrids_trainonly_fixet0_p60_28.1_probs.npz')
YEARS = os.path.join(ROOT, 'results', 'years_te.npy')
OUT = os.path.join(ROOT, 'results', 'local_exp35_equivalence_fewclusters.json')
PAIRS = ['RF_2273-PureLSTM', 'RF_2273-LSTM_QNN', 'PureLSTM-LSTM_QNN', 'CTRL_LINEAR4D-LSTM_QNN',
         'FOURIER1-LSTM_QNN', 'FOURIER2-LSTM_QNN', 'FOURIER_FULL2-LSTM_QNN', 'FOURIER_FULL1-LSTM_QNN',
         'FOURIER3-LSTM_QNN', 'FOURIER4-LSTM_QNN', 'QTL_CLASSICAL-QTL_FIXED']
WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])
B, SEED, MDE_Z = 9999, 2025, 1.959963984540054 + 0.8416212335729143


def influence(y, pa, pb):
    ca, cb = delong_components(y, pa).seed_mean(), delong_components(y, pb).seed_mean()
    delta = float(ca.auc[0] - cb.auc[0])
    psi = np.zeros(y.size)
    psi[y] = (ca.v10[0] - cb.v10[0] - delta) / y.sum()
    psi[~y] = (ca.v01[0] - cb.v01[0] - delta) / (~y).sum()
    return delta, psi


def smallest_margin(delta, se, q):
    """Smallest m with the (1 - 2 alpha) interval delta +- q*se inside (-m, m)."""
    return abs(delta) + q * se


def analyse(y, pa, pb, years, rng):
    delta, psi = influence(y, pa, pb)
    lv, inv = np.unique(years, return_inverse=True)
    K, N = lv.size, y.size
    S = np.bincount(inv, weights=psi)
    n_u = np.bincount(inv)
    se_iid = float(np.sqrt((psi ** 2).sum()))
    se_cr1 = float(np.sqrt(K / (K - 1) * (S ** 2).sum()))
    se_cr2 = float(np.sqrt(((S / np.sqrt(1 - n_u / N)) ** 2).sum()))
    t_cr1 = delta / se_cr1
    # studentized wild cluster bootstrap (contributions are centred: sum S = 0)
    W = WEBB[rng.integers(0, 6, size=(B, K))]
    num = (W * S).sum(1)
    den = np.sqrt(K / (K - 1) * ((W * S - (W * S).mean(1, keepdims=True)) ** 2).sum(1))
    t_star = num / den
    p_wild = float((np.abs(t_star) >= abs(t_cr1)).mean())
    q95_wild = float(np.quantile(np.abs(t_star), 0.95)); q90_wild = float(np.quantile(np.abs(t_star), 0.90))
    tq95, tq90 = student_t.ppf(0.975, K - 1), student_t.ppf(0.95, K - 1)
    mde = MDE_Z * se_iid
    out = dict(delta=delta, K=int(K), se_iid=se_iid, se_cr1=se_cr1, se_cr2=se_cr2,
               p_delong=float(2 * norm.sf(abs(delta) / se_iid)),
               p_cr1_t=float(2 * student_t.sf(abs(t_cr1), K - 1)),
               p_cr2_t=float(2 * student_t.sf(abs(delta / se_cr2), K - 1)),
               p_wild_cluster=p_wild,
               ci95_cr2_t=[delta - tq95 * se_cr2, delta + tq95 * se_cr2],
               ci95_wild=[delta - q95_wild * se_cr1, delta + q95_wild * se_cr1],
               mde_delong=float(mde),
               equivalence_margin_min=dict(delong=float(smallest_margin(delta, se_iid, norm.ppf(0.95))),
                                           cr2_t=float(smallest_margin(delta, se_cr2, tq90)),
                                           wild=float(smallest_margin(delta, se_cr1, q90_wild))))
    for name, m in [('0.02', 0.02), ('mde', mde)]:
        out[f'tost_equivalent_at_{name}'] = {k: bool(v < m) for k, v in out['equivalence_margin_min'].items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default=NPZ)
    ap.add_argument('--pairs', nargs='+', default=PAIRS)
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()
    z = dict(np.load(a.npz if os.path.exists(a.npz) else NPZ_FALLBACK))
    y = z['y_te'].astype(bool)
    years = np.load(YEARS)
    rng = np.random.default_rng(SEED)
    res = {'n': int(y.size), 'B_wild': B, 'pairs': {}}
    for pair in a.pairs:
        pa, pb = pair.split('-', 1)
        if f'{pa}_probs' not in z or f'{pb}_probs' not in z:
            print(f'{pair:28s} skipped (not in the archive)')
            continue
        r = analyse(y, np.atleast_2d(z[f'{pa}_probs']), np.atleast_2d(z[f'{pb}_probs']), years, rng)
        res['pairs'][pair] = r
        em = r['equivalence_margin_min']
        print(f'{pair:28s} d={r["delta"]:+.4f} | SE iid {r["se_iid"]:.4f} CR1 {r["se_cr1"]:.4f} CR2 {r["se_cr2"]:.4f} | '
              f'p DeLong {r["p_delong"]:.3f} CR2-t {r["p_cr2_t"]:.3f} wild {r["p_wild_cluster"]:.3f} | '
              f'equivalence margin m* DeLong {em["delong"]:.3f} CR2 {em["cr2_t"]:.3f} wild {em["wild"]:.3f}', flush=True)
    json.dump(res, open(a.out, 'w'), indent=2)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
