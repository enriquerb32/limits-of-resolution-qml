#!/usr/bin/env python3
"""
exp57_mde_ci_equivalence.py - uncertainty of the CR2 MDE and equivalence at the circuit interface (archived predictions only, no training)
===================================================================================================================
A. Uncertainty of the MDE. The MDE of the primary procedure, (t_{0.975,nu} + t_{0.80,nu}) SE_CR2, treats SE_CR2 as
   known. Two 95 % intervals for it: (i) the Satterthwaite approximation that defines the Bell-McCaffrey degrees of
   freedom, nu SE^2 / sigma^2 ~ chi^2_nu, so sigma lies in SE [sqrt(nu / chi2_{0.975,nu}), sqrt(nu / chi2_{0.025,nu})];
   (ii) a percentile bootstrap over the 19 calendar years (years resampled with replacement and relabelled, SE_CR2 and
   nu recomputed; B = 2,000). For each interval also the power of the design against the point MDE.
B. Equivalence (TOST) of the circuit against its interface controls with the primary procedure: two one-sided CR2 t
   tests at alpha = 0.05, i.e. the 90 % CR2 interval inside (-margin, +margin). The margin, 0.01 AUC, is the smallest
   difference of interest recorded before the external benchmark (audit/prediction_external.txt); it was not fixed
   for this panel before its analysis, so the test is post hoc. Also the smallest margin that the data would accept,
   max(|lo90|, |hi90|).

  python code/local_experiments/exp57_mde_ci_equivalence.py
Output: results/local_exp57_mde_ci_equivalence.json
"""
import json
import os
import sys

import numpy as np
from scipy.stats import chi2, t as student_t

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp55_review_closing import R, YEARS, archived_probs, mde_cr2  # noqa: E402
from qml_resolution import cluster_robust_delta  # noqa: E402

B, SEED, MARGIN = 2000, 0, 0.01
PRIMARY = [('RF', 'XGB'), ('RF', 'PureLSTM'), ('XGB', 'PureLSTM'),
           ('RF', 'LSTM_QNN'), ('CTRL_LINEAR4D', 'LSTM_QNN'), ('PureLSTM', 'LSTM_QNN')]
INTERFACE = [('CTRL_LINEAR4D', 'LSTM_QNN'), ('FOURIER2', 'LSTM_QNN'), ('FOURIER_FULL1', 'LSTM_QNN'),
             ('FOURIER_FULL2', 'LSTM_QNN'), ('QTL_FIXED', 'QTL_CLASSICAL'), ('QTL_FIXED', 'QTL_TENSOR1')]
# values printed in Tables 3, 6 and 7 of the paper (delta, CR2 MDE), reproduced as a check; QTL_FIXED is LSTM+QTL of the paper
PAPER = {'RF-XGB': (0.0100, 0.018), 'RF-PureLSTM': (0.0146, 0.020), 'XGB-PureLSTM': (0.0046, 0.024),
         'RF-LSTM_QNN': (0.0197, 0.025), 'CTRL_LINEAR4D-LSTM_QNN': (-0.0034, 0.019), 'PureLSTM-LSTM_QNN': (0.0051, 0.015),
         'FOURIER2-LSTM_QNN': (0.0041, 0.016), 'FOURIER_FULL1-LSTM_QNN': (0.0084, 0.017),
         'FOURIER_FULL2-LSTM_QNN': (0.0026, 0.017), 'QTL_FIXED-QTL_CLASSICAL': (-0.0027, 0.012),
         'QTL_FIXED-QTL_TENSOR1': (-0.0066, 0.012)}


def power(delta, se, nu, alpha=0.05):
    """Power of the two-sided CR2 t test against a true difference delta, with SE treated as known."""
    c = student_t.ppf(1 - alpha / 2, nu)
    return float(student_t.sf(c - delta / se, nu) + student_t.cdf(-c - delta / se, nu))


def boot_mde(y, a, b, rng):
    years = np.unique(YEARS); idx = {u: np.flatnonzero(YEARS == u) for u in years}
    out = []
    while len(out) < B:
        draw = rng.choice(years, size=years.size, replace=True)
        rows = np.concatenate([idx[u] for u in draw])
        lab = np.concatenate([np.full(idx[u].size, k) for k, u in enumerate(draw)])
        yy = y[rows]
        if yy.min() == yy.max():
            continue
        out.append(mde_cr2(cluster_robust_delta(yy, a[:, rows], b[:, rows], lab)))
    return np.array(out)


def part_a(y, P):
    rng = np.random.default_rng(SEED); res = {}
    for a, b in dict.fromkeys(PRIMARY + INTERFACE):
        r = cluster_robust_delta(y, P[a], P[b], YEARS); k = f'{a}-{b}'
        m = mde_cr2(r); nu = r.df
        assert abs(r.delta - PAPER[k][0]) < 5e-5 and abs(m - PAPER[k][1]) < 5e-4, f'{k}: does not reproduce the paper'
        f_lo, f_hi = np.sqrt(nu / chi2.ppf(0.975, nu)), np.sqrt(nu / chi2.ppf(0.025, nu))
        bs = boot_mde(y, P[a], P[b], rng); b_lo, b_hi = np.percentile(bs, [2.5, 97.5])
        res[k] = dict(delta=r.delta, se_cr2=r.se_cr2, df=nu, mde_cr2=m,
                      ci95_chi2=[m * f_lo, m * f_hi], ci95_year_boot=[float(b_lo), float(b_hi)],
                      power_at_mde_if_se_at_chi2_bounds=[power(m, r.se_cr2 * f_hi, nu), power(m, r.se_cr2 * f_lo, nu)])
        v = res[k]
        print(f'  A {k:26s} MDE {m:.3f}  chi2 [{v["ci95_chi2"][0]:.3f}, {v["ci95_chi2"][1]:.3f}]  '
              f'boot [{b_lo:.3f}, {b_hi:.3f}]  power at MDE [{v["power_at_mde_if_se_at_chi2_bounds"][0]:.2f}, '
              f'{v["power_at_mde_if_se_at_chi2_bounds"][1]:.2f}]')
    return res


def part_b(y, P):
    res = {}
    for a, b in INTERFACE:
        r = cluster_robust_delta(y, P[a], P[b], YEARS); lo, hi = r.ci(0.90)
        t_lo = (r.delta + MARGIN) / r.se_cr2; t_hi = (r.delta - MARGIN) / r.se_cr2
        p_tost = float(max(student_t.sf(t_lo, r.df), student_t.cdf(t_hi, r.df)))
        res[f'{a}-{b}'] = dict(delta=r.delta, ci90_cr2=[lo, hi], p_tost=p_tost, margin=MARGIN,
                               equivalent=bool(lo > -MARGIN and hi < MARGIN), smallest_margin=float(max(-lo, hi)))
        v = res[f'{a}-{b}']
        print(f'  B {a}-{b:14s} {r.delta:+.4f}  CR2 90% [{lo:+.4f}, {hi:+.4f}]  TOST p {p_tost:.3f}  '
              f'smallest margin {v["smallest_margin"]:.4f}')
    return res


def main():
    y, P = archived_probs()
    res = dict(B=B, seed=SEED, margin=MARGIN, mde_ci=part_a(y, P), tost=part_b(y, P))
    path = os.path.join(R, 'local_exp57_mde_ci_equivalence.json')
    json.dump(res, open(path, 'w'), indent=2)
    print('wrote', path)


if __name__ == '__main__':
    main()
