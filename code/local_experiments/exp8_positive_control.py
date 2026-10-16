"""
code/local_experiments/exp8_positive_control.py
Local Experiment 8: Positive control for the noise budget.

We take the train-only Random Forest test probabilities (15 seeds) and degrade them by a KNOWN
amount of AUC, delta in {0, 0.01, ..., 0.05}, by adding Gaussian noise on the logit scale with the
noise scale calibrated (bisection) so that AUC(degraded) = AUC(original) - delta exactly on the
528-row test set. We then ask how often the benchmark detects the known difference.

Two comparison designs:
  * paired   : RF seed j vs degraded RF seed j      -> sampling noise only (best case).
  * cross    : RF seed i vs degraded RF seed j, i!=j -> sampling + seed noise (realistic
               single-run comparison between two training runs).
Two tests:
  * iid   : DeLong paired test, p < 0.05 with the correct sign.
  * block : calendar-year block bootstrap (B = 1,000), 95% percentile CI excludes 0, correct sign.

Output: results/local_exp8_positive_control.json
"""

import json
import os
import numpy as np
from scipy.special import logit, expit
from scipy.stats import norm

from noise_utils import (Z975, load_test_partition, delong_cov, auc_matrix,
                         year_block_bootstrap_aucs)

DELTAS = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
R = 20          # noise draws per seed and delta
B = 1000        # year-block bootstrap resamples
N_SEEDS = 15


def degrade(y, p, delta, rng, tol=2e-4):
    """Return p' with AUC(p') = AUC(p) - delta (within tol), via logit-noise bisection."""
    base = auc_matrix(y, p[None, :])[0]
    if delta == 0:
        return p.copy(), 0.0
    target = base - delta
    lp = logit(np.clip(p, 1e-6, 1 - 1e-6))
    eps = rng.standard_normal(len(p))
    lo, hi = 0.0, 0.5
    while auc_matrix(y, expit(lp + hi * eps)[None, :])[0] > target and hi < 64:
        hi *= 2
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        a = auc_matrix(y, expit(lp + mid * eps)[None, :])[0]
        if abs(a - target) < tol:
            break
        if a > target:
            lo = mid
        else:
            hi = mid
    return expit(lp + mid * eps), mid


def main():
    os.makedirs('results', exist_ok=True)
    _, y, _, _, years, _ = load_test_partition()
    d = np.load('results/json/results_rerun_train_only_canonical_probs.npz')
    rf = d['train_only_RF_probs'].astype(float)
    rng = np.random.default_rng(2026)

    results = {'meta': {'deltas': DELTAS, 'draws_per_seed': R, 'B_bootstrap': B,
                        'n_seeds': N_SEEDS, 'base_model': 'Random Forest (train-only rerun)',
                        'degradation': 'Gaussian noise on logit scale, scale calibrated by bisection'},
               'by_delta': {}}

    for delta in DELTAS:
        deg, sig_noise, realised = [], [], []
        for j in range(N_SEEDS):
            for r in range(R if delta > 0 else 1):
                pd_, s = degrade(y, rf[j], delta, rng)
                deg.append((j, pd_))
                sig_noise.append(s)
        stack = np.vstack([rf] + [p for _, p in deg])  # rows 0..14 originals
        aucs, cov = delong_cov(y, stack)
        boot = year_block_bootstrap_aucs(y, stack, years, B=B, seed=7)

        def test(i, k):
            dd = aucs[i] - aucs[k]                       # original minus degraded (true sign > 0)
            se = np.sqrt(max(cov[i, i] + cov[k, k] - 2 * cov[i, k], 1e-300))
            p = 2 * norm.sf(abs(dd) / se)
            diffs = boot[:, i] - boot[:, k]
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            return dd, se, (p < 0.05 and dd > 0), (p < 0.05 and dd < 0), (lo > 0), (hi < 0)

        out = {}
        for design in ('paired', 'cross'):
            rows = []
            for kk, (j, _) in enumerate(deg):
                k = N_SEEDS + kk
                partners = [j] if design == 'paired' else [i for i in range(N_SEEDS) if i != j]
                for i in partners:
                    rows.append(test(i, k))
            rows = np.array(rows, dtype=float)
            out[design] = {
                'n_tests': int(len(rows)),
                'realised_delta_mean': float(rows[:, 0].mean()),
                'realised_delta_sd': float(rows[:, 0].std(ddof=1)) if len(rows) > 1 else 0.0,
                'se_diff_iid_mean': float(rows[:, 1].mean()),
                'detect_iid': float(rows[:, 2].mean()),
                'wrong_sign_iid': float(rows[:, 3].mean()),
                'detect_block': float(rows[:, 4].mean()),
                'wrong_sign_block': float(rows[:, 5].mean()),
            }
        se_p = out['paired']['se_diff_iid_mean']
        out['analytic_power_paired_iid'] = float(norm.cdf(delta / se_p - Z975) + norm.cdf(-delta / se_p - Z975))
        out['mean_logit_noise_scale'] = float(np.mean(sig_noise))
        results['by_delta'][f'{delta:.2f}'] = out
        print("delta=%.2f | paired: iid %5.1f%% block %5.1f%% (SEd %.4f, analytic %5.1f%%) | "
              "cross: iid %5.1f%% block %5.1f%% wrong-sign iid %4.1f%% block %4.1f%%" % (
                  delta, 100 * out['paired']['detect_iid'], 100 * out['paired']['detect_block'], se_p,
                  100 * out['analytic_power_paired_iid'], 100 * out['cross']['detect_iid'],
                  100 * out['cross']['detect_block'], 100 * out['cross']['wrong_sign_iid'],
                  100 * out['cross']['wrong_sign_block']))

    # delta at which detection first reaches 80% (linear interpolation)
    def d80(key, design):
        xs = DELTAS
        ys = [results['by_delta'][f'{x:.2f}'][design][key] for x in xs]
        for a, b, ya, yb in zip(xs[:-1], xs[1:], ys[:-1], ys[1:]):
            if ya < 0.8 <= yb:
                return float(a + (0.8 - ya) * (b - a) / (yb - ya))
        return None if ys[-1] < 0.8 else float(xs[0])

    results['delta_80pct_detection'] = {f'{des}_{k}': d80(k, des)
                                        for des in ('paired', 'cross') for k in ('detect_iid', 'detect_block')}
    print("delta for 80% detection:", results['delta_80pct_detection'])

    with open('results/local_exp8_positive_control.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print("Saved results/local_exp8_positive_control.json")


if __name__ == '__main__':
    main()
