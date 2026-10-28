#!/usr/bin/env python3
"""
exp22_same_arch_core.py - Same-architecture seed pairs for the train-only core models (no retraining)
=====================================================================================================
Repeats the same-architecture check of exp7_noise_budget.py (section F) on the 15-seed train-only core archive:
for every model, the 105 pairs of seeds (i < j) are compared with the paired DeLong test (p < 0.05) and with the
calendar-year block bootstrap (95 % interval excludes zero; B = 2,000, seed 42). Any significant result is a false
claim, since both runs are the same procedure.

Usage (from the repository root):  python code/local_experiments/exp22_same_arch_core.py
Output: results/local_exp22_same_arch_core.json
"""
import itertools
import json
import os
import sys

import numpy as np
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from noise_utils import ROOT, load_test_partition, delong_cov, year_block_bootstrap_aucs  # noqa: E402

B = 2000
OUT = os.path.join(ROOT, 'results', 'local_exp22_same_arch_core.json')
MODELS = ['RF_2273', 'PureLSTM', 'CTRL_LINEAR4D', 'LSTM_QNN', 'LBL_QNN', 'LSTM_PIQNN', 'LSTM_QTL']


def main():
    _, y, _, _, years, _ = load_test_partition()
    z = np.load(os.path.join(ROOT, 'results', 'json', 'results_hybrids_trainonly_fixet0_p60_28.1_probs.npz'))
    assert np.array_equal(z['y_te'].astype(int), y)
    out = {'meta': dict(B=B, boot_seed=42, n_pairs_per_model=105), 'models': {}}
    for k in MODELS:
        P = z[f'{k}_probs'].astype(float)
        aucs, cov = delong_cov(y, P)
        boot = year_block_bootstrap_aucs(y, P, years, B=B, seed=42)
        si, sb = [], []
        for i, j in itertools.combinations(range(len(P)), 2):
            se = np.sqrt(max(cov[i, i] + cov[j, j] - 2 * cov[i, j], 1e-300))
            si.append(2 * norm.sf(abs(aucs[i] - aucs[j]) / se) < 0.05)
            lo, hi = np.percentile(boot[:, i] - boot[:, j], [2.5, 97.5])
            sb.append(lo > 0 or hi < 0)
        out['models'][k] = dict(false_claim_rate_iid=float(np.mean(si)), false_claim_rate_block=float(np.mean(sb)),
                                seed_sd=float(np.std(aucs, ddof=1)), n_pairs=len(si))
        print(f"{k:14s} seed SD {np.std(aucs, ddof=1):.4f} | same-arch significant: DeLong {100*np.mean(si):5.1f}% "
              f"block {100*np.mean(sb):5.1f}%", flush=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print('saved', OUT)


if __name__ == '__main__':
    main()
