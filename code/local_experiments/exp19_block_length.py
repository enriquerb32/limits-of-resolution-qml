#!/usr/bin/env python3
"""
exp19_block_length.py - Sensitivity of the year-block bootstrap to the block length (no retraining)
===================================================================================================
The paper resamples single calendar years (block length L = 1). This script repeats the paired seed-mean
comparisons with a circular moving-block bootstrap over the 19 test years, blocks of L = 1, 2, 3 and 4 consecutive
years: in each replicate ceil(19/L) block starts are drawn uniformly from the 19 years, each block contributes the
L years that follow its start (wrapping 2026 -> 2008), the sequence is truncated to 19 years, and all stations of a
selected year enter together. L = 1 is the scheme of the paper (same estimator, different random stream).

Pairs: the three classical seed-mean comparisons (trees on 2,787 rows) and the three hybrid-case comparisons
(core experiment, 2,273 rows); plus the 15-seed RF and the training-only month x station climatology.
If exp17_hybrids_multiseed.py has produced results/json/results_core_plus_hybrids_probs.npz, the extra hybrids
are added against RF_2273 and CTRL_LINEAR4D.

Usage (from the repository root)
  python code/local_experiments/exp19_block_length.py            # about 1-3 min
Output: results/local_exp19_block_length.json
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from noise_utils import ROOT, load_test_partition  # noqa: E402
from exp10_paired_ci import auc_fast  # noqa: E402
from exp7_noise_budget import climatology_probs  # noqa: E402

B = 2000
SEED = 42
LENGTHS = [1, 2, 3, 4]
OUT = os.path.join(ROOT, 'results', 'local_exp19_block_length.json')


def mbb_diffs(PA, PB, y, years, L, rng):
    uy = np.unique(years)                      # sorted test years
    K = len(uy)
    idx_by = [np.where(years == u)[0] for u in uy]
    n_blocks = math.ceil(K / L)
    out = []
    for _ in range(B):
        starts = rng.integers(0, K, n_blocks)
        sel = np.concatenate([(s + np.arange(L)) % K for s in starts])[:K]
        idx = np.concatenate([idx_by[k] for k in sel])
        yy = y[idx]
        if yy.sum() in (0, len(yy)):
            continue
        out.append(np.mean([auc_fast(yy, p[idx]) for p in PA]) - np.mean([auc_fast(yy, p[idx]) for p in PB]))
    return np.asarray(out)


def main():
    ds, y, m_te, X_te, years, _ = load_test_partition()
    rr = np.load(os.path.join(ROOT, 'results', 'json', 'results_rerun_train_only_canonical_probs.npz'))
    core = np.load(os.path.join(ROOT, 'results', 'json', 'results_qml_core_trainonly_probs.npz'))
    assert np.array_equal(rr['y_te'].astype(int), y) and np.array_equal(core['y_te'].astype(int), y)
    P = {k: rr[f'train_only_{k}_probs'].astype(float) for k in ['RF', 'XGB', 'LSTM']}
    P.update({k: core[f'{k}_probs'].astype(float) for k in ['RF_2273', 'PureLSTM', 'CTRL_LINEAR4D', 'LSTM_QNN']})
    P['CLIM_TRAIN'] = climatology_probs(ds, y, m_te, X_te)[None, :]
    pairs = [('RF', 'XGB'), ('RF', 'LSTM'), ('XGB', 'LSTM'), ('RF_2273', 'LSTM_QNN'),
             ('CTRL_LINEAR4D', 'LSTM_QNN'), ('PureLSTM', 'LSTM_QNN'), ('RF', 'CLIM_TRAIN')]

    merged = os.path.join(ROOT, 'results', 'json', 'results_core_plus_hybrids_probs.npz')
    if os.path.exists(merged):
        z = np.load(merged)
        assert np.array_equal(z['y_te'].astype(int), y)
        for k in z.files:
            name = k[:-6]
            if k.endswith('_probs') and name not in P:
                P[name] = z[k].astype(float)
                if name != 'QTL_EXTRACTOR':
                    pairs += [('RF_2273', name), ('CTRL_LINEAR4D', name)]
        if 'QTL_EXTRACTOR' in P and 'LSTM_QTL' in P:
            pairs.append(('QTL_EXTRACTOR', 'LSTM_QTL'))

    res = {}
    print(f"{'pair':<28}{'Delta':>8} | " + ' | '.join(f'L={L}: SE   [95% CI]        ' for L in LENGTHS))
    for a, b in pairs:
        d = float(np.mean([auc_fast(y, p) for p in P[a]]) - np.mean([auc_fast(y, p) for p in P[b]]))
        row = dict(delta=d)
        line = f'{a + "-" + b:<28}{d:+8.4f} | '
        for L in LENGTHS:
            rng = np.random.default_rng(SEED + L)
            diffs = mbb_diffs(P[a], P[b], y, years, L, rng)
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            row[f'L{L}'] = dict(se=float(diffs.std(ddof=1)), ci95=[float(lo), float(hi)],
                                excludes_zero=bool(lo > 0 or hi < 0), B_valid=int(len(diffs)))
            line += f'{row[f"L{L}"]["se"]:.4f} [{lo:+.3f},{hi:+.3f}] | '
        res[f'{a}-{b}'] = row
        print(line, flush=True)

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(dict(meta=dict(B=B, seed=SEED, lengths=LENGTHS, n_years=int(len(np.unique(years))),
                                 scheme='circular moving-block bootstrap over calendar years'), pairs=res), f, indent=2)
    print('saved', OUT)


if __name__ == '__main__':
    main()
