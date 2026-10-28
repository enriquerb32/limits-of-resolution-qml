#!/usr/bin/env python3
"""
exp21_brier_trainonly.py - Brier score, skill against climatology and ECE of the train-only 15-seed models
==========================================================================================================
No retraining. Uses the archived test probabilities of the train-only pipeline:
  results/json/results_rerun_train_only_canonical_probs.npz   (XGBoost; RF and LSTM are identical to the core fits)
  results/json/results_hybrids_trainonly_fixet0_p60_28.1_probs.npz   (RF, LSTM, Linear-4D, the four hybrids)
For every model and seed: AUC, Brier score, Brier skill score against the month x station climatology fitted on the
training partition (BSS_clim = 1 - BS / BS_clim), and 10-bin ECE (same definition as exp3). Reports the 15-seed mean
and SD, the number of seeds with BS below the climatology, and the Brier score of the seed-averaged probabilities.
Also: paired year-block bootstrap (B=2,000, 19 years) of the 15-seed mean Brier difference model - climatology.

Usage (from the repository root):  python code/local_experiments/exp21_brier_trainonly.py
Output: results/local_exp21_brier_trainonly.json
"""
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from noise_utils import ROOT, load_test_partition  # noqa: E402
from exp7_noise_budget import climatology_probs  # noqa: E402
from exp3_brier_and_reliability import compute_calibration_metrics  # noqa: E402

B, SEED = 2000, 42
OUT = os.path.join(ROOT, 'results', 'local_exp21_brier_trainonly.json')


def main():
    ds, y, m_te, X_te, years, _ = load_test_partition()
    core = np.load(os.path.join(ROOT, 'results', 'json', 'results_hybrids_trainonly_fixet0_p60_28.1_probs.npz'))
    rr = np.load(os.path.join(ROOT, 'results', 'json', 'results_rerun_train_only_canonical_probs.npz'))
    assert np.array_equal(core['y_te'].astype(int), y) and np.array_equal(rr['y_te'].astype(int), y)
    P = {'Random Forest': core['RF_2273_probs'], 'XGBoost': rr['train_only_XGB_probs'],
         'Standalone LSTM': core['PureLSTM_probs'], 'Linear-4D control': core['CTRL_LINEAR4D_probs'],
         'LSTM+QNN': core['LSTM_QNN_probs'], 'LbL-QNN': core['LBL_QNN_probs'],
         'LSTM+PI-QNN': core['LSTM_PIQNN_probs'], 'LSTM+QTL': core['LSTM_QTL_probs']}
    p_clim = climatology_probs(ds, y, m_te, X_te)
    bs_clim = float(np.mean((p_clim - y) ** 2))
    p_base = float(np.asarray(ds['y_tr']).mean())
    out = {'meta': dict(n_test=int(len(y)), B=B, boot_seed=SEED, note='no retraining; archived train-only probabilities'),
           'climatology': dict(auc=float(roc_auc_score(y, p_clim)), brier=bs_clim,
                               ece_10=compute_calibration_metrics(y, p_clim)['ece_10'],
                               bss_vs_base_rate=1 - bs_clim / float(np.mean((p_base - y) ** 2))),
           'models': {}}
    uy = np.unique(years); idx_by = [np.where(years == u)[0] for u in uy]
    rng = np.random.default_rng(SEED)
    boots = [np.concatenate([idx_by[k] for k in rng.integers(0, len(uy), len(uy))]) for _ in range(B)]
    print(f"climatology AUC {out['climatology']['auc']:.4f} BS {bs_clim:.4f} ECE {out['climatology']['ece_10']:.3f}")
    for name, Pm in P.items():
        bs = np.mean((Pm - y) ** 2, axis=1)
        auc = np.array([roc_auc_score(y, p) for p in Pm])
        ece = np.array([compute_calibration_metrics(y, p)['ece_10'] for p in Pm])
        bss = 1 - bs / bs_clim
        # paired year-block bootstrap of mean_s BS_s - BS_clim
        d = np.array([np.mean((Pm[:, i] - y[i]) ** 2) - np.mean((p_clim[i] - y[i]) ** 2) for i in boots])
        lo, hi = np.percentile(d, [2.5, 97.5])
        r = dict(auc_mean=float(auc.mean()), brier_mean=float(bs.mean()), brier_sd=float(bs.std(ddof=1)),
                 bss_clim_mean=float(bss.mean()), bss_clim_min=float(bss.min()), bss_clim_max=float(bss.max()),
                 seeds_better_than_clim=int((bs < bs_clim).sum()),
                 ece_mean=float(ece.mean()), ece_sd=float(ece.std(ddof=1)),
                 brier_of_seed_average=float(np.mean((Pm.mean(axis=0) - y) ** 2)),
                 brier_minus_clim=float(bs.mean() - bs_clim), brier_minus_clim_block_ci=[float(lo), float(hi)])
        out['models'][name] = r
        print(f"{name:18s} AUC {r['auc_mean']:.4f} BS {r['brier_mean']:.4f} ({r['brier_sd']:.4f}) "
              f"BSS {r['bss_clim_mean']:+.3f} [{r['bss_clim_min']:+.3f},{r['bss_clim_max']:+.3f}] "
              f"seeds<clim {r['seeds_better_than_clim']:2d} ECE {r['ece_mean']:.3f} BS(avg) {r['brier_of_seed_average']:.4f} "
              f"dBS CI [{lo:+.4f},{hi:+.4f}]")
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print('saved', OUT)


if __name__ == '__main__':
    main()
