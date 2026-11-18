#!/usr/bin/env python3
"""
exp51_review_extras.py - archived-prediction analyses of the revision of 2026-11-18 (no training)
================================================================================================
1. Causal refit of the neural models (S1, lookahead audit): PureLSTM and LSTM+QNN retrained with V2_CAUSAL=1
   (run_v2.ps1 step 03b_core_causal, results/causal/) against the archived runs, on the 529 common test windows,
   with a calendar-year block bootstrap of the difference. Added to results/local_exp47_truncation_audit.json.
2. Primary procedure (CR2, Bell-McCaffrey df, calendar-year clusters) for the SEAS5 contrasts of Section 4.7.
   -> results/local_seas5_cr2.json
3. Sensitivity of the primary procedure to clusters of 1, 2 and 3 consecutive years for the pairs whose verdict
   depends on the procedure. -> results/local_cr2_block_sensitivity.json

  python code/local_experiments/exp51_review_extras.py      (after exp47 and run_v2.ps1 step 03b_core_causal)
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from qml_resolution import cluster_bootstrap_delta, cluster_robust_delta, percentile_ci, rank_auc  # noqa: E402

R, J = os.path.join(ROOT, 'results'), os.path.join(ROOT, 'results', 'json')
META = pd.read_csv(os.path.join(R, 'test_meta.csv'))
YEARS = pd.to_datetime(META.fecha).dt.year.values


def causal_neural():
    arc = np.load(os.path.join(J, 'results_hybrids_trainonly_fixet0_p60_28.1_probs.npz'))
    cau = np.load(os.path.join(R, 'causal', 'results_hybrids_trainonly_fixet0_probs.npz'))
    order = np.load(os.path.join(J, 'results_exp47_causal_trees_probs.npz'))      # causal test order (exp47)
    assert np.array_equal(cau['y_te'].astype(int), order['y_te'].astype(int)), 'causal test order differs'
    pos = {k: i for i, k in enumerate(zip(order['ciudad'], order['fecha']))}
    ib = [pos[k] for k in zip(META.ciudad, META.fecha.str[:10])]
    y = arc['y_te'].astype(int)
    assert np.array_equal(y, cau['y_te'].astype(int)[ib])
    out = {}
    for m in ('PureLSTM', 'LSTM_QNN'):
        a, b = arc[f'{m}_probs'], cau[f'{m}_probs'][:, ib]
        reps = cluster_bootstrap_delta(y, a, b, YEARS, n_boot=2000, random_state=0)
        out[m] = dict(auc_archived_common=float(rank_auc(y, a).mean()), auc_causal_common=float(rank_auc(y, b).mean()),
                      auc_causal_all=float(rank_auc(cau['y_te'].astype(int), cau[f'{m}_probs']).mean()),
                      delta_archived_minus_causal=float(rank_auc(y, a).mean() - rank_auc(y, b).mean()),
                      ci95_year_block=list(percentile_ci(reps)))
    return out


def seas5():
    out = {}
    for lab in ('orig', 'stationmonth'):
        z = np.load(os.path.join(J, f'results_exp33_seas5_{lab}_probs.npz')); y = z['y_te'].astype(int)
        for a, b in [('SEAS5_QM_L1', 'CLIM'), ('RF_2273', 'SEAS5_QM_L1'), ('LSTM_QNN', 'SEAS5_QM_L1'), ('SEAS5_QM_L2', 'CLIM')]:
            r = cluster_robust_delta(y, z[f'{a}_probs'], z[f'{b}_probs'], YEARS); lo, hi = r.ci()
            out[f'{lab}:{a}-{b}'] = dict(delta=r.delta, ci_cr2=[lo, hi], p_cr2=r.p_value, df=r.df)
    return out


def blocks():
    h = np.load(os.path.join(J, 'results_core_hybrids_controls_fixet0_p60_28.1_probs.npz'))
    c = np.load(os.path.join(J, 'results_rerun_train_only_canonical_probs.npz'))
    P = {k[:-6]: h[k] for k in h.files if k.endswith('_probs')}
    P.update(RF=c['train_only_RF_probs'], XGB=c['train_only_XGB_probs'], LSTM_QTL=h['QTL_FIXED_probs'])  # as exp42
    y = h['y_te'].astype(int); out = {}
    for a, b in [('RF', 'XGB'), ('RF', 'PureLSTM'), ('RF', 'LSTM_QNN'), ('RF', 'LSTM_QTL'),
                 ('FROZEN_RANDOM', 'LSTM_QNN'), ('FROZEN_IDENTITY', 'LSTM_QNN')]:
        row = {}
        for blk in (1, 2, 3):
            r = cluster_robust_delta(y, P[a], P[b], (YEARS - YEARS.min()) // blk); lo, hi = r.ci()
            row[f'{blk}yr'] = dict(ci=[lo, hi], p=r.p_value, df=r.df)
        out[f'{a}-{b}'] = row
    return out


def influential_years():
    """Leave one test year out: difference and CR2 p-value of the pairs whose verdict matters."""
    h = np.load(os.path.join(J, 'results_core_hybrids_controls_fixet0_p60_28.1_probs.npz'))
    c = np.load(os.path.join(J, 'results_rerun_train_only_canonical_probs.npz'))
    P = {k[:-6]: h[k] for k in h.files if k.endswith('_probs')}
    P.update(RF=c['train_only_RF_probs'], XGB=c['train_only_XGB_probs'], LSTM_QTL=h['QTL_FIXED_probs'])
    y = h['y_te'].astype(int); out = {}
    for a, b in [('RF', 'XGB'), ('RF', 'PureLSTM'), ('RF', 'LSTM_QNN'), ('RF', 'LSTM_QTL'),
                 ('FROZEN_RANDOM', 'LSTM_QNN'), ('FROZEN_IDENTITY', 'LSTM_QNN'), ('LSTM_QNN', 'CTRL_LINEAR4D')]:
        full = cluster_robust_delta(y, P[a], P[b], YEARS)
        rows = {}
        for u in np.unique(YEARS):
            keep = YEARS != u
            r = cluster_robust_delta(y[keep], P[a][:, keep], P[b][:, keep], YEARS[keep]); lo, hi = r.ci()
            rows[int(u)] = dict(delta=r.delta, p=r.p_value, resolved=bool(lo > 0 or hi < 0))
        lo, hi = full.ci(); res_full = bool(lo > 0 or hi < 0)
        flips = [u for u, v in rows.items() if v['resolved'] != res_full]
        out[f'{a}-{b}'] = dict(full=dict(delta=full.delta, p=full.p_value, resolved=res_full),
                               delta_range=[min(v['delta'] for v in rows.values()), max(v['delta'] for v in rows.values())],
                               p_range=[min(v['p'] for v in rows.values()), max(v['p'] for v in rows.values())],
                               years_that_flip_verdict=flips, by_year=rows)
    return out


def serial_dependence():
    """Lag-1 autocorrelation of the yearly influence sums S_u of the paired difference, and a Ljung-Box test (lags 1-3):
    is the panel in the serially correlated regime under which no calendar-year cluster procedure is calibrated?"""
    from scipy.stats import chi2
    from qml_resolution import delong_components
    h = np.load(os.path.join(J, 'results_core_hybrids_controls_fixet0_p60_28.1_probs.npz'))
    c = np.load(os.path.join(J, 'results_rerun_train_only_canonical_probs.npz'))
    P = {k[:-6]: h[k] for k in h.files if k.endswith('_probs')}
    P.update(RF=c['train_only_RF_probs'], XGB=c['train_only_XGB_probs'], LSTM_QTL=h['QTL_FIXED_probs'])
    y = h['y_te'].astype(bool); years = np.unique(YEARS); out = {}
    for a, b in [('RF', 'XGB'), ('RF', 'PureLSTM'), ('RF', 'LSTM_QNN'), ('RF', 'LSTM_QTL'),
                 ('FROZEN_RANDOM', 'LSTM_QNN'), ('FROZEN_IDENTITY', 'LSTM_QNN'), ('LSTM_QNN', 'CTRL_LINEAR4D')]:
        ca, cb = delong_components(y, P[a]).seed_mean(), delong_components(y, P[b]).seed_mean()
        d = float(ca.auc[0] - cb.auc[0]); psi = np.empty(y.size)
        psi[y] = (ca.v10[0] - cb.v10[0] - d) / y.sum(); psi[~y] = (ca.v01[0] - cb.v01[0] - d) / (~y).sum()
        S = np.array([psi[YEARS == u].sum() for u in years]); S = S - S.mean(); n = S.size
        acf = [float((S[k:] * S[:-k]).sum() / (S ** 2).sum()) for k in (1, 2, 3)]
        q = float(n * (n + 2) * sum(r ** 2 / (n - k) for k, r in zip((1, 2, 3), acf)))
        out[f'{a}-{b}'] = dict(acf=acf, ljung_box_q=q, ljung_box_p=float(chi2.sf(q, 3)),
                               white_noise_band_lag1=float(1.96 / np.sqrt(n)))
    return out


def panel2():
    """Primary procedure for the contrasts of Table 13 (second panel, 884 test rows, 19 years)."""
    d = os.path.join(R, 'panel2', 'med2v2')
    z = np.load(os.path.join(d, 'probs_pooled.npz')); y = z['y_te'].astype(int)
    m = pd.read_csv(os.path.join(d, 'test_meta_pooled.csv'))
    yrs = pd.to_datetime(m[[c for c in m.columns if c.startswith('fecha')][0]]).dt.year.values
    assert len(yrs) == len(y)
    out = {}
    for a, b in [('RF_2273', 'XGB'), ('RF_2273', 'PureLSTM'), ('RF_2273', 'LSTM_QNN'), ('XGB', 'LSTM_QNN'),
                 ('PureLSTM', 'LSTM_QNN'), ('CTRL_LINEAR4D', 'LSTM_QNN'), ('FOURIER2', 'LSTM_QNN')]:
        r = cluster_robust_delta(y, z[f'{a}_probs'], z[f'{b}_probs'], yrs); lo, hi = r.ci()
        out[f'{a}-{b}'] = dict(delta=r.delta, ci_cr2=[lo, hi], p_cr2=r.p_value, df=r.df)
    return out


def main():
    json.dump(panel2(), open(os.path.join(R, 'local_panel2_cr2.json'), 'w'), indent=2)
    json.dump(influential_years(), open(os.path.join(R, 'local_cr2_leave_one_year.json'), 'w'), indent=2)
    json.dump(serial_dependence(), open(os.path.join(R, 'local_serial_dependence.json'), 'w'), indent=2)
    path = os.path.join(R, 'local_exp47_truncation_audit.json')
    res = json.load(open(path)); res['refit_neural'] = causal_neural(); json.dump(res, open(path, 'w'), indent=2)
    json.dump(seas5(), open(os.path.join(R, 'local_seas5_cr2.json'), 'w'), indent=2)
    json.dump(blocks(), open(os.path.join(R, 'local_cr2_block_sensitivity.json'), 'w'), indent=2)
    print('updated', path, '| wrote local_seas5_cr2.json, local_cr2_block_sensitivity.json')


if __name__ == '__main__':
    main()
