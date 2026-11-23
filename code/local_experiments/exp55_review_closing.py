#!/usr/bin/env python3
"""
exp55_review_closing.py - causal refits, second-panel CR2 MDE, SEAS5 under CR2 (archived predictions only, no training)
================================================================================================================
A. Causal pipeline, primary families. The two Holm families of the paper (RF, XGBoost, LSTM; LSTM+QNN against RF,
   Linear-4D and the LSTM) recomputed with the primary procedure (CR2, Bell-McCaffrey df, calendar-year clusters) on
   the models refitted with causal imputation (V2_CAUSAL=1): RF and XGBoost from exp47, the LSTM and LSTM+QNN from
   run_v2.ps1 step 03b_core_causal (results/causal/), Linear-4D from step 03c_linear4d_causal (results/causal_linear4d/).
   Reported on the causal pipeline's own 533 test rows and on the 529 rows it shares with the archived test set, next
   to the archived verdicts, with Holm within each family.
B. Second panel: the MDE of the primary procedure (CR2) for each contrast of Table 11, next to the first-panel CR2
   MDE of the same contrast scaled by sqrt(529/884), the N^{-1/2} prediction that the DeLong column already reports.
C. SEAS5: the primary procedure for the reference contrasts at both leads, and for the incremental contrast
   SEAS5 + climatology + model against SEAS5 + climatology (combination fitted per seed on the validation partition,
   as exp49 part B, whose differences it must reproduce).

  python code/local_experiments/exp55_review_closing.py
Output: results/local_exp55_review_closing.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.linear_model import LogisticRegression

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from qml_resolution import cluster_robust_delta, rank_auc  # noqa: E402

R, J = os.path.join(ROOT, 'results'), os.path.join(ROOT, 'results', 'json')
META = pd.read_csv(os.path.join(R, 'test_meta.csv'))
YEARS = pd.to_datetime(META.fecha).dt.year.values
FAMILIES = {'classical': [('RF', 'XGB'), ('RF', 'PureLSTM'), ('XGB', 'PureLSTM')],
            'hybrid': [('RF', 'LSTM_QNN'), ('LSTM_QNN', 'CTRL_LINEAR4D'), ('PureLSTM', 'LSTM_QNN')]}
ALPHA = 0.05


def mde_cr2(r, alpha=ALPHA):
    """80 %-power MDE of the primary procedure, (t_{1-alpha/2,nu} + t_{0.80,nu}) SE_CR2, SE treated as known."""
    return float((student_t.ppf(1 - alpha / 2, r.df) + student_t.ppf(0.80, r.df)) * r.se_cr2)


def contrast(y, a, b, clusters):
    r = cluster_robust_delta(y, a, b, clusters); lo, hi = r.ci()
    return dict(delta=r.delta, ci_cr2=[lo, hi], p_cr2=r.p_value, df=r.df, se_cr2=r.se_cr2, se_iid=r.se_iid,
                ratio_cr1_iid=r.se_cr1 / r.se_iid, mde_cr2=mde_cr2(r), resolved=bool(lo > 0 or hi < 0))


def holm(pvals, alpha=ALPHA):
    """Holm step-down: list of booleans (rejected) in the input order."""
    order = np.argsort(pvals); m = len(pvals); rej = [False] * m
    for step, i in enumerate(order):
        if pvals[i] < alpha / (m - step):
            rej[i] = True
        else:
            break
    return rej


# ------------------------------------------------------------------ A. causal pipeline
def archived_probs():
    h = np.load(os.path.join(J, 'results_core_hybrids_controls_fixet0_p60_28.1_probs.npz'))
    c = np.load(os.path.join(J, 'results_rerun_train_only_canonical_probs.npz'))
    P = {k[:-6]: h[k] for k in h.files if k.endswith('_probs')}
    P.update(RF=c['train_only_RF_probs'], XGB=c['train_only_XGB_probs'])       # as exp42 / exp51
    return h['y_te'].astype(int), P


def causal_probs():
    t = np.load(os.path.join(J, 'results_exp47_causal_trees_probs.npz'), allow_pickle=True)
    h = np.load(os.path.join(R, 'causal', 'results_hybrids_trainonly_fixet0_probs.npz'))
    l4 = np.load(os.path.join(R, 'causal_linear4d', 'results_hybrids_trainonly_fixet0_probs.npz'))
    y = t['y_te'].astype(int)
    assert np.array_equal(y, h['y_te'].astype(int)) and np.array_equal(y, l4['y_te'].astype(int)), 'causal test orders differ'
    P = dict(RF=t['RF_causal_probs'], XGB=t['XGB_causal_probs'], PureLSTM=h['PureLSTM_probs'],
             LSTM_QNN=h['LSTM_QNN_probs'], CTRL_LINEAR4D=l4['CTRL_LINEAR4D_probs'])
    assert all(p.shape == (15, y.size) for p in P.values())
    keys = list(zip(t['ciudad'].astype(str), pd.to_datetime(t['fecha']).strftime('%Y-%m-%d')))
    return y, P, keys, pd.to_datetime(t['fecha']).year.values


def part_a():
    ya, Pa = archived_probs()
    yc, Pc, keys, yrs_c = causal_probs()
    pos = {k: i for i, k in enumerate(keys)}
    ib = np.array([pos[k] for k in zip(META.ciudad, META.fecha.str[:10])])     # archived row -> causal row
    assert np.array_equal(ya, yc[ib]), 'labels of the common rows differ'
    sets = {'archived_529': (ya, Pa, YEARS), 'causal_common_529': (yc[ib], {m: p[:, ib] for m, p in Pc.items()}, YEARS),
            'causal_own_533': (yc, Pc, yrs_c)}
    out = dict(n_causal=int(yc.size), n_common=int(ib.size), auc={}, families={})
    for name, (y, P, yr) in sets.items():
        out['auc'][name] = {m: float(rank_auc(y, P[m]).mean()) for m in ('RF', 'XGB', 'PureLSTM', 'LSTM_QNN', 'CTRL_LINEAR4D')}
        out['families'][name] = {}
        for fam, pairs in FAMILIES.items():
            rows = {f'{a}-{b}': contrast(y, P[a], P[b], yr) for a, b in pairs}
            rej = holm([v['p_cr2'] for v in rows.values()])
            for (k, v), r in zip(rows.items(), rej):
                v['resolved_holm'] = bool(r)
            out['families'][name][fam] = rows
    # Linear-4D refit, reported like the other four models of tab:lookahead (exp51 causal_neural)
    from qml_resolution import cluster_bootstrap_delta, percentile_ci
    a4, c4 = Pa['CTRL_LINEAR4D'], Pc['CTRL_LINEAR4D'][:, ib]
    reps = cluster_bootstrap_delta(ya, a4, c4, YEARS, n_boot=2000, random_state=0)
    out['refit_linear4d'] = dict(auc_archived_common=float(rank_auc(ya, a4).mean()), auc_causal_common=float(rank_auc(ya, c4).mean()),
                                 delta_archived_minus_causal=float(rank_auc(ya, a4).mean() - rank_auc(ya, c4).mean()),
                                 ci95_year_block=list(percentile_ci(reps)))
    print('  A Linear-4D refit', {k: (round(v, 4) if isinstance(v, float) else [round(x, 3) for x in v])
                                  for k, v in out['refit_linear4d'].items()})
    out['blocks'] = {}                      # clusters of 1-3 consecutive years for the two exploratory pairs
    for name, (y, P, yr) in sets.items():
        out['blocks'][name] = {f'{a}-{b}': {f'{k}yr': contrast(y, P[a], P[b], (yr - yr.min()) // k)['p_cr2'] for k in (1, 2, 3)}
                               for a, b in [('RF', 'PureLSTM'), ('RF', 'LSTM_QNN')]}
        print(f'  A blocks {name:18s}', {k: {b: round(p, 3) for b, p in v.items()} for k, v in out['blocks'][name].items()})
    for name in sets:
        for fam in FAMILIES:
            for k, v in out['families'][name][fam].items():
                print(f'  A {name:18s} {k:24s} {v["delta"]:+.4f}  CR2 p {v["p_cr2"]:.3f}  '
                      f'{"resolved" if v["resolved"] else "-":8s} {"Holm" if v["resolved_holm"] else ""}')
    return out


# ------------------------------------------------------------------ B. second panel, CR2 MDE
def part_b():
    d = os.path.join(R, 'panel2', 'med2v2')
    z = np.load(os.path.join(d, 'probs_pooled.npz')); y2 = z['y_te'].astype(int)
    m = pd.read_csv(os.path.join(d, 'test_meta_pooled.csv'))
    yrs2 = pd.to_datetime(m[[c for c in m.columns if c.startswith('fecha')][0]]).dt.year.values
    h = np.load(os.path.join(J, 'results_hybrids_trainonly_fixet0_p60_28.1_probs.npz'))       # first panel, as the
    c = np.load(os.path.join(J, 'results_rerun_train_only_canonical_probs.npz'))              # Table 11 test does
    k = np.load(os.path.join(J, 'results_controls_trainonly_fixet0_p60_28.1_probs.npz'))
    P1 = {mm: h[f'{mm}_probs'] for mm in ('RF_2273', 'PureLSTM', 'LSTM_QNN', 'CTRL_LINEAR4D')}
    P1.update(XGB=c['train_only_XGB_probs'], FOURIER2=k['FOURIER2_probs']); y1 = h['y_te'].astype(int)
    scale = np.sqrt(y1.size / y2.size)
    out = dict(n_first=int(y1.size), n_second=int(y2.size), years_first=int(np.unique(YEARS).size),
               years_second=int(np.unique(yrs2).size), scale=float(scale), pairs={})
    for a, b in [('RF_2273', 'XGB'), ('RF_2273', 'PureLSTM'), ('RF_2273', 'LSTM_QNN'), ('XGB', 'LSTM_QNN'),
                 ('PureLSTM', 'LSTM_QNN'), ('CTRL_LINEAR4D', 'LSTM_QNN'), ('FOURIER2', 'LSTM_QNN')]:
        r2 = contrast(y2, z[f'{a}_probs'], z[f'{b}_probs'], yrs2)
        r1 = contrast(y1, P1[a], P1[b], YEARS)
        out['pairs'][f'{a}-{b}'] = dict(mde_cr2_second=r2['mde_cr2'], mde_cr2_first=r1['mde_cr2'],
                                        mde_cr2_predicted=r1['mde_cr2'] * scale,
                                        ratio_observed_predicted=r2['mde_cr2'] / (r1['mde_cr2'] * scale),
                                        ratio_second_first=r2['mde_cr2'] / r1['mde_cr2'], ci_cr2_second=r2['ci_cr2'])
        v = out['pairs'][f'{a}-{b}']
        print(f'  B {a}-{b:14s} CR2 MDE panel 2 {v["mde_cr2_second"]:.3f} | panel 1 {v["mde_cr2_first"]:.3f} '
              f'| predicted {v["mde_cr2_predicted"]:.3f} (ratio {v["ratio_observed_predicted"]:.2f})')
    return out


# ------------------------------------------------------------------ C. SEAS5 with the primary procedure
def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def part_c():
    z = np.load(os.path.join(J, 'results_exp33_seas5_orig_probs.npz'))
    y_va, y = z['y_va'].astype(int), z['y_te'].astype(int)
    assert np.array_equal(y, META.y_te.values.astype(int))
    ref = json.load(open(os.path.join(R, 'local_exp49_review_minor.json')))['seas5_increment']
    out = dict(reference={}, increment={})
    for L in (2, 1):
        for a, b in [(f'SEAS5_QM_L{L}', 'CLIM'), ('RF_2273', f'SEAS5_QM_L{L}'), ('LSTM_QNN', f'SEAS5_QM_L{L}')]:
            out['reference'][f'{a}-{b}'] = contrast(y, z[f'{a}_probs'], z[f'{b}_probs'], YEARS)
        cv, ct = logit(z['CLIM_val'][0]), logit(z['CLIM_probs'][0])
        sv, st = logit(z[f'SEAS5_QM_L{L}_val'][0]), logit(z[f'SEAS5_QM_L{L}_probs'][0])
        base = LogisticRegression(C=1e6, max_iter=1000).fit(np.c_[sv, cv], y_va)
        p_base = base.predict_proba(np.c_[st, ct])[:, 1][None]
        out['increment'][f'L{L}'] = {'SEAS5+CLIM_auc': float(rank_auc(y, p_base)[0])}
        for m in ('RF_2273', 'XGB', 'PureLSTM', 'CTRL_LINEAR4D', 'LSTM_QNN'):
            pv, pt = z[f'{m}_val'], z[f'{m}_probs']
            P = np.array([LogisticRegression(C=1e6, max_iter=1000).fit(np.c_[sv, cv, logit(pv[s])], y_va)
                          .predict_proba(np.c_[st, ct, logit(pt[s])])[:, 1] for s in range(pv.shape[0])])
            r = contrast(y, P, p_base, YEARS)
            assert abs(r['delta'] - ref[f'L{L}'][m]['delta_auc']) < 1e-9, f'L{L} {m}: does not reproduce exp49'
            out['increment'][f'L{L}'][m] = r
            print(f'  C L{L} SEAS5+clim+{m:14s} - SEAS5+clim {r["delta"]:+.4f}  CR2 [{r["ci_cr2"][0]:+.3f}, '
                  f'{r["ci_cr2"][1]:+.3f}] p {r["p_cr2"]:.3f}  MDE {r["mde_cr2"]:.3f}')
    # replacement blends (model + SEAS5 against climatology + SEAS5), formerly reported with DeLong only
    out['replacement'] = {}
    for a, b in [(f'BLEND_{m}_L1', 'BLEND_CLIM_L1') for m in ('RF_2273', 'XGB', 'PureLSTM', 'CTRL_LINEAR4D', 'LSTM_QNN')] + \
                [('BLEND_LSTM_QNN_L1', 'BLEND_CTRL_LINEAR4D_L1')]:
        out['replacement'][f'{a}-{b}'] = contrast(y, z[f'{a}_probs'], z[f'{b}_probs'], YEARS)
    for k, v in {**out['reference'], **out['replacement']}.items():
        print(f'  C {k:26s} {v["delta"]:+.4f}  CR2 [{v["ci_cr2"][0]:+.3f}, {v["ci_cr2"][1]:+.3f}] p {v["p_cr2"]:.3f}')
    return out


def main():
    res = dict(causal=part_a(), panel2_mde=part_b(), seas5=part_c())
    path = os.path.join(R, 'local_exp55_review_closing.json')
    json.dump(res, open(path, 'w'), indent=2)
    print('wrote', path)


if __name__ == '__main__':
    main()
