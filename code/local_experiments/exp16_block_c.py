#!/usr/bin/env python3
"""
code/local_experiments/exp16_block_c.py
Reviewer-requested supplementary analyses (block C). Uses only archived test probabilities;
no model is retrained.

  C1  Seed share on the single-AUC scale AND on the paired-difference scale (single seeds and
      15-seed means), for the classical rerun and for the matched QML core experiment.
  C2  Station-only baseline (training base rate per station) and a month x station climatology
      fitted on training + validation (1938-2007), next to the training-only climatology.
  C3  Single-run (seed 42) Random Forest vs month x station climatology: DeLong and year-block
      bootstrap (point difference, bootstrap mean, percentile CI, bootstrap p as in exp4).
  C4  Leave-one-year-out cluster jackknife SE for the six seed-mean comparisons of the paper.
  C5  Audit: does the time interpolation in the monthly aggregation fill any value?
  C6  Gate count of Mottonen state preparation for a generic real 16-amplitude state.

Run from the repository root:
    python code/local_experiments/exp16_block_c.py
Output: results/local_exp16_block_c.json
"""
import json
import os
import sys
import itertools

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from noise_utils import (ROOT, load_test_partition, delong_cov, auc_matrix,  # noqa: E402
                         year_block_bootstrap_aucs)
from exp10_paired_ci import Model, paired_delong, auc_fast  # noqa: E402
from exp7_noise_budget import climatology_probs  # noqa: E402

B = 2000
BOOT_SEED = 42
ST_IDX = [31, 32, 33, 34]          # one-hot station columns of X (Alicante, Barcelona, San_Javier, Valencia)
OUT = os.path.join(ROOT, 'results', 'local_exp16_block_c.json')


def seed_sd(aucs):
    return float(np.std(aucs, ddof=1))


def boot_p(diffs):
    p = 2.0 * min(float(np.mean(diffs <= 0)), float(np.mean(diffs >= 0)))
    return min(1.0, max(1.0 / len(diffs), p))


def brier(y, p):
    return float(np.mean((np.asarray(p, float) - y) ** 2))


# ----------------------------------------------------------------------------- C1
def family_stats(name, probs, y, years):
    """probs: dict model -> (S, n). Returns per-model and per-pair seed-share statistics."""
    keys = list(probs)
    S = {k: probs[k].shape[0] for k in keys}
    stack = np.vstack([probs[k] for k in keys])
    offs = np.cumsum([0] + [S[k] for k in keys])
    rows = {k: np.arange(offs[i], offs[i + 1]) for i, k in enumerate(keys)}

    aucs, cov = delong_cov(y, stack)
    boot = year_block_bootstrap_aucs(y, stack, years, B=B, seed=BOOT_SEED)   # (B_valid, k)
    se_iid_row = np.sqrt(np.diag(cov))
    se_blk_row = boot.std(axis=0, ddof=1)

    per_model = {}
    for k in keys:
        r = rows[k]
        sd = seed_sd(aucs[r])
        se_iid, se_blk = float(se_iid_row[r].mean()), float(se_blk_row[r].mean())
        per_model[k] = dict(
            n_seeds=int(len(r)), mean_auc=float(aucs[r].mean()), seed_sd=sd,
            se_one_auc_iid=se_iid, se_one_auc_block=se_blk,
            seed_share_one_auc_iid=sd ** 2 / (sd ** 2 + se_iid ** 2),
            seed_share_one_auc_block=sd ** 2 / (sd ** 2 + se_blk ** 2))

    per_pair = {}
    for a, b in itertools.combinations(keys, 2):
        ra, rb = rows[a], rows[b]
        # single-seed pairings: sampling SE of each (i, j) difference, averaged over the S_a x S_b pairings
        v_iid = [cov[i, i] + cov[j, j] - 2 * cov[i, j] for i in ra for j in rb]
        se_pair_iid = float(np.mean(np.sqrt(np.maximum(v_iid, 0))))
        se_pair_blk = float(np.mean([np.std(boot[:, i] - boot[:, j], ddof=1) for i in ra for j in rb]))
        # seed variance of a single-seed difference (seeds independent across models)
        sda, sdb = per_model[a]['seed_sd'], per_model[b]['seed_sd']
        v_seed = sda ** 2 + sdb ** 2
        # 15-seed means: sampling SE (seed-averaged DeLong, as exp10) and block SE of the mean difference
        Ma, Mb = Model(probs[a], y), Model(probs[b], y)
        d_mean, se_mean_iid, _, p_mean = paired_delong(Ma, Mb, y)
        se_mean_blk = float(np.std(boot[:, ra].mean(axis=1) - boot[:, rb].mean(axis=1), ddof=1))
        v_seed_mean = sda ** 2 / len(ra) + sdb ** 2 / len(rb)
        per_pair[f'{a}-{b}'] = dict(
            delta_seed_means=float(d_mean),
            seed_sd_of_single_difference=float(np.sqrt(v_seed)),
            single_seed=dict(se_sampling_iid=se_pair_iid, se_sampling_block=se_pair_blk,
                             seed_share_iid=v_seed / (v_seed + se_pair_iid ** 2),
                             seed_share_block=v_seed / (v_seed + se_pair_blk ** 2)),
            seed_mean=dict(se_sampling_iid=float(se_mean_iid), se_sampling_block=se_mean_blk,
                           se_seed=float(np.sqrt(v_seed_mean)),
                           seed_share_iid=v_seed_mean / (v_seed_mean + se_mean_iid ** 2),
                           seed_share_block=v_seed_mean / (v_seed_mean + se_mean_blk ** 2)))
    return dict(family=name, per_model=per_model, per_pair=per_pair)


# ----------------------------------------------------------------------------- C4
def jackknife_seed_mean(PA, PB, y, years):
    uy = np.unique(years)
    th = []
    for g in uy:
        idx = years != g
        yy = y[idx]
        th.append(np.mean([auc_fast(yy, p[idx]) for p in PA]) - np.mean([auc_fast(yy, p[idx]) for p in PB]))
    th = np.array(th)
    G = len(uy)
    return float(np.sqrt((G - 1) / G * np.sum((th - th.mean()) ** 2)))


# ----------------------------------------------------------------------------- C2 helpers
def station_of(X):
    X = np.asarray(X)
    return np.array([ST_IDX.index([c for c in ST_IDX if X[i, c] == 1.0][0]) for i in range(len(X))])


def compare_single(y, years, pa, pb, label_a, label_b):
    aucs, cov = delong_cov(y, np.vstack([pa, pb]))
    d = float(aucs[0] - aucs[1])
    se = float(np.sqrt(max(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1], 1e-300)))
    bt = year_block_bootstrap_aucs(y, np.vstack([pa, pb]), years, B=B, seed=BOOT_SEED)
    diffs = bt[:, 0] - bt[:, 1]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return dict(A=label_a, B=label_b, auc_A=float(aucs[0]), auc_B=float(aucs[1]), point_diff=d,
                se_delong=se, z=d / se, p_delong=float(2 * norm.sf(abs(d) / se)),
                boot_mean_diff=float(diffs.mean()), se_block=float(diffs.std(ddof=1)),
                ci_block=[float(lo), float(hi)], p_block=boot_p(diffs), B_valid=int(len(diffs)))


def compare_seedmean_vs_fixed(y, years, P, p_fixed):
    """15-seed mean AUC of P (S, n) minus AUC of a fixed predictor (S=1)."""
    MA, MB = Model(P, y), Model(p_fixed[None, :], y)
    d, se, z, p = paired_delong(MA, MB, y)
    stack = np.vstack([P, p_fixed[None, :]])
    bt = year_block_bootstrap_aucs(y, stack, years, B=B, seed=BOOT_SEED)
    diffs = bt[:, :-1].mean(axis=1) - bt[:, -1]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return dict(delta=float(d), se_delong=float(se), p_delong=float(p), se_block=float(diffs.std(ddof=1)),
                ci_block=[float(lo), float(hi)], p_block=boot_p(diffs))


def main():
    ds, y, m_te, X_te, years, _ = load_test_partition()
    out = {'meta': dict(B=B, boot_seed=BOOT_SEED, n_test=int(len(y)), n_years=int(len(np.unique(years))),
                        note='No retraining; archived test probabilities only.')}

    rr = np.load(os.path.join(ROOT, 'results', 'json', 'results_rerun_train_only_canonical_probs.npz'))
    core = np.load(os.path.join(ROOT, 'results', 'json', 'results_qml_core_trainonly_probs.npz'))
    assert np.array_equal(rr['y_te'].astype(int), y) and np.array_equal(core['y_te'].astype(int), y)

    rerun = {k: rr[f'train_only_{k}_probs'].astype(float) for k in ['RF', 'XGB', 'LSTM']}
    qml = {k: core[f'{k}_probs'].astype(float) for k in ['RF_2273', 'PureLSTM', 'CTRL_LINEAR4D', 'LSTM_QNN']}

    # ---- C1
    print('C1 seed share ...')
    out['C1_seed_share'] = {'classical_rerun': family_stats('classical_rerun', rerun, y, years),
                            'qml_core': family_stats('qml_core', qml, y, years)}

    # ---- C4
    print('C4 jackknife ...')
    jk = {}
    for fam, P, pairs in [('classical_rerun', rerun, [('RF', 'XGB'), ('RF', 'LSTM'), ('XGB', 'LSTM')]),
                          ('qml_core', qml, [('RF_2273', 'LSTM_QNN'), ('CTRL_LINEAR4D', 'LSTM_QNN'),
                                             ('PureLSTM', 'LSTM_QNN')])]:
        for a, b in pairs:
            se_jk = jackknife_seed_mean(P[a], P[b], y, years)
            d = float(np.mean(auc_matrix(y, P[a])) - np.mean(auc_matrix(y, P[b])))
            jk[f'{a}-{b}'] = dict(family=fam, delta=d, se_jackknife=se_jk, mde_jackknife=2.80 * se_jk,
                                  ci95_t18=[d - 2.101 * se_jk, d + 2.101 * se_jk])
    out['C4_year_jackknife'] = jk

    # ---- C2
    print('C2 baselines ...')
    X_tr, y_tr, m_tr = np.array(ds['X_tr']), np.array(ds['y_tr']).astype(int), np.array(ds['m_tr']).astype(int)
    X_va, y_va, m_va = np.array(ds['X_va']), np.array(ds['y_va']).astype(int), np.array(ds['m_va']).astype(int)
    st_tr, st_te = station_of(X_tr), station_of(X_te)
    p_station = np.array([y_tr[st_tr == s].mean() for s in st_te])
    p_clim_tr = climatology_probs(ds, y, m_te, X_te)
    X_tv, y_tv, m_tv = np.vstack([X_tr, X_va]), np.concatenate([y_tr, y_va]), np.concatenate([m_tr, m_va])
    st_tv = station_of(X_tv)
    p_clim_tv = np.array([y_tv[(m_tv == m) & (st_tv == s)].mean() for m, s in zip(m_te, st_te)])
    rf_mean = rerun['RF']
    base = {}
    for name, p in [('station_only', p_station), ('clim_month_station_train', p_clim_tr),
                    ('clim_month_station_train_val', p_clim_tv)]:
        base[name] = dict(auc=float(auc_matrix(y, p[None, :])[0]), brier=brier(y, p),
                          rf_2787_seed_mean_minus_this=compare_seedmean_vs_fixed(y, years, rf_mean, p))
    base['clim_train_val_minus_clim_train'] = compare_single(y, years, p_clim_tv, p_clim_tr,
                                                             'clim_train_val', 'clim_train')
    base['clim_train_minus_station_only'] = compare_single(y, years, p_clim_tr, p_station,
                                                           'clim_train', 'station_only')
    out['C2_baselines'] = base

    # ---- C3
    print('C3 single-run RF vs climatology ...')
    b2 = json.load(open(os.path.join(ROOT, 'results', 'json', '02_baselines.json'), encoding='utf-8'))
    p_rf42 = np.array(b2['_probs']['Random Forest'], float)
    out['C3_single_run_rf_vs_clim'] = compare_single(y, years, p_rf42, p_clim_tr, 'RF_seed42', 'clim_train')

    # ---- C5
    print('C5 interpolation audit ...')
    daily = pd.read_csv(os.path.join(HERE, 'datos_limpios_multiestacion.csv'))
    cols = [c for c in ['precip', 'pmax', 'pmin', 'tmax', 'tmin', 'veloc_kmh', 'racha_kmh', 'sol'] if c in daily.columns]
    out['C5_interpolation_audit'] = dict(
        daily_rows=int(len(daily)), daily_columns=list(daily.columns),
        daily_nan_counts={c: int(daily[c].isna().sum()) for c in cols})
    date_col = 'fecha' if 'fecha' in daily.columns else daily.columns[0]
    daily[date_col] = pd.to_datetime(daily[date_col])
    daily['ym'] = daily[date_col].dt.to_period('M')
    g = daily.groupby(['ciudad', 'ym'])
    cnt = g['precip'].count()
    keep = cnt[cnt >= 10].index
    means = g[cols].mean().loc[keep]
    out['C5_interpolation_audit'].update(
        retained_station_months=int(len(keep)),
        station_months_with_an_all_missing_variable=int(means.isna().any(axis=1).sum()),
        all_missing_by_variable={c: int(means[c].isna().sum()) for c in cols})

    # ---- C6
    print('C6 Mottonen gate count ...')
    try:
        import pennylane as qml_
        rng = np.random.default_rng(0)
        v = rng.normal(size=16)
        v /= np.linalg.norm(v)
        ops = qml_.MottonenStatePreparation(v, wires=range(4)).decomposition()

        def flatten(op_list):
            res = []
            for op in op_list:
                if op.name in ('RY', 'RZ', 'CNOT', 'RX'):
                    res.append(op.name)
                else:
                    res.extend(flatten(op.decomposition()))
            return res
        names = flatten(ops)
        out['C6_mottonen_generic_real_state'] = {n: names.count(n) for n in sorted(set(names))}
        out['C6_mottonen_generic_real_state']['pennylane'] = qml_.__version__
    except Exception as e:  # pragma: no cover
        out['C6_mottonen_generic_real_state'] = {'error': repr(e)}

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print('saved', OUT)


if __name__ == '__main__':
    main()
