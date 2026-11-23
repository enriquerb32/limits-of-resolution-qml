#!/usr/bin/env python3
"""
exp47_truncation_audit.py - is every test window computable at its issue date?
===============================================================================================================
A. Truncation test. For every test target month t (2008 onwards) the monthly panel is rebuilt from the daily file
   cut at the end of month t-1, with the transformations fitted on the training partition (anomaly statistics,
   medians, P60: all from <= 1996, so unaffected by the cut). For each station, the 35 window inputs (12 precipitation
   lags t-12..t-1 and the physical features of t-1) and the eligibility of the window are compared with the full
   pipeline. The label of t is the outcome and is taken from the full record in both cases. Run for the archived
   imputation (V2_CAUSAL=0) and for the causal one (V2_CAUSAL=1); the causal pipeline must give zero changes.
   The daily file has no record of when each value became available, so this is a retrospective check of
   lookahead in the processing, not a reconstruction of what could have been issued in real time.
B. Label completeness. Test target months with 1-3 missing days have an observed sum that is a lower bound of the
   true total; with a fixed threshold they can only be false negatives. Counts them and recomputes the AUC of every
   archived model on the complete-month rows. Also documents the month set behind P60.
C. Refit. RF and XGB (tuned, 15 seeds, as exp41) on the archived and on the causal panel; AUC of both on the rows
   eligible in both, with a paired year-block bootstrap of the difference. (The LSTM-based models need run_v2.ps1
   with V2_CAUSAL=1.)

  LOR_DATASET=v2 python code/local_experiments/exp47_truncation_audit.py [--seeds 15] [--skip-refit]
Output: results/local_exp47_truncation_audit.json, results/exp47_truncation_rows.csv
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'src'))
import dataset_v2  # noqa: E402
import trainonly_dataset as tds  # noqa: E402

RAW_CSV = os.path.join(HERE, 'datos_limpios_multiestacion.csv')
CORE = os.path.join(ROOT, 'results', 'json', 'results_hybrids_trainonly_fixet0_p60_28.1_probs.npz')
META = os.path.join(ROOT, 'results', 'test_meta.csv')
FIRST_TEST = pd.Timestamp('2008-01-01')
W = tds.WINDOW_SIZE
FEATS = tds.FEATURES_FISICAS

# daily ET0 uses same-day values only, so it is computed once and looked up (it is the slow step)
_et0_orig, _et0_cache = dataset_v2._et0_daily, {}


def _et0_cached(df, tds_):
    key = pd.MultiIndex.from_arrays([df['ciudad'].values, df.index])
    if 'S' not in _et0_cache:
        _et0_cache['S'] = pd.Series(_et0_orig(df, tds_), index=key)
    return _et0_cache['S'].reindex(key).values


dataset_v2._et0_daily = _et0_cached


def panel(daily):
    return dataset_v2.construir_features_v2(daily, tds._anomalia_train_only)


def window(st, t):
    """Inputs and eligibility of the window with target t in one station's monthly rows `st` (date index)."""
    tm1 = t - pd.DateOffset(months=1)
    lag_idx = pd.date_range(t - pd.DateOffset(months=W), periods=W, freq='MS')
    if tm1 not in st.index or not lag_idx.isin(st.index).all():
        return None, False
    x = np.concatenate([st.loc[lag_idx, 'precip'].values, st.loc[tm1, FEATS].values.astype(float)])
    ok = bool(st.loc[tm1, 'feat_ok']) and bool(st.loc[lag_idx, 'precip_ok'].all())
    return x, ok


def truncation(daily, causal):
    dataset_v2.CAUSAL = causal
    full = panel(daily)
    by_st = {c: g.sort_index() for c, g in full.groupby('ciudad')}
    targets = sorted(full.index[(full.index >= FIRST_TEST)].unique())
    rows = []
    for t in targets:
        cut = panel(daily[daily.index < t])
        cut_st = {c: g.sort_index() for c, g in cut.groupby('ciudad')}
        for c, st in by_st.items():
            if t not in st.index:
                continue
            win_full = bool(st.loc[t, 'win_ok'])
            x_full, _ = window(st, t)
            x_cut, ok_cut = window(cut_st[c], t) if c in cut_st else (None, False)
            win_cut = bool(st.loc[t, 'label_ok']) and ok_cut       # label of t: outcome, from the full record
            if not (win_full or win_cut):
                continue
            diff = [] if (x_full is None or x_cut is None) else \
                [n for n, a, b in zip([f'lag{W - j}' for j in range(W)] + FEATS, x_full, x_cut)
                 if not np.isclose(a, b, rtol=1e-9, atol=1e-9)]
            rows.append(dict(causal=int(causal), ciudad=c, fecha=t.date().isoformat(), win_full=win_full,
                             win_cut=win_cut, n_diff=len(diff), diff=' '.join(diff),
                             max_abs=(float(np.max(np.abs(x_full - x_cut))) if diff else 0.0)))
        print(f'  [{"causal" if causal else "archived"}] {t.date()}', end='\r', flush=True)
    r = pd.DataFrame(rows)
    both = r[r.win_full & r.win_cut]
    feat_counts = pd.Series(' '.join(both['diff']).split()).value_counts().to_dict() if both['n_diff'].any() else {}
    summ = dict(test_rows_full=int(r.win_full.sum()), test_rows_cut=int(r.win_cut.sum()),
                eligible_full_only=int((r.win_full & ~r.win_cut).sum()),
                eligible_cut_only=int((~r.win_full & r.win_cut).sum()),
                rows_with_changed_inputs=int((both.n_diff > 0).sum()), rows_unchanged=int((r.win_full & r.win_cut & (r.n_diff == 0)).sum()),
                changed_by_feature=feat_counts,
                changed_by_station=both[both.n_diff > 0].groupby('ciudad').size().to_dict())
    print(f'\n  {json.dumps(summ)}')
    return r, summ


def interpolation_reach(daily):
    """Station-months whose covariates the interpolation fills, and the eligible windows whose last antecedent month
    is one of them, by partition of the target month (archived and causal imputation)."""
    base_cols = list(dataset_v2.BASE)
    dataset_v2.BASE = []
    raw = dataset_v2.agregar_mensual_v2(daily)
    dataset_v2.BASE = base_cols
    out = {}
    for causal in (False, True):
        dataset_v2.CAUSAL = causal
        agg = dataset_v2.agregar_mensual_v2(daily)
        cell = (raw[base_cols].isna() & agg[base_cols].notna()).any(axis=1)
        nxt = agg.groupby('ciudad')['win_ok'].shift(-1).astype('boolean').fillna(False).astype(bool)
        part = pd.cut((agg.index + pd.DateOffset(months=1)).year, [0, 1996, 2007, 2100], labels=['train', 'val', 'test'])
        used = pd.Series(part[(cell & nxt).values]).value_counts().to_dict()
        out['causal' if causal else 'archived'] = dict(filled_station_months=int(cell.sum()),
                                                        windows_with_filled_last_month={str(k): int(v) for k, v in used.items()})
    dataset_v2.CAUSAL = False
    lab = raw[raw['label_ok']]
    miss = lab.index.days_in_month.values - lab['dias'].values
    lp = pd.cut(lab.index.year, [0, 1996, 2007, 2100], labels=['train', 'val', 'test'])
    out['labelled_months'] = {str(k): int(v) for k, v in pd.Series(lp).value_counts().items()}
    out['labelled_months_with_missing_days'] = {str(k): int(v) for k, v in pd.Series(lp[miss > 0]).value_counts().items()}
    return out


def labels_part(daily):
    from qml_resolution import rank_auc
    dataset_v2.CAUSAL = False
    full = panel(daily)
    meta = pd.read_csv(META, parse_dates=['fecha'])
    key = full.reset_index().set_index(['ciudad', 'fecha'])
    nd = meta['fecha'].dt.days_in_month.values
    dias = key.loc[list(zip(meta.ciudad, meta.fecha)), 'dias'].values
    miss = nd - dias
    z = np.load(CORE); y = z['y_te'].astype(int)
    assert np.array_equal(y, meta['y_te'].values)
    comp = miss == 0
    auc = {k[:-6]: dict(all=float(rank_auc(y, z[k]).mean()), complete=float(rank_auc(y[comp], z[k][:, comp]).mean()))
           for k in z.files if k.endswith('_probs')}
    tr = (full.index.year <= 1996) & full['label_ok'].values.astype(bool)
    lab = full.loc[tr, 'precip_label']
    tr0 = tr & (full['dias'].values == full.index.days_in_month.values)
    return dict(
        test_missing_days_hist={int(k): int(v) for k, v in pd.Series(miss).value_counts().sort_index().items()},
        test_rows_incomplete=int((miss > 0).sum()), test_negatives_incomplete=int(((miss > 0) & (y == 0)).sum()),
        test_positives_incomplete=int(((miss > 0) & (y == 1)).sum()),
        auc_all_vs_complete_rows=auc,
        p60_month_set=dict(definition='all labelled training months (<=1996, at most 3 missing days) of the four '
                                      'stations, not only window targets',
                           n_months=int(tr.sum()), by_station=full[tr].groupby('ciudad').size().to_dict(),
                           p60=float(np.percentile(lab, 60)),
                           n_months_complete=int(tr0.sum()),
                           p60_complete_only=float(np.percentile(full.loc[tr0, 'precip_label'], 60))))


def refit(seeds):
    from sklearn.impute import SimpleImputer
    from qml_resolution import cluster_bootstrap_delta, percentile_ci, rank_auc
    out, P, M, Y = {}, {}, {}, {}
    for causal in (False, True):
        dataset_v2.CAUSAL = causal
        _, ds = tds.build_dataset(RAW_CSV, train_only=True, fix_et0_verano=True)
        imp = SimpleImputer(strategy='mean').fit(ds.X_sk)
        Xt, Xv, Xe = imp.transform(ds.X_sk), imp.transform(ds.X_va.numpy()), imp.transform(ds.X_te.numpy())
        yt, yv = ds.y_sk.astype(int), ds.y_va.numpy().astype(int)
        P[causal] = {'RF': np.array([dataset_v2.rf_v2(s).fit(Xt, yt).predict_proba(Xe)[:, 1] for s in range(seeds)]),
                     'XGB': np.array([dataset_v2.xgb_fit_v2(s, Xt, yt, Xv, yv).predict_proba(Xe)[:, 1] for s in range(seeds)])}
        M[causal] = list(zip(ds.meta_te['ciudad'], pd.to_datetime(ds.meta_te['fecha'])))
        Y[causal] = ds.y_te.numpy().astype(int)
        out['n_test_' + ('causal' if causal else 'archived')] = len(M[causal])
        out['n_train_' + ('causal' if causal else 'archived')] = int(len(yt))
        print(f'  refit {"causal" if causal else "archived"}: n_te={len(M[causal])}', flush=True)
    common = sorted(set(M[False]) & set(M[True]))
    ia = [M[False].index(k) for k in common]; ib = [M[True].index(k) for k in common]
    y = Y[False][ia]; assert np.array_equal(y, Y[True][ib])
    years = np.array([k[1].year for k in common])
    out['n_common'] = len(common)
    for m in ('RF', 'XGB'):
        a, b = P[False][m][:, ia], P[True][m][:, ib]
        reps = cluster_bootstrap_delta(y, a, b, years, n_boot=2000, random_state=0)
        out[m] = dict(auc_archived_all=float(rank_auc(Y[False], P[False][m]).mean()),
                      auc_causal_all=float(rank_auc(Y[True], P[True][m]).mean()),
                      auc_archived_common=float(rank_auc(y, a).mean()), auc_causal_common=float(rank_auc(y, b).mean()),
                      delta_archived_minus_causal=float(rank_auc(y, a).mean() - rank_auc(y, b).mean()),
                      ci95_year_block=list(percentile_ci(reps)))
        print(f'  {m}: {json.dumps(out[m])}', flush=True)
    np.savez_compressed(os.path.join(ROOT, 'results', 'json', 'results_exp47_causal_trees_probs.npz'),
                        **{f'{m}_causal_probs': P[True][m] for m in ('RF', 'XGB')}, y_te=Y[True],
                        ciudad=np.array([k[0] for k in M[True]]), fecha=np.array([str(k[1].date()) for k in M[True]]))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--seeds', type=int, default=15)
    ap.add_argument('--skip-refit', action='store_true'); a = ap.parse_args()
    if not dataset_v2.V2:
        raise SystemExit('set LOR_DATASET=v2')
    daily = pd.read_csv(RAW_CSV, index_col='fecha', parse_dates=['fecha'])
    res = {}
    rows = []
    for causal in (False, True):
        r, s = truncation(daily, causal)
        rows.append(r); res['truncation_' + ('causal' if causal else 'archived')] = s
    pd.concat(rows).to_csv(os.path.join(ROOT, 'results', 'exp47_truncation_rows.csv'), index=False)
    res['labels'] = labels_part(daily)
    res['interpolation_reach'] = interpolation_reach(daily)
    print(json.dumps(res['labels'], indent=1))
    if not a.skip_refit:
        res['refit_trees'] = refit(a.seeds)
    json.dump(res, open(os.path.join(ROOT, 'results', 'local_exp47_truncation_audit.json'), 'w'), indent=2)
    print('wrote results/local_exp47_truncation_audit.json')


if __name__ == '__main__':
    main()
