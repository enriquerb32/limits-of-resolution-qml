#!/usr/bin/env python3
"""
exp14_innerjoin_audit.py - How much observed precipitation does the inner join discard, and does it change labels / comparisons?

For every station-month IN THE PROJECT PANEL (joined daily table, dias_join >= 10) compare
    P_join = sum of precipitation over the days retained by the join       (what the project uses)
    P_raw  = sum of precipitation over all days with observed precipitation in the precipitation-only file
    L      = P_raw - P_join >= 0   (observed rain discarded by the join)
P_raw is the sum of OBSERVED days only; it is not a "complete" monthly total if days are also missing in the raw file.
Reports (JSON): months with L>0, distribution of L (quantiles, max), lost days, by partition and by station,
label changes at the FIXED project threshold (P60 of train P_join) and at a threshold RECOMPUTED from train P_raw,
overall and for the test partition, plus the number of project-panel months and the denominator used.
NOTE: the 'months_label_would_change...' figure of exp12 counts months of the precipitation-only table (including months
that the project dropped for having < 10 joined days, whose joined sum is treated as 0). This script restricts to the
project panel and reports both counts so they can be compared.

Writes --save_rows CSV = month_rows.csv (from exp12) + columns  p_join, p_raw, L, dias_raw, lab_raw_same_thr, lab_raw_new_thr
which exp13_label_sensitivity.py (updated) uses for scenarios S6 (raw labels, same threshold) and S7 (raw labels, new threshold).

USAGE
  python exp14_innerjoin_audit.py --csv datos_limpios_multiestacion.csv --precip_raw precip_raw.csv \
      --rows month_rows.csv --save_rows month_rows_raw.csv --out results_exp14_innerjoin.json
  python exp14_innerjoin_audit.py --selftest
"""
import argparse, json
import numpy as np, pandas as pd
from exp12_monthly_coverage_audit import monthly, synth


def run(df, pr, rows=None):
    mj = monthly(df); mr = monthly(pr)
    base = mj[mj['dias'] >= 10].copy()
    thr = float(np.percentile(base.loc[base['part'] == 'train', 'precip'], 60))
    r = mr[['ciudad', 'fecha', 'precip', 'dias']].rename(columns={'precip': 'p_raw', 'dias': 'dias_raw'})
    t = base.rename(columns={'precip': 'p_join', 'dias': 'dias_join'}).merge(r, on=['ciudad', 'fecha'], how='left')
    assert t['p_raw'].notna().all(), 'project months missing from the precipitation-only file'
    t['L'] = t['p_raw'] - t['p_join']; t['lost_days'] = t['dias_raw'] - t['dias_join']
    assert (t['L'] > -1e-9).all() and (t['lost_days'] >= 0).all(), 'raw file has fewer days/rain than joined: check the inputs'
    thr_raw = float(np.percentile(t.loc[t['part'] == 'train', 'p_raw'], 60))
    t['lab_project'] = (t['p_join'] > thr).astype(int)
    t['lab_raw_same_thr'] = (t['p_raw'] > thr).astype(int)
    t['lab_raw_new_thr'] = (t['p_raw'] > thr_raw).astype(int)
    res = {'n_project_panel_months': int(len(t)), 'P60_project_join': thr, 'P60_recomputed_raw': thr_raw}
    pos = t[t['L'] > 0]
    res['months_with_L_gt_0'] = int(len(pos)); res['share_with_L_gt_0'] = float(len(pos) / len(t))
    res['months_with_lost_days'] = int((t['lost_days'] > 0).sum()); res['total_lost_days'] = int(t['lost_days'].sum())
    res['total_L_mm'] = float(t['L'].sum()); res['max_L_mm'] = float(t['L'].max())
    res['L_quantiles_all_months'] = t['L'].quantile([.5, .75, .9, .95, .99, 1]).round(2).to_dict()
    res['L_quantiles_months_L_gt_0'] = pos['L'].quantile([.5, .75, .9, .95, .99, 1]).round(2).to_dict() if len(pos) else {}
    def block(d):
        return {'n': int(len(d)), 'months_L_gt_0': int((d['L'] > 0).sum()), 'total_L_mm': float(d['L'].sum()), 'max_L_mm': float(d['L'].max()),
                'flips_same_thr': int((d['lab_raw_same_thr'] != d['lab_project']).sum()),
                'flips_new_thr': int((d['lab_raw_new_thr'] != d['lab_project']).sum()),
                'prev_project': float(d['lab_project'].mean()), 'prev_raw_same_thr': float(d['lab_raw_same_thr'].mean()),
                'prev_raw_new_thr': float(d['lab_raw_new_thr'].mean())}
    res['by_partition'] = {p: block(d) for p, d in t.groupby('part')}
    res['by_city'] = {c: block(d) for c, d in t.groupby('ciudad')}
    res['by_city_and_partition'] = {f'{c}|{p}': block(d) for (c, p), d in t.groupby(['ciudad', 'part'])}
    res['overall'] = block(t)
    # exp12-style count (all months of the precip-only table, absent joined months counted as 0) for comparison
    j = mr.merge(mj, on=['ciudad', 'fecha'], how='left', suffixes=('_raw', '_join'))
    j = j[j['dias_raw'] >= 10]
    old = ((j['precip_raw'] > thr).astype(int) != (j['precip_join'].fillna(0) > thr).astype(int)).sum()
    res['exp12_style_count_precip_only_months'] = {'denominator_months_raw_ge10': int(len(j)), 'label_changes': int(old),
                                                  'of_which_months_not_in_project_panel': int(j['dias_join'].isna().sum() + (j['dias_join'] < 10).sum())}
    out_rows = None
    if rows is not None:
        rows = rows.copy(); rows['fecha'] = pd.to_datetime(rows['fecha'])
        out_rows = rows.merge(t[['ciudad', 'fecha', 'p_join', 'p_raw', 'L', 'dias_raw', 'lab_raw_same_thr', 'lab_raw_new_thr']], on=['ciudad', 'fecha'], how='left', validate='one_to_one')
        assert out_rows['p_raw'].notna().all(), 'rows (exp12) not all found'
        assert (out_rows['lab_project'].values == (out_rows['p_join'] > thr).astype(int).values).all(), 'labels of month_rows.csv differ from this run: use the same --csv'
    return res, out_rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv'); ap.add_argument('--precip_raw'); ap.add_argument('--rows'); ap.add_argument('--save_rows')
    ap.add_argument('--out', default='results_exp14_innerjoin.json'); ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        pr = synth(1); df = pr.copy(); rng = np.random.default_rng(2)
        df.loc[rng.random(len(df)) < 0.1, 'precip'] = np.nan; rows = None
    else:
        df = pd.read_csv(a.csv, index_col='fecha', parse_dates=['fecha']); pr = pd.read_csv(a.precip_raw, index_col='fecha', parse_dates=['fecha'])
        rows = pd.read_csv(a.rows) if a.rows else None
    res, out_rows = run(df, pr, rows)
    if out_rows is not None and a.save_rows: out_rows.to_csv(a.save_rows, index=False); print('saved', a.save_rows)
    json.dump(res, open('/tmp/exp14_selftest.json' if a.selftest else a.out, 'w'), indent=2, default=str)
    keys = ['n_project_panel_months', 'months_with_L_gt_0', 'months_with_lost_days', 'total_lost_days', 'total_L_mm', 'max_L_mm', 'P60_project_join', 'P60_recomputed_raw']
    print(json.dumps({k: res[k] for k in keys}, indent=2)); print('L quantiles (L>0):', res['L_quantiles_months_L_gt_0'])
    for p, v in res['by_partition'].items(): print(p, v)
    print('exp12-style:', res['exp12_style_count_precip_only_months'])
