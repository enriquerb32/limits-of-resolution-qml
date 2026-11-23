#!/usr/bin/env python3
"""
exp12_monthly_coverage_audit.py  -  Does the monthly aggregation distort the P60 labels?

Reproduces the aggregation of the project (code: _agregar_mensual):  per station and calendar month
    precip = daily precip SUM (NaN days contribute 0!),  dias = COUNT of non-null daily precip,  keep dias >= 10
so a month with only 10-11 observed days enters with a (possibly much) smaller total.
The audit answers, per partition (train <=1996, val 1997-2007, test >=2008):
  1. distribution of valid days per station-month and how many kept months are partial (dias < days in month)
  2. inner-join effect (needs the five raw per-variable daily files, optional): days with observed precip that are LOST
     because some other variable is missing, and how those lost days change monthly sums
  3. label sensitivity: the pooled P60 label (threshold from TRAIN months only, as in the project) recomputed under
       a) the project's rule (dias >= 10, plain sum)
       b) stricter coverage rules (dias >= 15, 20, 25, or complete month)
       c) coverage-corrected sums (sum / dias * days_in_month) for dias >= 10
     reporting: n months kept, prevalence by partition, and number of labels that FLIP relative to (a)
  4. test-set months with partial coverage (the 528 rows are targets of 12-month windows: the target month is the
     one whose label matters; antecedent months matter for the features)
INPUT: daily CSV with index column 'fecha' and columns ciudad, precip (+ pmax,pmin,tmax,tmin,veloc_kmh,racha_kmh,sol).
USAGE:
  python exp12_monthly_coverage_audit.py --csv datos_limpios_multiestacion.csv --out results_exp12_coverage.json
  python exp12_monthly_coverage_audit.py --selftest        # synthetic data, checks the script runs
  optional: --precip_raw path_to_daily_precip_only.csv  (columns fecha,ciudad,precip) to quantify inner-join losses
"""
import argparse, json
import numpy as np, pandas as pd

def partition(idx):
    y = idx.year
    return np.where(y <= 1996, 'train', np.where(y <= 2007, 'val', 'test'))

def monthly(df):
    g = df.groupby(['ciudad', pd.Grouper(freq='MS')])
    out = g.agg(precip=('precip', 'sum'), dias=('precip', 'count')).reset_index()
    out['dim'] = out['fecha'].dt.days_in_month
    out['part'] = partition(pd.DatetimeIndex(out['fecha']))
    return out

def label(out, thr):
    return (out['precip'] > thr).astype(int)

def audit(df, precip_raw=None, save_rows=None):
    res = {}
    m = monthly(df)
    base = m[m['dias'] >= 10].copy()
    res['n_station_months_before_filter'] = int(len(m)); res['n_kept_dias_ge_10'] = int(len(base))
    base['partial'] = base['dias'] < base['dim']
    res['kept_partial_total'] = int(base['partial'].sum())
    res['kept_partial_by_partition'] = base.groupby('part')['partial'].agg(['sum', 'count']).rename(columns={'sum': 'partial', 'count': 'n'}).to_dict('index')
    res['kept_partial_by_city'] = base.groupby('ciudad')['partial'].agg(['sum', 'count']).rename(columns={'sum': 'partial', 'count': 'n'}).to_dict('index')
    bins = [10, 15, 20, 25, 28, 32]
    res['dias_histogram_kept'] = {f'[{bins[i]},{bins[i+1]})': int(((base['dias'] >= bins[i]) & (base['dias'] < bins[i + 1])).sum()) for i in range(len(bins) - 1)}
    res['coverage_ratio_quantiles'] = (base['dias'] / base['dim']).quantile([0, .05, .25, .5, .75, 1]).round(3).to_dict()
    # project threshold: P60 of TRAIN station-months kept under the project's rule
    thr = float(np.percentile(base.loc[base['part'] == 'train', 'precip'], 60)); res['P60_project'] = thr
    lab0 = label(base, thr)
    sens = {}
    def summarize(name, d, lab, ref_lab=None, ref_idx=None):
        row = {'n_kept': int(len(d))}
        for p in ('train', 'val', 'test'):
            k = d['part'] == p
            row[f'n_{p}'] = int(k.sum()); row[f'prev_{p}'] = float(lab[k].mean()) if k.any() else None
        if ref_lab is not None:
            common = d.index.intersection(ref_idx)
            row['n_common_with_project'] = int(len(common))
            row['label_flips_vs_project'] = int((lab.loc[common] != ref_lab.loc[common]).sum())
            for p in ('train', 'val', 'test'):
                c = common[d.loc[common, 'part'] == p]
                row[f'flips_{p}'] = int((lab.loc[c] != ref_lab.loc[c]).sum())
        sens[name] = row
    summarize('project: dias>=10, plain sum', base, lab0)
    for k in (15, 20, 25):
        d = m[m['dias'] >= k].copy(); thr_k = float(np.percentile(d.loc[d['part'] == 'train', 'precip'], 60))
        summarize(f'dias>={k}, plain sum (P60 recomputed {thr_k:.1f})', d, label(d, thr_k), lab0, base.index)
    d = m[m['dias'] >= m['dim']].copy(); thr_k = float(np.percentile(d.loc[d['part'] == 'train', 'precip'], 60))
    summarize(f'complete months only (P60 recomputed {thr_k:.1f})', d, label(d, thr_k), lab0, base.index)
    d = base.copy(); d['precip'] = d['precip'] / d['dias'] * d['dim']
    thr_c = float(np.percentile(d.loc[d['part'] == 'train', 'precip'], 60))
    summarize(f'dias>=10, coverage-corrected sum (P60 {thr_c:.1f})', d, label(d, thr_c), lab0, base.index)
    res['label_sensitivity'] = sens
    # months whose label is "fragile": partial coverage AND label could change if the missing days had average rain
    d = base.copy(); d['corr'] = d['precip'] / d['dias'] * d['dim']
    d['lab_plain'] = (d['precip'] > thr).astype(int); d['lab_corr'] = (d['corr'] > thr).astype(int)
    fr = d[(d['lab_plain'] != d['lab_corr'])]
    res['fragile_labels_plain_vs_corrected'] = {'n': int(len(fr)), 'by_partition': fr.groupby('part').size().to_dict()}
    # attach columns for downstream row-level sensitivity analysis (exp13)
    base['corr'] = base['precip'] / base['dias'] * base['dim']
    base['lab_project'] = (base['precip'] > thr).astype(int)
    base['lab_corr_same_thr'] = (base['corr'] > thr).astype(int)
    base['lab_corr_new_thr'] = (base['corr'] > thr_c).astype(int)
    if save_rows:
        base.to_csv(save_rows, index=False)
        print('saved monthly rows to', save_rows)

    # inner-join effect
    if precip_raw is not None:
        pr = precip_raw.copy(); mp = monthly(pr).set_index(['ciudad', 'fecha']); mj = m.set_index(['ciudad', 'fecha'])
        j = mp.join(mj, lsuffix='_precip_only', rsuffix='_joined', how='left')
        j['lost_days'] = j['dias_precip_only'] - j['dias_joined'].fillna(0)
        j['sum_diff'] = j['precip_precip_only'] - j['precip_joined'].fillna(0)
        res['inner_join'] = {'months_with_lost_days': int((j['lost_days'] > 0).sum()),
                             'total_lost_days': int(j['lost_days'].sum()),
                             'months_with_lost_precip_mm_gt_0': int((j['sum_diff'] > 0).sum()),
                             'total_lost_precip_mm': float(j['sum_diff'].sum()),
                             'months_label_would_change_with_precip_only_sum': int((((j['precip_precip_only'] > thr).astype(int)) != ((j['precip_joined'].fillna(0) > thr).astype(int)))[j['dias_precip_only'] >= 10].sum())}
    return res

def synth(seed=0):
    rng = np.random.default_rng(seed); rows = []
    for city in ['Alicante', 'Barcelona']:
        idx = pd.date_range('1990-01-01', '2012-12-31', freq='D')
        pr = np.where(rng.random(len(idx)) < 0.15, rng.gamma(1.0, 8.0, len(idx)), 0.0)
        df = pd.DataFrame({'fecha': idx, 'ciudad': city, 'precip': pr})
        drop = rng.random(len(idx)) < 0.08
        bad_month = df['fecha'].dt.month.isin([3]) & (df['fecha'].dt.year.isin([2001, 2002]))
        df.loc[drop | (bad_month & (rng.random(len(idx)) < 0.6)), 'precip'] = np.nan
        rows.append(df)
    return pd.concat(rows).set_index('fecha')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv'); ap.add_argument('--precip_raw'); ap.add_argument('--out', default='results_exp12_coverage.json')
    ap.add_argument('--save_rows', nargs='?', const='month_rows.csv', default=None, help='Save monthly table rows to CSV (default: month_rows.csv)')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        df = synth(); pr = None
    else:
        df = pd.read_csv(a.csv, index_col='fecha', parse_dates=['fecha'])
        pr = pd.read_csv(a.precip_raw, index_col='fecha', parse_dates=['fecha']) if a.precip_raw else None
    res = audit(df, pr, a.save_rows)
    import tempfile, os
    out_path = a.out if not a.selftest else os.path.join(tempfile.gettempdir(), 'exp12_selftest.json')
    json.dump(res, open(out_path, 'w'), indent=2, default=str)
    print(json.dumps({k: res[k] for k in ('n_station_months_before_filter', 'n_kept_dias_ge_10', 'kept_partial_total', 'P60_project', 'fragile_labels_plain_vs_corrected')}, indent=2))
    for k, v in res['label_sensitivity'].items():
        print(f"{k:58s} kept {v['n_kept']:5d} | flips vs project: {v.get('label_flips_vs_project','-')} (test {v.get('flips_test','-')}) | prev test {v['prev_test']:.3f}")
