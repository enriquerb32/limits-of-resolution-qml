#!/usr/bin/env python3
"""
exp33_seas5_reference.py - an operational dynamical reference: ECMWF SEAS5 (C3S) in the same evaluation protocol
==================================================================================================================
Question: do the ML / QML models add discrimination or probabilistic skill to an operational dynamical seasonal
forecast at these stations, and can the protocol resolve it?

Data: C3S Climate Data Store, dataset 'seasonal-monthly-single-levels', ECMWF SEAS5 (system 51, a relabelling of
system 5 used for the current C3S provision), total precipitation, monthly means of the individual members, 1 x 1 deg.
Hindcasts 1981-2016 have 25 members, real-time forecasts (2017-) have 51; members 0-24 are used throughout so that
the ensemble size, and hence the probability resolution, is the same in every year.

Lead time and information set. The station models predict month M from data up to the end of month M-1.
  lead 1: SEAS5 started on the 1st of M (initial conditions up to the start of M) -> same information cut-off as the
          station models. Note that the forecast is released around the 5th of M, so operationally it arrives a few
          days into the target month.
  lead 2: SEAS5 started on the 1st of M-1 -> available before M begins (stricter for SEAS5).

Scores built from the 25 members interpolated (bilinear) to each station:
  SEAS5_RAW  fraction of members above the pooled 28.1 mm threshold (no bias correction; shows the 1-deg grid bias)
  SEAS5_QM   fraction of members above the model's own quantile at level 1 - p_clim, where p_clim is the training
             exceedance frequency of the station x calendar month (the climatology baseline of the paper) and the
             model quantile is taken from the hindcasts of 1981-1996 (training years only) for the same station,
             calendar month and lead. Probabilities are (k + 0.5) / (25 + 1) to keep logits finite.
  SEAS5_EM   percentile of the ensemble mean among the calibration-year ensemble means of the same station,
             calendar month and lead: an anomaly-only score with no base-rate information (the counterpart of the
             station x month label; with the pooled label compare it through the blends, not alone)
  *_PLATT    Platt map fitted on the validation partition (1997-2007), applied to the test partition.
  BLEND_<M>  per seed: logistic regression on the validation partition with inputs logit p_M and logit p_SEAS5_QM
             (M = RF_2273, XGB, PureLSTM, CTRL_LINEAR4D, LSTM_QNN, and CLIM for climatology + SEAS5).

Paired comparisons (same procedures as the paper): DeLong, calendar-year block and years x seeds bootstraps, MDE,
cluster-robust SE / iid ratio and the 12 phase-shifted pseudo-years; Brier scores with a year-block interval of the
difference against the Platt-recalibrated climatology.

Steps (run from the repository root)
  1. pip install "cdsapi>=0.7.4" xarray cfgrib eccodes scipy     (Windows: same; eccodes ships binary wheels)
     create an ECMWF account, put url/key in ~/.cdsapirc (Windows: %USERPROFILE%\\.cdsapirc) and ACCEPT THE LICENCE
     of the dataset once on its CDS web page ("Licence to use Copernicus Products").
  2. python code/local_experiments/exp33_seas5_reference.py --step download --dry-run   # prints the 46 requests
     python code/local_experiments/exp33_seas5_reference.py --step download             # ~1-3 h, mostly CDS queue
  3. python code/local_experiments/exp33_seas5_reference.py --step extract              # ~1 min
  4. python code/local_experiments/exp33_seas5_reference.py --step score                # ~1 min (needs exp28 npz)
  5. python code/local_experiments/exp33_seas5_reference.py --step analyze              # ~5 min
  4-5 again with --label stationmonth (skill beyond the seasonal and station base rates)
Requirement for step 4: trainonly_dataset.preparar_dataset must store the validation row metadata (one line, see
README: ret.meta_va = {'ciudad': ciudad_all[s1:s2], 'fecha': fechas_all[s1:s2]}).

Outputs
  data/seas5/seas5_tp_<year>.grib, data/seas5/seas5_station_members.csv
  results/json/results_exp33_seas5_<label>_probs.npz   keys '<NAME>_val', '<NAME>_probs', y_va, y_te
  results/local_exp33_seas5_<label>.json
"""
import argparse
import calendar
import datetime as dt
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(ROOT, 'code'))
sys.path.insert(2, os.path.join(ROOT, 'src'))

DATA = os.path.join(ROOT, 'data', 'seas5')
MEMBERS_CSV = os.path.join(DATA, 'seas5_station_members.csv')
EXP28_NPZ = os.path.join(ROOT, 'results', 'json', 'results_exp28_{label}_probs.npz')
OUT_NPZ = os.path.join(ROOT, 'results', 'json', 'results_exp33_seas5_{label}_probs.npz')
OUT_JSON = os.path.join(ROOT, 'results', 'local_exp33_seas5_{label}.json')
# station coordinates (decimal degrees; check against the AEMET station metadata of Table 1)
STATIONS = {'Alicante': (38.37, -0.49), 'Barcelona': (41.29, 2.07), 'San_Javier': (37.79, -0.80),
            'Valencia': (39.48, -0.37)}
AREA = [43, -2, 36, 4]               # N, W, S, E: every station has its four surrounding 1-deg grid points
N_MEM = 25
ML_MODELS = ['RF_2273', 'XGB', 'PureLSTM', 'CTRL_LINEAR4D', 'LSTM_QNN']
N_BOOT, N_PERM, SEED = 2000, 4000, 2025


# ====================================================================== 1. download
def request(years, months, system):
    return {'originating_centre': 'ecmwf', 'system': system, 'variable': ['total_precipitation'],
            'product_type': ['monthly_mean'], 'year': [str(y) for y in years], 'month': [f'{m:02d}' for m in months],
            'leadtime_month': ['1', '2'], 'area': AREA, 'data_format': 'grib'}


def covered_years():
    """Years already on disk, from seas5_tp_<y>.grib or seas5_tp_<y0>_<y1>.grib (the current year is never 'covered')."""
    import re
    out = set()
    for f in glob.glob(os.path.join(DATA, 'seas5_tp_*.grib')):
        m = re.search(r'seas5_tp_(\d{4})(?:_(\d{4}))?\.grib$', f)
        if m and os.path.getsize(f) > 0:
            y0 = int(m.group(1)); y1 = int(m.group(2) or y0)
            out.update(range(y0, y1 + 1))
    return out


def batches(years, size):
    """Consecutive runs of missing years, cut into blocks of at most `size` years."""
    out, cur = [], []
    for y in sorted(years):
        if cur and (y != cur[-1] + 1 or len(cur) == size):
            out.append(cur); cur = []
        cur.append(y)
    return out + ([cur] if cur else [])


def retrieve(c, years, months, target, system):
    for sysv in [system, '5'] if system != '5' else ['5']:
        try:
            c.retrieve('seasonal-monthly-single-levels', request(years, months, sysv), target)
            return True
        except Exception as e:  # noqa: BLE001
            print(f'{years[0]}-{years[-1]} months {months[0]}-{months[-1]} system {sysv}: {str(e)[:300]}')
    return False


def step_download(a):
    """One CDS request per block of `--batch` consecutive missing years (each request waits in the queue once)."""
    os.makedirs(DATA, exist_ok=True)
    today = dt.date.today()
    want = range(a.years[0], a.years[1] + 1)
    have = covered_years()
    past = [y for y in want if y < today.year and y not in have]
    blocks = batches(past, a.batch)
    cur = today.year if today.year in want else None
    print(f'years on disk: {len(have)}; past years to download: {len(past)} in {len(blocks)} request(s)'
          + (f'; plus {cur} (months 1-{today.month})' if cur else ''))
    if a.dry_run:
        for b in blocks:
            print('seasonal-monthly-single-levels', json.dumps(request(b, range(1, 13), a.system)))
        if cur:
            print('seasonal-monthly-single-levels', json.dumps(request([cur], range(1, today.month + 1), a.system)))
        return
    import cdsapi
    c = cdsapi.Client()
    for b in blocks:
        target = os.path.join(DATA, f'seas5_tp_{b[0]}_{b[-1]}.grib' if len(b) > 1 else f'seas5_tp_{b[0]}.grib')
        if not retrieve(c, b, list(range(1, 13)), target, a.system):
            raise SystemExit(f'{b[0]}-{b[-1]}: download failed (see the CDS error above); '
                             f'if the request is too large, rerun with a smaller --batch')
        print(f'{b[0]}-{b[-1]}: done', flush=True)
    if cur:                                   # the latest start may not be published yet: retry one month shorter
        target = os.path.join(DATA, f'seas5_tp_{cur}.grib')
        for last in (today.month, today.month - 1):
            if last >= 1 and retrieve(c, [cur], list(range(1, last + 1)), target, a.system):
                print(f'{cur}: done (months 1-{last})', flush=True)
                break
        else:
            raise SystemExit(f'{cur}: download failed')


# ====================================================================== 2. extract
def _open(path):
    import xarray as xr
    if path.endswith('.grib'):
        return xr.open_dataset(path, engine='cfgrib',
                               backend_kwargs={'time_dims': ('forecastMonth', 'time'), 'indexpath': ''})
    return xr.open_dataset(path)


def _months_add(ts, k):
    y, m = ts.year + (ts.month - 1 + k) // 12, (ts.month - 1 + k) % 12 + 1
    return pd.Timestamp(year=y, month=m, day=1)


def extract_dataset(ds):
    """Return a long DataFrame (ciudad, init, lead, valid, member, mm) from one opened SEAS5 file."""
    var = 'tprate' if 'tprate' in ds else list(ds.data_vars)[0]
    if var != 'tprate':
        print(f'WARNING: variable {var!r} (expected tprate, m s-1); check units')
    da = ds[var]
    ren = {k: v for k, v in {'forecast_reference_time': 'time', 'lat': 'latitude', 'lon': 'longitude',
                             'realization': 'number'}.items() if k in da.dims}
    da = da.rename(ren)
    if 'forecastMonth' not in da.dims:
        raise SystemExit('no forecastMonth dimension: open GRIB files with time_dims=("forecastMonth", "time")')
    if 'number' in da.dims:
        da = da.sel(number=[n for n in da['number'].values if int(n) < N_MEM])
    lon = da['longitude'].values
    rows = []
    for st, (la, lo) in STATIONS.items():
        lo_ = lo % 360 if lon.max() > 180 else lo
        p = da.interp(latitude=la, longitude=lo_)
        df = p.to_dataframe(name='rate').reset_index()[['time', 'forecastMonth', 'number', 'rate']]
        df['ciudad'] = st
        rows.append(df)
    df = pd.concat(rows).dropna(subset=['rate'])
    df['init'] = pd.to_datetime(df['time']).dt.to_period('M').dt.to_timestamp()
    df['lead'] = df['forecastMonth'].astype(int)
    df['valid'] = [_months_add(t, k - 1) for t, k in zip(df['init'], df['lead'])]
    days = np.array([calendar.monthrange(v.year, v.month)[1] for v in df['valid']])
    df['mm'] = df['rate'].values * 1000.0 * 86400.0 * days          # m s-1 -> mm per month
    df['member'] = df['number'].astype(int)
    return df[['ciudad', 'init', 'lead', 'valid', 'member', 'mm']]


def step_extract(a):
    files = sorted(glob.glob(os.path.join(DATA, 'seas5_tp_*.grib')) + glob.glob(os.path.join(DATA, 'seas5_tp_*.nc')))
    if not files:
        raise SystemExit(f'no SEAS5 files in {DATA}; run --step download first')
    df = pd.concat([extract_dataset(_open(f)) for f in files]).drop_duplicates(['ciudad', 'init', 'lead', 'member'])
    df.to_csv(MEMBERS_CSV, index=False)
    n = df.groupby(['ciudad', 'init', 'lead']).size()
    print(f'wrote {MEMBERS_CSV}: {len(df):,} rows, {df.init.min():%Y-%m} .. {df.init.max():%Y-%m}; '
          f'members per start: min {n.min()}, max {n.max()}')


# ====================================================================== 3. score
def logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def platt(p_va, y_va, p_te):
    m = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p_va)[:, None], y_va)
    return m.predict_proba(logit(p_va)[:, None])[:, 1], m.predict_proba(logit(p_te)[:, None])[:, 1]


def seas5_scores(members, meta, clim, lead, calib=(1981, 1996), thr_mm=28.1):
    """thr_mm=None skips SEAS5_RAW (station x month label, where a pooled mm threshold is meaningless)."""
    """members: long DataFrame; meta: DataFrame (ciudad, fecha) of the rows to score; clim: p_clim of those rows."""
    m = members[members['lead'] == lead]
    piv = m.pivot_table(index=['ciudad', 'valid'], columns='member', values='mm')
    if piv.shape[1] < N_MEM:
        raise SystemExit(f'lead {lead}: only {piv.shape[1]} members')
    piv = piv.iloc[:, :N_MEM]
    cal = piv[(piv.index.get_level_values('valid').year >= calib[0]) &
              (piv.index.get_level_values('valid').year <= calib[1])]
    cal_month = cal.index.get_level_values('valid').month
    key = list(zip(meta['ciudad'], pd.to_datetime(meta['fecha']).dt.to_period('M').dt.to_timestamp()))
    missing = [k for k in key if k not in piv.index]
    if missing:
        raise SystemExit(f'lead {lead}: SEAS5 missing for {len(missing)} rows, e.g. {missing[:3]}')
    X = piv.loc[key].values                                            # (N, 25)
    raw = (X > thr_mm).sum(1) if thr_mm is not None else None
    qm, em = np.zeros(len(key)), np.zeros(len(key))
    for i, ((st, ts), pc) in enumerate(zip(key, clim)):
        c = cal[(cal.index.get_level_values('ciudad') == st) & (cal_month == ts.month)].values
        thr = np.quantile(c.ravel(), 1 - pc)
        qm[i] = (X[i] > thr).sum()
        cm = c.mean(1)
        em[i] = (np.sum(cm < X[i].mean()) + 0.5 * np.sum(cm == X[i].mean())) / cm.size
    f = lambda k: (k + 0.5) / (N_MEM + 1)                               # noqa: E731
    out = {'QM': f(qm), 'EM': np.clip(em, 0.5 / 17, 1 - 0.5 / 17)}
    if thr_mm is not None:
        out['RAW'] = f(raw)
    return out


def build_scores(members, meta_va, meta_te, y_va, y_te, clim_va, clim_te, ml, leads, thr_mm=28.1):
    """ml: dict name -> (P_val (S, n_va), P_te (S, n_te)). Returns the dict of arrays saved to the npz."""
    y_va, y_te = np.asarray(y_va).astype(int), np.asarray(y_te).astype(int)
    clim_va, clim_te = np.asarray(clim_va, float), np.asarray(clim_te, float)
    out = {'y_va': y_va, 'y_te': y_te, 'CLIM_val': clim_va[None], 'CLIM_probs': clim_te[None]}
    cva, cte = platt(clim_va, y_va, clim_te)
    out.update(CLIM_PLATT_val=cva[None], CLIM_PLATT_probs=cte[None])
    for name, (pv, pt) in ml.items():
        out[f'{name}_val'], out[f'{name}_probs'] = pv, pt
    for L in leads:
        sv = seas5_scores(members, meta_va, clim_va, L, thr_mm=thr_mm)
        st = seas5_scores(members, meta_te, clim_te, L, thr_mm=thr_mm)
        for k in sv:
            out[f'SEAS5_{k}_L{L}_val'], out[f'SEAS5_{k}_L{L}_probs'] = sv[k][None], st[k][None]
            pv, pt = platt(sv[k], y_va, st[k])
            out[f'SEAS5_{k}_L{L}_PLATT_val'], out[f'SEAS5_{k}_L{L}_PLATT_probs'] = pv[None], pt[None]
        s_va, s_te = logit(sv['QM']), logit(st['QM'])
        for name, (pv, pt) in list(ml.items()) + [('CLIM', (clim_va[None], clim_te[None]))]:
            BV, BT = [], []
            for s in range(pv.shape[0]):
                Xv = np.c_[logit(pv[s]), s_va]
                Xt = np.c_[logit(pt[s]), s_te]
                lr = LogisticRegression(C=1e6, max_iter=1000).fit(Xv, y_va)
                BV.append(lr.predict_proba(Xv)[:, 1]); BT.append(lr.predict_proba(Xt)[:, 1])
            out[f'BLEND_{name}_L{L}_val'], out[f'BLEND_{name}_L{L}_probs'] = np.array(BV), np.array(BT)
    return out


def step_score(a):
    from exp28_labels_recalibration import load_dataset        # same datasets as the exp28 refits
    ds = load_dataset(a.label)
    if not hasattr(ds, 'meta_va'):
        raise SystemExit("add  ret.meta_va = {'ciudad': ciudad_all[s1:s2], 'fecha': fechas_all[s1:s2]}  next to "
                         "ret.meta_te in code/local_experiments/trainonly_dataset.py (validation row metadata)")
    z = np.load(a.exp28_npz.format(label=a.label))
    y_va, y_te = ds.y_va.numpy().astype(int), ds.y_te.numpy().astype(int)
    assert np.array_equal(z['y_va'].astype(int), y_va) and np.array_equal(z['y_te'].astype(int), y_te), \
        'exp28 npz and dataset labels differ'
    ml = {m: (z[f'{m}_val'], z[f'{m}_probs']) for m in ML_MODELS if f'{m}_val' in z.files}
    members = pd.read_csv(MEMBERS_CSV, parse_dates=['init', 'valid'])
    meta_va = pd.DataFrame({'ciudad': ds.meta_va['ciudad'], 'fecha': pd.to_datetime(ds.meta_va['fecha'])})
    meta_te = pd.DataFrame({'ciudad': ds.meta_te['ciudad'], 'fecha': pd.to_datetime(ds.meta_te['fecha'])})
    out = build_scores(members, meta_va, meta_te, y_va, y_te, z['clim_val'], z['clim_te'], ml, a.leads,
                       thr_mm=28.1 if a.label == 'orig' else None)
    npz = OUT_NPZ.format(label=a.label)
    np.savez_compressed(npz, **out)
    meta_te.assign(y_te=y_te).to_csv(npz.replace('_probs.npz', '_test_meta.csv'), index=False)
    print(f'wrote {npz} ({len([k for k in out if k.endswith("_probs")])} score matrices)')
    for k in sorted(k for k in out if k.startswith('SEAS5') and k.endswith('_probs') and 'PLATT' not in k):
        print(f'  {k[:-6]:16s} AUC {roc_auc_score(y_te, out[k][0]):.4f}')


# ====================================================================== 4. analyze
def _import_quiet(name):
    """Import a repository module with an empty command line, so that a module that parses arguments at import
    time (instead of under `if __name__ == '__main__'`) cannot read this script's arguments."""
    import importlib
    saved = sys.argv[:]
    sys.argv = [saved[0]]
    try:
        return importlib.import_module(name)
    finally:
        sys.argv = saved


def influence(y, pa, pb):
    """Paired DeLong influence values of the seed-averaged structural components (as in exp25)."""
    dc = _import_quiet('qml_resolution').delong_components
    ca, cb = dc(y, pa).seed_mean(), dc(y, pb).seed_mean()
    delta = float(ca.auc[0] - cb.auc[0])
    psi = np.zeros(y.size)
    psi[y] = (ca.v10[0] - cb.v10[0] - delta) / y.sum()
    psi[~y] = (ca.v01[0] - cb.v01[0] - delta) / (~y).sum()
    return delta, psi


def cluster_var(psi, groups, levels):
    k = len(levels)
    return k / (k - 1) * sum(psi[groups == u].sum() ** 2 for u in levels)


def pair_stats(y, pa, pb, years, ym, rng, n_boot, n_perm):
    compare = _import_quiet('exp10_paired_ci').compare
    r = compare(y, pa, pb, years, n_boot, rng)
    delta, psi = influence(y.astype(bool), pa, pb)
    levels = np.unique(years)
    se_iid = float(np.sqrt((psi ** 2).sum()))
    se_cl = float(np.sqrt(cluster_var(psi, years, levels)))
    null = np.array([np.sqrt(cluster_var(psi, rng.permutation(years), levels)) / se_iid for _ in range(n_perm)])
    n_ok = 0
    for start in range(1, 13):
        _, inv = np.unique((ym - start) // 12, return_inverse=True)
        s = np.bincount(inv, weights=psi); k = s.size
        se = np.sqrt(k / (k - 1) * (s ** 2).sum())
        n_ok += abs(delta) > student_t.ppf(0.975, k - 1) * se
    r.update(se_iid=se_iid, se_cluster=se_cl, ratio=se_cl / se_iid, perm_ratio_q025=float(np.quantile(null, .025)),
             perm_ratio_q975=float(np.quantile(null, .975)), phases_excluding_zero=int(n_ok))
    return r


def brier_block(y, P, ref, years, rng, n_boot):
    """Mean Brier score over seeds, and year-block 95 % interval of (BS_model - BS_ref)."""
    bs = ((P - y) ** 2).mean(0)                                  # seed-mean Brier contribution per row
    d = bs - (ref - y) ** 2
    lv = np.unique(years); idx = [np.where(years == u)[0] for u in lv]
    reps = []
    for _ in range(n_boot):
        rows = np.concatenate([idx[i] for i in rng.integers(0, lv.size, lv.size)])
        reps.append(d[rows].mean())
    seeds = ((P - y) ** 2).mean(1)
    bs_ref = float(((ref - y) ** 2).mean())
    return dict(brier=float(bs.mean()), bss=float(1 - bs.mean() / bs_ref), bss_seed_min=float(1 - seeds.max() / bs_ref),
                bss_seed_max=float(1 - seeds.min() / bs_ref), n_seeds_above=int((seeds < bs_ref).sum()),
                d_brier=float(d.mean()), d_brier_block_ci=[float(np.quantile(reps, .025)), float(np.quantile(reps, .975))])


def default_pairs(L):
    s = f'SEAS5_QM_L{L}'
    return ([f'{s}-CLIM', f'SEAS5_EM_L{L}-CLIM', f'RF_2273-{s}', f'LSTM_QNN-{s}'] +
            [f'BLEND_{m}_L{L}-BLEND_CLIM_L{L}' for m in ML_MODELS] +
            [f'BLEND_RF_2273_L{L}-RF_2273', f'BLEND_LSTM_QNN_L{L}-LSTM_QNN',
             f'BLEND_LSTM_QNN_L{L}-BLEND_CTRL_LINEAR4D_L{L}', f'BLEND_RF_2273_L{L}-BLEND_LSTM_QNN_L{L}'])


def analyze(z, meta, leads, n_boot=N_BOOT, n_perm=N_PERM, verbose=True):
    y = np.asarray(z['y_te']).astype(int)
    years = meta['fecha'].dt.year.values
    ym = years * 12 + meta['fecha'].dt.month.values
    rng = np.random.default_rng(SEED)
    names = sorted(k[:-6] for k in z if k.endswith('_probs'))
    res = {'n_test': int(y.size), 'K_years': int(np.unique(years).size), 'auc': {}, 'pairs': {}, 'brier': {}}
    for n in names:
        P = np.atleast_2d(z[n + '_probs'])
        a = np.array([roc_auc_score(y, p) for p in P])
        res['auc'][n] = dict(mean=float(a.mean()), sd=float(a.std(ddof=1)) if len(a) > 1 else None, S=int(len(a)))
    for L in leads:
        for pair in default_pairs(L):
            pa, pb = pair.split('-', 1)
            if pa + '_probs' not in z or pb + '_probs' not in z:
                continue
            r = pair_stats(y, np.atleast_2d(z[pa + '_probs']), np.atleast_2d(z[pb + '_probs']), years, ym, rng,
                           n_boot, n_perm)
            res['pairs'][pair] = r
            if verbose:
                print(f'{pair:42s} d={r["delta"]:+.4f} DeLong p={r["p_delong"]:.3f} '
                      f'block [{r["ci_block"][0]:+.3f},{r["ci_block"][1]:+.3f}] two-way [{r["ci_two_way"][0]:+.3f},'
                      f'{r["ci_two_way"][1]:+.3f}] MDE {r["mde_delong"]:.3f} ratio {r["ratio"]:.2f} '
                      f'phases {r["phases_excluding_zero"]}/12', flush=True)
    ref = np.atleast_2d(z['CLIM_PLATT_probs'])[0]
    for n in names:
        if n.endswith('PLATT') or n.startswith('BLEND') or n == 'CLIM':
            res['brier'][n] = brier_block(y, np.atleast_2d(z[n + '_probs']), ref, years, rng, n_boot)
    if verbose:
        print('\nAUC (seed mean):')
        for n, v in res['auc'].items():
            print(f'  {n:30s} {v["mean"]:.4f}' + (f' (SD {v["sd"]:.4f})' if v['sd'] else ''))
        print('\nBrier against the Platt-recalibrated climatology (BS %.4f):' % float(((ref - y) ** 2).mean()))
        for n, v in res['brier'].items():
            print(f'  {n:30s} BS {v["brier"]:.4f}  BSS {v["bss"]:+.4f}  dBS block CI '
                  f'[{v["d_brier_block_ci"][0]:+.4f},{v["d_brier_block_ci"][1]:+.4f}]')
    return res


def step_analyze(a):
    npz = OUT_NPZ.format(label=a.label)
    z = dict(np.load(npz))
    meta = pd.read_csv(npz.replace('_probs.npz', '_test_meta.csv'), parse_dates=['fecha'])
    res = analyze(z, meta, a.leads, a.boot, a.perm)
    res['label'] = a.label
    out = OUT_JSON.format(label=a.label)
    json.dump(res, open(out, 'w'), indent=2)
    print('wrote', out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--step', choices=['download', 'extract', 'score', 'analyze'], required=True)
    ap.add_argument('--years', type=int, nargs=2, default=[1981, dt.date.today().year])
    ap.add_argument('--system', default='51')
    ap.add_argument('--batch', type=int, default=10, help='years per CDS request')
    ap.add_argument('--leads', type=int, nargs='+', default=[1, 2])
    ap.add_argument('--label', choices=['orig', 'stationmonth'], default='orig')
    ap.add_argument('--exp28-npz', default=EXP28_NPZ, help='path; {label} is replaced')
    ap.add_argument('--boot', type=int, default=N_BOOT)
    ap.add_argument('--perm', type=int, default=N_PERM)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    {'download': step_download, 'extract': step_extract, 'score': step_score, 'analyze': step_analyze}[a.step](a)


if __name__ == '__main__':
    main()
