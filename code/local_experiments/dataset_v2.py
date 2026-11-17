"""
dataset_v2.py - v2 monthly panel and v2 tree baselines (active only with the environment variable LOR_DATASET=v2)
================================================================================================================
Monthly aggregation (replaces trainonly_dataset._agregar_mensual for v2):
  * every calendar month of every station is kept (windows are calendar-contiguous; v1 dropped months with < 10
    joined days and then concatenated the remaining months);
  * label: monthly precipitation sum from the precipitation file only, defined when at most V2_MAX_MISSING days
    (default 3) of the month are missing; otherwise the month has no label (no window targets it);
  * precipitation lags: sum over valid days rescaled to the month length, defined with >= 10 valid days;
  * covariates: means over the days on which they were observed (>= 10 days), sums (sunshine, ET0) rescaled to
    the month length; gaps of up to V2_INTERP_MONTHS (default 2) months are interpolated per station, longer gaps
    leave the month without features;
  * ET0 (FAO-56) uses the observed wind: u2 = 0.748 u10 (FAO-56 eq. 47, anemometer at 10 m), not u2 = 2 m/s;
  * a window (target month t) is kept only if t has a label, month t-1 has all covariates and the 12 lags
    t-12..t-1 are defined ('win_ok').
Tree baselines: tuned Random Forest and XGBoost whose configuration is read from results/json/v2_tuned_params.json
(written by exp40_v2_tune.py); XGBoost stops early on the validation AUC.
"""
import json
import math
import os

import numpy as np
import pandas as pd

V2 = os.environ.get('LOR_DATASET', 'v1').lower() == 'v2'
MAX_MISSING = int(os.environ.get('V2_MAX_MISSING', 3))
INTERP_MONTHS = int(os.environ.get('V2_INTERP_MONTHS', 2))
MIN_DAYS = 10
U2_FACTOR = 4.87 / math.log(67.8 * 10 - 5.42)          # 0.748
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TUNED = os.path.join(ROOT, 'results', 'json', 'v2_tuned_params.json')
BASE = ['pmed', 't_med', 'dp_std', 't_rango', 't_min', 'veloc_ms', 'racha_ms_max', 'sol_sum', 'et0_sum']


def _et0_daily(df, tds):
    ok = df[['tmax', 'tmin', 'pmed', 'sol']].notna().all(1) & df['veloc_ms'].notna()
    out = np.full(len(df), np.nan)
    for i in np.where(ok.values)[0]:
        r = df.iloc[i]
        out[i] = tds.et0_fao56(r['tmax'], r['tmin'], r['pmed'], r['sol'], int(r['dia_ano']),
                               lat=tds.LATITUDES.get(r['ciudad'], 38.37), alt=tds.ALTITUDES.get(r['ciudad'], 7.0),
                               u2=U2_FACTOR * r['veloc_ms'])
    return out


def agregar_mensual_v2(df):
    import trainonly_dataset as tds
    if 'precip_valid' not in df.columns:
        raise SystemExit('LOR_DATASET=v2 needs the daily file written by code/etl_v2.py (no precip_valid column)')
    df = df.copy()
    df['pmed'] = (df['pmax'] + df['pmin']) / 2.0
    df['t_med'] = (df['tmax'] + df['tmin']) / 2.0
    df['t_rango'] = df['tmax'] - df['tmin']
    df['veloc_ms'] = df['veloc_kmh'] / 3.6
    df['racha_ms_max'] = df['racha_kmh'] / 3.6
    df['dia_ano'] = df.index.dayofyear
    df['et0'] = _et0_daily(df, tds)
    df['precip_valid'] = df['precip_valid'].astype(str).str.lower().isin(['true', '1'])
    g = df.groupby(['ciudad', pd.Grouper(freq='MS')])
    agg = g.agg(precip_sum=('precip', 'sum'), dias=('precip', 'count'),
                pmed=('pmed', 'mean'), n_pmed=('pmed', 'count'), dp_std=('pmed', 'std'),
                t_med=('t_med', 'mean'), n_t=('t_med', 'count'), t_rango=('t_rango', 'mean'), t_min=('tmin', 'mean'),
                veloc_ms=('veloc_ms', 'mean'), n_v=('veloc_ms', 'count'),
                racha_ms_max=('racha_ms_max', 'max'), n_r=('racha_ms_max', 'count'),
                sol_mean=('sol', 'mean'), n_s=('sol', 'count'), et0_mean=('et0', 'mean'), n_e=('et0', 'count'),
                precip_max_dia=('precip', 'max')).reset_index().set_index('fecha').sort_index()
    nd = agg.index.days_in_month.values
    agg['sol_sum'] = agg['sol_mean'] * nd
    agg['et0_sum'] = agg['et0_mean'] * nd
    for col, n in [('pmed', 'n_pmed'), ('dp_std', 'n_pmed'), ('t_med', 'n_t'), ('t_rango', 'n_t'), ('t_min', 'n_t'),
                   ('veloc_ms', 'n_v'), ('racha_ms_max', 'n_r'), ('sol_sum', 'n_s'), ('et0_sum', 'n_e')]:
        agg.loc[agg[n] < MIN_DAYS, col] = np.nan
    missing = nd - agg['dias'].values
    agg['label_ok'] = missing <= MAX_MISSING
    agg['precip_label'] = np.where(agg['label_ok'], agg['precip_sum'], np.nan)
    agg['precip_ok'] = agg['dias'] >= MIN_DAYS
    agg['precip'] = np.where(agg['precip_ok'], agg['precip_sum'] / agg['dias'].clip(lower=1) * nd, np.nan)
    for c in BASE:                                  # short covariate gaps only, per station, no extrapolation
        agg[c] = agg.groupby('ciudad')[c].transform(
            lambda s: s.interpolate(method='time', limit=INTERP_MONTHS, limit_area='inside'))
    agg['feat_ok'] = agg[BASE].notna().all(1)
    agg['win_ok'] = False
    col = agg.columns.get_loc('win_ok')
    for _, ix in agg.groupby('ciudad').indices.items():   # positions in date order = chronological per station
        s = agg.iloc[ix]
        f_prev = s['feat_ok'].shift(1, fill_value=False)
        lags = s['precip_ok'].astype(int).rolling(12, min_periods=12).sum().shift(1) == 12
        agg.iloc[ix, col] = (s['label_ok'] & f_prev & lags).values
    agg['win_ok'] = agg['win_ok'].astype(bool)
    return agg


def fill_for_scalers(agg, feature_cols):
    """Rows without features or lags never enter a window, but MinMax scalers are fitted on all training months:
    give those cells the training median so that they do not stretch the scales."""
    agg = agg.copy()
    tr = agg.index.year <= 1996
    for c in feature_cols:
        bad = ~agg['feat_ok'].values if c != 'precip' else ~agg['precip_ok'].values
        med = float(np.nanmedian(agg.loc[tr & ~bad, c])) if (tr & ~bad).any() else 0.0
        agg.loc[bad, c] = med
    agg['precip_label_f'] = agg['precip_label'].fillna(-1.0)
    return agg


def construir_features_v2(df_diario, anomalia_fn, fix_et0_verano=True):
    import trainonly_dataset as tds
    agg = agregar_mensual_v2(df_diario)
    keep = agg[['label_ok', 'precip_ok', 'feat_ok', 'win_ok', 'precip_label', 'dias']].copy()
    base = agg.drop(columns=['label_ok', 'precip_ok', 'feat_ok', 'win_ok', 'precip_label'])
    base['precip'] = base['precip'].fillna(0.0)           # only for derived features of rows that are never used
    out = tds._indices_sinopticos(base, anomalia_fn, fix_et0_verano=True)
    # thermal memory: a JJA total needs all three months (v1 summed whatever months were present)
    jja = base.index.month.isin([6, 7, 8])
    tot = (base[jja].assign(_y=base.index[jja].year).groupby(['ciudad', '_y'])['et0_sum'].sum(min_count=3).to_dict())
    ref = base.index.year - (base.index.month < 9).astype(int)
    out['et0_verano'] = [tot.get((c, y), np.nan) for c, y in zip(base['ciudad'], ref)]
    col = out.columns.get_loc('et0_verano')
    for _, pos in out.groupby('ciudad').indices.items():
        out.iloc[pos, col] = out['et0_verano'].iloc[pos].interpolate(method='time').ffill().bfill().values
    out['et0_verano_prev'] = out['et0_verano']
    out['potencial_convectivo'] = out['et0_verano_prev'] * base['dp_std'].values
    out = tds._flag_ACP(out)
    for c in keep.columns:
        out[c] = keep[c].values
    out['precip'] = agg['precip'].values
    return fill_for_scalers(out, ['precip'] + tds.FEATURES_FISICAS)


# ------------------------------------------------------------------ tuned trees
def tuned(name):
    if not os.path.exists(TUNED):
        raise SystemExit(f'{TUNED} not found: run exp40_v2_tune.py first')
    return json.load(open(TUNED))[name]['params']


def rf_v2(seed):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(**tuned('RF'), class_weight='balanced', random_state=seed, n_jobs=-1)


def xgb_fit_v2(seed, X_tr, y_tr, X_va, y_va):
    from xgboost import XGBClassifier
    p = dict(tuned('XGB'))
    n_neg, n_pos = (y_tr == 0).sum(), (y_tr == 1).sum()
    m = XGBClassifier(**p, n_estimators=2000, early_stopping_rounds=50, eval_metric='auc',
                      scale_pos_weight=n_neg / max(n_pos, 1), random_state=seed, n_jobs=-1, tree_method='hist')
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return m
