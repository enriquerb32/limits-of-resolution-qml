"""
trainonly_dataset.py  -  Self-contained dataset builder extracted verbatim from the project's train-only rerun script
(results_rerun_train_only_canonical.json pipeline).  Build the TRAIN-ONLY dataset directly from the daily CSV:

    from trainonly_dataset import build_dataset
    agg, ds = build_dataset("datos_limpios_multiestacion.csv", train_only=True)

train_only=True  -> monthly anomalies standardized with year<=1996 statistics (no leakage)
train_only=False -> canonical full-record standardization (the pipeline of the hybrids so far)
Everything else (windows, scalers fitted on train, thresholds from train, split 1938-1996 / 1997-2007 / 2008-2026) is
unchanged.  CHECK after building: len(ds.X_tr)==2273, len(ds.X_va)==514, len(ds.X_te)==528, ds.umbral ~ 28.1.
"""
import math, sys, os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

RAW_CSV_PATH = "datos_limpios_multiestacion.csv"
OUTPUT_JSON_PATH = "results_rerun_train_only_canonical.json"
OUTPUT_NPZ_PATH = "results_rerun_train_only_canonical_probs.npz"

# Execution controls
# If GATE 0 detects a rebuild mismatch (U-9), it stops by default as required.
# Set BYPASS_GATE0_STOP = True if you wish to log the U-9 report and continue to GATE 1 and TRAIN-ONLY RUN.
STOP_ON_GATE0_MISMATCH = True

# Canonical constants
WINDOW_SIZE = 12
ANIO_CORTE_TRAIN = 1996
CANONICAL_P60 = 28.1
CANONICAL_P90 = 86.32
FEATURES_FISICAS = [
    'pmed', 'dp_std', 't_rango', 'et0_sum', 'et0_verano_prev',
    'et0_lag1m', 'et0_lag3m', 'iap', 'iir', 'ia_racha',
    'veloc_ms', 'ib', 't_anomaly',
    'choque_clausius',
    'potencial_convectivo',
    'fao_evapo_deficit',
    'vuln_costera'
]
LATITUDES = {'Barcelona': 41.38, 'Valencia': 39.46, 'Alicante': 38.37, 'San_Javier': 37.77}
ALTITUDES = {'Barcelona': 4.0, 'Valencia': 11.0, 'Alicante': 7.0, 'San_Javier': 4.0}
SOL_MAX_MENSUAL = {
    1: 9.5,  2: 10.5, 3: 11.8, 4: 13.3,
    5: 14.2, 6: 14.4, 7: 14.2, 8: 13.2,
    9: 11.8, 10: 10.3, 11: 9.6, 12: 9.4,
}

# Ground truth reference AUCs from 10_multisemilla_clasico.json
REF_10_MULTISEMILLA = {
    "Random Forest": [
        0.6836131403085038, 0.6861469616370560, 0.6872731044497460, 0.6834945989597996,
        0.6839687643546164, 0.6899106494584142, 0.6876583638330346, 0.6836131403085038,
        0.6842799353949650, 0.6851838131788345, 0.6854949842191830, 0.6809904129684236,
        0.6826499918502822, 0.6830056158963950, 0.6896291137552417
    ],
    "XGBoost": [
        0.6515032524782551, 0.6582304740172180, 0.6696697141671730, 0.6593121638241438,
        0.6622905152103368, 0.6603197652881295, 0.6567338894898277, 0.6581267503371019,
        0.6561708180834827, 0.6606902070028302, 0.6579637559826338, 0.6545260568702120,
        0.6548964985849127, 0.6465837865070310, 0.6558596470431343
    ],
    "LSTM pura": [
        0.6506882807059138, 0.6697734378472892, 0.6722331708329011, 0.6671655281757969,
        0.6837168639886201, 0.6631351223198543, 0.6559781883918383, 0.6615940847866998,
        0.6575784965993450, 0.6618459851526961, 0.6775823491931781, 0.6640093647665477,
        0.6643205358068962, 0.6580378443255739, 0.6772859958214176
    ]
}


# ------------------------------------------------------------------------------
@dataclass
class DatasetResult:
    X_all: np.ndarray
    y_all: np.ndarray
    y_all_ext: np.ndarray
    meses_all: np.ndarray
    dana_all: np.ndarray
    X_tr: torch.Tensor
    y_tr: torch.Tensor
    X_va: torch.Tensor
    y_va: torch.Tensor
    X_te: torch.Tensor
    y_te: torch.Tensor
    y_te_ext: torch.Tensor
    s1: int
    s2: int
    X_sk: np.ndarray
    y_sk: np.ndarray
    in_dim: int
    umbral: float
    umbral_ext: float
    m_tr: np.ndarray
    m_va: np.ndarray
    m_te: np.ndarray
    dana_te: np.ndarray
    precip_media_mes: Dict[int, float]
    scalers: Dict[str, MinMaxScaler]
    t_anomaly_mes: Dict[int, float]
    feature_names: List[str] = field(default_factory=list)

# Register shim so torch.load unpickles DatasetResult even without features.py on disk
import types
if 'features' not in sys.modules:
    mod_features = types.ModuleType('features')
    mod_features.DatasetResult = DatasetResult
    sys.modules['features'] = mod_features

# ------------------------------------------------------------------------------
def et0_fao56(tmax: float, tmin: float, pres_hpa: float,
              sol: float, dia_ano: int, lat: float, alt: float, u2: float = 2.0) -> float:
    t_med = (tmax + tmin) / 2.0
    pres_kpa = pres_hpa * 0.1
    e_tmed = 0.6108 * math.exp(17.27 * t_med / (t_med + 237.3))
    delta = 4098 * e_tmed / (t_med + 237.3) ** 2
    gamma = 0.000665 * pres_kpa
    e_tmax = 0.6108 * math.exp(17.27 * tmax / (tmax + 237.3))
    e_tmin = 0.6108 * math.exp(17.27 * tmin / (tmin + 237.3))
    es = (e_tmax + e_tmin) / 2.0
    ea = e_tmin
    lat_r = math.radians(lat)
    dr = 1 + 0.033 * math.cos(2 * math.pi * dia_ano / 365)
    decl = 0.409 * math.sin(2 * math.pi * dia_ano / 365 - 1.39)
    arg = max(-1.0, min(1.0, -math.tan(lat_r) * math.tan(decl)))
    ws = math.acos(arg)
    Ra = ((24 * 60 / math.pi) * 0.0820 * dr * (ws * math.sin(lat_r) * math.sin(decl) + math.sin(ws) * math.cos(lat_r) * math.cos(decl)))
    N_max = (24 / math.pi) * ws
    Rs = (0.25 + 0.50 * (sol / N_max)) * Ra if N_max > 0 else 0.0
    Rns = (1 - 0.23) * Rs
    Rso = (0.75 + 2e-5 * alt) * Ra
    fnu = max(0.05, min(1.0, 1.35 * (Rs / Rso) - 0.35)) if Rso > 0 else 0.05
    sigma = 4.903e-9
    Rnl = (sigma * ((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2 * (0.34 - 0.14 * math.sqrt(max(0.0, ea))) * fnu)
    Rn = Rns - Rnl
    num = 0.408 * delta * Rn + gamma * (900 / (t_med + 273)) * u2 * (es - ea)
    den = delta + gamma * (1 + 0.34 * u2)
    return max(0.0, num / den)

def _agregar_mensual(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['pmed'] = (df['pmax'] + df['pmin']) / 2.0
    df['t_med'] = (df['tmax'] + df['tmin']) / 2.0
    df['dp_dia'] = df.groupby('ciudad')['pmed'].diff().fillna(0.0)
    df['t_rango'] = df['tmax'] - df['tmin']
    df['veloc_ms'] = df['veloc_kmh'] / 3.6
    df['racha_ms_max'] = df['racha_kmh'] / 3.6
    df['dia_ano'] = df.index.map(lambda d: d.timetuple().tm_yday)
    df['et0'] = df.apply(lambda r: et0_fao56(
        r['tmax'], r['tmin'], r['pmed'], r['sol'], int(r['dia_ano']),
        lat=LATITUDES.get(r['ciudad'], 38.37), alt=ALTITUDES.get(r['ciudad'], 7.0)
    ), axis=1)
    p90_diario = df.groupby('ciudad')['precip'].transform(lambda x: x.quantile(0.90))
    df['es_extremo'] = (df['precip'] >= p90_diario).astype(int)
    df['dp_dt_neg'] = (df['dp_dia'] < -5).astype(int)

    agg = df.groupby(['ciudad', pd.Grouper(freq='MS')]).agg(
        precip=('precip', 'sum'),
        precip_max_dia=('precip', 'max'),
        dias_extremos=('es_extremo', 'sum'),
        pmed=('pmed', 'mean'),
        t_med=('t_med', 'mean'),
        dp_std=('pmed', 'std'),
        dp_dt_neg=('dp_dt_neg', 'sum'),
        t_rango=('t_rango', 'mean'),
        t_min=('tmin', 'mean'),
        veloc_ms=('veloc_ms', 'mean'),
        racha_ms_max=('racha_ms_max', 'max'),
        sol_sum=('sol', 'sum'),
        et0_sum=('et0', 'sum'),
        dias=('precip', 'count'),
    ).reset_index()

    agg = agg[agg['dias'] >= 10].copy().set_index('fecha').sort_index()
    # (patched for pandas >= 2.2/3.x: groupby.apply drops the grouping column; transform keeps it and gives the same values)
    _cols = [c for c in agg.columns if c != 'ciudad']
    agg[_cols] = agg.groupby('ciudad')[_cols].transform(lambda x: x.interpolate(method='time').ffill().bfill())
    agg = agg.sort_index()
    return agg

def _anomalia_canonical(agg: pd.DataFrame, col: str) -> pd.Series:
    """Full-record standardization by month and city (canonical pipeline)."""
    return agg.groupby(['ciudad', agg.index.month])[col].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-4)
    )

def _anomalia_train_only(agg: pd.DataFrame, col: str) -> pd.Series:
    """Train-only standardization from notebook cells 6 and 10 (strict pipeline)."""
    train_mask = (agg.index.year <= ANIO_CORTE_TRAIN)
    stats_tr = agg[train_mask].groupby(['ciudad', agg[train_mask].index.month])[col].agg(['mean', 'std'])
    stats_tr['std'] = stats_tr['std'].fillna(0) + 1e-4
    keys = list(zip(agg['ciudad'], agg.index.month))
    means = np.array([stats_tr.loc[k, 'mean'] if k in stats_tr.index else 0.0 for k in keys])
    stds = np.array([stats_tr.loc[k, 'std'] if k in stats_tr.index else 1e-4 for k in keys])
    return pd.Series((agg[col].values - means) / stds, index=agg.index)

def _indices_sinopticos(agg: pd.DataFrame, anomalia_fn, fix_et0_verano: bool = False) -> pd.DataFrame:
    """fix_et0_verano=False reproduces the archived core experiment (results_qml_core_trainonly*), in which
    et0_verano_prev / potencial_convectivo are misaligned: the groupby('ciudad').apply(...) result is assigned back
    on a date index that repeats across stations, so about 68 % of the rows receive the value of ANOTHER station on
    the SAME date (no temporal leakage). fix_et0_verano=True interpolates per station by position and matches the
    canonical 01_data.pt (and the classical train-only rerun)."""
    agg = agg.copy()
    agg['iap'] = anomalia_fn(agg, 'pmed')
    agg['iav'] = anomalia_fn(agg, 'veloc_ms')
    agg['ia_racha'] = anomalia_fn(agg, 'racha_ms_max')
    agg['ia_sol'] = anomalia_fn(agg, 'sol_sum')
    agg['ia_dp_std'] = anomalia_fn(agg, 'dp_std')

    dias_mes = agg.index.days_in_month
    sol_max_mes = agg.index.month.map(SOL_MAX_MENSUAL)
    agg['iir'] = (agg['sol_sum'] / (sol_max_mes.values * dias_mes.values + 1e-4)).clip(0, 1)
    agg['ib'] = agg['dp_std'] / (agg['t_rango'].replace(0, np.nan) + 0.1)

    # Thermal memory
    jja_mask = agg.index.month.isin([6, 7, 8])
    et0_jja = agg[jja_mask].groupby(['ciudad', agg[jja_mask].index.year])['et0_sum'].sum().to_dict()
    ref_year = agg.index.year - (agg.index.month < 9).astype(int)
    agg['et0_verano'] = [et0_jja.get((c, y), np.nan) for c, y in zip(agg['ciudad'], ref_year)]
    if fix_et0_verano:
        col = agg.columns.get_loc('et0_verano')
        for _, pos in agg.groupby('ciudad').indices.items():
            agg.iloc[pos, col] = agg['et0_verano'].iloc[pos].interpolate(method='time').ffill().bfill().values
    else:   # archived (misaligned) behaviour
        agg['et0_verano'] = agg.groupby('ciudad', group_keys=False)['et0_verano'].apply(lambda x: x.interpolate(method='time').ffill().bfill())
    agg['et0_verano_prev'] = agg['et0_verano']

    agg['et0_lag1m'] = agg.groupby('ciudad')['et0_sum'].shift(1).fillna(agg.groupby('ciudad')['et0_sum'].transform('mean'))
    agg['et0_lag3m'] = agg.groupby('ciudad')['et0_sum'].transform(lambda x: x.rolling(3, min_periods=1).sum().bfill())

    # Temperature anomaly
    roll_mean = agg.groupby('ciudad')['t_med'].transform(lambda x: x.rolling(12, min_periods=1).mean())
    roll_std = agg.groupby('ciudad')['t_med'].transform(lambda x: x.rolling(12, min_periods=1).std().fillna(1.0).replace(0, 1.0))
    agg['t_anomaly'] = (agg['t_med'] - roll_mean) / roll_std

    # Derived physics features
    agg['choque_clausius'] = agg.groupby('ciudad')['t_anomaly'].shift(3) - agg['t_anomaly']
    agg['potencial_convectivo'] = agg['et0_verano_prev'] * agg['dp_std']
    agg['fao_evapo_deficit'] = agg['et0_lag1m'] - agg.groupby('ciudad')['precip'].shift(1)
    agg['vuln_costera'] = agg['ib'] * agg['veloc_ms']

    # Anomaly lags
    agg['iap_lag1m'] = agg.groupby('ciudad')['iap'].shift(1).fillna(0)
    agg['ib_lag1m'] = agg.groupby('ciudad')['ib'].shift(1).fillna(0)
    agg['ia_sol_lag1m'] = agg.groupby('ciudad')['ia_sol'].shift(1).fillna(0)
    agg['ia_racha_lag1m'] = agg.groupby('ciudad')['ia_racha'].shift(1).fillna(0)
    return agg.fillna(0.0)

def _flag_ACP(agg: pd.DataFrame) -> pd.DataFrame:
    agg = agg.copy()
    agg['acp_flag'] = (
        agg.index.month.isin([9, 10, 11]) &
        (agg['ia_sol'] < 0.0) &
        (agg['ia_racha'] > 0.3)
    ).astype(float)
    return agg

def construir_features(df_diario: pd.DataFrame, anomalia_fn=_anomalia_canonical, fix_et0_verano: bool = False) -> pd.DataFrame:
    agg = _agregar_mensual(df_diario)
    agg = _indices_sinopticos(agg, anomalia_fn, fix_et0_verano=fix_et0_verano)
    agg = _flag_ACP(agg)
    return agg

def preparar_dataset(agg: pd.DataFrame,
                      umbral_override: Optional[float] = None,
                      umbral_ext_override: Optional[float] = None,
                      label_col: str = 'precip') -> DatasetResult:
    # label_col: column of `agg` that defines the target and its thresholds (default: the project's plain monthly sum).
    if 'ciudad' in agg.columns and not any(c.startswith('ciudad_') for c in agg.columns):
        agg = pd.get_dummies(agg, columns=['ciudad'], dtype=float)

    city_cols = [c for c in agg.columns if c.startswith('ciudad_')]
    train_mask = (agg.index.year <= ANIO_CORTE_TRAIN)

    # Thresholds
    precip_train = agg.loc[train_mask, label_col].values
    umbral = umbral_override if umbral_override is not None else float(np.percentile(precip_train, 60))
    umbral_x = umbral_ext_override if umbral_ext_override is not None else float(np.percentile(precip_train, 90))

    # Scalers fit strictly on training set
    scalers = {}
    df_sc = agg.copy()
    for col in ['precip'] + FEATURES_FISICAS:
        sc = MinMaxScaler(feature_range=(0, math.pi))
        sc.fit(agg.loc[train_mask, col].values.reshape(-1, 1))
        df_sc[col] = sc.transform(agg[col].values.reshape(-1, 1)).flatten()
        scalers[col] = sc

    # Sequence construction
    X_all, y_all, y_all_x, meses_all, ACP_all, t_anomaly_all, fechas_all = [], [], [], [], [], [], []
    ciudad_all = []
    for city_col in city_cols:
        mask = (df_sc[city_col].values == 1.0)
        df_city_sc = df_sc.iloc[mask].sort_index()
        df_city_raw = agg.iloc[mask].sort_index()
        fechas_city = df_city_sc.index
        precip_sc = df_city_sc['precip'].values
        precip_raw = df_city_raw[label_col].values
        n_muestras = len(df_city_sc) - WINDOW_SIZE

        for i in range(n_muestras):
            ventana = precip_sc[i:i + WINDOW_SIZE]
            ft = fechas_city[i + WINDOW_SIZE]
            s_sin = math.sin(2 * math.pi * ft.month / 12)
            s_cos = math.cos(2 * math.pi * ft.month / 12)
            k = i + WINDOW_SIZE - 1
            fisica = np.array([df_city_sc[c].values[k] for c in FEATURES_FISICAS])
            city_ohe = np.array([df_city_sc[c].values[k] for c in city_cols])
            x = np.concatenate([ventana, fisica, [s_sin, s_cos], city_ohe])

            X_all.append(x)
            y_all.append(1.0 if precip_raw[i + WINDOW_SIZE] > umbral else 0.0)
            y_all_x.append(1.0 if precip_raw[i + WINDOW_SIZE] > umbral_x else 0.0)
            meses_all.append(ft.month)
            acp_val = df_city_raw['acp_flag'].values[i + WINDOW_SIZE] if 'acp_flag' in df_city_raw else df_city_raw['dana_flag'].values[i + WINDOW_SIZE]
            ACP_all.append(float(acp_val))
            t_anomaly_all.append(float(df_city_raw['t_anomaly'].values[k]))
            fechas_all.append(ft)
            ciudad_all.append(city_col[len('ciudad_'):] if city_col.startswith('ciudad_') else city_col)

    sort_idx = np.argsort(fechas_all)
    X_all = np.array(X_all)[sort_idx]
    y_all = np.array(y_all)[sort_idx]
    y_all_x = np.array(y_all_x)[sort_idx]
    meses_all = np.array(meses_all)[sort_idx]
    ACP_all = np.array(ACP_all)[sort_idx]
    t_anomaly_all = np.array(t_anomaly_all)[sort_idx]
    fechas_all = np.array(fechas_all)[sort_idx]
    ciudad_all = np.array(ciudad_all)[sort_idx]

    years = np.array([d.year for d in fechas_all])
    s1 = (years <= 1996).sum()
    s2 = s1 + ((years >= 1997) & (years <= 2007)).sum()

    m_tr = meses_all[:s1]
    precip_media_mes = {
        m: float(y_all[:s1][m_tr == m].mean()) if (m_tr == m).sum() > 0 else 0.4
        for m in range(1, 13)
    }
    t_anomaly_mes = {
        m: float(t_anomaly_all[:s1][m_tr == m].mean()) if (m_tr == m).sum() > 0 else 0.0
        for m in range(1, 13)
    }

    feature_names = (
        [f'precip_lag_{WINDOW_SIZE - i}' for i in range(WINDOW_SIZE)] +
        FEATURES_FISICAS +
        ['mes_sin', 'mes_cos'] +
        city_cols
    )

    ret = DatasetResult(
        X_all=X_all, y_all=y_all, y_all_ext=y_all_x,
        meses_all=meses_all, dana_all=ACP_all,
        X_tr=torch.tensor(X_all[:s1], dtype=torch.float64),
        y_tr=torch.tensor(y_all[:s1], dtype=torch.float64),
        X_va=torch.tensor(X_all[s1:s2], dtype=torch.float64),
        y_va=torch.tensor(y_all[s1:s2], dtype=torch.float64),
        X_te=torch.tensor(X_all[s2:], dtype=torch.float64),
        y_te=torch.tensor(y_all[s2:], dtype=torch.float64),
        y_te_ext=torch.tensor(y_all_x[s2:], dtype=torch.float64),
        s1=s1, s2=s2,
        X_sk=X_all[:s1], y_sk=y_all[:s1],
        in_dim=X_all.shape[1],
        umbral=umbral, umbral_ext=umbral_x,
        m_tr=m_tr, m_va=meses_all[s1:s2], m_te=meses_all[s2:],
        dana_te=ACP_all[s2:],
        precip_media_mes=precip_media_mes,
        scalers=scalers,
        feature_names=feature_names,
        t_anomaly_mes=t_anomaly_mes
    )
    # row metadata of the test windows, in the SAME order as X_te / y_te (used to link predictions to monthly coverage)
    ret.meta_te = {'ciudad': ciudad_all[s2:], 'fecha': fechas_all[s2:]}
    return ret


def build_dataset(csv_path, train_only=True, label_scenario=None, fix_et0_verano=False, umbral_override=None):
    """label_scenario=None: project labels. 'S5': target from the coverage-corrected monthly sum
    (sum / valid days * days in month) with the 60th-percentile threshold recomputed on the training rows;
    predictors are unchanged (labels-only change)."""
    df = pd.read_csv(csv_path, index_col='fecha', parse_dates=['fecha'])
    fn = _anomalia_train_only if train_only else _anomalia_canonical
    agg = construir_features(df, anomalia_fn=fn, fix_et0_verano=fix_et0_verano)
    # umbral_override=28.1 (with fix_et0_verano=True) gives exactly the dataset of the classical train-only rerun
    # (results_rerun_train_only_canonical*); the default recomputes P60 on the training rows (28.00).
    if label_scenario is None:
        ds = preparar_dataset(agg, umbral_override=umbral_override)
    elif label_scenario == 'S5':
        agg = agg.copy(); agg['precip_corr'] = agg['precip'] / agg['dias'] * agg.index.days_in_month
        ds = preparar_dataset(agg, label_col='precip_corr')
    else:
        raise ValueError(label_scenario)
    return agg, ds
