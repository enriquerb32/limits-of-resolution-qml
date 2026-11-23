"""
features.py  —  Feature Engineering Module (Q1-READY)
MULTI-STATION VERSION: Prevents spatial data leakage and enforces strict chronological splitting.
"""

import math
import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass, field
from sklearn.preprocessing import MinMaxScaler
from typing import List, Dict


WINDOW_SIZE = 12
PERCENTIL_LLUVIA  = 60
PERCENTIL_EXTREMO = 90

FEATURES_FISICAS = [
    'pmed', 'dp_std', 't_rango', 'et0_sum', 'et0_verano_prev',
    'et0_lag1m', 'et0_lag3m', 'iap', 'iir', 'ia_racha',
    'veloc_ms', 'ib', 't_anomaly',
    'choque_clausius',
    'potencial_convectivo',
    'fao_evapo_deficit',
    'vuln_costera'
]


@dataclass
class DatasetResult:
    X_all: np.ndarray
    y_all: np.ndarray
    y_all_ext: np.ndarray
    meses_all: np.ndarray
    ACP_all: np.ndarray

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
    ACP_te: np.ndarray
    precip_media_mes: Dict[int, float]
    scalers: Dict[str, MinMaxScaler]
    t_anomaly_mes: Dict[int, float]
    feature_names: List[str] = field(default_factory=list)


def preparar_dataset(agg: pd.DataFrame) -> DatasetResult:
    # --- 1. One-Hot Encoding of station/city ---
    if 'ciudad' in agg.columns:
        agg = pd.get_dummies(agg, columns=['ciudad'], dtype=float)

    city_cols = [c for c in agg.columns if c.startswith('ciudad_')]
    if not city_cols:
        raise ValueError("No city indicator columns found (city_cols).")

    # --- 2. Thresholds computed EXCLUSIVELY on training set (prevents leakage) ---
    train_mask = (agg.index.year <= 1996)
    precip_train = agg.loc[train_mask, 'precip'].values

    umbral   = np.percentile(precip_train, PERCENTIL_LLUVIA)
    umbral_x = np.percentile(precip_train, PERCENTIL_EXTREMO)

    # --- 3. Scalers fitted strictly on training data ---
    scalers = {}
    df_sc = agg.copy()

    for col in ['precip'] + FEATURES_FISICAS:
        sc = MinMaxScaler(feature_range=(0, math.pi))
        sc.fit(agg.loc[train_mask, col].values.reshape(-1, 1))
        df_sc[col] = sc.transform(agg[col].values.reshape(-1, 1)).flatten()
        scalers[col] = sc

    # --- 4. Sequence construction per station/city ---
    X_all, y_all, y_all_x, meses_all, ACP_all, t_anomaly_all, fechas_all = [], [], [], [], [], [], []

    for city_col in city_cols:
        df_city_sc  = df_sc[df_sc[city_col] == 1.0].sort_index()
        df_city_raw = agg[agg[city_col] == 1.0].sort_index()

        fechas_city = df_city_sc.index
        precip_sc   = df_city_sc['precip'].values
        precip_raw  = df_city_raw['precip'].values

        n_muestras = len(df_city_sc) - WINDOW_SIZE

        for i in range(n_muestras):
            ventana = precip_sc[i : i + WINDOW_SIZE]
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
            ACP_all.append(float(df_city_raw['acp_flag'].values[i + WINDOW_SIZE]))
            t_anomaly_all.append(float(df_city_raw['t_anomaly'].values[k]))
            fechas_all.append(ft)

    X_all         = np.array(X_all)
    y_all         = np.array(y_all)
    y_all_x       = np.array(y_all_x)
    meses_all     = np.array(meses_all)
    ACP_all      = np.array(ACP_all)
    t_anomaly_all = np.array(t_anomaly_all)
    fechas_all    = np.array(fechas_all)

    # --- 5. Global chronological ordering ---
    sort_idx = np.argsort(fechas_all)
    X_all         = X_all[sort_idx]
    y_all         = y_all[sort_idx]
    y_all_x       = y_all_x[sort_idx]
    meses_all     = meses_all[sort_idx]
    ACP_all      = ACP_all[sort_idx]
    t_anomaly_all = t_anomaly_all[sort_idx]
    fechas_all    = fechas_all[sort_idx]

    # --- 6. Strict temporal train/validation/test split ---
    años = np.array([dt.year for dt in fechas_all])
    s1 = (años <= 1996).sum()
    s2 = s1 + ((años >= 1997) & (años <= 2007)).sum()

    m_tr = meses_all[:s1]

    precip_media_mes = {
        m: float(y_all[:s1][m_tr == m].mean()) if (m_tr == m).sum() > 0 else 0.4
        for m in range(1, 13)
    }

    t_anomaly_mes = {
        m: float(t_anomaly_all[:s1][m_tr == m].mean()) if (m_tr == m).sum() > 0 else 0.0
        for m in range(1, 13)
    }

    in_dim = X_all.shape[1]
    feature_names = (
        [f'precip_lag_{WINDOW_SIZE - i}' for i in range(WINDOW_SIZE)] +
        FEATURES_FISICAS +
        ['mes_sin', 'mes_cos'] +
        city_cols
    )

    print(f"  [Features] {len(X_all)} samples (Multi-Station) | dim={in_dim} | "
          f"threshold p60={umbral:.1f}mm | p90={umbral_x:.1f}mm | "
          f"balance={y_all.mean()*100:.1f}%")

    return DatasetResult(
        X_all=X_all, y_all=y_all, y_all_ext=y_all_x,
        meses_all=meses_all, ACP_all=ACP_all,
        X_tr=torch.tensor(X_all[:s1], dtype=torch.float64),
        y_tr=torch.tensor(y_all[:s1], dtype=torch.float64),
        X_va=torch.tensor(X_all[s1:s2], dtype=torch.float64),
        y_va=torch.tensor(y_all[s1:s2], dtype=torch.float64),
        X_te=torch.tensor(X_all[s2:], dtype=torch.float64),
        y_te=torch.tensor(y_all[s2:], dtype=torch.float64),
        y_te_ext=torch.tensor(y_all_x[s2:], dtype=torch.float64),
        s1=s1, s2=s2,

        # Training-only data for classical models (prevents leakage in XGBoost/RF)
        X_sk=X_all[:s1],
        y_sk=y_all[:s1],

        in_dim=in_dim,
        umbral=umbral, umbral_ext=umbral_x,
        m_tr=m_tr, m_va=meses_all[s1:s2], m_te=meses_all[s2:], ACP_te=ACP_all[s2:],
        precip_media_mes=precip_media_mes,
        scalers=scalers,
        feature_names=feature_names,
        t_anomaly_mes=t_anomaly_mes
    )


# English alias for public API
prepare_dataset = preparar_dataset