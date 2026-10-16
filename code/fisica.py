"""
fisica.py  —  Módulo de física diagnóstica (VERSIÓN MULTI-ESTACIÓN Q1-READY)
====================================================================
Transforma el DataFrame diario en variables físicas mensuales respetando la latitud
y aislando los "lags" temporales por ciudad para evitar Spatial Data Leakage.

  1. ET0 Penman-Monteith FAO-56 diaria (Dinámica por latitud)
  2. Índices sinópticos mensuales estandarizados por clima local
  3. Variables de Ingeniería Física Avanzada (Choque, Convección, Estrés)
  4. Indicador ACP basado EXCLUSIVAMENTE en precursores

Función pública:
  construir_features(df_diario) → pd.DataFrame mensual
"""

import math
import numpy as np
import pandas as pd
import torch

# ── Constantes físicas Multi-Estación ─────────────────────────────────────────

# Latitudes para el cálculo de la radiación astronómica (FAO-56)
LATITUDES = {
    'Barcelona': 41.38,
    'Valencia': 39.46,
    'Alicante': 38.37,
    'San_Javier': 37.77
}

# Altitudes aproximadas (m s.n.m.) para la presión atmosférica
ALTITUDES = {
    'Barcelona': 4.0,
    'Valencia': 11.0,
    'Alicante': 7.0,
    'San_Javier': 4.0
}

# Horas de sol máximas teóricas medias
SOL_MAX_MENSUAL = {
    1: 9.5,  2: 10.5, 3: 11.8, 4: 13.3,
    5: 14.2, 6: 14.4, 7: 14.2, 8: 13.2,
    9: 11.8, 10: 10.3, 11: 9.6, 12: 9.4,
}

# ── ET0 Penman-Monteith FAO-56 ────────────────────────────────────────────────

def et0_fao56(tmax: float, tmin: float, pres_hpa: float,
              sol: float, dia_ano: int,
              lat: float, alt: float,
              u2: float = 2.0) -> float:
    """Evapotranspiración de referencia diaria (mm/día) FAO 56."""
    t_med    = (tmax + tmin) / 2.0
    pres_kpa = pres_hpa * 0.1

    e_tmed = 0.6108 * math.exp(17.27 * t_med  / (t_med  + 237.3))
    delta  = 4098 * e_tmed / (t_med + 237.3) ** 2
    gamma  = 0.000665 * pres_kpa

    e_tmax = 0.6108 * math.exp(17.27 * tmax / (tmax + 237.3))
    e_tmin = 0.6108 * math.exp(17.27 * tmin / (tmin + 237.3))
    es     = (e_tmax + e_tmin) / 2.0
    ea     = e_tmin   

    lat_r = math.radians(lat)
    dr    = 1 + 0.033 * math.cos(2 * math.pi * dia_ano / 365)
    decl  = 0.409 * math.sin(2 * math.pi * dia_ano / 365 - 1.39)
    arg   = max(-1.0, min(1.0, -math.tan(lat_r) * math.tan(decl)))
    ws    = math.acos(arg)
    Ra    = ((24 * 60 / math.pi) * 0.0820 * dr
             * (ws * math.sin(lat_r) * math.sin(decl)
                + math.sin(ws) * math.cos(lat_r) * math.cos(decl)))
    N_max = (24 / math.pi) * ws

    Rs  = (0.25 + 0.50 * (sol / N_max)) * Ra if N_max > 0 else 0.0
    Rns = (1 - 0.23) * Rs
    Rso = (0.75 + 2e-5 * alt) * Ra
    fnu = max(0.05, min(1.0, 1.35 * (Rs / Rso) - 0.35)) if Rso > 0 else 0.05
    sigma = 4.903e-9
    Rnl = (sigma * ((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2
           * (0.34 - 0.14 * math.sqrt(max(0.0, ea))) * fnu)
    Rn  = Rns - Rnl

    num = 0.408 * delta * Rn + gamma * (900 / (t_med + 273)) * u2 * (es - ea)
    den = delta + gamma * (1 + 0.34 * u2)
    return max(0.0, num / den)

# ── Agregación mensual Multi-Estación ─────────────────────────────────────────

def _agregar_mensual(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega el DataFrame diario a frecuencia mensual agrupando por ciudad."""
    df = df.copy()
    df['pmed']     = (df['pmax'] + df['pmin']) / 2.0
    df['t_med']    = (df['tmax'] + df['tmin']) / 2.0  
    
    # CRÍTICO: La diferencia de presión de ayer a hoy se hace DENTRO de la misma ciudad
    df['dp_dia']   = df.groupby('ciudad')['pmed'].diff().fillna(0.0)
    
    df['t_rango']  = df['tmax'] - df['tmin']
    df['veloc_ms'] = df['veloc_kmh'] / 3.6
    df['racha_ms'] = df['racha_kmh'] / 3.6
    df['dia_ano']  = df.index.map(lambda d: d.timetuple().tm_yday)

    # 🎯 Calculamos ET0 inyectando la latitud/altitud correspondiente a cada ciudad
    df['et0'] = df.apply(lambda r: et0_fao56(
        r['tmax'], r['tmin'], r['pmed'], r['sol'], int(r['dia_ano']),
        lat=LATITUDES.get(r['ciudad'], 38.37),
        alt=ALTITUDES.get(r['ciudad'], 7.0)
    ), axis=1)

    # El umbral del percentil 90 diario se calcula respecto a la climatología de la ciudad
    p90_diario = df.groupby('ciudad')['precip'].transform(lambda x: x.quantile(0.90))
    df['es_extremo'] = (df['precip'] >= p90_diario).astype(int)
    df['dp_dt_neg']  = (df['dp_dia'] < -5).astype(int)
    
    # 🎯 Agregación a nivel mensual + ciudad usando Grouper
    agg = df.groupby(['ciudad', pd.Grouper(freq='MS')]).agg(
        precip          = ('precip',    'sum'),
        precip_max_dia  = ('precip',    'max'),      
        dias_extremos   = ('es_extremo','sum'), 
        pmed            = ('pmed',      'mean'),
        t_med           = ('t_med',     'mean'), 
        dp_std          = ('pmed',      'std'),       
        dp_dt_neg       = ('dp_dt_neg', 'sum'),
        t_rango         = ('t_rango',   'mean'),
        t_min           = ('tmin',      'mean'),
        veloc_ms        = ('veloc_ms',  'mean'),
        racha_ms_max    = ('racha_ms',  'max'),
        sol_sum         = ('sol',       'sum'),
        et0_sum         = ('et0',       'sum'),
        dias            = ('precip',    'count'),
    ).reset_index()

    agg = agg[agg['dias'] >= 10].copy()
    
    # Recuperamos el DatetimeIndex y ordenamos temporalmente
    agg = agg.set_index('fecha').sort_index()
    
    # Interpolamos huecos aisladamente para cada ciudad
    agg = agg.groupby('ciudad', group_keys=False).apply(lambda x: x.interpolate(method='time').ffill().bfill()).sort_index()
    
    return agg

# ── Índices sinópticos y Memoria Térmica (Aislados por ciudad) ───────────────

def _anomalia(agg: pd.DataFrame, col: str) -> pd.Series:
    """Estandariza una serie por mes Y POR CIUDAD para aislar microclimas."""
    return agg.groupby(['ciudad', agg.index.month])[col].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-4)
    )

def _calcular_et0_verano(agg: pd.DataFrame) -> pd.Series:
    """ET0 acumulada JJA del año anterior (por ciudad) → proxy de carga energética."""
    jja_mask  = agg.index.month.isin([6, 7, 8])
    
    # Suma de la ET0 en verano para cada ciudad y año
    et0_jja = agg[jja_mask].groupby(['ciudad', agg[jja_mask].index.year])['et0_sum'].sum()
    
    # Mapeo: si el mes actual es < Sep, usamos el verano del año anterior. Si no, el de este año.
    ref_year = agg.index.year - (agg.index.month < 9).astype(int)
    jja_dict = et0_jja.to_dict()
    
    # Asignamos el valor cruzando (ciudad, año_de_referencia)
    resultado = [jja_dict.get((c, y), np.nan) for c, y in zip(agg['ciudad'], ref_year)]
    agg['tmp_et0_verano'] = resultado
    
    # Rellenamos los NaN que puedan quedar a principios del dataset
    serie_res = agg.groupby('ciudad', group_keys=False)['tmp_et0_verano'].apply(lambda x: x.interpolate(method='time').ffill().bfill())
    agg.drop(columns=['tmp_et0_verano'], inplace=True)
    return serie_res

def _indices_sinopticos(agg: pd.DataFrame) -> pd.DataFrame:
    """Añade índices sinópticos, memoria térmica y las 4 variables de oro (Aislado)."""
    agg = agg.copy()

    # 1. Anomalías sinópticas base locales
    agg['iap']       = _anomalia(agg, 'pmed')
    agg['iav']       = _anomalia(agg, 'veloc_ms')
    agg['ia_racha']  = _anomalia(agg, 'racha_ms_max')
    agg['ia_sol']    = _anomalia(agg, 'sol_sum')
    agg['ia_dp_std'] = _anomalia(agg, 'dp_std')

    dias_mes     = agg.index.days_in_month
    sol_max_mes  = agg.index.month.map(SOL_MAX_MENSUAL)
    agg['iir']   = (agg['sol_sum'] / (sol_max_mes.values * dias_mes.values + 1e-4)).clip(0, 1)
    agg['ib']    = agg['dp_std'] / (agg['t_rango'].replace(0, np.nan) + 0.1)

    # 2. Memoria Térmica (El Combustible)
    agg['et0_verano']      = _calcular_et0_verano(agg)
    agg['et0_verano_prev'] = agg['et0_verano'] 
    
    # CRÍTICO: .shift(1) agrupado por ciudad
    agg['et0_lag1m'] = agg.groupby('ciudad')['et0_sum'].shift(1)
    agg['et0_lag1m'] = agg['et0_lag1m'].fillna(agg.groupby('ciudad')['et0_sum'].transform('mean'))
    
    agg['et0_lag3m'] = agg.groupby('ciudad')['et0_sum'].transform(lambda x: x.rolling(3, min_periods=1).sum().bfill())

    # 3. Anomalía Térmica Local
    roll_mean = agg.groupby('ciudad')['t_med'].transform(lambda x: x.rolling(12, min_periods=1).mean())
    roll_std  = agg.groupby('ciudad')['t_med'].transform(lambda x: x.rolling(12, min_periods=1).std().fillna(1.0).replace(0, 1.0))
    agg['t_anomaly'] = (agg['t_med'] - roll_mean) / roll_std

    # ══════════════════════════════════════════════════════════════
    # LAS 4 NUEVAS VARIABLES TERMODINÁMICAS (Physical Feature Engineering)
    # ══════════════════════════════════════════════════════════════
    
    agg['choque_clausius']      = agg.groupby('ciudad')['t_anomaly'].shift(3) - agg['t_anomaly']
    agg['potencial_convectivo'] = agg['et0_verano_prev'] * agg['dp_std']
    agg['fao_evapo_deficit']    = agg['et0_lag1m'] - agg.groupby('ciudad')['precip'].shift(1)
    agg['vuln_costera']         = agg['ib'] * agg['veloc_ms']

    # Lags sinópticos agrupados
    agg['iap_lag1m']      = agg.groupby('ciudad')['iap'].shift(1).fillna(0)
    agg['ib_lag1m']       = agg.groupby('ciudad')['ib'].shift(1).fillna(0)
    agg['ia_sol_lag1m']   = agg.groupby('ciudad')['ia_sol'].shift(1).fillna(0)
    agg['ia_racha_lag1m'] = agg.groupby('ciudad')['ia_racha'].shift(1).fillna(0)

    agg = agg.fillna(0.0)
    
    return agg

# ── Indicador ACP ───────────────────────────────────────────────────────────

def _flag_ACP(agg: pd.DataFrame) -> pd.DataFrame:
    """Etiqueta meses con firma observacional de ACP (Cero Leakage)."""
    agg = agg.copy()
    agg['acp_flag'] = (
        agg.index.month.isin([9, 10, 11]) &
        (agg['ia_sol']    < 0.0) &               # Anomalía de nubosidad local
        (agg['ia_racha']  > 0.3)                 # Anomalía de inestabilidad local
    ).astype(float)

    n = int(agg['acp_flag'].sum())
    print(f"  [Física] Riesgo Sinóptico ACP detectado globalmente: {n} eventos mes/ciudad")
    return agg

# ── Función pública principal ────────────────────────────────────────────────

def construir_features(df_diario: pd.DataFrame) -> pd.DataFrame:
    agg = _agregar_mensual(df_diario)
    agg = _indices_sinopticos(agg)
    agg = _flag_ACP(agg)
    return agg