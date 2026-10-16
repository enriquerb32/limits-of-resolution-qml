"""
etl.py  —  Módulo ETL
=====================
Carga, limpieza y unión de las 5 fuentes AEMET:
  datos.txt · datos_presion.txt · datos_temperatura.txt
  datos_sol.txt · datos_viento.txt

Función pública:
  cargar_datos(directorio) → pd.DataFrame diario limpio
"""

import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path


# ── Carga bruta de un fichero AEMET ──────────────────────────────────────────

def _leer_aemet(ruta: str, col_indices: list[int]) -> dict:
    """
    Parsea un fichero AEMET tabulado.
    Filtra filas con fecha DD-MM-YYYY en posición 0.
    col_indices: qué columnas (1-based) extraer como float.
    Devuelve dict {datetime: [val1, val2, ...]}.
    """
    registros = {}
    with open(ruta, encoding='utf-8', errors='ignore') as f:
        for linea in f:
            partes = linea.strip().split('\t')
            if (len(partes) > max(col_indices)
                    and len(partes[0]) == 10
                    and partes[0][2] == '-'):
                try:
                    dt = datetime.strptime(partes[0], '%d-%m-%Y')
                except ValueError:
                    continue
                fila = []
                for k in col_indices:
                    raw = partes[k].strip().replace(',', '.') if k < len(partes) else ''
                    try:
                        fila.append(float(raw))
                    except ValueError:
                        fila.append(np.nan)
                registros[dt] = fila
    return registros


# ── Función pública ───────────────────────────────────────────────────────────

def cargar_datos(dir_base: str = '.', ciudades: list = None) -> pd.DataFrame:
    """
    Une las 5 fuentes AEMET para múltiples ciudades en un único DataFrame.
    """
    if ciudades is None:
        ciudades = ['Barcelona', 'Valencia', 'Alicante', 'San_Javier']
        
    base_path = Path(dir_base)
    lista_dfs = []

    for ciudad in ciudades:
        d = base_path / ciudad
        print(f"Procesando {ciudad}...")
        
        # Leemos los 5 ficheros de la ciudad actual
        PR = _leer_aemet(d / 'datos_precipitacion.txt',  [1])
        PS = _leer_aemet(d / 'datos_presion.txt',     [1, 2])
        TM = _leer_aemet(d / 'datos_temperatura.txt', [1, 2])
        SL = _leer_aemet(d / 'datos_sol.txt',         [1, 2])
        VT = _leer_aemet(d / 'datos_viento.txt',      [1, 3])

        fechas = sorted(set(PR) & set(PS) & set(TM) & set(SL) & set(VT))

        rows = []
        for dt in fechas:
            rows.append({
                'fecha':      dt,
                'precip':     max(0.0, PR[dt][0]) if not np.isnan(PR[dt][0]) else 0.0,
                'pmax':       PS[dt][0],
                'pmin':       PS[dt][1],
                'tmax':       TM[dt][0],
                'tmin':       TM[dt][1],
                'sol':        SL[dt][1] if not np.isnan(SL[dt][1]) else 0.0,
                'racha_kmh':  VT[dt][0],
                'veloc_kmh':  VT[dt][1],
            })

        df_ciudad = pd.DataFrame(rows).set_index('fecha').sort_index()

        # Limites físicos
        df_ciudad['tmax'] = df_ciudad['tmax'].clip(lower=-5.0, upper=45.0)
        df_ciudad['tmin'] = df_ciudad['tmin'].clip(lower=-5.0, upper=45.0)
        
        mask_t_err = df_ciudad['tmax'] < df_ciudad['tmin']
        if mask_t_err.any():
            t_mean = (df_ciudad.loc[mask_t_err, 'tmax'] + df_ciudad.loc[mask_t_err, 'tmin']) / 2
            df_ciudad.loc[mask_t_err, 'tmax'] = t_mean
            df_ciudad.loc[mask_t_err, 'tmin'] = t_mean

        df_ciudad['pmax'] = df_ciudad['pmax'].clip(lower=950.0, upper=1060.0)
        df_ciudad['pmin'] = df_ciudad['pmin'].clip(lower=950.0, upper=1060.0)
        df_ciudad['racha_kmh'] = df_ciudad['racha_kmh'].clip(lower=0.0, upper=250.0)
        df_ciudad['veloc_kmh'] = df_ciudad['veloc_kmh'].clip(lower=0.0, upper=200.0)
        df_ciudad['sol'] = df_ciudad['sol'].clip(lower=0.0, upper=24.0)

        # Imputación de nulos y añadir la columna de ciudad
        df_ciudad = (df_ciudad.interpolate(method='time', limit=7).ffill().bfill())
        df_ciudad['ciudad'] = ciudad
        
        print(f"  [ETL] {ciudad}: {len(df_ciudad):,} días · NaN: {df_ciudad.isna().sum().sum()}")
        lista_dfs.append(df_ciudad)

    # Unimos todos los DataFrames en uno solo
    df_final = pd.concat(lista_dfs)
    
    # Exportamos el CSV maestro
    df_final.to_csv('datos_limpios_multiestacion.csv')

    return df_final
