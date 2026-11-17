"""
etl_v2.py - v2 daily table (replaces the inner join of etl.py)
=============================================================
Differences from etl.py (v1), all motivated by the label audit (supp, label audit) and review 4:
  1. Dates: full calendar per station, from the first to the last date found in ANY of the five files
     (v1: only dates present in all five files, so a missing pressure or wind day removed that day's rain).
  2. Precipitation comes from its own file only. 'Ip' (inapreciable, trace) -> 0.0; empty or unparseable -> NaN
     (v1: NaN -> 0.0, i.e. missing days counted as dry).
  3. Covariates stay NaN where missing (v1: sunshine NaN -> 0.0, then time interpolation over 7 days and
     forward/backward fill over any gap). Monthly aggregation in dataset_v2.py uses the available days only.
  4. Physical clipping as in v1.
Output: datos_limpios_multiestacion.csv with the same columns as v1 plus 'precip_valid' (bool). The file is only
read with LOR_DATASET=v2 (trainonly_dataset refuses a v2 file in v1 mode and vice versa).

  python code/etl_v2.py --datos_dir C:/Users/enriq/Documents/QML/datos --out code/local_experiments/datos_limpios_multiestacion.csv
"""
import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

FILES = {  # file -> {column name: 1-based column index}, as in etl.py
    'datos_precipitacion.txt': {'precip': 1},
    'datos_presion.txt': {'pmax': 1, 'pmin': 2},
    'datos_temperatura.txt': {'tmax': 1, 'tmin': 2},
    'datos_sol.txt': {'sol': 2},
    'datos_viento.txt': {'racha_kmh': 1, 'veloc_kmh': 3},
}
TRACE = {'ip', 'inap', 'inapreciable'}


def _read(path: Path, cols: dict, is_precip: bool, tokens: Counter) -> pd.DataFrame:
    recs = {}
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if not (len(parts[0]) == 10 and parts[0][2] == '-'):
                continue
            try:
                dt = datetime.strptime(parts[0], '%d-%m-%Y')
            except ValueError:
                continue
            row = {}
            for name, k in cols.items():
                raw = parts[k].strip() if k < len(parts) else ''
                try:
                    row[name] = float(raw.replace(',', '.'))
                except ValueError:
                    if is_precip and raw.lower() in TRACE:
                        row[name] = 0.0
                    else:
                        row[name] = np.nan
                    tokens[(name, raw[:12])] += 1
            recs[dt] = row
    return pd.DataFrame.from_dict(recs, orient='index')


def cargar_datos_v2(dir_base, ciudades=None, report=None):
    ciudades = ciudades or ['Barcelona', 'Valencia', 'Alicante', 'San_Javier']
    out, rep = [], {}
    for c in ciudades:
        tokens = Counter()
        parts = [_read(Path(dir_base) / c / fn, cols, fn == 'datos_precipitacion.txt', tokens) for fn, cols in FILES.items()]
        lo = min(p.index.min() for p in parts if len(p)); hi = max(p.index.max() for p in parts if len(p))
        df = pd.DataFrame(index=pd.date_range(lo, hi, freq='D', name='fecha'))
        for p in parts:
            df = df.join(p, how='left')
        df['precip'] = df['precip'].clip(lower=0.0)
        df['tmax'] = df['tmax'].clip(-5, 45); df['tmin'] = df['tmin'].clip(-5, 45)
        bad = df['tmax'] < df['tmin']
        df.loc[bad, ['tmax', 'tmin']] = np.repeat(((df.loc[bad, 'tmax'] + df.loc[bad, 'tmin']) / 2).values[:, None], 2, 1)
        df['pmax'] = df['pmax'].clip(950, 1060); df['pmin'] = df['pmin'].clip(950, 1060)
        df['racha_kmh'] = df['racha_kmh'].clip(0, 250); df['veloc_kmh'] = df['veloc_kmh'].clip(0, 200)
        df['sol'] = df['sol'].clip(0, 24)
        df['precip_valid'] = df['precip'].notna()
        df['ciudad'] = c
        rep[c] = {'first': str(lo.date()), 'last': str(hi.date()), 'days': int(len(df)),
                  'missing_share': {k: round(float(df[k].isna().mean()), 4) for k in
                                    ['precip', 'pmax', 'tmax', 'sol', 'veloc_kmh']},
                  'non_numeric_tokens': {f'{a}:{b}': n for (a, b), n in tokens.most_common(8)}}
        print(f"  [ETL v2] {c}: {lo.date()}..{hi.date()} | precip missing {rep[c]['missing_share']['precip']:.1%} "
              f"| tokens {rep[c]['non_numeric_tokens']}")
        out.append(df)
    if report:
        json.dump(rep, open(report, 'w'), indent=2)
    return pd.concat(out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--datos_dir', required=True)
    ap.add_argument('--stations', nargs='+', default=None)
    ap.add_argument('--out', default='datos_limpios_multiestacion.csv')
    ap.add_argument('--report', default='etl_v2_report.json')
    a = ap.parse_args()
    cargar_datos_v2(a.datos_dir, a.stations, a.report).to_csv(a.out)
    print('wrote', a.out, a.report)
