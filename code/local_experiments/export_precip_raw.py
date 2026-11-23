#!/usr/bin/env python3
"""
export_precip_raw.py - Extracts daily precipitation across stations directly from raw AEMET files
before the inner-join of meteorological variables.
Output feeds exp12_monthly_coverage_audit.py (--precip_raw).

USAGE:
  python export_precip_raw.py --datos_dir C:/Users/enriq/Documents/QML/datos --out precip_raw.csv
"""
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd


def parse_aemet_precip(path: Path, city: str):
    rows = []
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 1 and len(parts[0]) == 10 and parts[0][2] == '-':
                try:
                    dt = datetime.strptime(parts[0], '%d-%m-%Y').strftime('%Y-%m-%d')
                    val = float(parts[1].strip().replace(',', '.'))
                    rows.append((dt, city, max(0.0, val) if not np.isnan(val) else 0.0))
                except ValueError:
                    pass
    return rows


def build_precip_raw(datos_dir: Path, cities: list[str]) -> pd.DataFrame:
    rows = []
    for city in cities:
        fpath = datos_dir / city / 'datos_precipitacion.txt'
        if not fpath.exists():
            raise FileNotFoundError(f"Missing {fpath}")
        rows.extend(parse_aemet_precip(fpath, city))
    return pd.DataFrame(rows, columns=['fecha', 'ciudad', 'precip'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Extract daily precipitation from raw AEMET station files")
    ap.add_argument('--datos_dir', required=True, help='Path to AEMET raw stations folder')
    ap.add_argument('--cities', nargs='+', default=['Barcelona', 'Valencia', 'Alicante', 'San_Javier'])
    ap.add_argument('--out', default='precip_raw.csv')
    args = ap.parse_args()

    df = build_precip_raw(Path(args.datos_dir), args.cities)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} daily records to {args.out}")
