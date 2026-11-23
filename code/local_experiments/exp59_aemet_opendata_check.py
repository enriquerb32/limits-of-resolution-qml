#!/usr/bin/env python3
"""
exp59_aemet_opendata_check.py - daily station pressure of AEMET OpenData against the datosclima-based panel
===========================================================================================================
Is the end of the pressure records of San Javier (7031, after 2010)
and Valencia (8416, after February 2012) a gap of the datosclima.es mirror, or of the AEMET record itself?

Downloads the daily climatological values of both stations from AEMET OpenData for 2008-01 to 2026-05 (6-month
requests), and compares, per station and year, the days with both daily pressure extremes in each source and the
largest absolute difference of presMax / presMin on the days present in both. Needs an AEMET OpenData API key in
~/.aemet_key (not part of the repository). Only the yearly summary is archived.

  python code/local_experiments/exp59_aemet_opendata_check.py [--cache downloaded.csv]
Output: results/local_exp59_aemet_opendata_pressure.csv
"""
import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PANEL = os.path.join(HERE, 'datos_limpios_multiestacion.csv')
OUT = os.path.join(ROOT, 'results', 'local_exp59_aemet_opendata_pressure.csv')
BASE = 'https://opendata.aemet.es/opendata/api/valores/climatologicos/diarios/datos'
STATIONS = {'7031': 'San_Javier', '8416': 'Valencia'}


def get(url, key, tries=5):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={'api_key': key, 'Accept': 'application/json'})
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read()
            try:
                return json.loads(raw.decode('utf-8'))
            except UnicodeDecodeError:
                return json.loads(raw.decode('latin-1'))
        except Exception as e:  # rate limit or transient error
            print('  retry', i, e, flush=True)
            time.sleep(10 * (i + 1))
    raise RuntimeError(url)


def num(x):
    try:
        return float(str(x).replace(',', '.')) if x is not None else None
    except ValueError:
        return None


def download():
    key = Path.home().joinpath('.aemet_key').read_text().strip()
    rows = []
    for idema, name in STATIONS.items():
        for y in range(2008, 2027):
            for a, b in (('01-01', '06-30'), ('07-01', '12-31')):
                if y == 2026 and a == '07-01':
                    continue
                meta = get(f'{BASE}/fechaini/{y}-{a}T00:00:00UTC/fechafin/{y}-{b}T23:59:59UTC/estacion/{idema}', key)
                if meta.get('estado') == 200:
                    rows += [{'ciudad': name, 'fecha': d.get('fecha'), 'pmax_od': num(d.get('presMax')),
                              'pmin_od': num(d.get('presMin'))} for d in get(meta['datos'], key)]
                time.sleep(2)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default=None, help='CSV of an earlier download (columns ciudad, fecha, pmax_od, pmin_od)')
    a = ap.parse_args()
    od = pd.read_csv(a.cache) if a.cache else download()
    od['ciudad'] = od.ciudad.replace({'San Javier': 'San_Javier'})
    od = od[od.fecha <= '2026-05-31']  # end of the archived panel
    p = pd.read_csv(PANEL, usecols=['fecha', 'pmax', 'pmin', 'ciudad'])
    m = od.merge(p, on=['ciudad', 'fecha'], how='left')
    m['year'] = m.fecha.str[:4].astype(int)
    s = m.groupby(['ciudad', 'year']).apply(lambda g: pd.Series(dict(
        days=len(g),
        pressure_days_opendata=int((g.pmax_od.notna() & g.pmin_od.notna()).sum()),
        pressure_days_panel=int((g.pmax.notna() & g.pmin.notna()).sum()),
        max_abs_diff_pmax=(g.pmax_od - g.pmax).abs().max(),
        max_abs_diff_pmin=(g.pmin_od - g.pmin).abs().max())))
    print(s.to_string())
    s.to_csv(OUT)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
