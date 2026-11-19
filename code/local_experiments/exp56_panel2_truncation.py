#!/usr/bin/env python3
"""
exp56_panel2_truncation.py - the truncation test of exp47 on the second panel (review of 2026-11-19)
====================================================================================================
The second panel (Mallorca, Tortosa, Alcantarilla, Almeria; exp32, tag med2v2) is built by the same v2 pipeline as the
first, with the archived (two-sided) imputation. For every test target month t (2008 onwards) its monthly panel is
rebuilt from the daily file cut at the end of t-1, exactly as exp47 does for the first panel, and the inputs and the
eligibility of every test window are compared with those of the full pipeline; the same is repeated with causal
imputation (V2_CAUSAL semantics). No training.

  LOR_DATASET=v2 python code/local_experiments/exp56_panel2_truncation.py
Output: results/local_exp56_panel2_truncation.json, results/exp56_panel2_truncation_rows.csv
"""
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import exp47_truncation_audit as e47  # noqa: E402  (installs the ET0 cache and defines the truncation test)
import trainonly_dataset as tds  # noqa: E402

D = os.path.join(ROOT, 'results', 'panel2', 'med2v2')


def main():
    assert os.environ.get('LOR_DATASET', 'v1').lower() == 'v2', 'set LOR_DATASET=v2'
    for s in json.load(open(os.path.join(D, 'stations.json'))):      # as exp32.inject_stations
        tds.LATITUDES[s['name']] = float(s['lat']); tds.ALTITUDES[s['name']] = float(s['alt'])
    daily = pd.read_csv(os.path.join(D, 'datos.csv'), index_col='fecha', parse_dates=['fecha'])
    meta = pd.read_csv(os.path.join(D, 'test_meta_pooled.csv'))
    out, rows = dict(n_test_rows_archived=int(len(meta))), []
    for causal in (False, True):
        e47._et0_cache.clear()
        r, summ = e47.truncation(daily, causal)
        out['causal' if causal else 'archived'] = summ
        rows.append(r)
    pd.concat(rows).to_csv(os.path.join(ROOT, 'results', 'exp56_panel2_truncation_rows.csv'), index=False)
    json.dump(out, open(os.path.join(ROOT, 'results', 'local_exp56_panel2_truncation.json'), 'w'), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
