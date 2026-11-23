#!/usr/bin/env python3
"""
exp56_panel2_truncation.py - the truncation test of exp47 on the second panel
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


def impact():
    """Table 11 contrasts without the test windows that are eligible only with later data (archived pipeline)."""
    import numpy as np
    sys.path.insert(0, os.path.join(ROOT, 'src'))
    from qml_resolution import cluster_robust_delta
    r = pd.read_csv(os.path.join(ROOT, 'results', 'exp56_panel2_truncation_rows.csv'))
    late = r[(r.causal == 0) & r.win_full & ~r.win_cut]
    meta = pd.read_csv(os.path.join(D, 'test_meta_pooled.csv'))
    meta['key'] = meta['ciudad'] + '_' + pd.to_datetime(meta['fecha']).dt.strftime('%Y-%m-%d')
    drop = set(late['ciudad'] + '_' + late['fecha'])
    keep = ~meta['key'].isin(drop).values
    assert (~keep).sum() == len(late), 'late windows not all in the test set'
    z = np.load(os.path.join(D, 'probs_pooled.npz')); y = z['y_te'].astype(int)
    yrs = pd.to_datetime(meta['fecha']).dt.year.values
    out = dict(late_windows=sorted(drop), n_dropped=int((~keep).sum()), pairs={})
    for a, b in [('RF_2273', 'XGB'), ('RF_2273', 'PureLSTM'), ('RF_2273', 'LSTM_QNN'), ('XGB', 'LSTM_QNN'),
                 ('PureLSTM', 'LSTM_QNN'), ('CTRL_LINEAR4D', 'LSTM_QNN'), ('FOURIER2', 'LSTM_QNN')]:
        f = cluster_robust_delta(y, z[f'{a}_probs'], z[f'{b}_probs'], yrs)
        k = cluster_robust_delta(y[keep], z[f'{a}_probs'][:, keep], z[f'{b}_probs'][:, keep], yrs[keep])
        flo, fhi = f.ci(); klo, khi = k.ci()
        out['pairs'][f'{a}-{b}'] = dict(delta_all=f.delta, p_all=f.p_value, resolved_all=bool(flo > 0 or fhi < 0),
                                        delta_kept=k.delta, p_kept=k.p_value, resolved_kept=bool(klo > 0 or khi < 0))
    return out


def main():
    assert os.environ.get('LOR_DATASET', 'v1').lower() == 'v2', 'set LOR_DATASET=v2'
    if '--impact-only' in sys.argv:                        # reuse the rows written by a full run
        path = os.path.join(ROOT, 'results', 'local_exp56_panel2_truncation.json')
        out = json.load(open(path)); out['impact'] = impact(); json.dump(out, open(path, 'w'), indent=2)
        print(json.dumps(out['impact'], indent=2)); return
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
    out['impact'] = impact()
    json.dump(out, open(os.path.join(ROOT, 'results', 'local_exp56_panel2_truncation.json'), 'w'), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
