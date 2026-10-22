#!/usr/bin/env python3
"""
export_test_meta.py - writes test_meta.csv (ciudad, fecha_target) for the 528 test windows, in the SAME order as
X_te / y_te of the train-only dataset.  Run it where datos_limpios_multiestacion.csv lives (same code path as the
Colab / local reruns).  Output feeds exp13_label_sensitivity.py.

  python export_test_meta.py --csv datos_limpios_multiestacion.csv --out test_meta.csv
"""
import argparse
import numpy as np, pandas as pd
from trainonly_dataset import build_dataset

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--csv', required=True); ap.add_argument('--out', default='test_meta.csv')
    a = ap.parse_args()
    _, ds = build_dataset(a.csv, train_only=True)
    print('sizes', len(ds.X_tr), len(ds.X_va), len(ds.X_te), '| P60', round(ds.umbral, 2))
    m = pd.DataFrame({'ciudad': ds.meta_te['ciudad'], 'fecha': pd.to_datetime(ds.meta_te['fecha']), 'y_te': ds.y_te.numpy().astype(int)})
    assert len(m) == 528, len(m)
    m.to_csv(a.out, index=False); print('saved', a.out, '| prevalence', m['y_te'].mean().round(4))
