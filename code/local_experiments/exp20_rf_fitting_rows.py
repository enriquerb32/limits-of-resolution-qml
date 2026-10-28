#!/usr/bin/env python3
"""
exp20_rf_fitting_rows.py - Tree ensembles: pipeline (canonical / train-only) x fitting rows (2,273 / 2,787)
==========================================================================================================
The archived canonical tree ensembles were fitted on X_sk of the canonical checkpoint 01_data.pt, which holds the
training + validation rows (2,787). The train-only rerun (notebook 20260630_Estudio_Completo, cell 29) rebuilds the
dataset with preparar_dataset, whose X_sk is the training rows only (2,273). The archived train-only minus canonical
deltas of the trees therefore change the anomaly standardization AND the fitting rows. This script fits the four
combinations in ONE software environment, 15 seeds each, with the settings of the archived runs, and reports the row
and pipeline effects separately:
  RF   300 trees, depth 8, balanced class weights, mean imputation
  XGB  500 trees, lr 0.05, depth 4, subsample/colsample 0.8, scale_pos_weight = n_neg/n_pos, mean imputation
       (skipped if xgboost is not installed)

Usage (from the repository root):  python code/local_experiments/exp20_rf_fitting_rows.py
Output: results/local_exp20_rf_fitting_rows.json
"""
import json
import os
import sys

import numpy as np
import sklearn
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, os.path.join(ROOT, 'code')]
from trainonly_dataset import build_dataset  # noqa: E402

try:
    import xgboost
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    xgboost = None

OUT = os.path.join(ROOT, 'results', 'local_exp20_rf_fitting_rows.json')


def make_rf(seed, y):
    return RandomForestClassifier(n_estimators=300, max_depth=8, class_weight='balanced', random_state=seed, n_jobs=-1)


def make_xgb(seed, y):
    n_neg, n_pos = (y == 0).sum(), (y == 1).sum()
    return XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                         scale_pos_weight=n_neg / max(n_pos, 1), eval_metric='logloss', random_state=seed, n_jobs=-1)


def fit15(make, X, y, X_te, y_te):
    P = []
    for s in range(15):
        imp = SimpleImputer(strategy='mean')
        m = make(s, y)
        m.fit(imp.fit_transform(X), y)
        P.append(m.predict_proba(imp.transform(X_te))[:, 1])
    P = np.asarray(P)
    return P, np.array([roc_auc_score(y_te, p) for p in P])


def main():
    ck = torch.load(os.path.join(ROOT, 'checkpoints', '01_data.pt'), map_location='cpu', weights_only=False)
    dc = ck[1] if isinstance(ck, (tuple, list)) else ck['ds']
    assert len(dc.X_sk) == 2787 and len(dc.X_tr) == 2273
    _, dt = build_dataset(os.path.join(HERE, 'datos_limpios_multiestacion.csv'), train_only=True,
                          fix_et0_verano=True, umbral_override=28.1)
    rr = np.load(os.path.join(ROOT, 'results', 'json', 'results_rerun_train_only_canonical_probs.npz'))
    y = rr['y_te'].astype(int)
    assert np.array_equal(dc.y_te.numpy().astype(int), y) and np.array_equal(dt.y_te.numpy().astype(int), y)

    X_tv_t = np.vstack([dt.X_sk, dt.X_va.numpy()]); y_tv_t = np.concatenate([dt.y_sk, dt.y_va.numpy()])
    cells = {   # name: (X_fit, y_fit, X_test, archive key suffix or None)
        'canonical_2787': (dc.X_sk, dc.y_sk, dc.X_all[dc.s2:], 'canonical'),
        'canonical_2273': (dc.X_tr.numpy(), dc.y_tr.numpy(), dc.X_all[dc.s2:], None),
        'train_only_2787': (X_tv_t, y_tv_t, dt.X_te.numpy(), None),
        'train_only_2273': (dt.X_sk, dt.y_sk, dt.X_te.numpy(), 'train_only'),
    }
    models = {'RF': make_rf}
    if xgboost is not None:
        models['XGB'] = make_xgb
    else:
        print('xgboost not installed: XGB skipped')
    res = {'meta': dict(sklearn=sklearn.__version__, xgboost=getattr(xgboost, '__version__', None), seeds=15,
                        note='archived runs: scikit-learn 1.6.1 on Colab; other versions are not bit-identical')}
    for mname, make in models.items():
        r_m = {}
        for k, (X, yy, Xte, arch) in cells.items():
            P, a = fit15(make, X, yy, Xte, y)
            r = dict(n_fit=int(len(X)), mean_auc=float(a.mean()), sd=float(a.std(ddof=1)), aucs=a.tolist())
            if arch:
                ref = rr[f'{arch}_{mname}_probs']
                r['archived_mean_auc'] = float(np.mean([roc_auc_score(y, p) for p in ref]))
                r['max_abs_dp_vs_archive'] = float(np.abs(P - ref).max())
            r_m[k] = r
            print(f"{mname} {k:16s} n={len(X)} AUC {a.mean():.4f} +- {a.std(ddof=1):.4f}"
                  + (f" | archived {r['archived_mean_auc']:.4f}, max|dp| {r['max_abs_dp_vs_archive']:.1e}" if arch else ''),
                  flush=True)
        m = {k: r_m[k]['mean_auc'] for k in cells}
        r_m['effects'] = {
            'rows_2787_minus_2273_canonical': m['canonical_2787'] - m['canonical_2273'],
            'rows_2787_minus_2273_train_only': m['train_only_2787'] - m['train_only_2273'],
            'pipeline_to_minus_c_at_2273': m['train_only_2273'] - m['canonical_2273'],
            'pipeline_to_minus_c_at_2787': m['train_only_2787'] - m['canonical_2787'],
            'archived_style_delta_to2273_minus_c2787': m['train_only_2273'] - m['canonical_2787'],
        }
        for k, v in r_m['effects'].items():
            print(f'  {mname} {k:44s} {v:+.4f}')
        res[mname] = r_m
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2)
    print('saved', OUT)


if __name__ == '__main__':
    main()
