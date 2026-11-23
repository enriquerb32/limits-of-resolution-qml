#!/usr/bin/env python3
"""
exp41_v2_classical.py - v2: 15-seed tree baselines and the classical archive in the v1 file format
===================================================================================================
Fits, on the training rows only (the same rows as every neural model), with seeds 0..S-1:
  RF      tuned Random Forest (results/json/v2_tuned_params.json)
  XGB     tuned XGBoost, early stopping on the validation AUC
  RF_untuned / XGB_untuned   the v1 configurations (300 trees depth 8; 500 trees depth 4, lr 0.05), for the record
and writes results/json/results_rerun_train_only_canonical_probs.npz with the keys read by the analysis scripts
(train_only_RF, train_only_XGB, train_only_LSTM, y_te, y_te_ext, train_only_acp_te). train_only_LSTM is copied from
the v2 hybrids archive (PureLSTM, same seeds; file name kept from v1, see README.md, section 2): run exp17 first.
v2 has a single pipeline: the canonical_* keys are copies of the train_only_* keys (scripts that compare the two
pipelines are not part of the v2 run).
Usage (LOR_DATASET=v2):  python code/local_experiments/exp41_v2_classical.py [--seeds 15]     ~5 min
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import dataset_v2  # noqa: E402
from trainonly_dataset import build_dataset  # noqa: E402

RAW_CSV = os.path.join(HERE, 'datos_limpios_multiestacion.csv')
OUT = os.path.join(ROOT, 'results', 'json', 'results_rerun_train_only_canonical_probs.npz')
CORE = os.path.join(ROOT, 'results', 'json', 'results_hybrids_trainonly_fixet0_p60_28.1_probs.npz')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--seeds', type=int, default=15); a = ap.parse_args()
    if not dataset_v2.V2:
        raise SystemExit('set LOR_DATASET=v2')
    from xgboost import XGBClassifier
    _, ds = build_dataset(RAW_CSV, train_only=True, fix_et0_verano=True)
    imp = SimpleImputer(strategy='mean').fit(ds.X_sk)
    Xt, Xv, Xe = imp.transform(ds.X_sk), imp.transform(ds.X_va.numpy()), imp.transform(ds.X_te.numpy())
    yt, yv, ye = ds.y_sk.astype(int), ds.y_va.numpy().astype(int), ds.y_te.numpy().astype(int)
    n_neg, n_pos = (yt == 0).sum(), (yt == 1).sum()
    P = {k: [] for k in ['RF', 'XGB', 'RF_untuned', 'XGB_untuned']}
    for s in range(a.seeds):
        P['RF'].append(dataset_v2.rf_v2(s).fit(Xt, yt).predict_proba(Xe)[:, 1])
        P['XGB'].append(dataset_v2.xgb_fit_v2(s, Xt, yt, Xv, yv).predict_proba(Xe)[:, 1])
        P['RF_untuned'].append(RandomForestClassifier(n_estimators=300, max_depth=8, class_weight='balanced',
                                                      random_state=s, n_jobs=-1).fit(Xt, yt).predict_proba(Xe)[:, 1])
        P['XGB_untuned'].append(XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=4, subsample=0.8,
                                              colsample_bytree=0.8, scale_pos_weight=n_neg / max(n_pos, 1),
                                              eval_metric='logloss', random_state=s, n_jobs=-1).fit(Xt, yt).predict_proba(Xe)[:, 1])
        print(f'seed {s:2d} ' + ' '.join(f'{k} {roc_auc_score(ye, v[-1]):.4f}' for k, v in P.items()), flush=True)
    arr = {f'train_only_{k}_probs': np.asarray(v) for k, v in P.items()}
    if os.path.exists(CORE):
        core = np.load(CORE)
        assert np.array_equal(core['y_te'].astype(int), ye), 'core archive has other test labels'
        arr['train_only_LSTM_probs'] = core['PureLSTM_probs'][:a.seeds]
    else:
        print('WARNING: no core archive yet; train_only_LSTM missing (run exp17 first, then this script again)')
    for k in [k for k in arr if k.startswith('train_only_')]:
        arr[k.replace('train_only_', 'canonical_')] = arr[k]
    arr.update(y_te=ye, y_te_ext=ds.y_te_ext.numpy().astype(int), train_only_acp_te=ds.dana_te, canonical_acp_te=ds.dana_te)
    np.savez_compressed(OUT, **arr)
    summ = {k: dict(mean=float(np.mean([roc_auc_score(ye, p) for p in v])), sd=float(np.std([roc_auc_score(ye, p) for p in v], ddof=1)))
            for k, v in P.items()}
    json.dump(dict(summary=summ, n_test=int(len(ye)), threshold=float(ds.umbral)),
              open(os.path.join(ROOT, 'results', 'local_exp41_v2_classical.json'), 'w'), indent=2)
    print(json.dumps(summ, indent=1)); print('wrote', OUT)


if __name__ == '__main__':
    main()
