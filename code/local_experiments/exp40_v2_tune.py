#!/usr/bin/env python3
"""
exp40_v2_tune.py - v2: declared automated search for BOTH tree baselines (protocol item 6)
=========================================================================================
Objective: AUC on the validation partition (1997-2007) of a model fitted on the training rows (<= 1996), mean
imputation fitted on the training rows, model seed 0. Sampler TPE (seed 0), N trials per model (default 50).
  RF : n_estimators 100-500 (step 50), max_depth 3-16, min_samples_split 2-20, min_samples_leaf 1-10,
       max_features {sqrt, log2, 0.5}; class_weight balanced (as v1).
  XGB: learning_rate 0.01-0.3 (log), max_depth 2-8, min_child_weight 1-20 (log), subsample 0.5-1,
       colsample_bytree 0.5-1, reg_lambda 1e-3-10 (log), gamma 0-5; up to 2,000 trees with early stopping
       (50 rounds) on the validation AUC; scale_pos_weight = negatives/positives of the training rows.
The validation partition is used both for early stopping and for the search; the test partition is not touched.
Output: results/json/v2_tuned_params.json (read by dataset_v2.rf_v2 / xgb_fit_v2 in every v2 script).
Usage (LOR_DATASET=v2):  python code/local_experiments/exp40_v2_tune.py [--trials 50]      ~15-30 min CPU
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
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(HERE)), 'code'))
import dataset_v2  # noqa: E402
from trainonly_dataset import build_dataset  # noqa: E402

RAW_CSV = os.path.join(HERE, 'datos_limpios_multiestacion.csv')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=50)
    a = ap.parse_args()
    if not dataset_v2.V2:
        raise SystemExit('set LOR_DATASET=v2')
    import optuna
    from xgboost import XGBClassifier
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _, ds = build_dataset(RAW_CSV, train_only=True, fix_et0_verano=True)
    imp = SimpleImputer(strategy='mean').fit(ds.X_sk)
    X_tr, X_va = imp.transform(ds.X_sk), imp.transform(ds.X_va.numpy())
    y_tr, y_va = ds.y_sk.astype(int), ds.y_va.numpy().astype(int)
    print(f'rows train/val/test = {len(ds.X_tr)}/{len(ds.X_va)}/{len(ds.X_te)} | P60 = {ds.umbral:.2f} mm')

    def rf_obj(t):
        p = dict(n_estimators=t.suggest_int('n_estimators', 100, 500, step=50), max_depth=t.suggest_int('max_depth', 3, 16),
                 min_samples_split=t.suggest_int('min_samples_split', 2, 20),
                 min_samples_leaf=t.suggest_int('min_samples_leaf', 1, 10),
                 max_features=t.suggest_categorical('max_features', ['sqrt', 'log2', 0.5]))
        m = RandomForestClassifier(**p, class_weight='balanced', random_state=0, n_jobs=-1).fit(X_tr, y_tr)
        return roc_auc_score(y_va, m.predict_proba(X_va)[:, 1])

    def xgb_obj(t):
        p = dict(learning_rate=t.suggest_float('learning_rate', 0.01, 0.3, log=True), max_depth=t.suggest_int('max_depth', 2, 8),
                 min_child_weight=t.suggest_float('min_child_weight', 1, 20, log=True),
                 subsample=t.suggest_float('subsample', 0.5, 1.0), colsample_bytree=t.suggest_float('colsample_bytree', 0.5, 1.0),
                 reg_lambda=t.suggest_float('reg_lambda', 1e-3, 10, log=True), gamma=t.suggest_float('gamma', 0, 5))
        n_neg, n_pos = (y_tr == 0).sum(), (y_tr == 1).sum()
        m = XGBClassifier(**p, n_estimators=2000, early_stopping_rounds=50, eval_metric='auc', tree_method='hist',
                          scale_pos_weight=n_neg / max(n_pos, 1), random_state=0, n_jobs=-1)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        t.set_user_attr('best_iteration', int(m.best_iteration))
        return roc_auc_score(y_va, m.predict_proba(X_va)[:, 1])

    out = {}
    for name, obj in [('RF', rf_obj), ('XGB', xgb_obj)]:
        st = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=0))
        st.optimize(obj, n_trials=a.trials)
        out[name] = dict(params=st.best_params, best_val_auc=float(st.best_value), n_trials=a.trials, sampler='TPE(seed=0)',
                         best_iteration=st.best_trial.user_attrs.get('best_iteration'))
        print(f'{name}: {st.best_params} | validation AUC {st.best_value:.4f}', flush=True)
    import sklearn, xgboost
    out['versions'] = dict(sklearn=sklearn.__version__, xgboost=xgboost.__version__, optuna=optuna.__version__)
    os.makedirs(os.path.dirname(dataset_v2.TUNED), exist_ok=True)
    json.dump(out, open(dataset_v2.TUNED, 'w'), indent=2)
    print('wrote', dataset_v2.TUNED)


if __name__ == '__main__':
    main()
