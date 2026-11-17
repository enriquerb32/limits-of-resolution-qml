#!/usr/bin/env python3
"""
exp36_rf_optuna.py - documented automated search for the Random Forest baseline (train-only pipeline, 2,273 rows)
=================================================================================================================
The manuscript reports an Optuna-tuned Random Forest (archived in results/json/results_capacity_matched_lstm.json,
key 'optuna_tuned'; best 250 trees, depth 8, min_samples_split 8, max_features 'sqrt'; 15-seed test AUC
0.6798 +- 0.0045) whose search script was not preserved. This script reruns the search with a declared space so that
the result is reproducible:
  * objective: AUC on the validation partition (514 rows, 1997-2007) of a forest fitted on the 2,273 training rows
    (random_state 0, class_weight 'balanced', mean imputation fitted on the training rows);
  * sampler: TPE (seed 0), 50 trials;
  * space: n_estimators 100-500 (step 50), max_depth 3-16, min_samples_split 2-20, min_samples_leaf 1-10,
    max_features {'sqrt', 'log2', 0.5};
  * evaluation: the best configuration refitted with seeds 0-14, test AUC on the 528 test rows (same rows and seeds
    as the untuned Random Forest of the paper, 0.6829 +- 0.0035).
Requires: pip install optuna.   Runtime: ~5-10 min on a laptop CPU.
Usage: python code/local_experiments/exp36_rf_optuna.py      Output: results/local_exp36_rf_optuna.json
"""
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
sys.path.insert(1, os.path.join(ROOT, 'code'))
from exp27_controls import RAW_CSV, build_dataset  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'local_exp36_rf_optuna.json')
N_TRIALS, SEEDS = 50, 15


def main():
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _, ds = build_dataset(RAW_CSV, train_only=True, fix_et0_verano=True, umbral_override=28.1)
    assert (len(ds.X_tr), len(ds.X_va), len(ds.X_te)) == (2273, 514, 528)
    imp = SimpleImputer(strategy='mean').fit(ds.X_tr.numpy())
    X_tr, X_va, X_te = (imp.transform(x.numpy()) for x in (ds.X_tr, ds.X_va, ds.X_te))
    y_tr, y_va, y_te = (y.numpy().astype(int) for y in (ds.y_tr, ds.y_va, ds.y_te))

    def params(trial):
        return dict(n_estimators=trial.suggest_int('n_estimators', 100, 500, step=50),
                    max_depth=trial.suggest_int('max_depth', 3, 16),
                    min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                    min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
                    max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5]))

    def objective(trial):
        rf = RandomForestClassifier(**params(trial), class_weight='balanced', random_state=0, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        return roc_auc_score(y_va, rf.predict_proba(X_va)[:, 1])

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=N_TRIALS)
    best = study.best_params
    aucs = []
    for s in range(SEEDS):
        rf = RandomForestClassifier(**best, class_weight='balanced', random_state=s, n_jobs=-1).fit(X_tr, y_tr)
        aucs.append(float(roc_auc_score(y_te, rf.predict_proba(X_te)[:, 1])))
    res = dict(n_trials=N_TRIALS, sampler='TPE(seed=0)', best_params=best, best_val_auc=float(study.best_value),
               test_auc_seeds=aucs, test_auc_mean=float(np.mean(aucs)), test_auc_sd=float(np.std(aucs, ddof=1)),
               space=dict(n_estimators='100-500 step 50', max_depth='3-16', min_samples_split='2-20',
                          min_samples_leaf='1-10', max_features=['sqrt', 'log2', 0.5]))
    json.dump(res, open(OUT, 'w'), indent=2)
    print(f'best {best} | validation AUC {study.best_value:.4f} | test AUC {np.mean(aucs):.4f} +- {np.std(aucs, ddof=1):.4f}')
    print('wrote', OUT)


if __name__ == '__main__':
    main()
