"""exp44_xgb_iterations.py - number of boosting rounds of the 15 XGBoost fits of the paper (LOR_DATASET=v2).

Refits exp41's XGBoost (same data, imputer, tuned parameters, seeds, early stopping on the validation AUC), checks that
the test probabilities equal the archived ones exactly, and writes results/json/xgb_best_iterations.json.
Needs the daily file of code/etl_v2.py (not redistributed).  Usage: python code/local_experiments/exp44_xgb_iterations.py
"""
import json
import os
import sys

import numpy as np
from sklearn.impute import SimpleImputer

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
here = os.path.join(root, "code", "local_experiments")
sys.path.insert(0, here)
sys.path.insert(1, os.path.join(root, "code"))
import dataset_v2  # noqa: E402
from trainonly_dataset import build_dataset  # noqa: E402

_, ds = build_dataset(os.path.join(here, "datos_limpios_multiestacion.csv"), train_only=True, fix_et0_verano=True)
imp = SimpleImputer(strategy="mean").fit(ds.X_sk)
Xt, Xv, Xe = imp.transform(ds.X_sk), imp.transform(ds.X_va.numpy()), imp.transform(ds.X_te.numpy())
yt, yv = ds.y_sk.astype(int), ds.y_va.numpy().astype(int)
arch = np.load(os.path.join(root, "results", "json", "results_rerun_train_only_canonical_probs.npz"))["train_only_XGB_probs"]
iters, trees, maxdiff = [], [], 0.0
for s in range(15):
    m = dataset_v2.xgb_fit_v2(s, Xt, yt, Xv, yv)
    iters.append(m.best_iteration)
    trees.append(len(m.get_booster().get_dump()))
    maxdiff = max(maxdiff, float(np.abs(m.predict_proba(Xe)[:, 1] - arch[s]).max()))
print("best_iteration per seed:", iters)
print("trees in booster:", trees)
print("max |p - archived p|:", maxdiff)
print("train rows", len(yt), "val rows", len(yv), "test rows", len(Xe))
assert maxdiff == 0.0, "refit does not reproduce the archived XGBoost probabilities"
out = {"best_iteration": iters, "trees_used": [i + 1 for i in iters], "trees_fitted": trees,
       "trees_used_median": float(np.median([i + 1 for i in iters])), "max_abs_diff_archived_probs": maxdiff}
with open(os.path.join(root, "results", "json", "xgb_best_iterations.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1)
