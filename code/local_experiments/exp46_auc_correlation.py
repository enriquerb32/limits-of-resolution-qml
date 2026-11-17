"""exp46_auc_correlation.py - correlation between the AUCs of two models scored on the same test rows.

For each pair and seed s (seed s of A against seed s of B), the DeLong covariance gives the correlation of the two
single-run AUC estimates; the median over the 15 seeds is the empirical counterpart of the correlation that the
Hanley-McNeil approximation assumes (0.75 in the paper). Also the Hanley-McNeil MDE at that correlation, to compare
with the empirical resolution curve (exp43), and the resolution of the validation partition.
Archives only (no training).  Usage: python code/local_experiments/exp46_auc_correlation.py
Output: results/local_exp46_auc_correlation.json
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
from qml_resolution import delong_covariance, hanley_mcneil_se, paired_mde_hanley_mcneil  # noqa: E402

J = os.path.join(ROOT, "results", "json")
core = np.load(os.path.join(J, "results_core_hybrids_controls_fixet0_p60_28.1_probs.npz"))
cl = np.load(os.path.join(J, "results_rerun_train_only_canonical_probs.npz"))
e28 = np.load(os.path.join(J, "results_exp28_orig_probs.npz"))
y = core["y_te"].astype(int)
P = {"RF": cl["train_only_RF_probs"], "XGB": cl["train_only_XGB_probs"], "PureLSTM": core["PureLSTM_probs"],
     "LSTM_QNN": core["LSTM_QNN_probs"], "CTRL_LINEAR4D": core["CTRL_LINEAR4D_probs"]}
prev = float(y.mean())
out = {"pairs": {}}
for a, b in [("RF", "XGB"), ("RF", "PureLSTM"), ("RF", "LSTM_QNN"), ("PureLSTM", "LSTM_QNN"), ("LSTM_QNN", "CTRL_LINEAR4D")]:
    r = []
    for s in range(15):
        _, cov = delong_covariance(y, np.vstack([P[a][s], P[b][s]]))
        r.append(float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])))
    med = float(np.median(r))
    out["pairs"][f"{a}-{b}"] = {"r_median": med, "r_min": min(r), "r_max": max(r),
                                "hm_mde_at_r": paired_mde_hanley_mcneil(len(y), prevalence=prev, rho=med)}
yv = e28["y_va"].astype(int)
n1, n0 = int(yv.sum()), int(len(yv) - yv.sum())
out["validation"] = {"n": int(len(yv)), "positives": n1, "prevalence": n1 / len(yv),
                     "hm_se_one_auc_at_0.70": hanley_mcneil_se(0.70, n1, n0),
                     "hm_mde_paired_rho_0.75": paired_mde_hanley_mcneil(len(yv), auc=0.72, prevalence=n1 / len(yv), rho=0.75),
                     "hm_mde_paired_rho_0.90": paired_mde_hanley_mcneil(len(yv), auc=0.72, prevalence=n1 / len(yv), rho=0.90)}
with open(os.path.join(ROOT, "results", "local_exp46_auc_correlation.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1)
for k, v in out["pairs"].items():
    print(f"{k:24s} r median {v['r_median']:.3f} ({v['r_min']:.3f}-{v['r_max']:.3f})  HM MDE {v['hm_mde_at_r']:.4f}")
print("validation", {k: round(v, 4) if isinstance(v, float) else v for k, v in out["validation"].items()})
