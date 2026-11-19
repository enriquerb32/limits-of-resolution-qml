"""exp45_cluster_periods.py - cluster/iid ratio of the seed-mean AUC difference under year subsets (same estimator as exp42):
all years, without the partial year 2026, 2008-2012 (more than two stations; all four only in 2008-2011) and 2013-2026 (two stations).
Also the cross-station share of the cluster variance per period. AUCs are recomputed on each subset.
Archives only (no training).  Usage: python code/local_experiments/exp45_cluster_periods.py
Output: results/local_exp45_cluster_periods.json"""
import json
import os

import numpy as np
import pandas as pd

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
J = root + "/results/json/"
core = np.load(J + "results_core_hybrids_controls_fixet0_p60_28.1_probs.npz")
cl = np.load(J + "results_rerun_train_only_canonical_probs.npz")
meta = pd.read_csv(root + "/results/test_meta.csv")
y_all = core["y_te"].astype(int)
yr_all = pd.to_datetime(meta.fecha).dt.year.values
st_all = meta.ciudad.values
P = {"RF": cl["train_only_RF_probs"], "LSTM": core["PureLSTM_probs"], "QNN": core["LSTM_QNN_probs"],
     "XGB": cl["train_only_XGB_probs"], "L4D": core["CTRL_LINEAR4D_probs"]}


def comps(y, p):
    pos, neg = y == 1, y == 0
    a, b = p[pos], p[neg]
    psi = (a[:, None] > b[None]) + 0.5 * (a[:, None] == b[None])
    A = psi.mean()
    d = np.empty(len(p))
    d[pos] = (psi.mean(1) - A) / pos.sum()
    d[neg] = (psi.mean(0) - A) / neg.sum()
    return A, d


def stats(a, b, mask):
    y, yr, st = y_all[mask], yr_all[mask], st_all[mask]
    da = np.mean([comps(y, s[mask])[1] for s in P[a]], 0)
    db = np.mean([comps(y, s[mask])[1] for s in P[b]], 0)
    gap = np.mean([comps(y, s[mask])[0] for s in P[a]]) - np.mean([comps(y, s[mask])[0] for s in P[b]])
    d = da - db
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    viid = (d[y == 1] ** 2).sum() * n1 / (n1 - 1) + (d[y == 0] ** 2).sum() * n0 / (n0 - 1)
    G = np.unique(yr)
    S = np.array([d[yr == g].sum() for g in G])
    vcl = len(G) / (len(G) - 1) * (S ** 2).sum()
    within = sum((d[(yr == g) & (st == s)].sum()) ** 2 for g in G for s in np.unique(st[yr == g]))
    cross = (S ** 2).sum() - within
    return dict(n=int(mask.sum()), years=len(G), gap=gap, se_iid=viid ** .5, ratio=(vcl / viid) ** .5, cross_share=cross / within)


subsets = {"all": np.ones_like(yr_all, bool), "no2026": yr_all != 2026,
           "2008-2012": yr_all <= 2012, "2013-2026": yr_all >= 2013}
out = {}
for a, b in [("RF", "LSTM"), ("RF", "QNN"), ("RF", "XGB"), ("LSTM", "QNN"), ("L4D", "QNN")]:
    for k, m in subsets.items():
        r = stats(a, b, m)
        out.setdefault(f"{a}-{b}", {})[k] = {kk: (float(v) if not isinstance(v, int) else v) for kk, v in r.items()}
        print(f"{a}-{b:5s} {k:10s} n={r['n']:3d} K={r['years']:2d} gap={r['gap']:+.4f} se={r['se_iid']:.4f} "
              f"cluster/iid={r['ratio']:.2f} cross/within={r['cross_share']:+.2f}")
print("rows by period:", {k: int(m.sum()) for k, m in subsets.items()}, "2026 rows:", int((yr_all == 2026).sum()),
      "months 2026:", sorted(set(pd.to_datetime(meta.fecha[yr_all == 2026]).dt.month)))
with open(root + "/results/local_exp45_cluster_periods.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1)
