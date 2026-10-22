#!/usr/bin/env python3
"""
exp10_paired_ci.py  -  Estimand + paired uncertainty for the mean-AUC difference between two procedures.

ESTIMAND (conditional on the trained models, fixed 528-row test set):
    Delta = (1/S_A) sum_s AUC(p_{A,s})  -  (1/S_B) sum_s AUC(p_{B,s})
Three uncertainty estimates for THE SAME quantity:
  1. DeLong (iid rows). Mean-of-AUCs is linear in the DeLong structural components, so the components of each
     model are averaged over its seeds and the usual paired DeLong variance is applied to the averaged components.
  2. Common calendar-year block bootstrap (B replicates). Years are resampled with replacement, the SAME years for
     both models, and Delta is recomputed (mean over seeds) in every replicate. Conditional on the trained models.
  3. Two-way bootstrap (years + seeds). As (2) but seeds are also resampled with replacement, independently inside each
     model. Approximate 'unconditional on training' interval; assumes seeds are exchangeable and independent across
     models; with S=15 it is only indicative.
MDE = 2.80 * SE (alpha .05, power .80, normal approximation). It is a statement about design sensitivity, NOT a
significance threshold: report the CI first.

INPUT
  --probs   npz with keys '<prefix>_<MODEL>_probs' (S x N)  [e.g. results_rerun_train_only_canonical_probs.npz]
            optional: --hybrid json from colab_hybrids_perseed_probs.py (models[k]['probs'], any number of seeds)
            optional: --extra  npz with keys '<NAME>_probs' (S x N) and 'y_te'  [e.g. output of colab_qml_core_trainonly.py]
  --panel   panel_checkpoint.csv  (to rebuild the calendar year of every test row), or --years years_te.npy
  --pairs   e.g. RF-XGB RF-LSTM XGB-LSTM RF-LSTM+QNN
USAGE
  python exp10_paired_ci.py --probs results_rerun_train_only_canonical_probs.npz --prefix train_only \
      --panel panel_checkpoint.csv --pairs RF-XGB RF-LSTM XGB-LSTM --B 2000 --out results_exp10_paired_ci.json
"""
import argparse, json, sys
import numpy as np
from scipy.stats import rankdata, norm

def auc_fast(y, p):
    r = rankdata(p); m = int(y.sum()); n = len(y) - m
    return (r[y == 1].sum() - m * (m + 1) / 2) / (m * n)

def delong_components(y, p):
    pos = y == 1; m = int(pos.sum()); n = len(y) - m
    tz = rankdata(p); tx = rankdata(p[pos]); ty = rankdata(p[~pos])
    v10 = (tz[pos] - tx) / n                  # per positive
    v01 = 1.0 - (tz[~pos] - ty) / m           # per negative
    auc = tz[pos].sum() / (m * n) - (m + 1) / (2 * n)
    return auc, v10, v01

def test_years_from_panel(panel_csv, n_expected, window=12, test_start=2008):
    import pandas as pd
    df = pd.read_csv(panel_csv)
    dcol = 'fecha' if 'fecha' in df.columns else df.columns[0]
    df[dcol] = pd.to_datetime(df[dcol])
    city_cols = [c for c in df.columns if c.startswith('ciudad_')]
    if not city_cols:
        raise SystemExit("panel has no ciudad_* one-hot columns; pass --years instead")
    yrs = []
    for c in city_cols:
        d = df.loc[df[c] == 1, dcol].sort_values().reset_index(drop=True)
        yrs.extend(d.iloc[window:].dt.year.tolist())      # window targets = month i+window
    yrs = np.sort(np.array(yrs))
    yrs = yrs[yrs >= test_start]
    if len(yrs) != n_expected:
        raise SystemExit(f"rebuilt {len(yrs)} test years, expected {n_expected}: pass --years years_te.npy instead")
    return yrs

class Model:
    def __init__(self, probs, y):
        self.P = np.atleast_2d(np.asarray(probs, float)); self.S = self.P.shape[0]
        comp = [delong_components(y, p) for p in self.P]
        self.auc = np.array([c[0] for c in comp])
        self.v10 = np.mean([c[1] for c in comp], axis=0)   # averaged structural components (positives)
        self.v01 = np.mean([c[2] for c in comp], axis=0)   # (negatives)
    @property
    def mean_auc(self): return float(self.auc.mean())

def paired_delong(A, B, y):
    pos = y == 1; m = int(pos.sum()); n = len(y) - m
    S10 = np.cov(np.vstack([A.v10, B.v10])); S01 = np.cov(np.vstack([A.v01, B.v01]))
    var = (S10[0, 0] + S10[1, 1] - 2 * S10[0, 1]) / m + (S01[0, 0] + S01[1, 1] - 2 * S01[0, 1]) / n
    d = A.mean_auc - B.mean_auc; se = float(np.sqrt(max(var, 1e-18)))
    z = d / se
    return d, se, z, float(2 * (1 - norm.cdf(abs(z))))

def boot(A, B, y, years, B_rep, rng, resample_seeds):
    uy = np.unique(years); idx_by = {u: np.where(years == u)[0] for u in uy}
    out = np.empty(B_rep)
    for b in range(B_rep):
        ys = rng.choice(uy, size=len(uy), replace=True)
        idx = np.concatenate([idx_by[u] for u in ys]); yy = y[idx]
        if yy.sum() in (0, len(yy)): out[b] = np.nan; continue
        sa = rng.integers(0, A.S, A.S) if resample_seeds else np.arange(A.S)
        sb = rng.integers(0, B.S, B.S) if resample_seeds else np.arange(B.S)
        a = np.mean([auc_fast(yy, A.P[s][idx]) for s in sa]); bb = np.mean([auc_fast(yy, B.P[s][idx]) for s in sb])
        out[b] = a - bb
    return out[~np.isnan(out)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probs'); ap.add_argument('--prefix', default='train_only')
    ap.add_argument('--hybrid'); ap.add_argument('--extra')
    ap.add_argument('--panel'); ap.add_argument('--years')
    ap.add_argument('--pairs', nargs='+', required=True)
    ap.add_argument('--B', type=int, default=2000); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='results_exp10_paired_ci.json')
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    models, y = {}, None
    if a.probs:
        z = np.load(a.probs); y = z['y_te'].astype(int)
        for k in z.files:
            if k.startswith(a.prefix + '_') and k.endswith('_probs'):
                models[k[len(a.prefix) + 1:-6]] = z[k]
    if a.hybrid:
        J = json.load(open(a.hybrid)); yh = np.array(J['y_te'])
        if y is not None: assert (yh == y).all(), 'hybrid y_te differs'
        y = yh
        for k, v in J['models'].items(): models[k] = np.array(v['probs'])
    if a.extra:
        z = np.load(a.extra); ye = z['y_te'].astype(int)
        if y is not None: assert (ye == y).all(), 'extra y_te differs'
        y = ye
        for k in z.files:
            if k.endswith('_probs'): models[k[:-6]] = z[k]
    assert y is not None and models, 'no models loaded'
    if a.years: years = np.load(a.years)
    elif a.panel: years = test_years_from_panel(a.panel, len(y))
    else: sys.exit('need --panel or --years')
    assert len(years) == len(y)
    print('models:', {k: v.shape for k, v in models.items()}, '| test years', years.min(), '-', years.max(), '| clusters', len(np.unique(years)))
    M = {k: Model(v, y) for k, v in models.items()}
    res = {}
    print(f"\n{'pair':<22}{'Delta':>8}{'SE_DL':>8}{'z':>6}{'p':>7} | {'block 95% CI':^20}{'SE_blk':>8} | {'2-way 95% CI':^20} | MDE_iid MDE_blk")
    for pr in a.pairs:
        na, nb = pr.split('-', 1) if pr.count('-') == 1 else pr.rsplit('-', 1)
        if na not in M or nb not in M:
            # names that contain '-' (e.g. PI-QNN-LbL): try all split points
            for i in range(1, len(pr)):
                if pr[i] == '-' and pr[:i] in M and pr[i + 1:] in M: na, nb = pr[:i], pr[i + 1:]; break
        A, Bm = M[na], M[nb]
        d, se, zz, p = paired_delong(A, Bm, y)
        bs = boot(A, Bm, y, years, a.B, rng, False); b2 = boot(A, Bm, y, years, a.B, rng, True)
        ci = np.percentile(bs, [2.5, 97.5]); ci2 = np.percentile(b2, [2.5, 97.5]); sb = float(bs.std(ddof=1))
        res[pr] = dict(S_A=A.S, S_B=Bm.S, delta=d, se_delong=se, z=zz, p_delong=p, ci_delong=[d - 1.96 * se, d + 1.96 * se],
                       ci_block=ci.tolist(), se_block=sb, ci_two_way=ci2.tolist(), se_two_way=float(b2.std(ddof=1)),
                       mde_delong=2.8 * se, mde_block=2.8 * sb)
        print(f"{pr:<22}{d:8.4f}{se:8.4f}{zz:6.2f}{p:7.3f} | [{ci[0]:7.4f},{ci[1]:7.4f}]{sb:8.4f} | [{ci2[0]:7.4f},{ci2[1]:7.4f}] | {2.8*se:.3f}   {2.8*sb:.3f}")
    json.dump(dict(n_test=len(y), n_clusters=int(len(np.unique(years))), B=a.B, pairs=res), open(a.out, 'w'), indent=2)
    print('\nsaved', a.out)
    print('\nLaTeX rows (Delta [block CI] / SE_DeLong / MDE_DeLong):')
    for pr, r in res.items():
        print(f"{pr.replace('-', ' $-$ ')} & {r['delta']:+.3f} & [{r['ci_block'][0]:+.3f}, {r['ci_block'][1]:+.3f}] & {r['se_delong']:.4f} & {r['mde_delong']:.3f} \\\\")

if __name__ == '__main__':
    main()
