#!/usr/bin/env python3
"""
exp53_external_benchmark.py - which decisions does the protocol take on small clustered test sets when there is signal?
=====================================================================================================================
Implements the design recorded before any fit in audit/prediction_external.txt (commit 1e88a96): California Housing,
binary target (median house value above the training median), 0.25-degree spatial cells as clusters, split by cell
(50/15/35 %, seed 2026), models LR / RF / XGB / MLP (seeds 0-4), contrasts C1 RF-LR, C2 XGB-RF, C3 MLP-RF, margin
0.01 AUC, primary procedure CR2 with Bell-McCaffrey df over cells. Reference population: test cells with 10-60 rows;
1,000 small tests of 19 cells drawn without replacement from it.

  python code/local_experiments/exp53_external_benchmark.py          (~10 min)
Output: results/local_exp53_external_benchmark.json, results/json/results_exp53_external_probs.npz
Data: data/external/ (scikit-learn cache; not redistributed).
"""
import json
import os
import sys

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from qml_resolution import cluster_robust_delta, paired_delong  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'local_exp53_external_benchmark.json')
NPZ = os.path.join(ROOT, 'results', 'json', 'results_exp53_external_probs.npz')
SEEDS, MARGIN, DRAWS, K_SMALL, SPLIT_SEED = range(5), 0.01, 1000, 19, 2026
CONTRASTS = [('RF', 'LR'), ('XGB', 'RF'), ('MLP', 'RF')]


def data():
    d = fetch_california_housing(data_home=os.path.join(ROOT, 'data', 'external'), as_frame=True).frame
    cell = (np.floor(d.Latitude / 0.25).astype(int).astype(str) + '_' + np.floor(d.Longitude / 0.25).astype(int).astype(str)).values
    cells = np.unique(cell)
    rng = np.random.default_rng(SPLIT_SEED); perm = rng.permutation(cells)
    n_tr, n_va = int(0.50 * len(cells)), int(0.15 * len(cells))
    part = {c: 'train' for c in perm[:n_tr]} | {c: 'val' for c in perm[n_tr:n_tr + n_va]} | {c: 'test' for c in perm[n_tr + n_va:]}
    split = np.array([part[c] for c in cell])
    X = d.drop(columns='MedHouseVal').values
    thr = float(np.median(d.MedHouseVal.values[split == 'train']))
    y = (d.MedHouseVal.values > thr).astype(int)
    return X, y, cell, split, thr


def fit_all(X, y, split):
    tr, va, te = split == 'train', split == 'val', split == 'test'
    sc = StandardScaler().fit(X[tr]); Z = sc.transform(X)
    from xgboost import XGBClassifier
    P = {m: [] for m in ('LR', 'RF', 'XGB', 'MLP')}
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(Z[tr], y[tr]).predict_proba(Z[te])[:, 1]
    for s in SEEDS:
        P['LR'].append(lr)
        P['RF'].append(RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=s, n_jobs=-1)
                       .fit(X[tr], y[tr]).predict_proba(X[te])[:, 1])
        xg = XGBClassifier(tree_method='hist', max_depth=6, learning_rate=0.05, n_estimators=2000, early_stopping_rounds=50,
                           eval_metric='auc', random_state=s, n_jobs=-1)
        P['XGB'].append(xg.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False).predict_proba(X[te])[:, 1])
        P['MLP'].append(MLPClassifier((64, 32), early_stopping=True, max_iter=500, random_state=s)
                        .fit(Z[tr], y[tr]).predict_proba(Z[te])[:, 1])
    return {k: np.array(v) for k, v in P.items()}


def contrast(y, P, cl, a, b):
    cr = cluster_robust_delta(y, P[a], P[b], cl); lo, hi = cr.ci()
    dl = paired_delong(y, P[a], P[b]); dlo, dhi = dl.ci()
    return dict(delta=cr.delta, ci_cr2=[lo, hi], p_cr2=cr.p_value, df=cr.df, se_cr2=cr.se_cr2, se_delong=dl.se,
                ci_delong=[dlo, dhi], resolved_cr2=bool(lo > 0 or hi < 0), resolved_delong=bool(dlo > 0 or dhi < 0),
                mde_cr2=float(2.80 * cr.se_cr2))


def posthoc_all_cells(draws=DRAWS):
    """POST HOC (not in the recorded design): reference on all 182 test cells and small tests of 19 cells drawn from
    all of them, so that the sampling fraction is about 10 % (finite-population factor 0.95) instead of 19/39; uses the
    archived predictions only."""
    z = np.load(NPZ, allow_pickle=True)                     # our own archive: cell labels are stored as objects
    yt, ct = z['y_te'].astype(int), z['cell_te'].astype(str)
    P = {k[:-6]: z[k] for k in z.files if k.endswith('_probs')}
    cells = np.unique(ct)
    ref = {f'{a}-{b}': contrast(yt, P, ct, a, b) for a, b in CONTRASTS}
    rng = np.random.default_rng(530)
    small = {k: [] for k in ref}; n_rows = []
    while len(n_rows) < draws:
        m = np.isin(ct, rng.choice(cells, K_SMALL, replace=False))
        if yt[m].min() == yt[m].max():
            continue
        n_rows.append(int(m.sum()))
        for a, b in CONTRASTS:
            small[f'{a}-{b}'].append(contrast(yt[m], {k: v[:, m] for k, v in P.items()}, ct[m], a, b))
    out = dict(note='post hoc: reference = all test cells; 19 of 182 cells per small test',
               finite_population_factor=float(np.sqrt(1 - K_SMALL / len(cells))), cells=int(len(cells)),
               rows_median=float(np.median(n_rows)), rows_range=[int(min(n_rows)), int(max(n_rows))], by_contrast={})
    for key, rows in small.items():
        rd = ref[key]['delta']
        out['by_contrast'][key] = dict(
            reference=ref[key], resolved_cr2=float(np.mean([r['resolved_cr2'] for r in rows])),
            resolved_delong=float(np.mean([r['resolved_delong'] for r in rows])),
            coverage_cr2=float(np.mean([r['ci_cr2'][0] <= rd <= r['ci_cr2'][1] for r in rows])),
            coverage_delong=float(np.mean([r['ci_delong'][0] <= rd <= r['ci_delong'][1] for r in rows])))
    return out


def main():
    if '--posthoc' in sys.argv:
        res = json.load(open(OUT)); res['posthoc_all_cells'] = posthoc_all_cells()
        json.dump(res, open(OUT, 'w'), indent=2)
        print(json.dumps({k: v for k, v in res['posthoc_all_cells'].items() if k != 'by_contrast'}, indent=1))
        for k, v in res['posthoc_all_cells']['by_contrast'].items():
            print(k, {a: round(b, 3) for a, b in v.items() if a != 'reference'}, 'ref d', round(v['reference']['delta'], 4))
        return
    X, y, cell, split, thr = data()
    te = split == 'test'
    P = fit_all(X, y, split)
    yt, ct = y[te], cell[te]
    np.savez_compressed(NPZ, y_te=yt, cell_te=ct, **{f'{k}_probs': v for k, v in P.items()})
    sizes = {c: int((ct == c).sum()) for c in np.unique(ct)}
    ref_cells = np.array(sorted(c for c, n in sizes.items() if 10 <= n <= 60))
    ref = np.isin(ct, ref_cells)
    res = dict(threshold=thr, n={k: int((split == k).sum()) for k in ('train', 'val', 'test')},
               cells={k: int(len(np.unique(cell[split == k]))) for k in ('train', 'val', 'test')},
               reference_population=dict(cells=int(len(ref_cells)), rows=int(ref.sum())),
               auc_all_test={k: float(np.mean([roc_auc_score(yt, p) for p in v])) for k, v in P.items()},
               auc_reference={k: float(np.mean([roc_auc_score(yt[ref], p[ref]) for p in v])) for k, v in P.items()})
    Pr = {k: v[:, ref] for k, v in P.items()}
    res['contrasts_all_test'] = {f'{a}-{b}': contrast(yt, P, ct, a, b) for a, b in CONTRASTS}
    res['contrasts_reference'] = {f'{a}-{b}': contrast(yt[ref], Pr, ct[ref], a, b) for a, b in CONTRASTS}
    rng = np.random.default_rng(53)
    yr, cr_ = yt[ref], ct[ref]
    small = {f'{a}-{b}': [] for a, b in CONTRASTS}; n_small = []
    for _ in range(DRAWS):
        pick = rng.choice(ref_cells, K_SMALL, replace=False)
        m = np.isin(cr_, pick)
        if yr[m].min() == yr[m].max():
            continue
        n_small.append(int(m.sum()))
        for a, b in CONTRASTS:
            small[f'{a}-{b}'].append(contrast(yr[m], {k: v[:, m] for k, v in Pr.items()}, cr_[m], a, b))
    res['small_tests'] = dict(draws=len(n_small), rows_median=float(np.median(n_small)),
                              rows_range=[int(min(n_small)), int(max(n_small))], by_contrast={})
    for key, rows in small.items():
        rd = res['contrasts_reference'][key]['delta']
        d = np.array([r['delta'] for r in rows])
        sgn = np.sign(rd)
        res['small_tests']['by_contrast'][key] = dict(
            reference_delta=rd, reference_within_margin=bool(abs(rd) < MARGIN),
            resolved_cr2=float(np.mean([r['resolved_cr2'] for r in rows])),
            resolved_delong=float(np.mean([r['resolved_delong'] for r in rows])),
            wrong_sign_cr2=float(np.mean([r['resolved_cr2'] and np.sign(r['delta']) == -sgn for r in rows])),
            wrong_sign_delong=float(np.mean([r['resolved_delong'] and np.sign(r['delta']) == -sgn for r in rows])),
            coverage_cr2=float(np.mean([r['ci_cr2'][0] <= rd <= r['ci_cr2'][1] for r in rows])),
            coverage_delong=float(np.mean([r['ci_delong'][0] <= rd <= r['ci_delong'][1] for r in rows])),
            sd_delta=float(d.std(ddof=1)), mde_empirical=float(2.80 * d.std(ddof=1)),
            mde_predicted=float(res['contrasts_reference'][key]['mde_cr2'] * np.sqrt(ref.sum() / np.median(n_small))),
            mean_mde_cr2_small=float(np.mean([r['mde_cr2'] for r in rows])))
    s = res['small_tests']['by_contrast']
    c1 = res['contrasts_reference']['RF-LR']
    res['predictions'] = {
        'P1': bool(c1['resolved_cr2'] and c1['delta'] > MARGIN),
        'P2': bool(s['RF-LR']['resolved_cr2'] >= 0.80 and all(v['wrong_sign_cr2'] <= 0.025 for v in s.values())),
        'P3': bool(all(0.93 <= v['coverage_cr2'] <= 0.97 for v in s.values())),
        'P4': bool(all(v['resolved_cr2'] <= 0.10 for v in s.values() if v['reference_within_margin'])
                   and all(v['resolved_delong'] >= v['resolved_cr2'] for v in s.values())),
        'P5': bool(all(0.7 <= v['mde_predicted'] / v['mde_empirical'] <= 1.4 for v in s.values()))}
    json.dump(res, open(OUT, 'w'), indent=2)
    print(json.dumps({k: res[k] for k in ('n', 'cells', 'reference_population', 'auc_reference', 'predictions')}, indent=1))
    for k, v in s.items():
        print(k, json.dumps({a: round(b, 4) if isinstance(b, float) else b for a, b in v.items()}))
    print('wrote', OUT)


if __name__ == '__main__':
    main()
