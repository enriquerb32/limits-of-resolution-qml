#!/usr/bin/env python3
"""
exp50_teacher_grid.py - the positive control of exp38 repeated over teachers, signal strengths, label draws and a null
====================================================================================================================
Review of 2026-11-18: choosing, among 20 candidates, the teacher with the most signal outside the separable basis is a
legitimate stress test but does not show power in general; one teacher, one label draw and five seeds without a
paired interval are descriptive; and the teacher score (S5.3) had no LayerNorm while the student heads have one.
Here, in the interface mode of exp38 (every head receives the teacher angles q* directly):
  * teachers: the one exp38 selects (largest non-separable share among candidates 0-19) and candidates 0-3, which
    are NOT selected for anything;
  * teacher score s = w . LayerNorm(<Z>(q*; W*)), the same tail as the student heads (exp38 used w . <Z>);
  * strengths: noise set so that the oracle test AUC of s is 0.70 or 0.85;
  * label draws: two independent noise draws for the selected teacher;
  * null: labels from noise only (the same threshold rule, no teacher), two draws;
  * heads: CIRCUIT, LINEAR4D, FOURIER2 (separable) and FOURIER_FULL1 (81-function tensor basis), 5 seeds each.
For every configuration and control: seed-mean AUCs and the paired difference CIRCUIT - control with the primary
procedure of the paper (CR2 with Bell-McCaffrey degrees of freedom, calendar-year clusters) and with DeLong.
'Detected' = the CR2 interval excludes zero. Under the null the detection rate is a type-I error.

  LOR_DATASET=v2 python code/local_experiments/exp50_teacher_grid.py [--seeds 5]        ~2-3 h CPU
Output: results/local_exp50_teacher_grid.json (resumable), results/json/results_exp50_teacher_grid_probs.npz
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(ROOT, 'code'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from exp38_teacher_control import Head, separable, r2  # noqa: E402
from exp27_controls import RAW_CSV, build_dataset, seed_all  # noqa: E402
from qml_core_trainonly import make_qnode_circuit, _loop as core_loop  # noqa: E402
from qml_resolution import cluster_robust_delta, paired_delong  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'local_exp50_teacher_grid.json')
NPZ = os.path.join(ROOT, 'results', 'json', 'results_exp50_teacher_grid_probs.npz')
HEADS = {'CIRCUIT': ('circuit', 1), 'LINEAR4D': ('linear', 1), 'FOURIER2': ('sep', 2), 'FOURIER_FULL1': ('full', 1)}


def teacher(Z, s1, c, circ):
    rng = np.random.default_rng(1000 + c)                    # identical draws to exp38 candidate c
    P = rng.normal(0, 1 / math.sqrt(Z.shape[1]), (Z.shape[1], 4))
    q = math.pi / (1 + np.exp(-1.5 * Z @ P))
    W = torch.tensor(rng.uniform(-math.pi, math.pi, (2, 4, 3)))
    w = rng.normal(0, 1, 4)
    with torch.no_grad():
        ez = torch.stack(circ(W, torch.tensor(q)), -1).numpy()
    ln = (ez - ez.mean(1, keepdims=True)) / np.sqrt(ez.var(1, keepdims=True) + 1e-5)   # LayerNorm(4), no affine
    s = ln @ w
    return q, s, float(1 - r2(s[:s1], separable(q[:s1])))


def make_labels(s, s1, s2, target, draw, n):
    e = np.random.default_rng(7 + 100 * draw).standard_normal(n)
    if s is None:                                            # null: noise only
        v = e
        return (v > np.percentile(v[:s1], 60)).astype(float), float('nan')
    lab = lambda g: ((s + g * s[:s1].std() * e) > np.percentile((s + g * s[:s1].std() * e)[:s1], 60)).astype(float)
    sigs = np.linspace(0.05, 6, 240)
    orc = np.array([roc_auc_score(lab(g)[s2:], s[s2:]) for g in sigs])
    g = float(sigs[int(np.argmin(np.abs(orc - target)))])
    y = lab(g)
    return y, float(roc_auc_score(y[s2:], s[s2:]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--candidates', type=int, default=20); a = ap.parse_args()
    _, ds = build_dataset(RAW_CSV, train_only=True, fix_et0_verano=True)
    X = ds.X_all.astype(float); s1, s2 = ds.s1, ds.s2
    import pandas as pd
    years = pd.to_datetime(pd.Series(ds.meta_te['fecha'])).dt.year.values
    assert len(years) == len(X) - s2
    Z = (X - X[:s1].mean(0)) / (X[:s1].std(0) + 1e-8)
    circ = make_qnode_circuit(4, 2)
    T = {c: teacher(Z, s1, c, circ) for c in range(a.candidates)}
    sel = max(T, key=lambda c: T[c][2])
    configs = [dict(name=f'selected{sel}_auc{t}_draw{d}', teacher=sel, target=t, draw=d) for t in (0.70, 0.85) for d in (0, 1)]
    configs += [dict(name=f'teacher{c}_auc{t}', teacher=c, target=t, draw=0) for c in range(4) for t in (0.70, 0.85)]
    configs += [dict(name=f'null_draw{d}', teacher=None, target=None, draw=d) for d in (0, 1)]
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res.setdefault('configs', {}); res['selected_teacher'] = sel
    res['interaction_share'] = {str(c): v[2] for c, v in T.items()}
    probs = dict(np.load(NPZ)) if os.path.exists(NPZ) else {}
    for cf in configs:
        if cf['name'] in res['configs']:
            continue
        q, s, inter = T[cf['teacher'] if cf['teacher'] is not None else sel]
        y, oracle = make_labels(None if cf['teacher'] is None else s, s1, s2, cf['target'], cf['draw'], len(X))
        y_te = y[s2:].astype(int)
        Xt, y_t = torch.tensor(q), torch.tensor(y)
        dl_args = (Xt[s1:s2], y_t[s1:s2])
        P, t0 = {}, time.time()
        for name, (kd, dg) in HEADS.items():
            ps = []
            for seed in range(a.seeds):
                seed_all(seed)
                dl = DataLoader(TensorDataset(Xt[:s1], y_t[:s1]), batch_size=16, shuffle=True, drop_last=True)
                m = core_loop(Head(kd, dg).double(), dl, *dl_args, epochs=25, patience=6, lr=5e-3); m.eval()
                with torch.no_grad():
                    out = m(Xt[s2:]); ps.append(torch.sigmoid(out[0] if isinstance(out, tuple) else out).numpy())
            P[name] = np.array(ps); probs[f"{cf['name']}__{name}"] = P[name]
        probs[f"{cf['name']}__y_te"] = y_te
        row = dict(cf, oracle_auc_test=oracle, interaction_share=None if cf['teacher'] is None else inter,
                   prevalence_test=float(y_te.mean()),
                   auc={k: float(np.mean([roc_auc_score(y_te, p) for p in v])) for k, v in P.items()}, contrasts={})
        for k in ('LINEAR4D', 'FOURIER2', 'FOURIER_FULL1'):
            cr = cluster_robust_delta(y_te, P['CIRCUIT'], P[k], years)
            dl = paired_delong(y_te, P['CIRCUIT'], P[k])
            lo, hi = cr.ci()
            row['contrasts'][f'CIRCUIT-{k}'] = dict(delta=cr.delta, ci_cr2_bm=[lo, hi], df=cr.df, p_cr2=cr.p_value,
                                                    p_delong=dl.p_value, detected=bool(lo > 0 or hi < 0))
        res['configs'][cf['name']] = row
        json.dump(res, open(OUT, 'w'), indent=2); np.savez_compressed(NPZ, **probs)
        print(f"{cf['name']:24s} oracle {oracle:.3f} | " + ' '.join(f'{k} {v:.3f}' for k, v in row['auc'].items())
              + ' | detected ' + ' '.join(f"{k.split('-')[1]}:{int(v['detected'])}" for k, v in row['contrasts'].items())
              + f' | {(time.time() - t0) / 60:.1f} min', flush=True)
    uniq = {}                       # the selected teacher is also one of candidates 0-3: count each configuration once
    for r in res['configs'].values():
        uniq.setdefault((r['teacher'], r['target'], r['draw']), r)
    sig = [r for r in uniq.values() if r['teacher'] is not None]
    nul = [r for r in uniq.values() if r['teacher'] is None]
    res['n_distinct'] = dict(signal=len(sig), null=len(nul), teachers=sorted({r['teacher'] for r in sig}))
    res['detection'] = {k: dict(circuit_above=int(sum(r['contrasts'][f'CIRCUIT-{k}']['ci_cr2_bm'][0] > 0 for r in sig)),
                                control_above=int(sum(r['contrasts'][f'CIRCUIT-{k}']['ci_cr2_bm'][1] < 0 for r in sig)),
                                null_detected=int(sum(r['contrasts'][f'CIRCUIT-{k}']['detected'] for r in nul)))
                        for k in ('LINEAR4D', 'FOURIER2', 'FOURIER_FULL1')}
    res.pop('detection_rate', None)
    json.dump(res, open(OUT, 'w'), indent=2)
    print(json.dumps(dict(n=res['n_distinct'], detection=res['detection']), indent=1)); print('wrote', OUT)


if __name__ == '__main__':
    main()
