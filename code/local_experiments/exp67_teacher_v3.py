#!/usr/bin/env python3
"""
exp67_teacher_v3.py - refit v3, section 4: the repeated planted-signal control of exp50 under the searched settings
==================================================================================================================
Plan: audit/refit_plan_v3.txt (commit adfcd21, doi:10.5281/zenodo.24466944), section 4. Configurations of exp50
(the selected teacher at oracle test AUC 0.70 and 0.85 with two label draws, candidates 0-3 at 0.70 and 0.85, two
nulls), the teacher score through the same LayerNorm as the students, interface mode (heads receive the teacher
angles). Heads and the setting each takes from section 2 (results/refit_v3/search_<MODEL>.json):
  CIRCUIT <- LSTM_QNN, LINEAR4D <- CTRL_LINEAR4D, FOURIER2 <- FOURIER2, FOURIER_FULL1 <- FOURIER_FULL1,
  FOURIER_FULL4 <- FOURIER_FULL4.
5 seeds per head; clarification: each seed is the best of 3 starts by validation AUC (ties: validation loss), as in
the final fits of section 2. The common training loop of exp66 is used.
  --heads <HEAD>    fits one head over all configurations: results/refit_v3/teacher_<HEAD>.json / _probs.npz
  --combine         recovery per head and configuration, (AUC - 0.5) >= 0.9 (oracle - 0.5) on the 5-seed mean
                    test AUC; paired CR2 (calendar-year clusters) and DeLong for CIRCUIT minus each control.
                    Writes results/refit_v3/teacher_summary.json
"""
import argparse
import json
import math
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(ROOT, 'code'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from exp66_refit_v3 import OUT, STARTS, loop, seed_all, selected_setting, RAW_CSV  # noqa: E402
from exp38_teacher_control import Head  # noqa: E402
from exp50_teacher_grid import teacher, make_labels  # noqa: E402
from trainonly_dataset import build_dataset  # noqa: E402
from qml_core_trainonly import make_qnode_circuit  # noqa: E402
from qml_resolution import cluster_robust_delta, paired_delong  # noqa: E402

HEADS = {'CIRCUIT': (('circuit', 1), 'LSTM_QNN'), 'LINEAR4D': (('linear', 1), 'CTRL_LINEAR4D'),
         'FOURIER2': (('sep', 2), 'FOURIER2'), 'FOURIER_FULL1': (('full', 1), 'FOURIER_FULL1'),
         'FOURIER_FULL4': (('full', 4), 'FOURIER_FULL4')}
SEEDS = range(5)
if os.environ.get('REFIT_SMOKE') == '1':
    SEEDS = range(1)


def setup():
    _, ds = build_dataset(RAW_CSV, train_only=True, fix_et0_verano=True)
    X = ds.X_all.astype(float); s1, s2 = ds.s1, ds.s2
    import pandas as pd
    years = pd.to_datetime(pd.Series(ds.meta_te['fecha'])).dt.year.values
    Z = (X - X[:s1].mean(0)) / (X[:s1].std(0) + 1e-8)
    circ = make_qnode_circuit(4, 2)
    T = {c: teacher(Z, s1, c, circ) for c in range(20)}
    sel = max(T, key=lambda c: T[c][2])
    configs = [dict(name=f'selected{sel}_auc{t}_draw{d}', teacher=sel, target=t, draw=d) for t in (0.70, 0.85)
               for d in (0, 1)]
    configs += [dict(name=f'teacher{c}_auc{t}', teacher=c, target=t, draw=0) for c in range(4) for t in (0.70, 0.85)]
    configs += [dict(name=f'null_draw{d}', teacher=None, target=None, draw=d) for d in (0, 1)]
    if os.environ.get('REFIT_SMOKE') == '1':
        configs = configs[:1] + configs[-1:]
    return ds, X, s1, s2, years, T, sel, configs


def fit_head(kind, cfg, seed, D):
    seed_all(seed)
    m = Head(*kind).double()
    t0 = time.time()
    m, info = loop(m, D, cfg)
    m.eval()
    with torch.no_grad():
        pv = torch.sigmoid(m(D.X_va)).numpy()
        pt = torch.sigmoid(m(D.X_te)).numpy()
    info.update(val_auc=float(roc_auc_score(D.y_va.numpy().astype(int), pv)), seed=seed,
                minutes=(time.time() - t0) / 60)
    return pt, info


def run_head(head):
    out = os.path.join(OUT, f'teacher_{head}.json')
    if os.path.exists(out):
        print('done already (this run):', out); return
    kind, src = HEADS[head]
    cfg = selected_setting(src)[0]
    ds, X, s1, s2, years, T, sel, configs = setup()
    res, probs = dict(head=head, setting=cfg, setting_from=src, starts=STARTS, seeds=list(SEEDS), configs={}), {}
    for cf in configs:
        q, s, inter = T[cf['teacher'] if cf['teacher'] is not None else sel]
        y, oracle = make_labels(None if cf['teacher'] is None else s, s1, s2, cf['target'], cf['draw'], len(X))
        Xt, yt = torch.tensor(q), torch.tensor(y)
        D = SimpleNamespace(X_tr=Xt[:s1], y_tr=yt[:s1], X_va=Xt[s1:s2], y_va=yt[s1:s2], X_te=Xt[s2:])
        y_te = y[s2:].astype(int)
        P, rows = [], []
        for seed in SEEDS:
            fits = [fit_head(kind, cfg, seed + 1000 * k, D) for k in range(STARTS)]
            k = max(range(STARTS), key=lambda i: (fits[i][1]['val_auc'], -fits[i][1]['val_loss']))
            P.append(fits[k][0])
            rows.append(dict(seed=seed, start_chosen=k, test_auc=float(roc_auc_score(y_te, fits[k][0])),
                             starts=[f[1] for f in fits]))
        probs[f"{cf['name']}__{head}"] = np.array(P)
        probs[f"{cf['name']}__y_te"] = y_te
        res['configs'][cf['name']] = dict(cf, oracle_auc_test=oracle, fits=rows,
                                          mean_test_auc=float(np.mean([r['test_auc'] for r in rows])))
        print(f"{head} {cf['name']:24s} oracle {oracle:.3f} mean test AUC {res['configs'][cf['name']]['mean_test_auc']:.3f}",
              flush=True)
    res['selected_teacher'] = sel
    np.savez_compressed(os.path.join(OUT, f'teacher_{head}_probs.npz'), **probs)
    json.dump(res, open(out, 'w', encoding='utf-8'), indent=2)


def combine():
    out = os.path.join(OUT, 'teacher_summary.json')
    if os.path.exists(out):
        print('done already (this run):', out); return
    ds, X, s1, s2, years, T, sel, configs = setup()
    R = {h: json.load(open(os.path.join(OUT, f'teacher_{h}.json'), encoding='utf-8')) for h in HEADS}
    Z = {h: np.load(os.path.join(OUT, f'teacher_{h}_probs.npz')) for h in HEADS}
    summ = dict(plan_rule='recovers if (mean test AUC - 0.5) >= 0.9 (oracle - 0.5); 5-seed means',
                settings={h: R[h]['setting'] for h in HEADS}, configs={})
    for cf in configs:
        n = cf['name']
        y_te = Z['CIRCUIT'][f'{n}__y_te']
        oracle = R['CIRCUIT']['configs'][n]['oracle_auc_test']
        P = {h: Z[h][f'{n}__{h}'] for h in HEADS}
        auc = {h: float(np.mean([roc_auc_score(y_te, p) for p in P[h]])) for h in HEADS}
        row = dict(cf, oracle_auc_test=oracle, auc=auc, contrasts={})
        if cf['teacher'] is not None:
            row['recovers'] = {h: bool(auc[h] - 0.5 >= 0.9 * (oracle - 0.5)) for h in HEADS}
        for h in HEADS:
            if h == 'CIRCUIT':
                continue
            cr = cluster_robust_delta(y_te, P['CIRCUIT'], P[h], years)
            lo, hi = cr.ci()
            row['contrasts'][f'CIRCUIT-{h}'] = dict(delta=cr.delta, ci_cr2_bm=[lo, hi], df=cr.df, p_cr2=cr.p_value,
                                                    p_delong=paired_delong(y_te, P['CIRCUIT'], P[h]).p_value,
                                                    resolved=bool(lo > 0 or hi < 0))
        summ['configs'][n] = row
        print(f"{n:24s} oracle {oracle:.3f} | " + ' '.join(f'{h} {v:.3f}' for h, v in auc.items()), flush=True)
    uniq = {}                       # the selected teacher may also be one of candidates 0-3: count it once (exp50)
    for r in summ['configs'].values():
        uniq.setdefault((r['teacher'], r['target'], r['draw']), r)
    sig = [r for r in uniq.values() if r['teacher'] is not None]
    summ['recovery_counts'] = {h: int(sum(r['recovers'][h] for r in sig)) for h in HEADS}
    summ['n_signal_configs'] = len(sig)
    json.dump(summ, open(out, 'w', encoding='utf-8'), indent=2)
    print(json.dumps(summ['recovery_counts']))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--heads', nargs='+', default=[], choices=list(HEADS))
    ap.add_argument('--combine', action='store_true'); ap.add_argument('--threads', type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    if os.environ.get('LOR_DATASET') != 'v2' or os.environ.get('V2_CAUSAL') != '1':
        raise SystemExit('set LOR_DATASET=v2 and V2_CAUSAL=1')
    for h in a.heads:
        run_head(h)
    if a.combine:
        combine()


if __name__ == '__main__':
    main()
