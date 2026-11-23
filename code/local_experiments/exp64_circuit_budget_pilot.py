#!/usr/bin/env python3
"""
exp64_circuit_budget_pilot.py - does the trained circuit recover its own teacher with a larger training budget?
================================================================================================================
Review point: in the repeated positive control (exp50) the trained circuit stays below the 81-function tensor map in
all 10 configurations and far below the oracle, although the teacher has its own architecture and the planted signal
is exactly representable. This pilot asks whether that is an optimisation limit, before the training protocol of the
causal re-run is fixed. Interface mode of exp50 (heads receive the teacher angles q*), strong signal (oracle test
AUC 0.85), teachers 0-3 (not selected for anything), the teacher score through the same LayerNorm as the students.

Budgets for the circuit head (AdamW + ReduceLROnPlateau + early stopping on validation loss, as in the paper):
  base       25 epochs, patience 6, lr 5e-3, one start            (the budget of the paper)
  long       100 epochs, patience 15, lr 5e-3
  long_lr2   100 epochs, patience 15, lr 2e-2
  long_lr1   100 epochs, patience 15, lr 1e-3
  restart3   'long', best of 3 random starts by validation loss
  warm       'long' started at the teacher weights W* (a ceiling: what this loss and data reach from the right basin)
Reference heads: the 81-function tensor map (FOURIER_FULL1) at 'base' and 'long'.
Per fit: test AUC, correlation of the student logit with the teacher score on the test rows, validation loss, epochs.

  LOR_DATASET=v2 V2_CAUSAL=1 python code/local_experiments/exp64_circuit_budget_pilot.py [--seeds 3]
Output: results/local_exp64_circuit_budget_pilot.json (resumable per configuration x head x budget x seed)
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
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(ROOT, 'code'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from exp38_teacher_control import Head  # noqa: E402
from exp27_controls import RAW_CSV, build_dataset, seed_all  # noqa: E402
from exp50_teacher_grid import make_labels  # noqa: E402
from qml_core_trainonly import make_qnode_circuit, _loop as core_loop  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'local_exp64_circuit_budget_pilot.json')
BUDGETS = {
    'base': dict(epochs=25, patience=6, lr=5e-3, starts=1),
    'long': dict(epochs=100, patience=15, lr=5e-3, starts=1),
    'long_lr2': dict(epochs=100, patience=15, lr=2e-2, starts=1),
    'long_lr1': dict(epochs=100, patience=15, lr=1e-3, starts=1),
    'restart3': dict(epochs=100, patience=15, lr=5e-3, starts=3),
    'warm': dict(epochs=100, patience=15, lr=5e-3, starts=1, warm=True),
}
RUNS = [('CIRCUIT', b) for b in BUDGETS] + [('FOURIER_FULL1', 'base'), ('FOURIER_FULL1', 'long')]
KIND = {'CIRCUIT': ('circuit', 1), 'FOURIER_FULL1': ('full', 1)}


def teacher_full(Z, c, circ):
    """exp50.teacher, also returning the teacher weights (same draws: P, W, w from default_rng(1000 + c))."""
    rng = np.random.default_rng(1000 + c)
    P = rng.normal(0, 1 / math.sqrt(Z.shape[1]), (Z.shape[1], 4))
    q = math.pi / (1 + np.exp(-1.5 * Z @ P))
    W = torch.tensor(rng.uniform(-math.pi, math.pi, (2, 4, 3)))
    w = rng.normal(0, 1, 4)
    with torch.no_grad():
        ez = torch.stack(circ(W, torch.tensor(q)), -1).numpy()
    ln = (ez - ez.mean(1, keepdims=True)) / np.sqrt(ez.var(1, keepdims=True) + 1e-5)
    return q, ln @ w, W


def val_loss(m, X_va, y_va, y_tr):
    pos_w = torch.tensor([float((y_tr == 0).sum()) / (float((y_tr == 1).sum()) + 1e-5)])
    m.eval()
    with torch.no_grad():
        return float(nn.BCEWithLogitsLoss(pos_weight=pos_w)(m(X_va), y_va))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--teachers', type=int, nargs='+', default=[0, 1, 2, 3])
    ap.add_argument('--target', type=float, default=0.85); a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
    _, ds = build_dataset(RAW_CSV, train_only=True, fix_et0_verano=True)
    X = ds.X_all.astype(float); s1, s2 = ds.s1, ds.s2
    Z = (X - X[:s1].mean(0)) / (X[:s1].std(0) + 1e-8)
    circ = make_qnode_circuit(4, 2)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res['causal'] = os.environ.get('V2_CAUSAL', '0') == '1'; res['budgets'] = BUDGETS
    res.setdefault('fits', {})
    for c in a.teachers:
        q, s, Wt = teacher_full(Z, c, circ)
        y, oracle = make_labels(s, s1, s2, a.target, 0, len(X))
        cfg = f'teacher{c}_auc{a.target}'
        res.setdefault('oracle', {})[cfg] = oracle
        Xt, yt = torch.tensor(q), torch.tensor(y)
        y_te = y[s2:].astype(int)
        for head, bname in RUNS:
            b = BUDGETS[bname]
            for seed in range(a.seeds):
                key = f'{cfg}|{head}|{bname}|{seed}'
                if key in res['fits']:
                    continue
                t0, best = time.time(), None
                for start in range(b['starts']):
                    seed_all(1000 * start + seed)
                    m = Head(*KIND[head]).double()
                    if b.get('warm'):
                        with torch.no_grad():
                            m.W.copy_(Wt)
                    dl = DataLoader(TensorDataset(Xt[:s1], yt[:s1]), batch_size=16, shuffle=True, drop_last=True)
                    m = core_loop(m, dl, Xt[s1:s2], yt[s1:s2], epochs=b['epochs'], patience=b['patience'], lr=b['lr'])
                    vl = val_loss(m, Xt[s1:s2], yt[s1:s2], yt[:s1])
                    if best is None or vl < best[0]:
                        best = (vl, m)
                vl, m = best
                m.eval()
                with torch.no_grad():
                    logit = m(Xt[s2:]).numpy()
                res['fits'][key] = dict(auc=float(roc_auc_score(y_te, logit)), val_loss=vl,
                                        corr_teacher=float(np.corrcoef(logit, s[s2:])[0, 1]),
                                        minutes=(time.time() - t0) / 60)
                json.dump(res, open(OUT, 'w'), indent=2)
                f = res['fits'][key]
                print(f"{key:40s} auc {f['auc']:.3f} (oracle {oracle:.3f}) corr {f['corr_teacher']:+.3f} "
                      f"val {vl:.4f} {f['minutes']:.1f} min", flush=True)
    summ = {}
    for key, f in res['fits'].items():
        cfg, head, bname, _ = key.split('|')
        summ.setdefault(f'{head}|{bname}', []).append((f['auc'] - res['oracle'][cfg], f['corr_teacher'], f['auc']))
    res['summary'] = {k: dict(n=len(v), auc_mean=float(np.mean([x[2] for x in v])),
                              gap_to_oracle_mean=float(np.mean([x[0] for x in v])),
                              corr_teacher_mean=float(np.mean([x[1] for x in v]))) for k, v in summ.items()}
    json.dump(res, open(OUT, 'w'), indent=2)
    print(json.dumps(res['summary'], indent=1)); print('wrote', OUT)


if __name__ == '__main__':
    main()
