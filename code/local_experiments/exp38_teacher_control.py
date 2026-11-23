#!/usr/bin/env python3
"""
exp38_teacher_control.py - positive control for the quantum controls
==========================================================================================
On the real benchmark no model beats the climatology, so 'the Fourier surrogates match the circuit' could hold
simply because there is nothing for the circuit to learn. This control plants a circuit-generated signal and asks
whether the controls of the paper can tell a circuit contribution when there is one.
  * inputs: the real X of the panel (same rows, splits and scalers as the benchmark);
  * teacher angles: q* = pi * sigmoid(c * Z P), Z = inputs standardized with training statistics, P a fixed
    Gaussian 35 x 4 projection (the model's own projection layer can represent it);
  * teacher score: s = w . <Z>(q*; W*), the LSTM+QNN circuit (re-uploading R_Y R_Z, two entangling layers) with fixed
    random weights W* and readout w; among 20 candidate teachers the one with the largest share of variance NOT
    explained by a separable degree-4 Fourier basis is kept (non-separable signal by construction);
  * labels: y = 1[s + e > training 60th percentile], e ~ N(0, sigma^2), sigma set so that the oracle AUC of s on
    the test rows is about 0.85;
  * models (same code and training loop as the paper): LSTM_QNN, CTRL_LINEAR4D, FOURIER2, FOURIER4 (separable),
    FOURIER_FULL1, FOURIER_FULL2 (tensor), PureLSTM, RF (tuned in v2).
Expected if the controls work: the separable surrogates and Linear-4D fall clearly below the circuit, the circuit
approaches the oracle. Caveat stated in the paper: the full tensor basis of degree 4 spans the circuit exactly, so
no surrogate of that size can be beaten by it (dequantization); FULL1/FULL2 measure how much of the planted signal
is low-degree.
Two modes:
  --mode interface (default): every head receives the teacher angles q* directly (no LSTM, no projection), so the
      comparison isolates what the paper compares - the map applied at the 4-angle interface: trained circuit
      (CIRCUIT), Linear-4D, separable Fourier-2/4, full tensor Fourier-1/2, each followed by LayerNorm and a linear
      output, as in the paper's models. ~15 min CPU.
  --mode end-to-end: the paper's full models on the raw inputs (they must also learn the projection). ~1-1.5 h CPU.
Usage:  python code/local_experiments/exp38_teacher_control.py [--mode interface] [--seeds 5] [--target-auc 0.85]
Output: results/local_exp38_teacher_control.json
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(ROOT, 'code'))
from qml_core_trainonly import PureLSTM, HybridQNNControl, LinearControl4D, make_qnode_circuit, _loop as core_loop  # noqa: E402
from exp27_controls import FourierControl4D, RAW_CSV, build_dataset, seed_all  # noqa: E402
from exp30_fourier_full import FourierFull4D  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'local_exp38_teacher_control_{mode}.json')
V2 = os.environ.get('LOR_DATASET', 'v1').lower() == 'v2'


class Head(torch.nn.Module):
    """Map at the 4-angle interface + LayerNorm + linear output (the tail of every model of the paper)."""
    def __init__(self, kind, degree=1):
        super().__init__()
        self.kind, self.degree = kind, degree
        if kind == 'circuit':
            self.circ = make_qnode_circuit(4, 2)
            self.W = torch.nn.Parameter(torch.empty(2, 4, 3).uniform_(-math.pi, math.pi))
        else:
            n = {'linear': 4, 'sep': 8 * degree, 'full': (2 * degree + 1) ** 4}[kind]
            self.ctrl = torch.nn.Linear(n, 4)
        self.norm = torch.nn.LayerNorm(4)
        self.clf = torch.nn.Linear(4, 1)

    def feats(self, q):
        if self.kind == 'linear':
            return q
        if self.kind == 'sep':
            return torch.cat([f(k * q) for k in range(1, self.degree + 1) for f in (torch.sin, torch.cos)], 1)
        cols = [torch.ones_like(q)] + [f(k * q) for k in range(1, self.degree + 1) for f in (torch.cos, torch.sin)]
        b = torch.stack(cols, -1); out = b[:, 0]
        for j in range(1, 4):
            out = (out[:, :, None] * b[:, j, None, :]).reshape(len(q), -1)
        return out

    def forward(self, q):
        z = torch.stack(self.circ(self.W, q), -1).to(q.dtype) if self.kind == 'circuit' else self.ctrl(self.feats(q))
        return self.clf(self.norm(z)).squeeze(-1)


HEADS = {'CIRCUIT': ('circuit', 1), 'LINEAR4D': ('linear', 1), 'FOURIER2': ('sep', 2), 'FOURIER4': ('sep', 4),
         'FOURIER_FULL1': ('full', 1), 'FOURIER_FULL2': ('full', 2)}


def separable(q, d=4):
    return np.concatenate([np.ones((len(q), 1))] + [f(k * q) for k in range(1, d + 1) for f in (np.cos, np.sin)], 1)


def r2(y, X):
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    return 1 - ((y - X @ c) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--target-auc', type=float, default=0.85)
    ap.add_argument('--candidates', type=int, default=20)
    ap.add_argument('--mode', choices=['interface', 'end-to-end'], default='interface')
    ap.add_argument('--models', nargs='+', default=None,
                    help='default: all heads (interface) or LSTM_QNN CTRL_LINEAR4D FOURIER2 FOURIER4 FOURIER_FULL1 FOURIER_FULL2 PureLSTM RF')
    ap.add_argument('--epochs-scale', type=float, default=1.0, help='<1 only for a smoke test')
    a = ap.parse_args()
    if a.models is None:
        a.models = list(HEADS) if a.mode == 'interface' else ['LSTM_QNN', 'CTRL_LINEAR4D', 'FOURIER2', 'FOURIER4',
                                                              'FOURIER_FULL1', 'FOURIER_FULL2', 'PureLSTM', 'RF']
    out_path = OUT.format(mode=a.mode.replace('-', '_'))
    _, ds = build_dataset(RAW_CSV, train_only=True, fix_et0_verano=True, **({} if V2 else {'umbral_override': 28.1}))
    X = ds.X_all.astype(float); s1, s2 = ds.s1, ds.s2
    mu, sd = X[:s1].mean(0), X[:s1].std(0) + 1e-8
    Z = (X - mu) / sd
    circ = make_qnode_circuit(4, 2)
    best = None
    for c in range(a.candidates):
        rng = np.random.default_rng(1000 + c)
        P = rng.normal(0, 1 / math.sqrt(Z.shape[1]), (Z.shape[1], 4))
        q = math.pi / (1 + np.exp(-1.5 * Z @ P))
        W = torch.tensor(rng.uniform(-math.pi, math.pi, (2, 4, 3)))
        w = rng.normal(0, 1, 4)
        with torch.no_grad():
            ez = torch.stack(circ(W, torch.tensor(q)), -1).numpy()
        s = ez @ w
        inter = 1 - r2(s[:s1], separable(q[:s1]))
        cand = dict(candidate=c, interaction_share=float(inter), s=s, q=q)
        if best is None or inter > best['interaction_share']:
            best = cand
        print(f'teacher {c:2d}: share of variance outside separable degree-4 basis {inter:.3f}', flush=True)
    s = best['s']; rng = np.random.default_rng(7)
    e0 = rng.standard_normal(len(s))
    def labels(sig):
        v = s + sig * s[:s1].std() * e0
        return (v > np.percentile(v[:s1], 60)).astype(float)
    sigs = np.linspace(0.05, 3, 120)
    oracle = [roc_auc_score(labels(g)[s2:], s[s2:]) for g in sigs]
    sig = float(sigs[int(np.argmin(np.abs(np.array(oracle) - a.target_auc)))])
    y = labels(sig); y_te = y[s2:].astype(int)
    res = dict(mode=a.mode, teacher=best['candidate'], interaction_share=best['interaction_share'], noise_sigma_rel=sig,
               oracle_auc_test=float(roc_auc_score(y_te, s[s2:])), prevalence_test=float(y_te.mean()), models={})
    print(f"teacher {best['candidate']} | interaction share {best['interaction_share']:.3f} | oracle AUC {res['oracle_auc_test']:.3f}")
    if os.path.exists(out_path):                              # resume: same teacher and labels -> keep finished models
        prev = json.load(open(out_path))
        if (prev.get('teacher'), prev.get('noise_sigma_rel'), prev.get('mode')) == (res['teacher'], sig, a.mode):
            res['models'] = {k: v for k, v in prev.get('models', {}).items() if len(v.get('auc', [])) == a.seeds}
            print('[resume]', list(res['models']))
    Xt = torch.tensor(best['q'] if a.mode == 'interface' else X); y_t = torch.tensor(y)
    X_tr, y_tr, X_va, y_va, X_te = Xt[:s1], y_t[:s1], Xt[s1:s2], y_t[s1:s2], Xt[s2:]
    build = {'LSTM_QNN': lambda: HybridQNNControl(in_dim=ds.in_dim, mode='trained'),
             'CTRL_LINEAR4D': lambda: LinearControl4D(in_dim=ds.in_dim),
             'FOURIER2': lambda: FourierControl4D(in_dim=ds.in_dim, degree=2),
             'FOURIER4': lambda: FourierControl4D(in_dim=ds.in_dim, degree=4),
             'FOURIER_FULL1': lambda: FourierFull4D(in_dim=ds.in_dim, degree=1),
             'FOURIER_FULL2': lambda: FourierFull4D(in_dim=ds.in_dim, degree=2),
             'PureLSTM': lambda: PureLSTM(in_dim=ds.in_dim)}
    if a.mode == 'interface':     # the 4-angle heads share names with the full models: only one set per mode
        build.update({k: (lambda kd=kd, dg=dg: Head(kd, dg)) for k, (kd, dg) in HEADS.items()})
    ep = max(1, int(round(25 * a.epochs_scale)))
    for name in a.models:
        if name in res['models']:
            continue
        aucs, t0 = [], time.time()
        for seed in range(a.seeds):
            if name == 'RF':
                imp = SimpleImputer().fit(X[:s1])
                m = (__import__('dataset_v2').rf_v2(seed) if V2 else
                     RandomForestClassifier(n_estimators=300, max_depth=8, class_weight='balanced', random_state=seed, n_jobs=-1))
                p = m.fit(imp.transform(X[:s1]), y[:s1]).predict_proba(imp.transform(X[s2:]))[:, 1]
            else:
                seed_all(seed)
                dl = DataLoader(TensorDataset(X_tr, y_tr), batch_size=16, shuffle=True, drop_last=True)
                m = core_loop(build[name]().double(), dl, X_va, y_va, epochs=ep, patience=6, lr=5e-3)
                m.eval()
                with torch.no_grad():
                    out = m(X_te); p = torch.sigmoid(out[0] if isinstance(out, tuple) else out).numpy()
            aucs.append(float(roc_auc_score(y_te, p)))
        res['models'][name] = dict(auc=aucs, mean=float(np.mean(aucs)), sd=float(np.std(aucs, ddof=1)) if len(aucs) > 1 else None)
        print(f'{name:14s} AUC {np.mean(aucs):.4f} ({(time.time() - t0) / 60:.1f} min)', flush=True)
        json.dump(res, open(out_path, 'w'), indent=2)
    q = res['models'].get('CIRCUIT' if a.mode == 'interface' else 'LSTM_QNN', {}).get('mean')
    if q:
        for k, v in res['models'].items():
            v['gap_to_qnn'] = v['mean'] - q
    json.dump(res, open(out_path, 'w'), indent=2)
    print('wrote', out_path)


if __name__ == '__main__':
    main()
