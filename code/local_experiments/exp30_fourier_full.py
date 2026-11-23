#!/usr/bin/env python3
"""
exp30_fourier_full.py - full (non-separable) Fourier surrogates of the 4-qubit circuit (15 seeds)
================================================================================================
The Fourier-1/-2 controls of exp27 are separable: they contain sin/cos of each angle but no products across angles,
which the entangling layers of the circuit can generate. This script adds the full tensor-product trigonometric
basis, i.e. the complete space of real trigonometric polynomials with the circuit's per-coordinate frequencies:

  FOURIER_FULL1   per-coordinate basis [1, cos q, sin q]                      -> 3^4 = 81 features  (frequencies {0,+-1})
  FOURIER_FULL2   per-coordinate basis [1, cos q, sin q, cos 2q, sin 2q]      -> 5^4 = 625 features (frequencies {0,+-1,+-2})

FOURIER_FULL2 spans every function the 2-layer re-uploading circuit of LSTM+QNN can output for each <Z_j>
(Schuld et al. 2021): it is a classical model that contains the circuit's function class. The backbone, interface,
LayerNorm, output layer and training loop are those of LSTM_QNN / CTRL_LINEAR4D / FOURIER1-2 (core_loop: AdamW 5e-3,
batch 16, <=25 epochs, patience 6). Parameter counts are larger than the circuit's (Linear(81->4) = 328,
Linear(625->4) = 2,504); this is the point of the control: a classical model that can represent everything the circuit
can, trained on the same rows.

Outputs (resumable)
  results/json/results_fourier_full_trainonly_fixet0_p60_28.1.json / _probs.npz
  and the keys FOURIER_FULL1_probs / FOURIER_FULL2_probs are added to
  results/json/results_core_hybrids_controls_fixet0_p60_28.1_probs.npz (written by exp27) when it exists.

Usage (from the repository root)
  python code/local_experiments/exp30_fourier_full.py --dry-run
  python code/local_experiments/exp30_fourier_full.py                # ~35-45 min on CPU (2 models x 15 seeds)
Then
  python code/local_experiments/exp10_paired_ci.py --extra results/json/results_core_hybrids_controls_fixet0_p60_28.1_probs.npz \
      --pairs FOURIER_FULL1-LSTM_QNN FOURIER_FULL2-LSTM_QNN FOURIER_FULL2-FOURIER2 --B 2000 \
      --out results/json/results_exp10_fourier_full.json
  python code/local_experiments/exp29_phase_intervals.py --npz results/json/results_core_hybrids_controls_fixet0_p60_28.1_probs.npz \
      --pairs FOURIER_FULL2-LSTM_QNN FOURIER_FULL1-LSTM_QNN --out results/local_exp29_fourier_full.json
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(ROOT, 'code'))

from exp27_controls import (RAW_CSV, ARCHIVE, OUT_DIR, TAG, seed_all, build_dataset,  # noqa: E402
                            core_loop, DataLoader, TensorDataset, device)

MERGED = os.path.join(OUT_DIR, f'results_core_hybrids_controls_{TAG}_probs.npz')
MODELS = {'FOURIER_FULL1': 1, 'FOURIER_FULL2': 2}


class FourierFull4D(nn.Module):
    """LinearControl4D backbone; the 4->4 map acts on the full tensor-product trigonometric basis of the 4 angles."""
    def __init__(self, in_dim=35, n_q=4, degree=2):
        super().__init__()
        self.degree = degree
        self.lstm = nn.LSTM(1, 8, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
        self.proj = nn.Sequential(nn.Linear(16 + in_dim, n_q), nn.Dropout(0.1))
        n_feat = (2 * degree + 1) ** n_q
        self.ctrl = nn.Linear(n_feat, n_q)
        self.norm = nn.LayerNorm(n_q)
        self.clf = nn.Linear(n_q, 1)

    def basis(self, qi):
        cols = [torch.ones_like(qi)]
        for k in range(1, self.degree + 1):
            cols += [torch.cos(k * qi), torch.sin(k * qi)]
        b = torch.stack(cols, dim=-1)                       # (B, n_q, 2d+1)
        f = b[:, 0, :]
        for j in range(1, b.shape[1]):
            f = (f.unsqueeze(-1) * b[:, j, :].unsqueeze(1)).reshape(f.shape[0], -1)
        return f                                            # (B, (2d+1)^n_q)

    def forward(self, x):
        _, (hn, _) = self.lstm(x[:, :12].unsqueeze(-1))
        h = torch.cat([hn[-2], hn[-1]], dim=1)
        qi = torch.sigmoid(self.proj(torch.cat([h, x], dim=1))) * math.pi
        return self.clf(self.norm(self.ctrl(self.basis(qi)))).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=list(MODELS), choices=list(MODELS))
    ap.add_argument('--seeds', type=int, default=15)
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    _, ds = build_dataset(RAW_CSV, train_only=True, fix_et0_verano=True, umbral_override=28.1)
    y_te = ds.y_te.numpy().astype(int)
    assert os.environ.get('LOR_DATASET', 'v1').lower() == 'v2' or (len(ds.X_tr), len(ds.X_va), len(ds.X_te)) == (2273, 514, 528)
    if os.path.exists(ARCHIVE):
        assert np.array_equal(np.load(ARCHIVE)['y_te'].astype(int), y_te), 'test labels differ from the archive'
    X_tr, y_tr = ds.X_tr.double(), ds.y_tr.double()
    X_va, y_va, X_te = ds.X_va.double(), ds.y_va.double(), ds.X_te.double()

    if a.dry_run:
        for name, deg in MODELS.items():
            m = FourierFull4D(ds.in_dim, degree=deg).double()
            m.train()
            loss = nn.functional.binary_cross_entropy_with_logits(m(X_tr[:32]), y_tr[:32])
            loss.backward()
            # the basis must be exactly the product of the per-coordinate bases
            q = torch.rand(3, 4, dtype=torch.float64) * math.pi
            f = m.basis(q)
            assert f.shape == (3, (2 * deg + 1) ** 4) and torch.allclose(f[:, 0], torch.ones(3, dtype=torch.float64))
            print(f'  dry-run {name}: features {f.shape[1]}, trainable params '
                  f'{sum(p.numel() for p in m.parameters() if p.requires_grad)}, loss {loss.item():.4f}')
        print('  dry-run OK')
        return

    out_json = os.path.join(OUT_DIR, f'results_fourier_full_trainonly_{TAG}.json')
    out_npz = os.path.join(OUT_DIR, f'results_fourier_full_trainonly_{TAG}_probs.npz')
    state = {'y_te': y_te.tolist(), 'models': {}}
    if os.path.exists(out_json):
        state = json.load(open(out_json, encoding='utf-8'))
    for n in a.models:
        state['models'].setdefault(n, {'probs': [], 'auc': [], 'seconds': []})

    def save():
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(state, f)
        arr = {'y_te': y_te, **{f'{k}_probs': np.asarray(v['probs']) for k, v in state['models'].items() if v['probs']}}
        np.savez_compressed(out_npz, **arr)
        if os.path.exists(MERGED):
            merged = dict(np.load(MERGED))
            for k, v in state['models'].items():
                if len(v['probs']) == a.seeds:
                    merged[f'{k}_probs'] = np.asarray(v['probs'])
            np.savez_compressed(MERGED, **merged)

    for name in a.models:
        st = state['models'][name]
        for seed in range(len(st['probs']), a.seeds):
            t0 = time.time()
            seed_all(seed)
            dl = DataLoader(TensorDataset(X_tr, y_tr), batch_size=16, shuffle=True, drop_last=True)
            m = core_loop(FourierFull4D(ds.in_dim, degree=MODELS[name]).to(device).double(), dl, X_va, y_va,
                          epochs=25, patience=6, lr=5e-3)
            m.eval()
            with torch.no_grad():
                p = torch.sigmoid(m(X_te)).numpy().astype(float)
            st['probs'].append(p.tolist()); st['auc'].append(float(roc_auc_score(y_te, p)))
            st['seconds'].append(time.time() - t0)
            save()
            print(f'{name} seed {seed:2d} AUC {st["auc"][-1]:.4f}  {(time.time() - t0) / 60:.1f} min', flush=True)
    for k, v in state['models'].items():
        au = np.asarray(v['auc'])
        print(f'{k}: n={au.size} mean AUC {au.mean():.4f} SD {au.std(ddof=1):.4f}')
    if not os.path.exists(MERGED):
        print(f'note: {MERGED} not found (run exp27 first); exp10/exp29 need the merged archive')


if __name__ == '__main__':
    main()
