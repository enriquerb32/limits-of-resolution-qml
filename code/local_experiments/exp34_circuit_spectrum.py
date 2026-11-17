#!/usr/bin/env python3
"""
exp34_circuit_spectrum.py - the frequency spectrum of the trained circuit, and Fourier controls that cover it
=============================================================================================================
Correction of the paper's spectral claim. In LSTM+QNN each angle q_j enters FOUR rotation gates (R_Y and R_Z in each
of the two re-uploading layers), so by Schuld, Sweke & Meyer (2021) each readout <Z_k> is a trigonometric polynomial
with per-coordinate frequencies {0, +-1, ..., +-4}, not {0, +-1, +-2}. The controls Fourier-1/2 and Full Fourier-1/2
therefore use SUBSETS of the circuit's spectrum. This script quantifies how much that matters and adds the separable
controls with the full per-coordinate spectrum.

Part A (no training, ~2 min): for the 15 trained LSTM+QNN models of exp31 (checkpoints/exp31/lstm_qnn_seed{s}.pt)
  A1. spectrum along each coordinate: for 100 test rows per model, sweep q_j over one period (64 points) with the
      other angles at the row's values, FFT the four readouts, and report the share of the readout variance at
      frequencies 1, 2, 3, 4 (also for random weights, as a reference);
  A2. how much of the trained readout map the degree-2 tensor basis captures: least-squares fit of the 625-feature
      Full Fourier-2 basis to the exact readouts <Z_k>(q) on the training angles, R^2 on the test angles, and the same
      for the degree-1 basis (81) and for the separable degree-4 basis (33 features).
Part B (training, ~20-30 min): separable controls Fourier-3 and Fourier-4 (FourierControl4D of exp27 with degree 3, 4;
  per-coordinate frequencies up to the circuit's), 15 seeds, same loop as Fourier-1/2; written into the merged archive
  for exp10 (pairs FOURIER3-LSTM_QNN, FOURIER4-LSTM_QNN, FOURIER4-FOURIER2).

Usage (repository root; needs the exp31 checkpoints for part A)
  python code/local_experiments/exp34_circuit_spectrum.py --part A
  python code/local_experiments/exp34_circuit_spectrum.py --part B            # add --threads 4 if useful
  python code/local_experiments/exp10_paired_ci.py --extra results/json/results_core_hybrids_controls_fixet0_p60_28.1_probs.npz \
      --pairs FOURIER3-LSTM_QNN FOURIER4-LSTM_QNN FOURIER4-FOURIER2 --B 2000 --out results/json/results_exp10_fourier_sep34.json
Outputs: results/local_exp34_circuit_spectrum.json, results/json/results_fourier_sep34_trainonly_fixet0_p60_28.1.json
"""
import argparse
import itertools
import json
import math
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(ROOT, 'code'))

from exp27_controls import (RAW_CSV, OUT_DIR, TAG, seed_all, build_dataset, core_loop,  # noqa: E402
                            DataLoader, TensorDataset, device, FourierControl4D)
from qml_core_trainonly import HybridQNNControl  # noqa: E402

CKPT = os.path.join(ROOT, 'checkpoints', 'exp31')
MERGED = os.path.join(OUT_DIR, f'results_core_hybrids_controls_{TAG}_probs.npz')
OUT_A = os.path.join(ROOT, 'results', 'local_exp34_circuit_spectrum.json')
OUT_B = os.path.join(OUT_DIR, f'results_fourier_sep34_trainonly_{TAG}.json')
MODELS_B = {'FOURIER3': 3, 'FOURIER4': 4}


def angles(m, X):
    with torch.no_grad():
        _, (hn, _) = m.lstm(X[:, :12].unsqueeze(-1))
        h = torch.cat([hn[-2], hn[-1]], dim=1)
        return torch.sigmoid(m.proj(torch.cat([h, X], dim=1))) * math.pi


def readout(m, q):
    with torch.no_grad():
        return torch.stack(m.circ(m.W, q), dim=-1).detach().numpy()          # (B, 4)


def tensor_basis(q, d):
    """(B, (2d+1)^4) product basis, as FOURIER_FULL<d> of exp30."""
    cols = [np.ones_like(q)] + [f(k * q) for k in range(1, d + 1) for f in (np.cos, np.sin)]
    b = np.stack(cols, -1)                                                    # (B, 4, 2d+1)
    out = b[:, 0, :]
    for j in range(1, 4):
        out = (out[:, :, None] * b[:, j, None, :]).reshape(len(q), -1)
    return out


def separable_basis(q, d):
    return np.concatenate([np.ones((len(q), 1))] + [f(k * q) for k in range(1, d + 1) for f in (np.cos, np.sin)], 1)


def r2(y, yhat):
    return float(1 - ((y - yhat) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())


def part_a(ds, n_rows=100, n_grid=64):
    rng = np.random.default_rng(34)
    X_tr, X_te = ds.X_tr.double(), ds.X_te.double()
    t = torch.arange(n_grid, dtype=torch.float64) * 2 * math.pi / n_grid
    res = {'trained': [], 'fits': []}
    paths = [os.path.join(CKPT, f'lstm_qnn_seed{s}.pt') for s in range(15)]
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        raise SystemExit(f'no checkpoints in {CKPT}: run exp31_shot_noise.py first')
    for p in paths + ['random'] * 5:
        m = HybridQNNControl(in_dim=ds.in_dim, mode='trained').double()
        if p == 'random':
            with torch.no_grad():
                m.W.copy_(torch.empty_like(m.W).uniform_(-math.pi, math.pi))
        else:
            m.load_state_dict(torch.load(p))
        m.eval()
        q_te = angles(m, X_te)
        rows = rng.choice(len(q_te), size=min(n_rows, len(q_te)), replace=False)
        power = np.zeros(n_grid // 2 + 1)
        for r, j in itertools.product(rows, range(4)):
            q = q_te[r].repeat(n_grid, 1)
            q[:, j] = t
            z = readout(m, q)
            power += (np.abs(np.fft.rfft(z - z.mean(0), axis=0)) ** 2).sum(1)
        share = power[1:6] / power[1:].sum()
        key = 'trained' if p != 'random' else 'random'
        res.setdefault(key, []).append(share.tolist())
        if p != 'random':
            q_tr = angles(m, X_tr).numpy(); qt = q_te.numpy()
            z_tr, z_te = readout(m, torch.as_tensor(q_tr)), readout(m, torch.as_tensor(qt))
            fit = {}
            res.setdefault('angle_hist_test_18bins', []).append(np.histogram(qt, bins=18, range=(0, math.pi))[0].tolist())
            res.setdefault('angle_quantiles_test', []).append(np.quantile(qt, [.01, .05, .25, .5, .75, .95, .99], axis=0).tolist())
            m_rand = HybridQNNControl(in_dim=ds.in_dim, mode='trained').double()
            m_rand.load_state_dict(m.state_dict())
            with torch.no_grad():
                m_rand.W.copy_(torch.empty_like(m_rand.W).uniform_(-math.pi, math.pi))
            zr_tr, zr_te = readout(m_rand, torch.as_tensor(q_tr)), readout(m_rand, torch.as_tensor(qt))
            A1, B1 = tensor_basis(q_tr, 1), tensor_basis(qt, 1)
            c1, *_ = np.linalg.lstsq(A1, zr_tr, rcond=None)
            fit['full_d1_randomW_same_angles'] = dict(r2_train=r2(zr_tr, A1 @ c1), r2_test=r2(zr_te, B1 @ c1), n_features=81)
            for name, fb in [('full_d1', lambda q: tensor_basis(q, 1)), ('full_d2', lambda q: tensor_basis(q, 2)),
                             ('separable_d2', lambda q: separable_basis(q, 2)), ('separable_d4', lambda q: separable_basis(q, 4))]:
                A, B = fb(q_tr), fb(qt)
                coef, *_ = np.linalg.lstsq(A, z_tr, rcond=None)
                fit[name] = dict(r2_train=r2(z_tr, A @ coef), r2_test=r2(z_te, B @ coef), n_features=int(A.shape[1]))
            res['fits'].append(fit)
            print(f'{os.path.basename(p)}: variance share f=1..4 {np.round(share[:4], 3)} | R2 test '
                  + ', '.join(f'{k} {v["r2_test"]:.4f}' for k, v in fit.items()), flush=True)
    summ = {'n_trained_models': len(res['trained'])}
    for key in ('trained', 'random'):
        s = np.asarray(res[key])
        summ[f'{key}_variance_share_f1_f5'] = s.mean(0).round(4).tolist()
        summ[f'{key}_share_f_le_2_min'] = float(s[:, :2].sum(1).min())
    for name in res['fits'][0]:
        v = np.array([f[name]['r2_test'] for f in res['fits']])
        summ[f'r2_test_{name}'] = dict(mean=float(v.mean()), min=float(v.min()), max=float(v.max()))
    res['summary'] = summ
    json.dump(res, open(OUT_A, 'w'), indent=2)
    print(json.dumps(summ, indent=2))
    print('wrote', OUT_A)


def part_b(ds, seeds):
    y_te = ds.y_te.numpy().astype(int)
    X_tr, y_tr = ds.X_tr.double(), ds.y_tr.double()
    X_va, y_va, X_te = ds.X_va.double(), ds.y_va.double(), ds.X_te.double()
    state = json.load(open(OUT_B)) if os.path.exists(OUT_B) else {'models': {}}
    for name, deg in MODELS_B.items():
        st = state['models'].setdefault(name, {'probs': [], 'auc': []})
        for seed in range(len(st['probs']), seeds):
            t0 = time.time()
            seed_all(seed)
            dl = DataLoader(TensorDataset(X_tr, y_tr), batch_size=16, shuffle=True, drop_last=True)
            m = core_loop(FourierControl4D(ds.in_dim, degree=deg).to(device).double(), dl, X_va, y_va,
                          epochs=25, patience=6, lr=5e-3)
            m.eval()
            with torch.no_grad():
                p = torch.sigmoid(m(X_te)).numpy().astype(float)
            st['probs'].append(p.tolist()); st['auc'].append(float(roc_auc_score(y_te, p)))
            json.dump(state, open(OUT_B, 'w'))
            print(f'{name} seed {seed:2d} AUC {st["auc"][-1]:.4f} ({(time.time() - t0) / 60:.1f} min)', flush=True)
    merged = dict(np.load(MERGED))
    assert np.array_equal(merged['y_te'].astype(int), y_te)
    for name, st in state['models'].items():
        if len(st['probs']) == seeds:
            merged[f'{name}_probs'] = np.asarray(st['probs'])
        au = np.asarray(st['auc'])
        print(f'{name}: mean AUC {au.mean():.4f} SD {au.std(ddof=1):.4f} (n={au.size})')
    np.savez_compressed(MERGED, **merged)
    print('added FOURIER3/FOURIER4 to', MERGED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', choices=['A', 'B'], required=True)
    ap.add_argument('--seeds', type=int, default=15)
    ap.add_argument('--threads', type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    _, ds = build_dataset(RAW_CSV, train_only=True, fix_et0_verano=True, umbral_override=28.1)
    assert os.environ.get('LOR_DATASET', 'v1').lower() == 'v2' or (len(ds.X_tr), len(ds.X_va), len(ds.X_te)) == (2273, 514, 528)
    part_a(ds) if a.part == 'A' else part_b(ds, a.seeds)


if __name__ == '__main__':
    main()
