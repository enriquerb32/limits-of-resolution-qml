#!/usr/bin/env python3
"""
exp52_emulation_substitution.py - emulation, substitution and learning, separated (review of 2026-11-18, addition 3)
==================================================================================================================
The end-to-end comparisons of the paper (LSTM+QNN against Linear-4D or Fourier surrogates trained end to end) share the
interface dimension but not the learned angles. This script separates three questions on the 15 trained LSTM+QNN models
(checkpoints/exp31/lstm_qnn_seed{s}.pt):

E. Emulation - can this trained circuit be reproduced classically? Each readout <Z_j>(q) of the 2-layer re-uploading
   circuit is a trigonometric polynomial with frequencies |k| <= 4 in each angle, so its values on the 9^4 = 6,561-point
   grid q_i in {2 pi m / 9} determine it exactly (4-D DFT). The reconstructed polynomial is evaluated on the actual test
   angles and pushed through the SAME LayerNorm and output layer: logit and AUC parity with the circuit. Also reported:
   the error when the reconstruction is truncated to |k| <= 1 or 2 per angle (on the whole torus and on visited angles).
S. Substitution and learning on common angles - with the LSTM and projection of seed s FROZEN (evaluation mode), so
   that every head sees exactly the same angles q, train from scratch: the circuit (new random weights), Linear-4D,
   separable Fourier-2 and the 81- and 625-function tensor bases, each followed by LayerNorm and a linear output, with
   the training loop of the paper. Compared with the original end-to-end LSTM+QNN of the same seed and with each
   other: paired differences of 15-seed means with CR2 (Bell-McCaffrey df, calendar years) and a years x seeds
   bootstrap with seeds drawn jointly (all heads of seed s share its frozen extractor).

  LOR_DATASET=v2 python code/local_experiments/exp52_emulation_substitution.py [--part E S] [--seeds 15]
Output: results/local_exp52_emulation_substitution.json, results/json/results_exp52_common_angles_probs.npz
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(ROOT, 'code'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from exp27_controls import RAW_CSV, build_dataset, seed_all  # noqa: E402
from exp34_circuit_spectrum import angles, readout  # noqa: E402
from exp38_teacher_control import Head  # noqa: E402
from qml_core_trainonly import HybridQNNControl, _loop as core_loop  # noqa: E402
from qml_resolution import cluster_bootstrap_delta, cluster_robust_delta, percentile_ci  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

CKPT = os.path.join(ROOT, 'checkpoints', 'exp31')
ARCH = os.path.join(ROOT, 'results', 'json', 'results_exp31_lstm_qnn_probs.npz')
OUT = os.path.join(ROOT, 'results', 'local_exp52_emulation_substitution.json')
NPZ = os.path.join(ROOT, 'results', 'json', 'results_exp52_common_angles_probs.npz')
HEADS = {'CIRCUIT_NEW': ('circuit', 1), 'LINEAR4D': ('linear', 1), 'FOURIER2': ('sep', 2),
         'FOURIER_FULL1': ('full', 1), 'FOURIER_FULL2': ('full', 2)}
L = 4                                   # maximal frequency per angle of the circuit
G = 2 * L + 1                           # grid points per angle for an exact reconstruction


def load(ds, s):
    m = HybridQNNControl(in_dim=ds.in_dim, mode='trained').double()
    m.load_state_dict(torch.load(os.path.join(CKPT, f'lstm_qnn_seed{s}.pt')))
    return m.eval()


def tail(m, z):
    with torch.no_grad():
        return m.clf(m.norm(torch.as_tensor(z, dtype=torch.float64))).squeeze(-1).numpy()


def fourier_coefficients(m):
    """(4, G, G, G, G) complex DFT coefficients of the four readouts from the 9^4 grid."""
    t = 2 * math.pi * np.arange(G) / G
    grid = np.stack(np.meshgrid(t, t, t, t, indexing='ij'), -1).reshape(-1, 4)
    z = np.concatenate([readout(m, torch.tensor(grid[i:i + 729])) for i in range(0, len(grid), 729)])
    z = z.reshape(G, G, G, G, 4)
    return np.moveaxis(np.fft.fftn(z, axes=(0, 1, 2, 3)) / G ** 4, -1, 0)


def evaluate(coef, q, kmax=L):
    """Evaluate the trigonometric polynomial of coefficients `coef` at angles q (B, 4), keeping |k_j| <= kmax."""
    k = np.fft.fftfreq(G, 1 / G).astype(int)            # 0..4, -4..-1
    keep = np.abs(k) <= kmax
    c = coef[:, keep][:, :, keep][:, :, :, keep][:, :, :, :, keep]
    kk = k[keep]
    E = [np.exp(1j * np.outer(q[:, j], kk)) for j in range(4)]          # (B, K) per angle
    out = np.einsum('rabcd,na,nb,nc,nd->nr', c, E[0], E[1], E[2], E[3], optimize=True)
    return out.real


def part_e(ds, seeds):
    X_te = ds.X_te.double(); y = ds.y_te.numpy().astype(int)
    arch = np.load(ARCH)['LSTM_QNN31_probs'] if os.path.exists(ARCH) else None
    rows = []
    for s in seeds:
        t0 = time.time(); m = load(ds, s)
        q = angles(m, X_te).numpy(); z = readout(m, torch.tensor(q))
        logit = tail(m, z)
        if arch is not None:
            assert np.allclose(1 / (1 + np.exp(-logit)), arch[s], atol=1e-6), f'seed {s}: checkpoint does not reproduce'
        coef = fourier_coefficients(m)
        row = dict(seed=s, auc_circuit=float(roc_auc_score(y, logit)))
        rng = np.random.default_rng(52 + s); q_rand = rng.uniform(0, 2 * math.pi, (2000, 4))
        z_rand = readout(m, torch.tensor(q_rand))
        for kmax in (1, 2, L):
            ze = evaluate(coef, q, kmax); le = tail(m, ze)
            row[f'k{kmax}'] = dict(max_abs_readout_err_test=float(np.abs(ze - z).max()),
                                   max_abs_readout_err_torus=float(np.abs(evaluate(coef, q_rand, kmax) - z_rand).max()),
                                   max_abs_logit_err_test=float(np.abs(le - logit).max()),
                                   auc=float(roc_auc_score(y, le)))
        rows.append(row)
        print(f"  E seed {s:2d}: exact |dz| {row[f'k{L}']['max_abs_readout_err_test']:.1e} |dlogit| "
              f"{row[f'k{L}']['max_abs_logit_err_test']:.1e} | k<=1 AUC {row['k1']['auc']:.4f} vs {row['auc_circuit']:.4f}"
              f" | {time.time() - t0:.0f}s", flush=True)
    return dict(n_coefficients_per_readout=G ** 4, seeds=rows,
                max_logit_err_exact=max(r[f'k{L}']['max_abs_logit_err_test'] for r in rows),
                auc_equal_exact=all(abs(r[f'k{L}']['auc'] - r['auc_circuit']) < 1e-12 for r in rows))


def part_s(ds, seeds, res_prev, probs):
    X_tr, X_va, X_te = ds.X_tr.double(), ds.X_va.double(), ds.X_te.double()
    y_tr, y_va = ds.y_tr.double(), ds.y_va.double(); y = ds.y_te.numpy().astype(int)
    out = res_prev.get('S', {'per_seed': {}})
    for s in seeds:
        if str(s) in out['per_seed']:
            continue
        m = load(ds, s)
        q_tr, q_va, q_te = angles(m, X_tr), angles(m, X_va), angles(m, X_te)
        row = {'ORIGINAL': float(roc_auc_score(y, tail(m, readout(m, q_te))))}
        probs[f'ORIGINAL_s{s}'] = 1 / (1 + np.exp(-tail(m, readout(m, q_te))))
        for name, (kd, dg) in HEADS.items():
            seed_all(s)
            dl = DataLoader(TensorDataset(q_tr, y_tr), batch_size=16, shuffle=True, drop_last=True)
            h = core_loop(Head(kd, dg).double(), dl, q_va, y_va, epochs=25, patience=6, lr=5e-3); h.eval()
            with torch.no_grad():
                o = h(q_te); p = torch.sigmoid(o[0] if isinstance(o, tuple) else o).numpy()
            probs[f'{name}_s{s}'] = p; row[name] = float(roc_auc_score(y, p))
        out['per_seed'][str(s)] = row
        res_prev['S'] = out
        json.dump(res_prev, open(OUT, 'w'), indent=2); np.savez_compressed(NPZ, **probs)
        print(f'  S seed {s:2d}: ' + ' '.join(f'{k} {v:.4f}' for k, v in row.items()), flush=True)
    done = sorted(int(k) for k in out['per_seed'])
    years = pd.to_datetime(pd.Series(ds.meta_te['fecha'])).dt.year.values
    P = {n: np.array([probs[f'{n}_s{s}'] for s in done]) for n in ['ORIGINAL'] + list(HEADS)}
    out['mean_auc'] = {n: float(np.mean([out['per_seed'][str(s)][n] for s in done])) for n in P}
    out['contrasts'] = {}
    for a, b in [('ORIGINAL', 'CIRCUIT_NEW'), ('CIRCUIT_NEW', 'LINEAR4D'), ('CIRCUIT_NEW', 'FOURIER2'),
                 ('CIRCUIT_NEW', 'FOURIER_FULL1'), ('CIRCUIT_NEW', 'FOURIER_FULL2'), ('ORIGINAL', 'LINEAR4D'),
                 ('ORIGINAL', 'FOURIER2'), ('ORIGINAL', 'FOURIER_FULL1'), ('ORIGINAL', 'FOURIER_FULL2')]:
        cr = cluster_robust_delta(y, P[a], P[b], years); lo, hi = cr.ci()
        tw = percentile_ci(cluster_bootstrap_delta(y, P[a], P[b], years, n_boot=2000, random_state=0,
                                                   resample_seeds=True, shared_seeds=True))
        out['contrasts'][f'{a}-{b}'] = dict(delta=cr.delta, ci_cr2=[lo, hi], p_cr2=cr.p_value, ci_two_way=list(tw),
                                            resolved=bool(lo > 0 or hi < 0))
    out['n_seeds'] = len(done)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--part', nargs='+', default=['E', 'S'])
    ap.add_argument('--seeds', type=int, default=15); a = ap.parse_args()
    _, ds = build_dataset(RAW_CSV, train_only=True, fix_et0_verano=True)
    seeds = [s for s in range(a.seeds) if os.path.exists(os.path.join(CKPT, f'lstm_qnn_seed{s}.pt'))]
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    probs = dict(np.load(NPZ)) if os.path.exists(NPZ) else {}
    if 'E' in a.part:
        res['E'] = part_e(ds, seeds); json.dump(res, open(OUT, 'w'), indent=2)
    if 'S' in a.part:
        res['S'] = part_s(ds, seeds, res, probs); json.dump(res, open(OUT, 'w'), indent=2)
        print(json.dumps({k: v for k, v in res['S'].items() if k != 'per_seed'}, indent=1))
    print('wrote', OUT)


if __name__ == '__main__':
    main()
