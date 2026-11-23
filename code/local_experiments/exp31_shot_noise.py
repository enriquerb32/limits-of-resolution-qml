#!/usr/bin/env python3
"""
exp31_shot_noise.py - the third term of the noise budget on hardware: finite-shot estimation of <Z_j> (15 seeds)
================================================================================================================
The paper's hybrids are evaluated with exact expectation values (state-vector simulation). On a quantum device each
<Z_j> is estimated from n_shots measurements of the 4-qubit register, which adds per-row noise to the predicted
probabilities. This script

  1. refits LSTM_QNN with the 15 seeds of the core experiment (HybridQNNControl, core_loop) and saves every model
     (checkpoints/exp31/lstm_qnn_seed{s}.pt), reporting the AUC agreement with the archived LSTM_QNN;
  2. for every seed and test row, computes the exact 16-outcome distribution of the circuit (qml.probs), checks that
     it reproduces the exact <Z_j>, and draws n_shots joint measurements (multinomial; the 4 qubits are measured
     together, so the estimates of the four <Z_j> are correlated as on hardware);
  3. propagates the shot estimates through LayerNorm and the output layer and recomputes the test AUC,
     R times per seed and shot budget.

Reported per shot budget (default 32, 100, 1,000, 10,000): mean AUC loss with respect to exact expectations,
SD of the AUC across repetitions of the same trained model (the shot term of the noise budget, on the scale of one AUC),
the same for the 15-seed mean, and these SDs next to the seed SD (about 0.010) and the sampling SE of one AUC (0.024).
Also: total circuit executions and shots for one test evaluation and, for orientation, for one training epoch with
parameter-shift gradients.

Outputs
  results/local_exp31_shot_noise.json, checkpoints/exp31/*.pt, results/json/results_exp31_lstm_qnn_probs.npz

Usage (from the repository root)
  python code/local_experiments/exp31_shot_noise.py --dry-run
  python code/local_experiments/exp31_shot_noise.py                 # ~45-60 min training + ~5 min sampling (CPU)
  python code/local_experiments/exp31_shot_noise.py --skip-training # re-run the sampling on the saved checkpoints
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pennylane as qml
import torch
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(ROOT, 'code'))

from exp27_controls import (RAW_CSV, ARCHIVE, seed_all, build_dataset, core_loop,  # noqa: E402
                            DataLoader, TensorDataset, device)
from qml_core_trainonly import HybridQNNControl  # noqa: E402

CKPT = os.path.join(ROOT, 'checkpoints', 'exp31')
OUT = os.path.join(ROOT, 'results', 'local_exp31_shot_noise.json')
NPZ = os.path.join(ROOT, 'results', 'json', 'results_exp31_lstm_qnn_probs.npz')
N_Q, N_L = 4, 2


def make_probs_qnode():
    """Same gate sequence as qml_core_trainonly.make_qnode_circuit, returning the 16 outcome probabilities."""
    dev = qml.device('default.qubit', wires=N_Q)

    @qml.qnode(dev, interface='torch')
    def circ(W, x):
        for l in range(N_L):
            for i in range(N_Q):
                qml.RY(x[:, i], wires=i)
                qml.RZ(x[:, i], wires=i)
            qml.StronglyEntanglingLayers(W[l:l + 1], wires=range(N_Q))
        return qml.probs(wires=range(N_Q))
    return circ


# Z eigenvalue of qubit j for each computational basis state (PennyLane: wire 0 is the most significant bit)
SIGNS = np.array([[1 - 2 * ((b >> (N_Q - 1 - j)) & 1) for j in range(N_Q)] for b in range(2 ** N_Q)], float)  # (16, 4)


def angles(model, X):
    with torch.no_grad():
        _, (hn, _) = model.lstm(X[:, :12].unsqueeze(-1))
        h = torch.cat([hn[-2], hn[-1]], dim=1)
        return torch.sigmoid(model.proj(torch.cat([h, X], dim=1))) * math.pi


def head(model, z):
    with torch.no_grad():
        zt = torch.as_tensor(z, dtype=torch.float64)
        return torch.sigmoid(model.clf(model.norm(zt)).squeeze(-1)).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=15)
    ap.add_argument('--shots', type=int, nargs='+', default=[32, 100, 1000, 10000])
    ap.add_argument('--reps', type=int, default=50)
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--skip-training', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    _, ds = build_dataset(RAW_CSV, train_only=True, fix_et0_verano=True, umbral_override=28.1)
    y_te = ds.y_te.numpy().astype(int)
    assert os.environ.get('LOR_DATASET', 'v1').lower() == 'v2' or (len(ds.X_tr), len(ds.X_va), len(ds.X_te)) == (2273, 514, 528)
    X_tr, y_tr = ds.X_tr.double(), ds.y_tr.double()
    X_va, y_va, X_te = ds.X_va.double(), ds.y_va.double(), ds.X_te.double()
    probs_qnode = make_probs_qnode()
    os.makedirs(CKPT, exist_ok=True)
    seeds = 1 if a.dry_run else a.seeds

    # ---------------------------------------------------------------- 1. models
    models, exact = [], []
    for s in range(seeds):
        path = os.path.join(CKPT, f'lstm_qnn_seed{s}.pt')
        m = HybridQNNControl(in_dim=ds.in_dim, mode='trained').double()
        if (a.skip_training or a.dry_run) and os.path.exists(path):
            m.load_state_dict(torch.load(path))
        elif a.dry_run:
            pass                                                  # untrained model: wiring check only
        else:
            t0 = time.time()
            seed_all(s)
            dl = DataLoader(TensorDataset(X_tr, y_tr), batch_size=16, shuffle=True, drop_last=True)
            m = core_loop(m.to(device), dl, X_va, y_va, epochs=25, patience=6, lr=5e-3)
            torch.save(m.state_dict(), path)
            print(f'seed {s:2d} trained in {(time.time() - t0) / 60:.1f} min', flush=True)
        m.eval()
        with torch.no_grad():
            exact.append(torch.sigmoid(m(X_te)).numpy())
        models.append(m)
    exact = np.asarray(exact)
    auc_exact = np.array([roc_auc_score(y_te, p) for p in exact])
    if not a.dry_run:
        np.savez_compressed(NPZ, y_te=y_te, LSTM_QNN31_probs=exact)
    agree = None
    if os.path.exists(ARCHIVE) and not a.dry_run:
        arch = np.load(ARCHIVE)['LSTM_QNN_probs'][:seeds]
        agree = dict(max_abs_prob_diff=float(np.abs(arch - exact).max()),
                     archived_mean_auc=float(np.mean([roc_auc_score(y_te, p) for p in arch])),
                     refit_mean_auc=float(auc_exact.mean()))
        print('agreement with the archived LSTM_QNN:', agree)

    # ---------------------------------------------------------------- 2-3. shots
    rng = np.random.default_rng(31)
    P = []
    for m in models:
        with torch.no_grad():
            p16 = probs_qnode(m.W, angles(m, X_te)).numpy()               # (528, 16)
        z_from_p = p16 @ SIGNS
        z_exact = torch.stack(m.circ(m.W, angles(m, X_te)), dim=-1).detach().numpy()
        err = np.abs(z_from_p - z_exact).max()
        assert err < 1e-8, f'probability readout does not reproduce <Z>: {err}'
        assert np.allclose(head(m, z_exact), torch.sigmoid(m(X_te)).detach().numpy(), atol=1e-10)
        P.append(np.clip(p16, 0, None) / np.clip(p16, 0, None).sum(1, keepdims=True))
    print(f'readout check OK (max |<Z> from probs - exact| < 1e-8 for all {len(models)} models)')

    res = {'n_test': int(y_te.size), 'seeds': seeds, 'reps': a.reps, 'agreement_with_archive': agree,
           'auc_exact_mean': float(auc_exact.mean()), 'auc_exact_sd_seeds': float(auc_exact.std(ddof=1)) if seeds > 1 else None,
           'by_shots': {}}
    for n in a.shots:
        auc = np.zeros((seeds, a.reps))
        for i, (m, p16) in enumerate(zip(models, P)):
            for r in range(a.reps):
                counts = np.stack([rng.multinomial(n, row) for row in p16])  # (528, 16)
                z_hat = (counts / n) @ SIGNS
                auc[i, r] = roc_auc_score(y_te, head(m, z_hat))
        loss = auc_exact[:, None] - auc
        shot_sd_single = auc.std(axis=1, ddof=1).mean()                       # per trained model
        shot_sd_mean = auc.mean(axis=0).std(ddof=1)                           # 15-seed mean AUC across repetitions
        res['by_shots'][str(n)] = dict(
            mean_auc=float(auc.mean()), mean_auc_loss=float(loss.mean()), max_seed_mean_loss=float(loss.mean(1).max()),
            shot_sd_single_model=float(shot_sd_single), shot_sd_seed_mean=float(shot_sd_mean),
            shots_per_test_evaluation=int(n * y_te.size), circuit_executions_per_test_evaluation=int(y_te.size))
        print(f'n_shots={n:6d}: AUC {auc.mean():.4f} (exact {auc_exact.mean():.4f}; loss {loss.mean():+.4f}) | '
              f'shot SD one model {shot_sd_single:.4f}, 15-seed mean {shot_sd_mean:.4f}', flush=True)
    res['training_cost_parameter_shift'] = dict(
        quantum_parameters=24, encoding_gate_occurrences=16, circuits_per_row_per_step=1 + 2 * 24 + 2 * 16,
        note='parameter shift: 1 forward circuit, 2 per circuit weight (24) and 2 per encoding-gate occurrence '
             '(4 angles x RY,RZ x 2 layers = 16; needed to backpropagate into the classical layers) = 81 circuits per '
             'row and step; per epoch multiply by 2,273 rows and by the shots per circuit')
    if not a.dry_run:
        with open(OUT, 'w') as f:
            json.dump(res, f, indent=2)
        print('wrote', OUT)
    else:
        print('dry-run OK')


if __name__ == '__main__':
    main()
