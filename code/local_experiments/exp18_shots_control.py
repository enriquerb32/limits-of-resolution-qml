#!/usr/bin/env python3
"""
exp18_shots_control.py - Finite-shot noiseless control for the inference-time noise check
=========================================================================================
The paper's noise check evaluates the trained LSTM+QNN and LbL-QNN with 1% depolarizing noise AND 1,024 shots
(experimento_ruido.py, Qiskit Aer) and compares the result with exact (analytic) expectation values. That comparison
mixes two effects. This script separates them, using the archived weights (no retraining):

  exact      default.qubit, analytic expectations (the model as trained)
  shots      default.qubit, 1,024 shots, NO noise, R repetitions with seeds 0..R-1     <- the missing control
  noisy      qiskit.aer, depolarizing p on ry/rz/rx/u1/u2/u3 (1q) and cx/cz (2q), 1,024 shots, R repetitions
             (same noise model as experimento_ruido.py; only with --noisy, needs `pip install qiskit-aer pennylane-qiskit`)

For every condition it reports AUC, Brier score and accuracy at 0.5 (mean and SD over repetitions), the correlation
of the test probabilities with the exact ones, and the mean |<Z_i>| entering the LayerNorm. Because the LayerNorm
re-standardises the four expectation values row by row, a uniform contraction of <Z> (as produced by global
depolarizing noise) is largely removed; the |<Z>| column shows how much contraction the LayerNorm absorbed.

Inputs (archived, canonical pipeline): checkpoints/01_data.pt (dataset), checkpoints/06_pesos_ruido.pt (weights:
'LSTM+QNN_state', 'PI-QNN-LbL_state'); archived noisy metrics checkpoints/07_metricas_ruido.pt are printed for reference.
Note: main.py:230 looks for the key 'LbL-QNN_state', but the archived file stores 'PI-QNN-LbL_state'; and
experimento_ruido.py:127 reads ds.ACP_te, which the archived dataset does not have (it has dana_te). This script uses
the archived names.

Usage (from the repository root)
  python code/local_experiments/exp18_shots_control.py                 # exact + noiseless shots, R=15 (minutes)
  python code/local_experiments/exp18_shots_control.py --noisy         # adds the Aer noisy condition (slow)
Output: results/local_exp18_shots_control.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import pennylane as qml
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'code'))
from modelos_cuanticos import LstmQNN  # noqa: E402

CKPT = os.path.join(ROOT, 'checkpoints')
OUT = os.path.join(ROOT, 'results', 'local_exp18_shots_control.json')
N_Q = 4


def circuit_body(W, x, n_l):
    """Identical gate sequence to modelos_cuanticos._make_qnode and experimento_ruido.circ_ruidoso."""
    is_batched = (x.ndim == 2)
    for l in range(n_l):
        for i in range(N_Q):
            val = x[:, i % x.shape[1]] if is_batched else x[i % x.shape[0]]
            qml.RY(val, wires=i)
            qml.RZ(val, wires=i)
        qml.StronglyEntanglingLayers(W[l:l + 1], wires=range(N_Q))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_Q)]


def make_qnode(dev, n_l, shots=None):
    @qml.qnode(dev, interface='torch')
    def circ(W, x):
        return circuit_body(W, x, n_l)
    return qml.set_shots(circ, shots=shots) if shots else circ


def load_model(ds, state, n_l, physics):
    m = LstmQNN(ds.in_dim, n_q=N_Q, n_l=n_l, physics=physics).double()
    m.load_state_dict(state)
    m.eval()
    return m


def evaluate(model, X, y):
    """Returns probabilities and the four expectation values that enter the LayerNorm."""
    captured = {}
    h = model.norm.register_forward_hook(lambda mod, inp, out: captured.__setitem__('z', inp[0].detach()))
    with torch.no_grad():
        out = model(X)
    h.remove()
    logits = out[0] if isinstance(out, tuple) else out
    p = torch.sigmoid(logits).numpy().astype(float)
    return p, captured['z'].numpy()


def metrics(y, p, p_exact, z):
    return dict(auc=float(roc_auc_score(y, p)), brier=float(np.mean((p - y) ** 2)),
                acc=float(np.mean((p >= 0.5) == y)), corr_with_exact=float(np.corrcoef(p, p_exact)[0, 1]),
                mean_abs_z=float(np.mean(np.abs(z))))


def summarise(reps):
    keys = reps[0].keys()
    return {k: dict(mean=float(np.mean([r[k] for r in reps])), sd=float(np.std([r[k] for r in reps], ddof=1)))
            for k in keys}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--reps', type=int, default=15)
    ap.add_argument('--shots', type=int, default=1024)
    ap.add_argument('--noisy', action='store_true', help='add the Qiskit Aer depolarizing condition')
    ap.add_argument('--p', type=float, default=0.01, help='depolarizing probability for --noisy')
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()

    _, ds = torch.load(os.path.join(CKPT, '01_data.pt'), weights_only=False)
    W = torch.load(os.path.join(CKPT, '06_pesos_ruido.pt'), weights_only=False)
    X, y = ds.X_te.double(), ds.y_te.numpy().astype(int)
    specs = {'LSTM+QNN': ('LSTM+QNN_state', 2, False), 'LbL-QNN': ('PI-QNN-LbL_state', 3, True)}

    noisy_dev = None
    if a.noisy:
        from qiskit_aer.noise import NoiseModel, depolarizing_error
        nm = NoiseModel()
        nm.add_all_qubit_quantum_error(depolarizing_error(a.p, 1), ['ry', 'rz', 'rx', 'u1', 'u2', 'u3'])
        nm.add_all_qubit_quantum_error(depolarizing_error(a.p, 2), ['cx', 'cz'])

    out = dict(meta=dict(n_test=int(len(y)), shots=a.shots, reps=a.reps, pennylane=qml.__version__,
                         noisy=bool(a.noisy), p=a.p if a.noisy else None), models={})
    for name, (key, n_l, physics) in specs.items():
        model = load_model(ds, W[key], n_l, physics)
        t0 = time.time()
        p_exact, z_exact = evaluate(model, X, y)
        res = {'exact': metrics(y, p_exact, p_exact, z_exact)}
        print(f'{name:9s} exact      AUC {res["exact"]["auc"]:.4f}  Brier {res["exact"]["brier"]:.4f}  '
              f'mean|Z| {res["exact"]["mean_abs_z"]:.3f}')

        reps = []
        for r in range(a.reps):
            model.circ = make_qnode(qml.device('default.qubit', wires=N_Q, seed=r), n_l, shots=a.shots)
            p, z = evaluate(model, X, y)
            reps.append(metrics(y, p, p_exact, z))
        res['shots_noiseless'] = dict(summary=summarise(reps), reps=reps)
        s = res['shots_noiseless']['summary']
        print(f'{name:9s} shots      AUC {s["auc"]["mean"]:.4f} ± {s["auc"]["sd"]:.4f}  Brier {s["brier"]["mean"]:.4f}  '
              f'r(exact) {s["corr_with_exact"]["mean"]:.3f}  mean|Z| {s["mean_abs_z"]["mean"]:.3f}')

        if a.noisy:
            reps = []
            for r in range(a.reps):
                dev = qml.device('qiskit.aer', wires=N_Q, noise_model=nm, shots=a.shots, seed=r)
                model.circ = make_qnode(dev, n_l)
                p, z = evaluate(model, X, y)
                reps.append(metrics(y, p, p_exact, z))
            res['shots_noisy'] = dict(summary=summarise(reps), reps=reps)
            s = res['shots_noisy']['summary']
            print(f'{name:9s} noisy      AUC {s["auc"]["mean"]:.4f} ± {s["auc"]["sd"]:.4f}  Brier {s["brier"]["mean"]:.4f}  '
                  f'r(exact) {s["corr_with_exact"]["mean"]:.3f}  mean|Z| {s["mean_abs_z"]["mean"]:.3f}')
        res['seconds'] = time.time() - t0
        out['models'][name] = res

    try:
        arch = torch.load(os.path.join(CKPT, '07_metricas_ruido.pt'), weights_only=False)
        out['archived_noisy_metrics'] = {k: {m: float(v[m]) for m in ('auc', 'auc_std', 'acc', 'fpr') if m in v}
                                         for k, v in arch.items()}
        print('archived noisy (07_metricas_ruido.pt):',
              {k: round(v['auc'], 4) for k, v in out['archived_noisy_metrics'].items()})
    except Exception as e:  # pragma: no cover
        out['archived_noisy_metrics'] = {'error': repr(e)}

    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print('saved', a.out)


if __name__ == '__main__':
    main()
