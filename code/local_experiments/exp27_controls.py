#!/usr/bin/env python3
"""
exp27_controls.py - corrected transfer-learning hybrid and circuit-free / frozen-circuit controls (15 seeds)
============================================================================================================
Same data, split, fitting rows (2,273), early-stopping rows (514), test rows (528), label threshold (28.1),
corrected et0 feature and seed indices as the archived core (results_hybrids_trainonly_fixet0_p60_28.1_probs.npz),
so every output pairs row by row and seed by seed with RF_2273 / PureLSTM / CTRL_LINEAR4D / LSTM_QNN.

Models (--models; default: all)
  QTL_FIXED       transfer learning done as described in the paper: per seed s, the PureLSTM extractor of seed s
                  (stored as QTL_EXTRACTOR27, so that it is paired with the head trained on it) is frozen AND kept in evaluation mode (fixed batch-norm statistics, no
                  dropout); trainable adapter 67->8->4 with sigmoid, angles = adapter * pi (in [0, pi], scaled once),
                  RY angle encoding, 2 StronglyEntanglingLayers (24 parameters), LayerNorm, linear output.
                  Plain weighted BCE, no heuristic penalty. Loop of entrenar_qtl: AdamW 2e-3, batch 8, <=15 epochs,
                  patience 4 on the validation loss.
  QTL_CLASSICAL   identical to QTL_FIXED with the circuit replaced by Linear(4->4) (20 parameters): the control that
                  isolates the quantum head.
  FROZEN_RANDOM   LSTM_QNN with circuit weights drawn U(-pi, pi) and frozen (HybridQNNControl mode='frozen_random').
  FROZEN_IDENTITY LSTM_QNN with circuit weights frozen at 0 (mode='frozen_identity'; Rot(0) = I, CNOT ring kept).
  FOURIER1        LinearControl4D with the linear map applied to [sin q, cos q] (8 -> 4, 36 parameters).
  FOURIER2        idem with [sin q, cos q, sin 2q, cos 2q] (16 -> 4, 68 parameters): the frequencies a 2-layer
                  re-uploading circuit can represent per coordinate (no cross-coordinate products).
  FROZEN_*, FOURIER* use exactly the training loop of LSTM_QNN (core_loop: AdamW 5e-3, batch 16, <=25 epochs,
  patience 6).

Outputs (resumable; rewritten after every seed)
  results/json/results_controls_trainonly_fixet0_p60_28.1.json
  results/json/results_controls_trainonly_fixet0_p60_28.1_probs.npz           keys '<MODEL>_probs' (S x 528), 'y_te'
  results/json/results_core_hybrids_controls_fixet0_p60_28.1_probs.npz        the above merged with the archive

Usage (from the repository root)
  python code/local_experiments/exp27_controls.py --dry-run           # builds every model, 1 forward/backward
  python code/local_experiments/exp27_controls.py                     # all models x 15 seeds (about 4 h on CPU)
  python code/local_experiments/exp27_controls.py --models QTL_FIXED QTL_CLASSICAL
Then
  python code/local_experiments/exp25_within_year_structure.py \
      --npz results/json/results_core_hybrids_controls_fixet0_p60_28.1_probs.npz \
      --pairs QTL_EXTRACTOR27-QTL_FIXED QTL_CLASSICAL-QTL_FIXED CTRL_LINEAR4D-LSTM_QNN FOURIER1-LSTM_QNN \
              FOURIER2-LSTM_QNN FROZEN_RANDOM-LSTM_QNN FROZEN_IDENTITY-LSTM_QNN RF_2273-QTL_FIXED \
      --out results/local_exp25_controls.json
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
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import pennylane as qml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(ROOT, 'code'))

# qml_core_trainonly sets float64 as default dtype and imports trainonly_dataset
from qml_core_trainonly import PureLSTM, HybridQNNControl, _loop as core_loop  # noqa: E402
from trainonly_dataset import build_dataset  # noqa: E402

RAW_CSV = os.path.join(HERE, 'datos_limpios_multiestacion.csv')
ARCHIVE = os.path.join(ROOT, 'results', 'json', 'results_hybrids_trainonly_fixet0_p60_28.1_probs.npz')
OUT_DIR = os.path.join(ROOT, 'results', 'json')
TAG = 'fixet0_p60_28.1'
ALL_MODELS = ['QTL_FIXED', 'QTL_CLASSICAL', 'QTL_TENSOR1', 'FROZEN_RANDOM', 'FROZEN_IDENTITY', 'FOURIER1', 'FOURIER2']
device = torch.device('cpu')


def seed_all(s):
    torch.manual_seed(s)
    np.random.seed(s)


# ------------------------------------------------------------------ models
class FrozenExtractor(nn.Module):
    """PureLSTM state + raw features after the extractor's batch norm, always in evaluation mode, no gradients."""
    def __init__(self, encoder):
        super().__init__()
        self.enc = encoder
        for p in self.enc.parameters():
            p.requires_grad_(False)
        self.enc.eval()

    def train(self, mode=True):           # model.train() must not switch the extractor back to training mode
        super().train(mode)
        self.enc.eval()
        return self

    def forward(self, x):
        with torch.no_grad():
            _, (hn, _) = self.enc.lstm(x[:, :12].unsqueeze(-1))
            h = torch.cat([hn[-2], hn[-1]], dim=1)
            return self.enc.bn(torch.cat([h, x], dim=1))


class TransferHead(nn.Module):
    def __init__(self, encoder, in_embed, head='quantum', n_q=4, n_l=2):
        super().__init__()
        self.extractor = FrozenExtractor(encoder)
        self.adapter = nn.Sequential(nn.Linear(in_embed, 2 * n_q), nn.GELU(), nn.Dropout(0.1),
                                     nn.Linear(2 * n_q, n_q), nn.Sigmoid())
        self.head = head
        if head == 'quantum':
            dev = qml.device('default.qubit', wires=n_q)

            @qml.qnode(dev, interface='torch', diff_method='backprop')
            def circ(W, x):
                qml.AngleEmbedding(x, wires=range(n_q), rotation='Y')     # x already in [0, pi]
                qml.StronglyEntanglingLayers(W, wires=range(n_q))
                return [qml.expval(qml.PauliZ(i)) for i in range(n_q)]
            self.circ = circ
            self.W = nn.Parameter(torch.empty(n_l, n_q, 3).uniform_(-0.1, 0.1))
        elif head == 'tensor1':     # exact span of the R_Y-embedding head: {1, cos q_j, sin q_j} tensor basis, 3^4 = 81
            self.ctrl = nn.Linear(3 ** n_q, n_q)
        elif head == 'classical':
            self.ctrl = nn.Linear(n_q, n_q)
        else:
            raise ValueError(head)
        self.norm = nn.LayerNorm(n_q)
        self.clf = nn.Linear(n_q, 1)

    def forward(self, x):
        qi = self.adapter(self.extractor(x)) * math.pi
        if self.head == 'quantum':
            res = self.circ(self.W, qi)
            z = (torch.stack(res, dim=-1) if isinstance(res, (list, tuple)) else res).to(qi.dtype)
        elif self.head == 'tensor1':
            b = torch.stack([torch.ones_like(qi), torch.cos(qi), torch.sin(qi)], -1)
            f = b[:, 0]
            for j in range(1, b.shape[1]):
                f = (f[:, :, None] * b[:, j, None, :]).reshape(len(qi), -1)
            z = self.ctrl(f)
        else:
            z = self.ctrl(qi)
        return self.clf(self.norm(z)).squeeze(-1)


class FourierControl4D(nn.Module):
    """Backbone of LinearControl4D / HybridQNNControl; the circuit is replaced by a linear map of Fourier features."""
    def __init__(self, in_dim=35, n_q=4, degree=1):
        super().__init__()
        self.degree = degree
        self.lstm = nn.LSTM(1, 8, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
        self.proj = nn.Sequential(nn.Linear(16 + in_dim, n_q), nn.Dropout(0.1))
        self.ctrl = nn.Linear(2 * degree * n_q, n_q)
        self.norm = nn.LayerNorm(n_q)
        self.clf = nn.Linear(n_q, 1)

    def forward(self, x):
        _, (hn, _) = self.lstm(x[:, :12].unsqueeze(-1))
        h = torch.cat([hn[-2], hn[-1]], dim=1)
        qi = torch.sigmoid(self.proj(torch.cat([h, x], dim=1))) * math.pi
        feats = torch.cat([f(k * qi) for k in range(1, self.degree + 1) for f in (torch.sin, torch.cos)], dim=1)
        return self.clf(self.norm(self.ctrl(feats))).squeeze(-1)


def transfer_loop(model, X_tr, y_tr, X_va, y_va, epochs=15, patience=4, lr=2e-3, batch=8):
    """The loop of modelos_cuanticos.entrenar_qtl with use_physics=False (no scheduler)."""
    n_neg, n_pos = (y_tr == 0).sum(), (y_tr == 1).sum()
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float(n_neg / (n_pos + 1e-5))]))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    dl = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch, shuffle=True, drop_last=True)
    best, best_state, ni = float('inf'), None, 0
    for _ in range(epochs):
        model.train()
        for bX, by in dl:
            opt.zero_grad()
            loss = crit(model(bX), by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(X_va), y_va).item()
        if vl < best - 1e-4:
            best, best_state, ni = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            ni += 1
            if ni >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def n_trainable(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def make_trainers(ds, epochs_scale=1.0):
    X_tr, y_tr = ds.X_tr.double(), ds.y_tr.double()
    X_va, y_va, X_te = ds.X_va.double(), ds.y_va.double(), ds.X_te.double()
    in_embed = 32 + ds.in_dim
    ep = lambda e: max(1, int(round(e * epochs_scale)))  # noqa: E731

    def predict(m):
        m.eval()
        with torch.no_grad():
            return torch.sigmoid(m(X_te)).cpu().numpy().astype(float)

    def fit_core(model):
        dl = DataLoader(TensorDataset(X_tr, y_tr), batch_size=16, shuffle=True, drop_last=True)
        m = core_loop(model.to(device).double(), dl, X_va, y_va, epochs=ep(25), patience=6, lr=5e-3)
        return m, predict(m)

    cache = {}   # extractor per seed, shared by QTL_FIXED and QTL_CLASSICAL

    def extractor(seed):
        if seed not in cache:
            seed_all(seed)
            cache[seed] = fit_core(PureLSTM(in_dim=ds.in_dim))
        return cache[seed]

    def t_transfer(head, name):
        def run(seed):
            enc, p_ext = extractor(seed)
            seed_all(seed)
            m = TransferHead(enc, in_embed, head=head).double()
            m = transfer_loop(m, X_tr, y_tr, X_va, y_va, epochs=ep(15))
            return {'QTL_EXTRACTOR27': p_ext, name: predict(m)}
        return run

    def t_core(build, name):
        def run(seed):
            seed_all(seed)
            return {name: fit_core(build())[1]}
        return run

    return {
        'QTL_FIXED': t_transfer('quantum', 'QTL_FIXED'),
        'QTL_CLASSICAL': t_transfer('classical', 'QTL_CLASSICAL'),
        'QTL_TENSOR1': t_transfer('tensor1', 'QTL_TENSOR1'),
        'FROZEN_RANDOM': t_core(lambda: HybridQNNControl(in_dim=ds.in_dim, mode='frozen_random'), 'FROZEN_RANDOM'),
        'FROZEN_IDENTITY': t_core(lambda: HybridQNNControl(in_dim=ds.in_dim, mode='frozen_identity'), 'FROZEN_IDENTITY'),
        'FOURIER1': t_core(lambda: FourierControl4D(in_dim=ds.in_dim, degree=1), 'FOURIER1'),
        'FOURIER2': t_core(lambda: FourierControl4D(in_dim=ds.in_dim, degree=2), 'FOURIER2'),
    }


def outputs_of(name):
    return ['QTL_EXTRACTOR27', name] if name.startswith('QTL_') else [name]


def dry_run(ds):
    X, y = ds.X_tr[:32].double(), ds.y_tr[:32].double()
    enc = PureLSTM(ds.in_dim).double()
    checks = {
        'QTL_FIXED': TransferHead(enc, 32 + ds.in_dim, 'quantum').double(),
        'QTL_CLASSICAL': TransferHead(enc, 32 + ds.in_dim, 'classical').double(),
        'FROZEN_RANDOM': HybridQNNControl(ds.in_dim, mode='frozen_random').double(),
        'FROZEN_IDENTITY': HybridQNNControl(ds.in_dim, mode='frozen_identity').double(),
        'FOURIER1': FourierControl4D(ds.in_dim, degree=1).double(),
        'FOURIER2': FourierControl4D(ds.in_dim, degree=2).double(),
    }
    for name, m in checks.items():
        m.train()
        loss = nn.functional.binary_cross_entropy_with_logits(m(X), y)
        loss.backward()
        extra = ''
        if name.startswith('QTL'):
            extra = f' | extractor in eval mode during training: {not m.extractor.enc.training}'
        print(f'  dry-run {name:16s} loss {loss.item():.4f} trainable params {n_trainable(m):5d}{extra}')
    print('  dry-run OK')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', nargs='+', default=ALL_MODELS, choices=ALL_MODELS)
    ap.add_argument('--seeds', type=int, default=15)
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--out-dir', default=OUT_DIR)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)

    _, ds = build_dataset(RAW_CSV, train_only=True, fix_et0_verano=True, umbral_override=28.1)
    y_te = ds.y_te.numpy().astype(int)
    assert os.environ.get('LOR_DATASET', 'v1').lower() == 'v2' or (len(ds.X_tr), len(ds.X_va), len(ds.X_te)) == (2273, 514, 528), 'split sizes differ from the paper'
    if os.path.exists(ARCHIVE):
        assert np.array_equal(np.load(ARCHIVE)['y_te'].astype(int), y_te), 'test labels differ from the archive'
    print(f'rows {len(ds.X_tr)}/{len(ds.X_va)}/{len(ds.X_te)} | threshold {ds.umbral:.2f} | test positives {y_te.sum()}')
    if a.dry_run:
        dry_run(ds)
        return

    os.makedirs(a.out_dir, exist_ok=True)
    out_json = os.path.join(a.out_dir, f'results_controls_trainonly_{TAG}.json')
    out_npz = os.path.join(a.out_dir, f'results_controls_trainonly_{TAG}_probs.npz')
    out_merged = os.path.join(a.out_dir, f'results_core_hybrids_controls_{TAG}_probs.npz')
    state = {'y_te': y_te.tolist(), 'models': {}}
    if os.path.exists(out_json):
        old = json.load(open(out_json, encoding='utf-8'))
        if old.get('y_te') != state['y_te']:
            raise SystemExit(f'{out_json} has different test labels')
        state['models'] = old['models']
    for n in a.models:
        for o in outputs_of(n):
            state['models'].setdefault(o, {'probs': [], 'auc': [], 'seconds': []})

    def save():
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(state, f)
        arr = {'y_te': y_te, **{f'{k}_probs': np.asarray(v['probs']) for k, v in state['models'].items() if v['probs']}}
        np.savez_compressed(out_npz, **arr)
        if os.path.exists(ARCHIVE):
            merged = dict(np.load(ARCHIVE))
            for k, v in state['models'].items():
                if len(v['probs']) == a.seeds:
                    merged[f'{k}_probs'] = np.asarray(v['probs'])
            np.savez_compressed(out_merged, **merged)

    trainers = make_trainers(ds)
    for name in a.models:
        st_main = state['models'][name]
        for seed in range(len(st_main['probs']), a.seeds):
            t0 = time.time()
            res = trainers[name](seed)
            dt = time.time() - t0
            for o, p in res.items():
                p = np.asarray(p, float)
                assert p.shape == (len(y_te),) and np.isfinite(p).all(), o
                st = state['models'][o]
                if len(st['probs']) > seed:
                    st['probs'][seed], st['auc'][seed], st['seconds'][seed] = p.tolist(), float(roc_auc_score(y_te, p)), dt
                else:
                    st['probs'].append(p.tolist()); st['auc'].append(float(roc_auc_score(y_te, p))); st['seconds'].append(dt)
            save()
            print(f'{name:16s} seed {seed:2d} ' + ' '.join(f'{o} {state["models"][o]["auc"][seed]:.4f}' for o in res)
                  + f'  {dt / 60:.1f} min', flush=True)

    if 'QTL_EXTRACTOR27' in state['models'] and os.path.exists(ARCHIVE):
        ext = np.asarray(state['models']['QTL_EXTRACTOR27']['probs'])
        ref = np.load(ARCHIVE)['QTL_EXTRACTOR_probs'][:len(ext)]
        print(f'extractor vs archived QTL_EXTRACTOR: max |dp| = {np.abs(ext - ref).max():.2e} '
              '(0 if the environment reproduces the archive; the pairing below uses the new extractor either way)')
    print('\nsummary')
    for o, v in state['models'].items():
        au = np.asarray(v['auc'])
        print(f'  {o:16s} n={len(au):2d} mean AUC {au.mean():.4f} SD {au.std(ddof=1) if len(au) > 1 else float("nan"):.4f}')


if __name__ == '__main__':
    main()
