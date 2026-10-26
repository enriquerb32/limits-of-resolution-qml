#!/usr/bin/env python3
"""
exp17_hybrids_multiseed.py - 15-seed train-only runs of the remaining hybrids, storing every test probability
==============================================================================================================
Closes the gap "only the seed-0 probabilities of LbL-QNN, LSTM+QTL and LSTM+PI-QNN were archived".
Same data, split, fitting rows (2,273), early-stopping rows (514), test rows (528) and seed indices as
qml_core_trainonly.py, so the outputs can be paired with RF_2273 / PureLSTM / CTRL_LINEAR4D / LSTM_QNN.

Models (choose with --models; default: the three missing hybrids)
  LSTM_PIQNN     LstmQNN(physics=True) + thermodynamic loss + _loop_with_physics   (modelos_cuanticos.entrenar_qml, 2nd half)
  LBL_QNN        modelos_cuanticos.entrenar_qnn_layerbylayer (3 layers x 10 epochs + fine-tune)
  LSTM_QTL       per seed s: PureLSTM extractor trained with seed s (exactly as qml_core_trainonly.PureLSTM),
                 then modelos_cuanticos.entrenar_qtl on that extractor. The extractor's own test probabilities are
                 stored too, as QTL_EXTRACTOR (paired extractor-vs-hybrid comparison).
  RF_2273, PureLSTM, CTRL_LINEAR4D, LSTM_QNN
                 the four core models of qml_core_trainonly.py (use them with --label S5 to retrain under the
                 coverage-corrected label definition).

Notes
  * The training code of the hybrids is imported unchanged from code/modelos_cuanticos.py. Seeds are set with
    torch.manual_seed(s) and np.random.seed(s) immediately before each model is built. In the canonical multi-seed run
    (robustez_estadistica.py) PI-QNN was trained after LSTM+QNN in the same RNG stream and QTL used ONE fixed
    extractor for all seeds, so seed-by-seed values are not expected to match earlier runs.
  * entrenar_qtl calls qtl.train(), which also puts the "frozen" extractor in train mode: its BatchNorm running
    statistics are updated and its LSTM dropout is active during QTL training (gradients are off). This is the
    behaviour of the code used in the paper and is reproduced as is.
  * Environment: the core archive (results_qml_core_trainonly_probs.npz) was produced on Colab with
    scikit-learn 1.6.1. Locally, PyTorch training reproduces Colab exactly (PureLSTM seed 0: 0.6529), and RF_2273
    reproduces it bit for bit with scikit-learn 1.6.1 (with 1.9.1 it differs: seed 0 AUC 0.6832 vs 0.6849).
    Install it before running RF: pip install scikit-learn==1.6.1
  * Dataset: trainonly_dataset.py (used by the archived core) misaligns et0_verano_prev / potencial_convectivo
    (about 68 % of rows take another station's value on the same date) and recomputes P60 = 28.00 on the training
    rows. The classical train-only rerun used the correct feature and P60 = 28.1. Together these two differences
    are the whole difference between the two datasets (3 training/validation labels, 2 features), i.e. the source
    of PureLSTM 0.6630 (core) vs 0.6682 (rerun). Default: the archived core dataset, so that the new hybrids pair
    with the archived core. --fix-et0 --p60 28.1: exactly the rerun dataset; then rerun the core models too.
  * --label S5: target from the coverage-corrected monthly sum, threshold recomputed on training rows
    (trainonly_dataset.build_dataset(label_scenario='S5')); predictors unchanged. Outputs get the suffix _S5.

Outputs (resumable; rewritten after every seed)
  results/json/results_hybrids_trainonly[_S5].json         probs, AUC and seconds per model and seed
  results/json/results_hybrids_trainonly[_S5]_probs.npz    keys '<MODEL>_probs' (S x 528) and 'y_te'
  results/json/results_core_plus_hybrids[_S5]_probs.npz    the above merged with the core archive
                                                           (results_qml_core_trainonly_probs.npz) when labels match

Usage (from the repository root)
  python code/local_experiments/exp17_hybrids_multiseed.py --dry-run          # wiring check, ~1 min
  python code/local_experiments/exp17_hybrids_multiseed.py                    # 3 hybrids x 15 seeds (many hours)
  python code/local_experiments/exp17_hybrids_multiseed.py --models LSTM_PIQNN --seeds 15
  python code/local_experiments/exp17_hybrids_multiseed.py --label S5 --models RF_2273 PureLSTM CTRL_LINEAR4D LSTM_QNN
Then
  python code/local_experiments/exp10_paired_ci.py --extra results/json/results_core_plus_hybrids_probs.npz \
      --pairs RF_2273-LSTM_PIQNN RF_2273-LBL_QNN RF_2273-LSTM_QTL CTRL_LINEAR4D-LSTM_PIQNN QTL_EXTRACTOR-LSTM_QTL \
      --B 2000 --out results/json/results_exp10_hybrids.json
  python code/local_experiments/exp16_block_c.py       # picks up the merged archive automatically
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CODE = os.path.join(ROOT, 'code')
sys.path.insert(0, HERE)
sys.path.insert(1, CODE)

# qml_core_trainonly sets float64 as default dtype and imports trainonly_dataset (which registers the 'features' shim)
from qml_core_trainonly import PureLSTM, HybridQNNControl, LinearControl4D, _loop as core_loop  # noqa: E402
from trainonly_dataset import build_dataset  # noqa: E402
import modelos_cuanticos as mq  # noqa: E402

RAW_CSV = os.path.join(HERE, 'datos_limpios_multiestacion.csv')
CORE_NPZ = os.path.join(ROOT, 'results', 'json', 'results_qml_core_trainonly_probs.npz')
OUT_DIR = os.path.join(ROOT, 'results', 'json')
ALL_MODELS = ['RF_2273', 'PureLSTM', 'CTRL_LINEAR4D', 'LSTM_QNN', 'LSTM_PIQNN', 'LBL_QNN', 'LSTM_QTL']
DEFAULT_MODELS = ['LSTM_PIQNN', 'LBL_QNN', 'LSTM_QTL']
device = torch.device('cpu')


def seed_all(s):
    torch.manual_seed(s)
    np.random.seed(s)


def make_trainers(ds):
    """Each trainer takes a seed and returns {output_name: test probabilities (528,)}."""
    X_tr, y_tr = ds.X_tr.double(), ds.y_tr.double()
    X_va, y_va, X_te = ds.X_va.double(), ds.y_va.double(), ds.X_te.double()

    def loader():
        return DataLoader(TensorDataset(X_tr, y_tr), batch_size=16, shuffle=True, drop_last=True)

    def predict(m):
        m.eval()
        with torch.no_grad():
            out = m(X_te)
            logits = out[0] if isinstance(out, tuple) else out
            return torch.sigmoid(logits).cpu().numpy().astype(float)

    def fit_core(model):
        m = core_loop(model.to(device).double(), loader(), X_va, y_va, epochs=25, patience=6, lr=5e-3)
        return m, predict(m)

    def t_rf(seed):
        imp = SimpleImputer(strategy='mean')
        rf = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight='balanced', random_state=seed, n_jobs=-1)
        rf.fit(imp.fit_transform(ds.X_sk), ds.y_sk)
        return {'RF_2273': rf.predict_proba(imp.transform(ds.X_te.numpy()))[:, 1].astype(float)}

    def t_lstm(seed):
        seed_all(seed)
        return {'PureLSTM': fit_core(PureLSTM(in_dim=ds.in_dim))[1]}

    def t_ctrl(seed):
        seed_all(seed)
        return {'CTRL_LINEAR4D': fit_core(LinearControl4D(in_dim=ds.in_dim))[1]}

    def t_qnn(seed):
        seed_all(seed)
        return {'LSTM_QNN': fit_core(HybridQNNControl(in_dim=ds.in_dim, mode='trained'))[1]}

    def t_piqnn(seed):
        # second half of modelos_cuanticos.entrenar_qml, unchanged
        seed_all(seed)
        n_neg, n_pos = (ds.y_tr == 0).sum().float(), (ds.y_tr == 1).sum().float()
        pos_w = torch.tensor([n_neg / (n_pos + 1e-5)])
        T_tr = torch.tensor([ds.t_anomaly_mes[m] for m in ds.m_tr], dtype=torch.float64)
        dl = DataLoader(TensorDataset(X_tr, y_tr, T_tr), batch_size=16, shuffle=True, drop_last=True)
        model = mq.LstmQNN(ds.in_dim, physics=True).double()
        model.physics_fn = mq.make_thermodynamic_physics_loss(pos_w)
        model, _ = mq._loop_with_physics(model, dl, X_va, y_va)
        return {'LSTM_PIQNN': predict(model)}

    def t_lbl(seed):
        seed_all(seed)
        res = mq.entrenar_qnn_layerbylayer(ds)
        return {'LBL_QNN': np.asarray(res['LbL-QNN']['_prob'], float)}

    def t_qtl(seed):
        seed_all(seed)
        extractor, p_ext = fit_core(PureLSTM(in_dim=ds.in_dim))
        seed_all(seed)
        res = mq.entrenar_qtl(ds, extractor)
        return {'QTL_EXTRACTOR': p_ext, 'LSTM_QTL': np.asarray(res['LSTM+QTL']['_prob'], float)}

    return {'RF_2273': t_rf, 'PureLSTM': t_lstm, 'CTRL_LINEAR4D': t_ctrl, 'LSTM_QNN': t_qnn,
            'LSTM_PIQNN': t_piqnn, 'LBL_QNN': t_lbl, 'LSTM_QTL': t_qtl}


def outputs_of(name):
    return ['QTL_EXTRACTOR', 'LSTM_QTL'] if name == 'LSTM_QTL' else [name]


def dry_run(ds):
    """Builds every model and pushes 32 rows through forward + one backward pass. No training, nothing saved."""
    X, y = ds.X_tr[:32].double(), ds.y_tr[:32].double()
    T = torch.tensor([ds.t_anomaly_mes[m] for m in ds.m_tr[:32]], dtype=torch.float64)
    pos_w = torch.tensor([1.5], dtype=torch.float64)
    phys = mq.make_thermodynamic_physics_loss(pos_w)
    checks = {
        'PureLSTM': PureLSTM(ds.in_dim).double(),
        'CTRL_LINEAR4D': LinearControl4D(ds.in_dim).double(),
        'LSTM_QNN': HybridQNNControl(ds.in_dim, mode='trained').double(),
        'LSTM_PIQNN': mq.LstmQNN(ds.in_dim, physics=True).double(),
        'LBL_QNN': mq.LstmQNN(ds.in_dim, n_q=4, n_l=3, physics=True).double(),
        'LSTM_QTL': mq.QuantumTransferLearning(PureLSTM(ds.in_dim).double(), in_embed=32 + ds.in_dim,
                                               n_q=4, n_l=2, use_physics=True),
    }
    for name, m in checks.items():
        t0 = time.time()
        m.train()
        out = m(X)
        logits, qo = out if isinstance(out, tuple) else (out, None)
        loss = phys(logits, y, qo, T, 1, 10) if qo is not None else torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        n_par = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f'  dry-run {name:14s} logits {tuple(logits.shape)} loss {loss.item():.4f} trainable params {n_par:6d} ({time.time() - t0:.1f}s)')
    print('  dry-run OK: all models build, run forward and backward on 32 training rows.')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', nargs='+', default=DEFAULT_MODELS, choices=ALL_MODELS)
    ap.add_argument('--seeds', type=int, default=15, help='number of seeds (indices 0..seeds-1)')
    ap.add_argument('--label', choices=['orig', 'S5'], default='orig')
    ap.add_argument('--threads', type=int, default=0, help='torch threads (0 = all cores)')
    ap.add_argument('--fix-et0', action='store_true',
                    help='correct the et0_verano_prev misalignment of trainonly_dataset (outputs get _fixet0; '
                         'then also rerun the core models, the archived core used the misaligned feature)')
    ap.add_argument('--p60', type=float, default=None,
                    help='fix the label threshold (28.1 = the classical rerun and the paper); default: training P60 (28.00, as the core)')
    ap.add_argument('--out-dir', default=OUT_DIR, help='output folder (default results/json)')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)

    import sklearn
    if 'RF_2273' in a.models and sklearn.__version__ != '1.6.1':
        print(f'WARNING: scikit-learn {sklearn.__version__}; the core archive used 1.6.1 (Colab). RF_2273 reproduces the '
              f'archive bit for bit only with 1.6.1 (pip install scikit-learn==1.6.1).')
    print('Building TRAIN-ONLY dataset', '(label S5)' if a.label == 'S5' else '(project labels)',
          '| et0_verano_prev corrected' if a.fix_et0 else '| et0_verano_prev as archived (misaligned)')
    _, ds = build_dataset(RAW_CSV, train_only=True, label_scenario=None if a.label == 'orig' else 'S5',
                          fix_et0_verano=a.fix_et0, umbral_override=a.p60)
    if a.p60 is not None and a.label != 'orig':
        raise SystemExit('--p60 applies to the project labels only')
    y_te = ds.y_te.numpy().astype(int)
    print(f'rows train/val/test = {len(ds.X_tr)}/{len(ds.X_va)}/{len(ds.X_te)} | threshold = {ds.umbral:.2f} '
          f'| test positives = {int(y_te.sum())}')
    assert (len(ds.X_tr), len(ds.X_va), len(ds.X_te)) == (2273, 514, 528), 'split sizes differ from the paper'
    if a.label == 'orig':
        assert abs(ds.umbral - 28.1) < 0.2, 'P60 differs from 28.1'

    if a.dry_run:
        dry_run(ds)
        return

    suffix = ('' if a.label == 'orig' else '_S5') + ('_fixet0' if a.fix_et0 else '') + (f'_p60_{a.p60:g}' if a.p60 else '')
    out_json = os.path.join(a.out_dir, f'results_hybrids_trainonly{suffix}.json')
    out_npz = os.path.join(a.out_dir, f'results_hybrids_trainonly{suffix}_probs.npz')
    out_merged = os.path.join(a.out_dir, f'results_core_plus_hybrids{suffix}_probs.npz')
    os.makedirs(a.out_dir, exist_ok=True)

    outs = [o for n in a.models for o in outputs_of(n)]
    state = {'y_te': y_te.tolist(), 'label': a.label, 'threshold': float(ds.umbral), 'models': {}}
    if os.path.exists(out_json):
        old = json.load(open(out_json, encoding='utf-8'))
        if old.get('y_te') == state['y_te']:
            state['models'] = old['models']
            print('[resume]', {k: len(v['probs']) for k, v in state['models'].items()})
        else:
            raise SystemExit(f'{out_json} has different test labels; move it away before running')
    for o in outs:
        state['models'].setdefault(o, {'probs': [], 'auc': [], 'seconds': []})

    def save():
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(state, f)
        arr = {'y_te': y_te}
        for k, v in state['models'].items():
            if v['probs']:
                arr[f'{k}_probs'] = np.asarray(v['probs'])
        np.savez_compressed(out_npz, **arr)
        # merged archive with the 15-seed core models (only complete models, only if the labels match)
        if a.label == 'orig' and not a.fix_et0 and a.p60 is None and os.path.exists(CORE_NPZ):
            core = np.load(CORE_NPZ)
            if np.array_equal(core['y_te'].astype(int), y_te):
                merged = {k: core[k] for k in core.files}
                for k, v in state['models'].items():
                    if len(v['probs']) == a.seeds and f'{k}_probs' not in merged:
                        merged[f'{k}_probs'] = np.asarray(v['probs'])
                np.savez_compressed(out_merged, **merged)

    trainers = make_trainers(ds)
    for name in a.models:
        first = outputs_of(name)[-1]
        for seed in range(len(state['models'][first]['probs']), a.seeds):
            t0 = time.time()
            res = trainers[name](seed)
            dt = time.time() - t0
            for o, p in res.items():
                p = np.asarray(p, float)
                assert p.shape == (528,) and np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1, o
                st = state['models'][o]
                if len(st['probs']) > seed:      # QTL_EXTRACTOR may already exist from an interrupted seed
                    st['probs'][seed], st['auc'][seed], st['seconds'][seed] = p.tolist(), float(roc_auc_score(y_te, p)), dt
                else:
                    st['probs'].append(p.tolist()); st['auc'].append(float(roc_auc_score(y_te, p))); st['seconds'].append(dt)
            save()
            print(f'{name:14s} seed {seed:2d} ' + ' '.join(f'{o} AUC {state["models"][o]["auc"][seed]:.4f}' for o in res)
                  + f'  {dt / 60:.1f} min', flush=True)

    # QTL extractor vs the core PureLSTM (same class, same seeds); equal only if run in the same environment as the core
    if a.label == 'orig' and not a.fix_et0 and a.p60 is None and 'LSTM_QTL' in a.models and os.path.exists(CORE_NPZ):
        core = np.load(CORE_NPZ)
        ext = np.asarray(state['models']['QTL_EXTRACTOR']['probs'])
        ref = core['PureLSTM_probs'][:len(ext)]
        print(f'QTL extractor vs core PureLSTM (Colab archive): max |dp| = {np.abs(ext - ref).max():.2e} over {len(ext)} seeds '
              f'| mean AUC {np.mean([roc_auc_score(y_te, p) for p in ext]):.4f} vs {np.mean([roc_auc_score(y_te, p) for p in ref]):.4f}')

    print('\nsummary')
    for o in outs:
        au = np.asarray(state['models'][o]['auc'])
        print(f'  {o:14s} n={len(au):2d} mean AUC {au.mean():.4f} SD {au.std(ddof=1) if len(au) > 1 else float("nan"):.4f}')
    print('saved', out_json, out_npz, sep='\n  ')
    if os.path.exists(out_merged):
        print('  ' + out_merged)


if __name__ == '__main__':
    main()
