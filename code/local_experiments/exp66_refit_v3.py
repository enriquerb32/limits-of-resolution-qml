#!/usr/bin/env python3
"""
exp66_refit_v3.py - refit v3 of the first panel: one declared search and best-of-3 final fits for every neural model
====================================================================================================================
Implements section 2 of audit/refit_plan_v3.txt (commit adfcd21, doi:10.5281/zenodo.24466944). Run with
LOR_DATASET=v2 V2_CAUSAL=1 in a clean worktree (results/, logs_v2/, checkpoints/ empty at the start).

  --step check    prints and verifies the search space (log-uniform learning rate and weight decay), no training
  --step search   Optuna TPE, 30 trials (sampler seed 0, training seed 100), validation AUC; the 3 best
                  configurations refitted with seeds 101-103; the highest mean validation AUC wins (ties: lower mean
                  validation loss). Writes results/refit_v3/search_<MODEL>.json
  --step final    the selected setting, seeds 0-14, each the best of 3 starts (start k uses seed s + 1000 k) by the
                  validation AUC of the restored early-stopping checkpoint (ties: lower validation loss). Writes
                  results/refit_v3/final_<MODEL>.json and _probs.npz (test and validation probabilities)

Models searched (plan, section 2): PureLSTM, LSTM_QNN, QTL_FIXED (LSTM+QTL), CTRL_LINEAR4D, FOURIER1-4 (separable),
FOURIER_FULL1, FOURIER_FULL2, FOURIER_FULL4 (tensor bases of 81, 625, 6,561 functions).
Not searched (plan: frozen circuits take the LSTM+QNN setting): FROZEN_RANDOM, FROZEN_IDENTITY. Clarification: the
two heads that replace the circuit of LSTM+QTL (QTL_CLASSICAL, QTL_TENSOR1) take the LSTM+QTL setting by the same
rule; they were not named in the plan.

Clarifications of the plan (written to every output):
  * One training loop for every neural model: AdamW (searched learning rate and weight decay), ReduceLROnPlateau
    (patience 3, factor 0.5), gradient clipping at 1.0, weighted BCE, early stopping on the validation loss with the
    searched patience and maximum epochs, best-validation-loss weights restored. In v2 the LSTM+QTL head had its own
    loop (no scheduler, batch 8); here it uses the common loop with the searched setting.
  * LSTM+QTL and its two control heads sit on a frozen PureLSTM extractor of the same seed: in the search, a PureLSTM
    fitted with the selected PureLSTM setting at the trial seed (one start); in the final fits, the final PureLSTM of
    seed s (itself the best of 3 starts), stored in checkpoints/refit_v3/.
  * Labels: pooled training 60th percentile (no override). Dropout keeps the value of each architecture.
No resume across runs: a model whose output exists is skipped (it was written by this run, which the date audit
checks); a partly finished search continues from its own Optuna database in results/refit_v3/.
"""
import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(ROOT, 'code'))

from qml_core_trainonly import PureLSTM, HybridQNNControl, LinearControl4D  # noqa: E402
from trainonly_dataset import build_dataset  # noqa: E402
from exp27_controls import FourierControl4D, TransferHead  # noqa: E402
from exp30_fourier_full import FourierFull4D  # noqa: E402

RAW_CSV = os.path.join(HERE, 'datos_limpios_multiestacion.csv')
OUT = os.path.join(ROOT, 'results', 'refit_v3')
CKPT = os.path.join(ROOT, 'checkpoints', 'refit_v3')
SEARCHED = ['PureLSTM', 'LSTM_QNN', 'QTL_FIXED', 'CTRL_LINEAR4D', 'FOURIER1', 'FOURIER2', 'FOURIER3', 'FOURIER4',
            'FOURIER_FULL1', 'FOURIER_FULL2', 'FOURIER_FULL4']
DONOR = {'FROZEN_RANDOM': 'LSTM_QNN', 'FROZEN_IDENTITY': 'LSTM_QNN', 'QTL_CLASSICAL': 'QTL_FIXED',
         'QTL_TENSOR1': 'QTL_FIXED'}
ALL = SEARCHED + list(DONOR)
TRANSFER = {'QTL_FIXED': 'quantum', 'QTL_CLASSICAL': 'classical', 'QTL_TENSOR1': 'tensor1'}
N_TRIALS, TOP, TRIAL_SEED, REEVAL_SEEDS, FINAL_SEEDS, STARTS = 30, 3, 100, (101, 102, 103), range(15), 3
SPACE = dict(lr=(1e-3, 1e-1), wd=(1e-5, 1e-2), epochs=[25, 50, 100, 200], patience=[6, 15, 30], batch=[16, 32])
WD_EXCEPTION = {'FOURIER_FULL4': (1e-4, 1e-1)}      # declared exception of the plan
if os.environ.get('REFIT_SMOKE') == '1':             # wiring test only: tiny budget, separate folders
    N_TRIALS, TOP, REEVAL_SEEDS, FINAL_SEEDS, STARTS = 2, 1, (101,), range(1), 2
    SPACE = dict(SPACE, epochs=[1, 2])
    OUT, CKPT = OUT + '_smoke', CKPT + '_smoke'


def seed_all(s):
    torch.manual_seed(s)
    np.random.seed(s)


def suggest(t, name):
    lo, hi = WD_EXCEPTION.get(name, SPACE['wd'])
    return dict(lr=t.suggest_float('lr', *SPACE['lr'], log=True), wd=t.suggest_float('wd', lo, hi, log=True),
                epochs=t.suggest_categorical('epochs', SPACE['epochs']),
                patience=t.suggest_categorical('patience', SPACE['patience']),
                batch=t.suggest_categorical('batch', SPACE['batch']))


def check_space():
    """The learning rate and the weight decay must be log-uniform; verified on the distributions and by sampling."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    out = {}
    for name in ('LSTM_QNN', 'FOURIER_FULL4'):
        st = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
        st.optimize(lambda t: (suggest(t, name), 0.0)[1], n_trials=4000)
        d = st.trials[0].distributions
        assert isinstance(d['lr'], optuna.distributions.FloatDistribution) and d['lr'].log, d['lr']
        assert isinstance(d['wd'], optuna.distributions.FloatDistribution) and d['wd'].log, d['wd']
        lr = np.log10([t.params['lr'] for t in st.trials])
        wd = np.log10([t.params['wd'] for t in st.trials])
        # log-uniform on [1e-3, 1e-1]: log10 uniform on [-3, -1], median -2, quartiles -2.5 / -1.5
        q = np.percentile(lr, [25, 50, 75])
        assert np.allclose(q, [-2.5, -2.0, -1.5], atol=0.06), q
        out[name] = dict(lr=repr(d['lr']), wd=repr(d['wd']), lr_log10_quartiles=q.round(3).tolist(),
                         wd_log10_range=[round(float(wd.min()), 2), round(float(wd.max()), 2)],
                         epochs=repr(d['epochs']), patience=repr(d['patience']), batch=repr(d['batch']))
    return out


class Data:
    def __init__(self):
        _, ds = build_dataset(RAW_CSV, train_only=True, fix_et0_verano=True)
        self.ds = ds
        self.X_tr, self.y_tr = ds.X_tr.double(), ds.y_tr.double()
        self.X_va, self.y_va, self.X_te = ds.X_va.double(), ds.y_va.double(), ds.X_te.double()
        self.y_va_np, self.y_te_np = ds.y_va.numpy().astype(int), ds.y_te.numpy().astype(int)
        self.in_dim = ds.in_dim


def build(name, D, extractor=None):
    d = D.in_dim
    if name == 'PureLSTM':
        return PureLSTM(d)
    if name == 'LSTM_QNN':
        return HybridQNNControl(d, mode='trained')
    if name == 'FROZEN_RANDOM':
        return HybridQNNControl(d, mode='frozen_random')
    if name == 'FROZEN_IDENTITY':
        return HybridQNNControl(d, mode='frozen_identity')
    if name == 'CTRL_LINEAR4D':
        return LinearControl4D(d)
    if name.startswith('FOURIER_FULL'):
        return FourierFull4D(d, degree=int(name[-1]))
    if name.startswith('FOURIER'):
        return FourierControl4D(d, degree=int(name[-1]))
    if name in TRANSFER:
        return TransferHead(copy.deepcopy(extractor), 32 + d, head=TRANSFER[name])
    raise ValueError(name)


def loop(model, D, cfg):
    """The common loop (see the docstring); returns the model with the best-validation-loss weights restored."""
    n_neg, n_pos = float((D.y_tr == 0).sum()), float((D.y_tr == 1).sum())
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / (n_pos + 1e-5)]))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg['lr'], weight_decay=cfg['wd'])
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    dl = DataLoader(TensorDataset(D.X_tr, D.y_tr), batch_size=cfg['batch'], shuffle=True, drop_last=True)
    best, state, bad, ep = float('inf'), None, 0, 0
    for ep in range(1, cfg['epochs'] + 1):
        model.train()
        for bX, by in dl:
            opt.zero_grad()
            loss = crit(model(bX), by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(D.X_va), D.y_va).item()
        sch.step(vl)
        if vl < best - 1e-4:
            best, state, bad, best_ep = vl, copy.deepcopy(model.state_dict()), 0, ep
        else:
            bad += 1
            if bad >= cfg['patience']:
                break
    if state is not None:
        model.load_state_dict(state)
    return model, dict(val_loss=best, epochs_run=ep, best_epoch=best_ep if state is not None else 0)


def predict(m, X):
    m.eval()
    with torch.no_grad():
        return torch.sigmoid(m(X)).numpy().astype(float)


def fit_one(name, cfg, seed, D, extractor=None):
    seed_all(seed)
    m = build(name, D, extractor).double()
    t0 = time.time()
    m, info = loop(m, D, cfg)
    pv = predict(m, D.X_va)
    info.update(val_auc=float(roc_auc_score(D.y_va_np, pv)), seed=seed, minutes=(time.time() - t0) / 60)
    return m, pv, info


def best_of_starts(name, cfg, s, D, extractor=None):
    fits = [fit_one(name, cfg, s + 1000 * k, D, extractor) for k in range(STARTS)]
    k = max(range(STARTS), key=lambda i: (fits[i][2]['val_auc'], -fits[i][2]['val_loss']))
    return fits[k], k, [f[2] for f in fits]


def lstm_extractor(cfg, seed, D):
    return fit_one('PureLSTM', cfg, seed, D)[0]


def read_json(p):
    return json.load(open(p, encoding='utf-8'))


def selected_setting(name):
    src = DONOR.get(name, name)
    p = os.path.join(OUT, f'search_{src}.json')
    if not os.path.exists(p):
        raise SystemExit(f'{p} missing: run --step search --models {src} first')
    return read_json(p)['selected']['params'], src


def step_search(name, D, threads):
    import optuna
    out = os.path.join(OUT, f'search_{name}.json')
    if os.path.exists(out):
        print('done already (this run):', out); return
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    ext_cfg = selected_setting('PureLSTM')[0] if name in TRANSFER else None
    ext_cache = {}

    def ext(seed):
        if ext_cfg is None:
            return None
        if seed not in ext_cache:
            ext_cache[seed] = lstm_extractor(ext_cfg, seed, D)
        return ext_cache[seed]

    db = f"sqlite:///{os.path.join(OUT, f'optuna_{name}.db')}"
    st = optuna.create_study(study_name=name, storage=db, load_if_exists=True, direction='maximize',
                             sampler=optuna.samplers.TPESampler(seed=0))

    def obj(t):
        cfg = suggest(t, name)
        _, _, info = fit_one(name, cfg, TRIAL_SEED, D, ext(TRIAL_SEED))
        for k, v in info.items():
            t.set_user_attr(k, v)
        print(f'{name} trial {t.number:2d} {cfg} val AUC {info["val_auc"]:.4f} ({info["minutes"]:.1f} min, '
              f'{info["epochs_run"]} epochs)', flush=True)
        return info['val_auc']

    done = len([t for t in st.trials if t.state == optuna.trial.TrialState.COMPLETE])
    st.optimize(obj, n_trials=max(0, N_TRIALS - done))
    trials = [t for t in st.trials if t.state == optuna.trial.TrialState.COMPLETE][:N_TRIALS]
    assert st.trials[0].distributions['lr'].log and st.trials[0].distributions['wd'].log
    top = sorted(trials, key=lambda t: (-t.value, t.user_attrs['val_loss']))[:TOP]
    reeval = []
    for t in top:
        infos = [fit_one(name, dict(t.params), s, D, ext(s))[2] for s in REEVAL_SEEDS]
        reeval.append(dict(trial=t.number, params=dict(t.params), trial_val_auc=t.value,
                           mean_val_auc=float(np.mean([i['val_auc'] for i in infos])),
                           mean_val_loss=float(np.mean([i['val_loss'] for i in infos])), fits=infos))
        print(f'{name} re-eval trial {t.number}: mean val AUC {reeval[-1]["mean_val_auc"]:.4f}', flush=True)
    sel = max(reeval, key=lambda r: (r['mean_val_auc'], -r['mean_val_loss']))
    on_edge = {k: sel['params'][k] in (SPACE[k][0], SPACE[k][-1]) for k in ('epochs', 'patience', 'batch')}
    lo, hi = WD_EXCEPTION.get(name, SPACE['wd'])
    on_edge.update(lr_near_bound=bool(sel['params']['lr'] < 1.2e-3 or sel['params']['lr'] > 0.083),
                   wd_near_bound=bool(sel['params']['wd'] < 1.2 * lo or sel['params']['wd'] > hi / 1.2))
    res = dict(model=name, plan='audit/refit_plan_v3.txt (commit adfcd21; doi:10.5281/zenodo.24466944)',
               space=dict(SPACE, wd=[lo, hi]), sampler='TPE(seed=0)', trial_seed=TRIAL_SEED,
               reeval_seeds=list(REEVAL_SEEDS), extractor_setting=ext_cfg, threads=threads,
               trials=[dict(number=t.number, params=t.params, value=t.value, **t.user_attrs) for t in trials],
               top=reeval, selected=sel, selected_on_edge=on_edge, causal=os.environ.get('V2_CAUSAL') == '1',
               threshold=float(D.ds.umbral), n=dict(train=len(D.X_tr), val=len(D.X_va), test=len(D.X_te)))
    json.dump(res, open(out, 'w', encoding='utf-8'), indent=2)
    print(f'{name}: selected trial {sel["trial"]} {sel["params"]} mean val AUC {sel["mean_val_auc"]:.4f}', flush=True)


def step_final(name, D, threads):
    out = os.path.join(OUT, f'final_{name}.json')
    if os.path.exists(out):
        print('done already (this run):', out); return
    cfg, src = selected_setting(name)
    rows, P_te, P_va = [], [], []
    for s in FINAL_SEEDS:
        extractor = None
        if name in TRANSFER:
            p = os.path.join(CKPT, f'PureLSTM_s{s}.pt')
            if not os.path.exists(p):
                raise SystemExit(f'{p} missing: run --step final --models PureLSTM first')
            extractor = PureLSTM(D.in_dim).double()
            extractor.load_state_dict(torch.load(p))
        (m, pv, info), k, starts = best_of_starts(name, cfg, s, D, extractor)
        pt = predict(m, D.X_te)
        if name == 'PureLSTM':
            os.makedirs(CKPT, exist_ok=True)
            torch.save(m.state_dict(), os.path.join(CKPT, f'PureLSTM_s{s}.pt'))
        P_te.append(pt); P_va.append(pv)
        rows.append(dict(seed=s, start_chosen=k, test_auc=float(roc_auc_score(D.y_te_np, pt)), starts=starts))
        print(f'{name} seed {s:2d} start {k} val AUC {info["val_auc"]:.4f} test AUC {rows[-1]["test_auc"]:.4f} '
              f'({sum(x["minutes"] for x in starts):.1f} min)', flush=True)
    np.savez_compressed(os.path.join(OUT, f'final_{name}_probs.npz'), test=np.array(P_te), val=np.array(P_va),
                        y_te=D.y_te_np, y_va=D.y_va_np)
    json.dump(dict(model=name, setting=cfg, setting_from=src, starts=STARTS, seeds=list(FINAL_SEEDS), fits=rows,
                   mean_test_auc=float(np.mean([r['test_auc'] for r in rows])), threads=threads,
                   causal=os.environ.get('V2_CAUSAL') == '1', threshold=float(D.ds.umbral)),
              open(out, 'w', encoding='utf-8'), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--step', choices=['check', 'search', 'final', 'time'], required=True)
    ap.add_argument('--models', nargs='+', default=[], choices=ALL)
    ap.add_argument('--threads', type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_default_dtype(torch.float64)
    if a.step == 'check':
        print(json.dumps(check_space(), indent=1)); return
    if os.environ.get('LOR_DATASET') != 'v2' or os.environ.get('V2_CAUSAL') != '1':
        raise SystemExit('set LOR_DATASET=v2 and V2_CAUSAL=1 (causal pipeline is primary in refit v3)')
    os.makedirs(OUT, exist_ok=True)
    D = Data()
    print(f'rows {len(D.X_tr)}/{len(D.X_va)}/{len(D.X_te)} | threshold {D.ds.umbral:.2f} | test positives '
          f'{D.y_te_np.sum()}', flush=True)
    if a.step == 'time':          # one epoch per model at batch 16, for planning only (nothing written)
        for name in a.models:
            ext = PureLSTM(D.in_dim).double() if name in TRANSFER else None
            _, _, info = fit_one(name, dict(lr=5e-3, wd=1e-4, epochs=1, patience=1, batch=16), 0, D, ext)
            print(f'{name:16s} 1 epoch {info["minutes"] * 60:.1f} s', flush=True)
        return
    for name in a.models:
        if a.step == 'search':
            if name not in SEARCHED:
                raise SystemExit(f'{name} is not searched (it takes the setting of {DONOR[name]})')
            step_search(name, D, a.threads)
        else:
            step_final(name, D, a.threads)


if __name__ == '__main__':
    main()
