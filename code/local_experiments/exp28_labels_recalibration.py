#!/usr/bin/env python3
"""
exp28_labels_recalibration.py - station x month exceedance labels and recalibrated proper scores (15 seeds)
==========================================================================================================
Two reviewer requests, one training run per label definition. Every model stores validation AND test probabilities.

  --label orig          the paper's target (monthly sum > pooled training P60 = 28.1 mm; corrected et0 feature).
                        Used for the recalibration analysis: Platt (logistic on logit p) and isotonic maps are fitted
                        on the validation partition (1997-2007) and applied to the test partition.
  --label stationmonth  target = monthly sum > the 60th percentile of the training years (<= 1996) of the SAME station
                        and calendar month. Every station x month cell has a training exceedance rate of about 0.4,
                        so the month x station climatology has no discrimination by construction (AUC about 0.5) and
                        any AUC above 0.5 is skill beyond the seasonal and station base rates.

Models: RF_2273 (as the core), XGB (configuration of modelos_clasicos.py, fitted on the 2,273 rows), PureLSTM,
CTRL_LINEAR4D, LSTM_QNN (exactly the core classes and loop). Optional: --models ... FOURIER2 (from exp27).

Outputs
  results/json/results_exp28_<label>.json / _probs.npz     keys '<MODEL>_val' (S x 514), '<MODEL>_probs' (S x 528),
                                                           'y_va', 'y_te', 'clim_val', 'clim_te'
  results/local_exp28_<label>_summary.json                 written by --analyze

Usage (from the repository root)
  python code/local_experiments/exp28_labels_recalibration.py --label orig --dry-run
  python code/local_experiments/exp28_labels_recalibration.py --label orig             # ~1.5 h on CPU
  python code/local_experiments/exp28_labels_recalibration.py --label stationmonth     # ~1.5 h on CPU
  python code/local_experiments/exp28_labels_recalibration.py --label orig --analyze
  python code/local_experiments/exp28_labels_recalibration.py --label stationmonth --analyze
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(ROOT, 'code'))
sys.path.insert(2, os.path.join(ROOT, 'src'))

from qml_core_trainonly import PureLSTM, HybridQNNControl, LinearControl4D, _loop as core_loop  # noqa: E402
import trainonly_dataset as tds  # noqa: E402

RAW_CSV = os.path.join(HERE, 'datos_limpios_multiestacion.csv')
OUT_DIR = os.path.join(ROOT, 'results', 'json')
ALL_MODELS = ['RF_2273', 'XGB', 'PureLSTM', 'CTRL_LINEAR4D', 'LSTM_QNN', 'FOURIER2']
DEFAULT_MODELS = ['RF_2273', 'XGB', 'PureLSTM', 'CTRL_LINEAR4D', 'LSTM_QNN']
device = torch.device('cpu')


def seed_all(s):
    torch.manual_seed(s)
    np.random.seed(s)


def load_dataset(label, floor=0.1):
    df = pd.read_csv(RAW_CSV, index_col='fecha', parse_dates=['fecha'])
    agg = tds.construir_features(df, anomalia_fn=tds._anomalia_train_only, fix_et0_verano=True)
    if label == 'orig':
        return tds.preparar_dataset(agg, umbral_override=28.1)
    # station x month 60th percentile of the training years, then label = precip / threshold > 1
    train = agg.index.year <= tds.ANIO_CORTE_TRAIN
    key = list(zip(agg['ciudad'].values, agg.index.month))
    thr = (agg.loc[train].assign(_m=agg.index[train].month)
           .groupby(['ciudad', '_m'])['precip_label' if 'label_ok' in agg.columns else 'precip'].quantile(0.60).to_dict())
    # Use a strictly positive floor (default 0.1 mm, standard AEMET rain detection limit)
    # because San Javier July P60 is 0.0 mm in training years.
    t = np.array([max(thr[k], floor) for k in key])
    agg = agg.copy()
    agg['precip_ratio'] = (agg['precip_label_f'] if 'label_ok' in agg.columns else agg['precip']).values / t
    return tds.preparar_dataset(agg, umbral_override=1.0, label_col='precip_ratio')


def climatology(ds):
    """Training exceedance frequency per station x calendar month, for validation and test rows."""
    city_idx = [i for i, n in enumerate(ds.feature_names) if n.startswith('ciudad_')]
    st = lambda X: np.asarray(X)[:, city_idx].argmax(1)  # noqa: E731
    s_tr, s_va, s_te = st(ds.X_tr), st(ds.X_va), st(ds.X_te)
    y_tr = ds.y_tr.numpy()
    freq = {}
    for s in np.unique(s_tr):
        for m in range(1, 13):
            k = (s_tr == s) & (ds.m_tr == m)
            freq[(s, m)] = y_tr[k].mean() if k.any() else y_tr.mean()
    f = lambda S, M: np.array([freq[(s, m)] for s, m in zip(S, M)])  # noqa: E731
    return f(s_va, ds.m_va), f(s_te, ds.m_te)


def make_trainers(ds, epochs_scale=1.0):
    X_tr, y_tr = ds.X_tr.double(), ds.y_tr.double()
    X_va, y_va, X_te = ds.X_va.double(), ds.y_va.double(), ds.X_te.double()
    ep = lambda e: max(1, int(round(e * epochs_scale)))  # noqa: E731

    def predict(m, X):
        m.eval()
        with torch.no_grad():
            return torch.sigmoid(m(X)).cpu().numpy().astype(float)

    def neural(build):
        def run(seed):
            seed_all(seed)
            dl = DataLoader(TensorDataset(X_tr, y_tr), batch_size=16, shuffle=True, drop_last=True)
            m = core_loop(build().to(device).double(), dl, X_va, y_va, epochs=ep(25), patience=6, lr=5e-3)
            return predict(m, X_va), predict(m, X_te)
        return run

    def rf(seed):
        imp = SimpleImputer(strategy='mean')
        m = __import__('dataset_v2').rf_v2(seed) if os.environ.get('LOR_DATASET', 'v1').lower() == 'v2' else RandomForestClassifier(n_estimators=300, max_depth=8, class_weight='balanced', random_state=seed, n_jobs=-1)
        m.fit(imp.fit_transform(ds.X_sk), ds.y_sk)
        return (m.predict_proba(imp.transform(ds.X_va.numpy()))[:, 1], m.predict_proba(imp.transform(ds.X_te.numpy()))[:, 1])

    def xgb(seed):
        if os.environ.get('LOR_DATASET', 'v1').lower() == 'v2':   # v2: tuned configuration, early stopping on the validation AUC
            import dataset_v2
            imp = SimpleImputer(strategy='mean')
            Xt = imp.fit_transform(ds.X_sk)
            Xv, Xe = imp.transform(ds.X_va.numpy()), imp.transform(ds.X_te.numpy())
            m = dataset_v2.xgb_fit_v2(seed, Xt, ds.y_sk, Xv, ds.y_va.numpy())
            return m.predict_proba(Xv)[:, 1], m.predict_proba(Xe)[:, 1]
        from xgboost import XGBClassifier
        imp = SimpleImputer(strategy='mean')
        n_neg, n_pos = (ds.y_sk == 0).sum(), (ds.y_sk == 1).sum()
        m = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                          scale_pos_weight=n_neg / max(n_pos, 1), eval_metric='logloss', random_state=seed, n_jobs=-1)
        m.fit(imp.fit_transform(ds.X_sk), ds.y_sk, verbose=False)
        return (m.predict_proba(imp.transform(ds.X_va.numpy()))[:, 1], m.predict_proba(imp.transform(ds.X_te.numpy()))[:, 1])

    trainers = {'RF_2273': rf, 'XGB': xgb,
                'PureLSTM': neural(lambda: PureLSTM(in_dim=ds.in_dim)),
                'CTRL_LINEAR4D': neural(lambda: LinearControl4D(in_dim=ds.in_dim)),
                'LSTM_QNN': neural(lambda: HybridQNNControl(in_dim=ds.in_dim, mode='trained'))}
    try:
        from exp27_controls import FourierControl4D
        trainers['FOURIER2'] = neural(lambda: FourierControl4D(in_dim=ds.in_dim, degree=2))
    except ImportError:
        pass
    return trainers


# ------------------------------------------------------------------ analysis
def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


def recalibrate(y_va, p_va, p_te):
    eps = 1e-6
    lg = lambda p: np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps))).reshape(-1, 1)  # noqa: E731
    platt = LogisticRegression(C=1e6).fit(lg(p_va), y_va).predict_proba(lg(p_te))[:, 1]
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(p_va, y_va).predict(p_te)
    return platt, iso


def analyze(npz_path, out_path):
    from qml_resolution.auc import rank_auc, paired_delong
    d = np.load(npz_path)
    y_va, y_te = d['y_va'].astype(int), d['y_te'].astype(int)
    clim_te = d['clim_te']
    b_clim = float(np.mean((clim_te - y_te) ** 2))
    summary = {'n_test': int(y_te.size), 'test_rate': float(y_te.mean()),
               'climatology': {'auc': float(roc_auc_score(y_te, clim_te)), 'brier': b_clim, 'ece': ece(y_te, clim_te)},
               'models': {}, 'pairs': {}}
    print(f"climatology: AUC {summary['climatology']['auc']:.4f} Brier {b_clim:.4f} ECE {summary['climatology']['ece']:.3f}")
    models = [k[:-6] for k in d.files if k.endswith('_probs')]
    for m in models:
        P_va, P_te = d[f'{m}_val'], d[f'{m}_probs']
        rows = []
        for pv, pt in zip(P_va, P_te):
            pl, iso = recalibrate(y_va, pv, pt)
            rows.append([roc_auc_score(y_te, pt), roc_auc_score(y_te, pl), roc_auc_score(y_te, iso),
                         np.mean((pt - y_te) ** 2), np.mean((pl - y_te) ** 2), np.mean((iso - y_te) ** 2),
                         ece(y_te, pt), ece(y_te, pl), ece(y_te, iso)])
        r = np.asarray(rows)
        names = ['auc_raw', 'auc_platt', 'auc_iso', 'brier_raw', 'brier_platt', 'brier_iso', 'ece_raw', 'ece_platt', 'ece_iso']
        res = {n: {'mean': float(r[:, i].mean()), 'sd': float(r[:, i].std(ddof=1)) if len(r) > 1 else 0.0,
                   'min': float(r[:, i].min()), 'max': float(r[:, i].max())} for i, n in enumerate(names)}
        for kind in ('raw', 'platt', 'iso'):
            bss = 1 - r[:, names.index(f'brier_{kind}')] / b_clim
            res[f'bss_{kind}'] = {'mean': float(bss.mean()), 'min': float(bss.min()), 'max': float(bss.max()),
                                  'n_seeds_above_clim': int((bss > 0).sum())}
        summary['models'][m] = res
        print(f"{m:14s} n={len(r):2d} AUC {res['auc_raw']['mean']:.4f} | Brier raw {res['brier_raw']['mean']:.4f} "
              f"Platt {res['brier_platt']['mean']:.4f} iso {res['brier_iso']['mean']:.4f} | BSS Platt "
              f"{res['bss_platt']['mean']:+.3f} [{res['bss_platt']['min']:+.3f},{res['bss_platt']['max']:+.3f}] "
              f"({res['bss_platt']['n_seeds_above_clim']} seeds > clim) | ECE raw {res['ece_raw']['mean']:.3f} "
              f"Platt {res['ece_platt']['mean']:.3f}")
    yb = y_te.astype(bool)
    for a_, b_ in [('RF_2273', 'XGB'), ('RF_2273', 'PureLSTM'), ('RF_2273', 'LSTM_QNN'), ('CTRL_LINEAR4D', 'LSTM_QNN'),
                   ('PureLSTM', 'LSTM_QNN'), ('FOURIER2', 'LSTM_QNN')]:
        if f'{a_}_probs' in d.files and f'{b_}_probs' in d.files:
            r = paired_delong(yb, d[f'{a_}_probs'], d[f'{b_}_probs'])
            summary['pairs'][f'{a_}-{b_}'] = {'delta': float(r.delta), 'se': float(r.se), 'p': float(r.p_value)}
            print(f'  {a_}-{b_}: delta {r.delta:+.4f} SE {r.se:.4f} p {r.p_value:.3f} (DeLong, seed means)')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print('wrote', out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--label', choices=['orig', 'stationmonth'], required=True)
    ap.add_argument('--models', nargs='+', default=DEFAULT_MODELS, choices=ALL_MODELS)
    ap.add_argument('--seeds', type=int, default=15)
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--out-dir', default=OUT_DIR)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--analyze', action='store_true')
    a = ap.parse_args()
    out_json = os.path.join(a.out_dir, f'results_exp28_{a.label}.json')
    out_npz = os.path.join(a.out_dir, f'results_exp28_{a.label}_probs.npz')
    if a.analyze:
        analyze(out_npz, os.path.join(ROOT, 'results', f'local_exp28_{a.label}_summary.json'))
        return
    if a.threads:
        torch.set_num_threads(a.threads)

    ds = load_dataset(a.label)
    y_va, y_te = ds.y_va.numpy().astype(int), ds.y_te.numpy().astype(int)
    assert os.environ.get('LOR_DATASET', 'v1').lower() == 'v2' or (len(ds.X_tr), len(ds.X_va), len(ds.X_te)) == (2273, 514, 528), 'split sizes differ from the paper'
    archive = os.path.join(ROOT, 'results', 'json', 'results_hybrids_trainonly_fixet0_p60_28.1_probs.npz')
    if a.label == 'orig' and os.path.exists(archive):
        assert np.array_equal(np.load(archive)['y_te'].astype(int), y_te), 'test labels differ from the archive'
    clim_va, clim_te = climatology(ds)
    print(f'label {a.label}: training positive rate {ds.y_tr.numpy().mean():.3f} | test positives {y_te.sum()} / 528 '
          f'| climatology test AUC {roc_auc_score(y_te, clim_te):.4f}')
    trainers = make_trainers(ds, epochs_scale=0.04 if a.dry_run else 1.0)
    if a.dry_run:
        for name in a.models:
            pv, pt = trainers[name](0)
            print(f'  dry-run {name:14s} val {pv.shape} test {pt.shape} test AUC {roc_auc_score(y_te, pt):.3f}')
        return

    os.makedirs(a.out_dir, exist_ok=True)
    state = {'y_va': y_va.tolist(), 'y_te': y_te.tolist(), 'models': {}}
    if os.path.exists(out_json):
        old = json.load(open(out_json, encoding='utf-8'))
        if old.get('y_te') != state['y_te']:
            raise SystemExit(f'{out_json} has different test labels')
        state['models'] = old['models']
    for n in a.models:
        state['models'].setdefault(n, {'val': [], 'probs': [], 'auc': []})

    def save():
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(state, f)
        arr = {'y_va': y_va, 'y_te': y_te, 'clim_val': clim_va, 'clim_te': clim_te}
        for k, v in state['models'].items():
            if v['probs']:
                arr[f'{k}_val'] = np.asarray(v['val'])
                arr[f'{k}_probs'] = np.asarray(v['probs'])
        np.savez_compressed(out_npz, **arr)

    for name in a.models:
        st = state['models'][name]
        for seed in range(len(st['probs']), a.seeds):
            t0 = time.time()
            pv, pt = trainers[name](seed)
            st['val'].append(np.asarray(pv, float).tolist())
            st['probs'].append(np.asarray(pt, float).tolist())
            st['auc'].append(float(roc_auc_score(y_te, pt)))
            save()
            print(f'{name:14s} seed {seed:2d} test AUC {st["auc"][-1]:.4f}  {(time.time() - t0) / 60:.1f} min', flush=True)
    print('saved', out_npz)


if __name__ == '__main__':
    main()
