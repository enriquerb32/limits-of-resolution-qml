#!/usr/bin/env python3
"""
exp32_second_panel.py - out-of-panel replication: the full protocol on a second set of AEMET stations (15 seeds)
================================================================================================================
The paper's claims rest on one four-station panel. This script reruns the protocol on another panel with exactly
the same code path (ETL -> monthly features -> train-only scalers and thresholds -> split 1938-1996 / 1997-2007 /
2008-) and the same models, so the resolution statements (noise budget, paired intervals, MDE, cluster ratio,
phase robustness, climatology AUC) can be checked out of panel.

Steps (each can be run on its own; outputs go to results/panel2/<tag>/)
  etl      reads <data-dir>/<station>/datos_{precipitacion,presion,temperatura,sol,viento}.txt (the same AEMET daily
           files and format as the core panel; code/etl.py) and writes datos.csv. Latitude and altitude of every
           station (for ET0) come from the --stations JSON and are injected into trainonly_dataset.
  train    labels: pooled (monthly sum > pooled training P60, the paper's target) and stationmonth (> station x month
           training P60; floor 0.1 mm, as in exp28). Models: RF_2273, XGB, PureLSTM, CTRL_LINEAR4D, LSTM_QNN, FOURIER2.
           Saves '<MODEL>_val' / '<MODEL>_probs' after every model (resumable) plus test meta.
  analyze  for each label: per-model AUC (seed mean, seed SD), climatology AUC, within-station AUC; for each pair:
           paired DeLong, calendar-year block and years x seeds bootstraps, MDE, cluster/i.i.d. SE ratio with a
           year-permutation null, and the 12 phase-shifted pseudo-year intervals.

Stations JSON (example; check availability and the indicativo on AEMET OpenData before downloading):
  [{"name": "Malaga",   "lat": 36.67, "alt": 7.0},
   {"name": "Almeria",  "lat": 36.85, "alt": 21.0},
   {"name": "Palma",    "lat": 39.55, "alt": 8.0},
   {"name": "Tortosa",  "lat": 40.82, "alt": 44.0}]
Requirement: daily series that start before ~1960 for the five variables (the train split ends in 1996).

Usage (from the repository root)
  python code/local_experiments/exp32_second_panel.py --stations panel2.json --data-dir data_panel2 --tag med2 --step etl
  python code/local_experiments/exp32_second_panel.py --tag med2 --step train --dry-run
  python code/local_experiments/exp32_second_panel.py --tag med2 --step train          # ~1.5-2 h per label (CPU)
  python code/local_experiments/exp32_second_panel.py --tag med2 --step analyze        # ~5 min
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import t as student_t

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(ROOT, 'code'))
sys.path.insert(2, os.path.join(ROOT, 'src'))

import trainonly_dataset as tds  # noqa: E402

MODELS = ['RF_2273', 'XGB', 'PureLSTM', 'CTRL_LINEAR4D', 'LSTM_QNN', 'FOURIER2']
PAIRS = ['RF_2273-PureLSTM', 'RF_2273-LSTM_QNN', 'XGB-LSTM_QNN', 'CTRL_LINEAR4D-LSTM_QNN',
         'PureLSTM-LSTM_QNN', 'FOURIER2-LSTM_QNN', 'RF_2273-XGB']
LABELS = ['pooled', 'stationmonth']
N_BOOT, N_PERM, SEED = 2000, 4000, 2025


def outdir(tag):
    d = os.path.join(ROOT, 'results', 'panel2', tag)
    os.makedirs(d, exist_ok=True)
    return d


def inject_stations(tag):
    cfg = json.load(open(os.path.join(outdir(tag), 'stations.json')))
    for s in cfg:
        tds.LATITUDES[s['name']] = float(s['lat'])
        tds.ALTITUDES[s['name']] = float(s['alt'])
    return cfg


# ------------------------------------------------------------------ etl
def step_etl(a):
    if os.environ.get('LOR_DATASET', 'v1').lower() == 'v2':
        from etl_v2 import cargar_datos_v2 as cargar_datos
    else:
        from etl import cargar_datos
    d = outdir(a.tag)
    shutil.copy(a.stations, os.path.join(d, 'stations.json'))
    cfg = inject_stations(a.tag)
    data_dir = os.path.abspath(a.data_dir)
    cwd = os.getcwd()
    os.chdir(d)                                   # cargar_datos writes datos_limpios_multiestacion.csv to the cwd
    try:
        df = cargar_datos(data_dir, [s['name'] for s in cfg])
        if os.environ.get('LOR_DATASET', 'v1').lower() == 'v2':
            df.to_csv('datos_limpios_multiestacion.csv')
    finally:
        os.chdir(cwd)
    os.replace(os.path.join(d, 'datos_limpios_multiestacion.csv'), os.path.join(d, 'datos.csv'))
    cov = df.groupby('ciudad').apply(lambda g: dict(first=str(g.index.min())[:10], last=str(g.index.max())[:10],
                                                     days=int(len(g)))).to_dict()
    json.dump(cov, open(os.path.join(d, 'coverage.json'), 'w'), indent=2)
    print(json.dumps(cov, indent=2))
    for k, v in cov.items():
        if v['first'] > '1960':
            print(f'WARNING: {k} starts in {v["first"][:4]}; the training split (<= 1996) will be short')


# ------------------------------------------------------------------ data
def load_dataset(tag, label):
    inject_stations(tag)
    df = pd.read_csv(os.path.join(outdir(tag), 'datos.csv'), index_col='fecha', parse_dates=['fecha'])
    agg = tds.construir_features(df, anomalia_fn=tds._anomalia_train_only, fix_et0_verano=True)
    if label == 'pooled':
        return tds.preparar_dataset(agg)                     # pooled training P60 of THIS panel
    train = agg.index.year <= tds.ANIO_CORTE_TRAIN
    thr = (agg.loc[train].assign(_m=agg.index[train].month)
           .groupby(['ciudad', '_m'])['precip_label' if 'label_ok' in agg.columns else 'precip'].quantile(0.60).clip(lower=0.1).to_dict())
    t = np.array([thr[k] for k in zip(agg['ciudad'].values, agg.index.month)])
    agg = agg.copy()
    agg['precip_ratio'] = (agg['precip_label_f'] if 'label_ok' in agg.columns else agg['precip']).values / t
    return tds.preparar_dataset(agg, umbral_override=1.0, label_col='precip_ratio')


# ------------------------------------------------------------------ train
def step_train(a):
    from exp28_labels_recalibration import make_trainers, climatology
    d = outdir(a.tag)
    for label in a.labels:
        ds = load_dataset(a.tag, label)
        print(f'[{label}] train/val/test = {len(ds.X_tr)}/{len(ds.X_va)}/{len(ds.X_te)}, '
              f'positives test {float(ds.y_te.mean()):.3f}', flush=True)
        path = os.path.join(d, f'probs_{label}.npz')
        store = dict(np.load(path)) if os.path.exists(path) else {}
        c_va, c_te = climatology(ds)
        store.update(y_va=ds.y_va.numpy().astype(int), y_te=ds.y_te.numpy().astype(int),
                     clim_val=c_va, clim_te=c_te)
        pd.DataFrame({'ciudad': ds.meta_te['ciudad'], 'fecha': pd.to_datetime(ds.meta_te['fecha']),
                      'y_te': store['y_te']}).to_csv(os.path.join(d, f'test_meta_{label}.csv'), index=False)
        trainers = make_trainers(ds, epochs_scale=0.08 if a.dry_run else 1.0)
        seeds = 1 if a.dry_run else a.seeds
        for name in a.models:
            if f'{name}_probs' in store and store[f'{name}_probs'].shape[0] >= seeds and not a.dry_run:
                print(f'  {name}: already done'); continue
            t0, V, T = time.time(), [], []
            for s in range(seeds):
                pv, pt = trainers[name](s)
                V.append(pv); T.append(pt)
            store[f'{name}_val'], store[f'{name}_probs'] = np.asarray(V), np.asarray(T)
            auc = np.mean([roc_auc_score(store['y_te'], p) for p in T])
            print(f'  {name}: AUC {auc:.4f} ({seeds} seeds, {(time.time() - t0) / 60:.1f} min)', flush=True)
            if not a.dry_run:
                np.savez_compressed(path, **store)
    print('dry-run OK' if a.dry_run else 'training done')


def _import_quiet(name):
    """Import a repository module with an empty command line, so that a module that parses arguments at import
    time (instead of under `if __name__ == '__main__'`) cannot read this script's arguments."""
    import importlib
    saved = sys.argv[:]
    sys.argv = [saved[0]]
    try:
        return importlib.import_module(name)
    finally:
        sys.argv = saved


def influence(y, pa, pb):
    """Paired DeLong influence values of the seed-averaged structural components (as in exp25)."""
    dc = _import_quiet('qml_resolution').delong_components
    ca, cb = dc(y, pa).seed_mean(), dc(y, pb).seed_mean()
    delta = float(ca.auc[0] - cb.auc[0])
    psi = np.zeros(y.size)
    psi[y] = (ca.v10[0] - cb.v10[0] - delta) / y.sum()
    psi[~y] = (ca.v01[0] - cb.v01[0] - delta) / (~y).sum()
    return delta, psi


def cluster_var(psi, groups, levels):
    k = len(levels)
    return k / (k - 1) * sum(psi[groups == u].sum() ** 2 for u in levels)


# ------------------------------------------------------------------ analyze
def step_analyze(a):
    compare = _import_quiet('exp10_paired_ci').compare
    d = outdir(a.tag)
    out = {}
    for label in a.labels:
        z = np.load(os.path.join(d, f'probs_{label}.npz'))
        meta = pd.read_csv(os.path.join(d, f'test_meta_{label}.csv'), parse_dates=['fecha'])
        y = z['y_te'].astype(int)
        yb = y.astype(bool)
        years = meta['fecha'].dt.year.values
        ym = years * 12 + meta['fecha'].dt.month.values
        st = meta['ciudad'].values
        names = [m for m in MODELS if f'{m}_probs' in z.files]
        res = {'n_test': int(y.size), 'prevalence': float(y.mean()), 'K_years': int(np.unique(years).size),
               'climatology_auc': float(roc_auc_score(y, z['clim_te'])), 'models': {}, 'pairs': {}}
        for m in names:
            P = z[f'{m}_probs']
            aucs = np.array([roc_auc_score(y, p) for p in P])
            within = {s: float(roc_auc_score(y[st == s], P[:, st == s].mean(0)))
                      for s in np.unique(st) if 0 < y[st == s].sum() < (st == s).sum()}
            res['models'][m] = dict(auc_seed_mean=float(aucs.mean()), auc_seed_sd=float(aucs.std(ddof=1)) if len(P) > 1 else None,
                                    auc_of_mean_prob=float(roc_auc_score(y, P.mean(0))),
                                    within_station_auc=within, within_station_mean=float(np.mean(list(within.values()))))
        rng = np.random.default_rng(SEED)
        levels = np.unique(years)
        for pair in a.pairs:
            pa, pb = pair.split('-', 1)
            if pa not in names or pb not in names:
                continue
            r = compare(y, z[f'{pa}_probs'], z[f'{pb}_probs'], years, a.boot, rng)
            delta, psi = influence(yb, z[f'{pa}_probs'], z[f'{pb}_probs'])
            se_iid = float(np.sqrt((psi ** 2).sum()))
            se_cl = float(np.sqrt(cluster_var(psi, years, levels)))
            null = np.array([np.sqrt(cluster_var(psi, rng.permutation(years), levels)) / se_iid for _ in range(a.perm)])
            phases = []
            for start in range(1, 13):
                g = (ym - start) // 12
                _, inv = np.unique(g, return_inverse=True)
                s_ = np.bincount(inv, weights=psi); k = s_.size
                se = float(np.sqrt(k / (k - 1) * (s_ ** 2).sum())); q = student_t.ppf(0.975, k - 1)
                phases.append(dict(start=start, se=se, excludes_zero=bool(abs(delta) > q * se)))
            r.update(se_iid=se_iid, se_cluster=se_cl, ratio=se_cl / se_iid,
                     perm_ratio_q025=float(np.quantile(null, .025)), perm_ratio_q975=float(np.quantile(null, .975)),
                     phases_excluding_zero=int(sum(p['excludes_zero'] for p in phases)), phases=phases)
            res['pairs'][pair] = r
            print(f'[{label}] {pair:24s} d={r["delta"]:+.4f} DeLong [{r["ci_delong"][0]:+.3f},{r["ci_delong"][1]:+.3f}] '
                  f'block [{r["ci_block"][0]:+.3f},{r["ci_block"][1]:+.3f}] two-way [{r["ci_two_way"][0]:+.3f},'
                  f'{r["ci_two_way"][1]:+.3f}] MDE {r["mde_delong"]:.3f} ratio {r["ratio"]:.2f} '
                  f'phases {r["phases_excluding_zero"]}/12', flush=True)
        for m, v in res['models'].items():
            print(f'[{label}] {m:14s} AUC {v["auc_seed_mean"]:.4f} (seed SD {v["auc_seed_sd"] or 0:.4f}) '
                  f'within-station {v["within_station_mean"]:.4f}')
        print(f'[{label}] climatology AUC {res["climatology_auc"]:.4f}, n={res["n_test"]}, K={res["K_years"]}')
        out[label] = res
    p = os.path.join(ROOT, 'results', f'local_exp32_panel2_{a.tag}.json')
    json.dump(out, open(p, 'w'), indent=2)
    print('wrote', p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--step', choices=['etl', 'train', 'analyze'], required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--stations')
    ap.add_argument('--data-dir')
    ap.add_argument('--labels', nargs='+', default=LABELS, choices=LABELS)
    ap.add_argument('--models', nargs='+', default=MODELS, choices=MODELS)
    ap.add_argument('--pairs', nargs='+', default=PAIRS)
    ap.add_argument('--seeds', type=int, default=15)
    ap.add_argument('--boot', type=int, default=N_BOOT)
    ap.add_argument('--perm', type=int, default=N_PERM)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    if a.step == 'etl' and not (a.stations and a.data_dir):
        ap.error('--step etl needs --stations and --data-dir')
    {'etl': step_etl, 'train': step_train, 'analyze': step_analyze}[a.step](a)


if __name__ == '__main__':
    main()
