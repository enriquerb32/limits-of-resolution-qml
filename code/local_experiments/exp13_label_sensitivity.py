#!/usr/bin/env python3
"""
exp13_label_sensitivity.py - Are the model comparisons sensitive to incomplete months and to the label definition?

Fixed predictors (the saved test probabilities, NO retraining) are re-evaluated on
  S0  all test rows, project labels                         (reproduces the main analysis)
  S1  test rows whose TARGET month is complete               (composition change, original labels)
  S2  test rows whose target month has >= 25 valid days
  S3  drop rows whose label is ambiguous (differs between plain and coverage-corrected sum at the project threshold)
  S4  all rows, labels from the coverage-corrected sum, same threshold      (HYPOTHETICAL scenario)
  S5  all rows, labels from the coverage-corrected sum, recomputed threshold (HYPOTHETICAL scenario)
For every scenario: n, prevalence, mean AUC and Brier per model, and for each requested pair the seed-mean difference with the
paired DeLong SE (seed-averaged components), year-block bootstrap CI.  S4/S5 evaluate predictors trained on the project labels
against alternative labels; they do not evaluate a pipeline retrained under those labels.  Antecedent (feature-side) months
with partial coverage are NOT handled here.

INPUTS
  --probs  results_rerun_train_only_canonical_probs.npz  (+ --prefix train_only)   and/or  --extra results_qml_core_trainonly_probs.npz
  --meta   test_meta.csv          from export_test_meta.py  (ciudad, fecha, y_te; same row order as y_te)
  --rows   month_rows.csv         from exp12 --save_rows
USAGE
  python exp13_label_sensitivity.py --probs results_rerun_train_only_canonical_probs.npz --prefix train_only \
      --meta test_meta.csv --rows month_rows.csv --pairs RF-XGB RF-LSTM XGB-LSTM --B 2000 --out results_exp13_label_sensitivity.json
"""
import argparse, json
import numpy as np, pandas as pd
from exp10_paired_ci import Model, paired_delong, boot


def load_models(a):
    models, y = {}, None
    if a.probs:
        z = np.load(a.probs); y = z['y_te'].astype(int)
        for k in z.files:
            if k.startswith(a.prefix + '_') and k.endswith('_probs'): models[k[len(a.prefix) + 1:-6]] = z[k]
    if a.extra:
        z = np.load(a.extra); ye = z['y_te'].astype(int)
        if y is not None: assert (ye == y).all(), 'extra y_te differs'
        y = ye
        for k in z.files:
            if k.endswith('_probs'): models[k[:-6]] = z[k]
    assert y is not None and models
    return models, y


def link(meta, rows):
    meta = meta.copy(); meta['fecha'] = pd.to_datetime(meta['fecha']); rows = rows.copy(); rows['fecha'] = pd.to_datetime(rows['fecha'])
    j = meta.merge(rows, on=['ciudad', 'fecha'], how='left', validate='one_to_one')
    assert len(j) == len(meta) and j['dias'].notna().all(), 'some test rows not found in month table'
    assert (j['part'] == 'test').all(), 'linked rows are not all in the test partition'
    return j


def split_pair(pr, M):
    for i in range(1, len(pr)):
        if pr[i] == '-' and pr[:i] in M and pr[i + 1:] in M: return pr[:i], pr[i + 1:]
    raise SystemExit(f'cannot parse pair {pr}')


def main():
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument('--probs'); ap.add_argument('--prefix', default='train_only'); ap.add_argument('--extra')
    ap.add_argument('--meta', default='results/test_meta.csv')
    ap.add_argument('--rows', default='month_rows.csv')
    ap.add_argument('--pairs', nargs='*', default=None); ap.add_argument('--B', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--out', default='results_exp13_label_sensitivity.json')
    a = ap.parse_args(); rng = np.random.default_rng(a.seed)

    if not a.pairs:
        if a.extra:
            a.pairs = ['RF-XGB', 'RF-LSTM', 'XGB-LSTM', 'RF_2273-LSTM_QNN', 'CTRL_LINEAR4D-LSTM_QNN', 'PureLSTM-LSTM_QNN']
        else:
            a.pairs = ['RF-XGB', 'RF-LSTM', 'XGB-LSTM']

    models, y0 = load_models(a)

    meta_path = a.meta
    if not os.path.exists(meta_path) and os.path.exists(os.path.join('results', 'test_meta.csv')):
        meta_path = os.path.join('results', 'test_meta.csv')
    if not os.path.exists(meta_path):
        raise SystemExit(f"test_meta file '{meta_path}' not found; generate it or pass --meta results/test_meta.csv")

    if not os.path.exists(a.rows):
        raise SystemExit(f"month rows file '{a.rows}' not found. Note: month_rows.csv is produced from raw daily AEMET records (exp12), which are not redistributed due to AEMET licensing.")

    meta = pd.read_csv(meta_path); rows = pd.read_csv(a.rows)
    assert len(meta) == len(y0), (len(meta), len(y0))
    if 'y_te' in meta: assert (meta['y_te'].values == y0).all(), 'meta y_te differs from probs y_te: row order or dataset differs'
    j = link(meta, rows); years = pd.to_datetime(j['fecha']).dt.year.values
    agree = float((j['lab_project'].values == y0).mean())
    print(f'row link OK: {len(j)} rows | agreement project-label (exp12) vs y_te: {agree:.4f} | partial target months {int(j.partial.sum())}')
    fragile = (j['lab_project'] != j['lab_corr_same_thr']).values
    scen = {
        'S0 all rows, project labels': (np.ones(len(j), bool), y0),
        'S1 complete target months only': ((j['partial'] == 0).values, y0),
        'S2 target month >=25 valid days': ((j['dias'] >= 25).values, y0),
        'S3 drop ambiguous labels': (~fragile, y0),
        'S4 corrected-sum labels, same threshold (hypothetical)': (np.ones(len(j), bool), j['lab_corr_same_thr'].values),
        'S5 corrected-sum labels, new threshold (hypothetical)': (np.ones(len(j), bool), j['lab_corr_new_thr'].values),
    }
    if 'lab_raw_same_thr' in j:   # from exp14: labels from the unjoined (precipitation-only) monthly sum
        scen['S6 unjoined-sum labels, same threshold (hypothetical)'] = (np.ones(len(j), bool), j['lab_raw_same_thr'].values)
        scen['S7 unjoined-sum labels, new threshold (hypothetical)'] = (np.ones(len(j), bool), j['lab_raw_new_thr'].values)
    out = {'agreement_exp12_vs_yte': agree, 'n_rows': len(j), 'scenarios': {}}
    for name, (keep, lab) in scen.items():
        idx = np.where(keep)[0]; yy = lab[idx].astype(int); yr = years[idx]
        r = {'n': int(len(idx)), 'prevalence': float(yy.mean()), 'label_flips_vs_y_te': int((lab[idx] != y0[idx]).sum()), 'models': {}, 'pairs': {}}
        M = {k: Model(v[:, idx], yy) for k, v in models.items()}
        for k, m in M.items():
            r['models'][k] = dict(auc=m.mean_auc, auc_sd_seeds=float(m.auc.std(ddof=1)) if m.S > 1 else None,
                                  brier=float(np.mean([np.mean((p - yy) ** 2) for p in models[k][:, idx]])))
        for pr in a.pairs:
            na, nb = split_pair(pr, M); A, Bm = M[na], M[nb]
            d, se, zz, p = paired_delong(A, Bm, yy)
            bs = boot(A, Bm, yy, yr, a.B, rng, False); ci = np.percentile(bs, [2.5, 97.5])
            r['pairs'][pr] = dict(delta=d, se_delong=se, z=zz, p=p, ci_block=ci.tolist(), se_block=float(bs.std(ddof=1)))
        out['scenarios'][name] = r
        print(f"\n{name} | n={r['n']} prev={r['prevalence']:.3f} flips={r['label_flips_vs_y_te']}")
        print('  AUC:   ' + '  '.join(f"{k} {v['auc']:.4f}" for k, v in r['models'].items()))
        for pr, v in r['pairs'].items():
            print(f"  {pr:<14} d={v['delta']:+.4f} SE={v['se_delong']:.4f} z={v['z']:+.2f} p={v['p']:.3f} block CI [{v['ci_block'][0]:+.4f},{v['ci_block'][1]:+.4f}]")
    json.dump(out, open(a.out, 'w'), indent=2); print('\nsaved', a.out)
    print('\nLaTeX rows (scenario & n & prevalence & AUC models... & first pair delta [block CI]):')
    for name, r in out['scenarios'].items():
        aucs = ' & '.join(f"{v['auc']:.3f}" for v in r['models'].values())
        first = next(iter(r['pairs'].values()))
        print(f"{name.split(' ')[0]} & {r['n']} & {r['prevalence']:.3f} & {aucs} & {first['delta']:+.3f} [{first['ci_block'][0]:+.3f}, {first['ci_block'][1]:+.3f}] \\\\")


if __name__ == '__main__':
    main()