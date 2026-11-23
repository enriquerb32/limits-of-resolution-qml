#!/usr/bin/env python3
"""
exp65_external_v2.py - external benchmark v2 (semi-synthetic), as recorded in audit/prediction_external_v2.txt
==============================================================================================================
Record: commit b100bea, deposited at doi:10.5281/zenodo.24459864 (audit/deposits.txt). This script implements it.

  --step truth   steps 1, 1b, 2, 3 and 5: generator, calibration of s_u, models, TRUE differences and a priori MDEs.
                 Writes audit/truth_external_v2.json, results/local_exp65_external_v2.json and
                 results/json/results_exp65_external_v2_probs.npz. No small test is scored.
  --step score   step 4 (small tests). Refuses to run unless audit/truth_external_v2.json is committed in git AND the
                 DOI of its independent deposit is listed in audit/deposits.txt.

No resume: every step refuses to start if one of its outputs already exists (delete them by hand to re-run).

Interpretations of the record (where it does not fix a detail; written to the output as well):
  * Step 5, 'inflated by the design effect 1 + (m - 1) rho': the design effect multiplies the variance, so the MDE
    is multiplied by sqrt(1 + (m - 1) rho); rho is the intraclass correlation of the influence values of the
    contrast (one-way ANOVA estimator for unequal cluster sizes), not of the labels; m is the median size of the 39
    blocks; the prevalence is that of the reference population.
  * C0: the second RF fit uses an independent label draw of the training and validation cells with seed 2028.
  * Calibration seeds (disjoint from steps 2-4): training labels 9000 + k, test-block draws 9500 + k for grid value k.
  * 'Settings of v1': models fitted on the training rows, XGBoost early-stopped on the validation rows, as in exp53.

  python code/local_experiments/exp65_external_v2.py --step truth          (~20-40 min)
  python code/local_experiments/exp65_external_v2.py --step score          (writes results/json/results_exp65_external_v2_score.json)
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'src'))
from exp53_external_benchmark import data  # noqa: E402  (same features, cells and split as v1)
from qml_resolution import cluster_robust_delta, delong_components, paired_delong  # noqa: E402
from qml_resolution.power import paired_mde_hanley_mcneil  # noqa: E402

TRUTH = os.path.join(ROOT, 'audit', 'truth_external_v2.json')
OUT = os.path.join(ROOT, 'results', 'local_exp65_external_v2.json')
NPZ = os.path.join(ROOT, 'results', 'json', 'results_exp65_external_v2_probs.npz')
SEEDS = range(5)
GRID = [0.25 * k for k in range(1, 9)]            # 0.25, 0.50, ..., 2.00
# REF_CELLS: 20,000 as recorded gave a Monte Carlo SE above 0.001 in S2 (0.0013-0.0014 for C1, C3, C0); the record
# prescribes enlarging the reference sample in that case, so it is 80,000 (still 20 batches, of 4,000 cells).
CAL_DRAWS, REF_CELLS, REF_BATCHES, SMALL_K = 200, 80_000, 20, 19
C0_NOTE = ('The record states that the true difference of C0 is zero by construction. It is zero only in expectation '
           'over training label draws; conditional on the two fitted RF models, which is the estimand of CR2 and of '
           'the reference, it is not zero (see truth). C0 is therefore not a null, and P4 cannot be evaluated as '
           'recorded; its coverage is reported like that of C1-C3. Found when the truth was computed, before any '
           'small test was scored.')
CONTRASTS = {'C1': ('RF', 'LR'), 'C2': ('XGB', 'RF'), 'C3': ('MLP', 'RF'), 'C0': ('RF', 'RF_B')}
DECLARED = ('C1', 'C2', 'C3')


def refuse_if_exists(paths):
    hit = [p for p in paths if os.path.exists(p)]
    if hit:
        raise SystemExit('outputs already exist (no resume by design): ' + ', '.join(hit))


def generator(X, y_real, split):
    """Step 1: f = logit of XGBoost (depth 4, lr 0.05, 300 rounds, seed 0) on the real target of the training rows."""
    from xgboost import XGBClassifier
    tr = split == 'train'
    g = XGBClassifier(tree_method='hist', max_depth=4, learning_rate=0.05, n_estimators=300, random_state=0, n_jobs=-1)
    g.fit(X[tr], y_real[tr])
    return g.predict(X, output_margin=True).astype(float)


def draw_labels(f, cell_id, s_u, rng):
    """y = 1[f (1 + v_c) + u_c + logistic noise > 0], s_v = s_u / 2; cell_id indexes the synthetic cells."""
    n_cells = int(cell_id.max()) + 1
    u = rng.normal(0.0, s_u, n_cells) if s_u > 0 else np.zeros(n_cells)
    v = rng.normal(0.0, s_u / 2, n_cells) if s_u > 0 else np.zeros(n_cells)
    eta = f * (1.0 + v[cell_id]) + u[cell_id]
    return (rng.random(f.size) < expit(eta)).astype(int)


def fit_models(X, y, split, X_eval, names=('LR', 'RF', 'XGB', 'MLP'), seeds=SEEDS):
    """The four models of v1 (seeds 0-4) on synthetic labels; returns {name: (5, len(X_eval)) probabilities}."""
    from xgboost import XGBClassifier
    tr, va = split == 'train', split == 'val'
    sc = StandardScaler().fit(X[tr])
    Z, Ze = sc.transform(X), sc.transform(X_eval)
    P = {m: [] for m in names}
    if 'LR' in names:
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(Z[tr], y[tr]).predict_proba(Ze)[:, 1]
    for s in seeds:
        if 'LR' in names:
            P['LR'].append(lr)
        if 'RF' in names:
            P['RF'].append(RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=s, n_jobs=-1)
                           .fit(X[tr], y[tr]).predict_proba(X_eval)[:, 1])
        if 'XGB' in names:
            xg = XGBClassifier(tree_method='hist', max_depth=6, learning_rate=0.05, n_estimators=2000,
                               early_stopping_rounds=50, eval_metric='auc', random_state=s, n_jobs=-1)
            P['XGB'].append(xg.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False).predict_proba(X_eval)[:, 1])
        if 'MLP' in names:
            P['MLP'].append(MLPClassifier((64, 32), early_stopping=True, max_iter=500, random_state=s)
                            .fit(Z[tr], y[tr]).predict_proba(Ze)[:, 1])
    return {k: np.array(v) for k, v in P.items()}


def train_cells_ids(cell, mask):
    _, idx = np.unique(cell[mask], return_inverse=True)
    return idx


def calibrate(X, f, cell, split):
    """Step 1b: s* such that the median SE_CR2 / SE_DeLong over C1-C3 on the 182 test blocks is closest to 5.5."""
    trva = np.isin(split, ['train', 'val'])
    te = split == 'test'
    te_ids = train_cells_ids(cell, te)
    rows = []
    for k, s in enumerate(GRID, start=1):
        y = np.zeros(len(X), dtype=int)
        y[trva] = draw_labels(f[trva], train_cells_ids(cell, trva), s, np.random.default_rng(9000 + k))
        P = fit_models(X, y, split, X[te], seeds=(0,))                 # seed 0, as recorded
        rng = np.random.default_rng(9500 + k)
        ratios = {c: [] for c in DECLARED}
        for _ in range(CAL_DRAWS):
            yt = draw_labels(f[te], te_ids, s, rng)
            for c in DECLARED:
                a, b = CONTRASTS[c]
                cr = cluster_robust_delta(yt, P[a], P[b], te_ids)
                ratios[c].append(cr.se_cr2 / paired_delong(yt, P[a], P[b]).se)
        med = {c: float(np.median(v)) for c, v in ratios.items()}
        rows.append(dict(s_u=s, median_ratio=med, mean_over_contrasts=float(np.mean(list(med.values())))))
        print(f'calibration s_u={s:.2f} ratio ' + ' '.join(f'{c} {v:.2f}' for c, v in med.items()), flush=True)
    reach = [r for r in rows if r['mean_over_contrasts'] >= 4.4]
    if reach:
        s_star = min(rows, key=lambda r: abs(r['mean_over_contrasts'] - 5.5))['s_u']
        note = 'closest to 5.5'
    else:
        s_star, note = GRID[-1], 'no grid value reaches 4.4: largest used (as recorded)'
    return float(s_star), note, rows


def auc_mean(y, M):
    return float(np.mean([roc_auc_score(y, p) for p in M]))


def icc_anova(psi, groups):
    """One-way ANOVA intraclass correlation for unequal group sizes."""
    _, g = np.unique(groups, return_inverse=True)
    G, N = g.max() + 1, psi.size
    n = np.bincount(g, minlength=G).astype(float)
    means = np.bincount(g, weights=psi, minlength=G) / n
    grand = psi.mean()
    msb = float((n * (means - grand) ** 2).sum() / (G - 1))
    msw = float(((psi - means[g]) ** 2).sum() / (N - G))
    n0 = (N - (n ** 2).sum() / N) / (G - 1)
    return (msb - msw) / (msb + (n0 - 1) * msw)


def auc_correlation(y, A, B):
    """Correlation of the two seed-mean AUC estimates from the DeLong covariance."""
    ca, cb = delong_components(y, A).seed_mean(), delong_components(y, B).seed_mean()
    m, n = ca.v10.shape[1], ca.v01.shape[1]
    S10 = np.cov(np.vstack([ca.v10[0], cb.v10[0]]))
    S01 = np.cov(np.vstack([ca.v01[0], cb.v01[0]]))
    S = S10 / m + S01 / n
    return float(S[0, 1] / np.sqrt(S[0, 0] * S[1, 1]))


def step_truth():
    refuse_if_exists([TRUTH, OUT, NPZ])
    t0 = time.time()
    X, y_real, cell, split, thr = data()
    f = generator(X, y_real, split)
    y_real = None                                                  # no real outcome is used after step 1
    te = split == 'test'
    ct = cell[te]
    sizes = {c: int((ct == c).sum()) for c in np.unique(ct)}
    blocks = np.array(sorted(c for c, n in sizes.items() if 10 <= n <= 60))
    assert len(blocks) == 39, len(blocks)
    s_star, s_note, cal = calibrate(X, f, cell, split)
    scen = {'S0': 0.0, 'S1': s_star / 2, 'S2': s_star}
    trva = np.isin(split, ['train', 'val'])
    va = split == 'val'
    block_rows = [np.where(cell == b)[0] for b in blocks]
    block_n = np.array([len(r) for r in block_rows])
    blk_idx = np.concatenate(block_rows)
    res = dict(record='audit/prediction_external_v2.txt (commit b100bea; doi:10.5281/zenodo.24459864)',
               interpretations=__doc__.split('Interpretations of the record')[1].split('python code')[0].strip(),
               threshold_real=thr, calibration=dict(grid=cal, s_star=s_star, note=s_note), scenarios={},
               population=dict(blocks=int(len(blocks)), block_rows=block_n.tolist(),
                               median_rows_per_cell=float(np.median(block_n))))
    probs = {}
    rng_sizes = np.random.default_rng(3002)                        # small-test SIZES only (no labels): for step 5
    n_small = np.median([block_n[rng_sizes.integers(0, 39, SMALL_K)].sum() for _ in range(2000)])
    res['population']['median_small_test_rows'] = float(n_small)
    for name, s in scen.items():
        ts = time.time()
        ids_trva = train_cells_ids(cell, trva)
        y = np.zeros(len(X), dtype=int)
        y[trva] = draw_labels(f[trva], ids_trva, s, np.random.default_rng(2027))
        yB = np.zeros(len(X), dtype=int)
        yB[trva] = draw_labels(f[trva], ids_trva, s, np.random.default_rng(2028))
        X_eval = np.vstack([X[va], X[blk_idx]])
        P = fit_models(X, y, split, X_eval)
        P['RF_B'] = fit_models(X, yB, split, X_eval, names=('RF',))['RF']
        nva = int(va.sum())
        Pva = {k: v[:, :nva] for k, v in P.items()}
        Pbl = {k: v[:, nva:] for k, v in P.items()}                # rows of the 39 blocks, in blk_idx order
        for k, v in Pbl.items():
            probs[f'{name}__{k}'] = v
        # step 3: reference, 20,000 synthetic cells in 20 batches, same population as the small tests
        rng = np.random.default_rng(3001)
        starts = np.concatenate([[0], np.cumsum(block_n)[:-1]])
        per_batch = REF_CELLS // REF_BATCHES
        batch_d = {c: [] for c in CONTRASTS}
        ys, picks_all = [], []
        for _ in range(REF_BATCHES):
            pick = rng.integers(0, 39, per_batch)
            rows = np.concatenate([np.arange(starts[b], starts[b] + block_n[b]) for b in pick])
            cid = np.repeat(np.arange(per_batch), block_n[pick])
            yb = draw_labels(f[blk_idx][rows], cid, s, rng)
            ys.append(yb); picks_all.append(rows)
            for c, (a, b) in CONTRASTS.items():
                batch_d[c].append(auc_mean(yb, Pbl[a][:, rows]) - auc_mean(yb, Pbl[b][:, rows]))
        y_ref, r_ref = np.concatenate(ys), np.concatenate(picks_all)
        truth = {}
        for c, (a, b) in CONTRASTS.items():
            d = auc_mean(y_ref, Pbl[a][:, r_ref]) - auc_mean(y_ref, Pbl[b][:, r_ref])
            mc = float(np.std(batch_d[c], ddof=1) / np.sqrt(REF_BATCHES))
            truth[c] = dict(contrast=f'{a} - {b}', true_difference=float(d), mc_se=mc, mc_se_ok=bool(mc < 0.001),
                            within_margin=bool(abs(d) < 0.01))
        prevalence = float(y_ref.mean())
        # step 5: a priori MDE from the validation cells
        yva = y[va]
        va_ids = cell[va]
        mde = {}
        for c in DECLARED:
            a, b = CONTRASTS[c]
            cr = cluster_robust_delta(yva, Pva[a], Pva[b], va_ids)
            ca, cb = delong_components(yva, Pva[a]).seed_mean(), delong_components(yva, Pva[b]).seed_mean()
            lab = yva.astype(bool)
            psi = np.empty(yva.size)
            psi[lab] = (ca.v10[0] - cb.v10[0] - cr.delta) / lab.sum()
            psi[~lab] = (ca.v01[0] - cb.v01[0] - cr.delta) / (~lab).sum()
            rho_icc = icc_anova(psi, va_ids)
            r_auc = auc_correlation(yva, Pva[a], Pva[b])
            auc_ab = (auc_mean(yva, Pva[a]) + auc_mean(yva, Pva[b])) / 2
            base = paired_mde_hanley_mcneil(n_small, auc=auc_ab, prevalence=prevalence, rho=r_auc)
            m = float(np.median(block_n))
            deff = 1 + (m - 1) * rho_icc
            mde[c] = dict(mde_iid=float(base), icc_influence=float(rho_icc), auc_correlation=r_auc, auc_mean=auc_ab,
                          design_effect=float(deff), mde_a_priori=float(base * np.sqrt(max(deff, 0.0))))
        res['scenarios'][name] = dict(s_u=s, s_v=s / 2, prevalence_reference=prevalence,
                                      reference_rows=int(y_ref.size), reference_cells=REF_CELLS, truth=truth,
                                      auc_reference={k: auc_mean(y_ref, v[:, r_ref]) for k, v in Pbl.items()},
                                      a_priori_mde=mde)
        print(f'{name} s_u={s:.3f} prevalence {prevalence:.3f} | ' +
              ' '.join(f"{c} {v['true_difference']:+.4f} (MC {v['mc_se']:.4f})" for c, v in truth.items()) +
              f' | {(time.time() - ts) / 60:.1f} min', flush=True)
    res['p1_true_c1_above_margin'] = {k: v['truth']['C1']['true_difference'] > 0.01 for k, v in res['scenarios'].items()}
    res['minutes'] = (time.time() - t0) / 60
    os.makedirs(os.path.dirname(NPZ), exist_ok=True)
    np.savez_compressed(NPZ, blocks=blocks, block_rows=block_n, f_blocks=f[blk_idx], **probs)
    json.dump(res, open(OUT, 'w'), indent=2)
    keep = ('record', 'interpretations', 'calibration', 'population')
    truth_doc = {k: res[k] for k in keep}
    truth_doc['scenarios'] = {k: dict(s_u=v['s_u'], s_v=v['s_v'], prevalence_reference=v['prevalence_reference'],
                                      truth=v['truth'], a_priori_mde=v['a_priori_mde'])
                              for k, v in res['scenarios'].items()}
    import hashlib
    truth_doc['predictions_npz_sha256'] = hashlib.sha256(open(NPZ, 'rb').read()).hexdigest()
    truth_doc['note'] = 'Computed before any small test of step 4 was scored.'
    truth_doc['deviations'] = dict(
        reference_size=('Reference enlarged from 20,000 to 80,000 synthetic cells (20 batches of 4,000), as the '
                        'record prescribes when the Monte Carlo SE exceeds 0.001 (it did in S2 with 20,000).'),
        c0_not_null=C0_NOTE,
        prevalence=('The record fixes no prevalence. The label rule implies about 0.5 on the training rows (f was fitted '
                    'to the above-median target of v1); the population of the 39 test blocks has the prevalence given '
                    'per scenario.'))
    json.dump(truth_doc, open(TRUTH, 'w'), indent=2)
    print('wrote', TRUTH, OUT, NPZ, f"({res['minutes']:.1f} min)")


SCORE = os.path.join(ROOT, 'results', 'json', 'results_exp65_external_v2_score.json')
SCORE_CLARIFICATIONS = dict(
    c0=('C0 is reported as a contrast like C1-C3 (coverage of its true difference, resolutions, wrong signs); P4 is '
        'not evaluable as recorded, because the two RF fits use different training label draws (seeds 2027 and '
        '2028), so conditional on the fitted models their true difference is not zero (truth, deviation '
        'c0_not_null). Its CR2 and DeLong rejection rates are reported descriptively only.'),
    s_star=('s* = 1.25: the mean ratio over C1-C3 was 6.26 at s_u = 1.25 and 4.73 at 1.00, at distances 0.76 and 0.77 '
            'from 5.5, equal within Monte Carlo error (the calibration grid is non-monotonic by up to 1.0 between '
            'neighbouring values; one training draw, 200 test draws per value). 1.25 was taken by the recorded '
            'rule (closest) and gives a mean ratio inside the v1 range 4.4-6.6.'),
    prevalence=('The record fixes no prevalence; the population of the 39 blocks has the prevalence of the reference '
                '(truth), 0.37-0.39.'),
    small_tests=('Step 4 draws, per scenario, from one generator with seed 3002 (restarted for each scenario): 19 '
                 'block indices with replacement, then cell effects and labels of those cells. Each appearance of '
                 'a block is its own cluster. Draws with a single class would be skipped and counted (none '
                 'expected).'),
    decisions=('Resolved: 95 % interval excludes 0. Wrong sign: resolved with the sign opposite to the true '
               'difference. Coverage: the interval contains the true difference of the truth file. CR2 without the '
               'floor rule (the record names CR2 only). Empirical MDE: 2.80 x SD of the 2,000 small-test estimates. '
               'P2, P3 and P6 are evaluated on the declared contrasts C1-C3; P5 on those of C1-C3 within the margin '
               '(wrong-sign part: all of C1-C3).'),
)


def check_deposited():
    tracked = subprocess.run(['git', 'ls-files', '--error-unmatch', TRUTH], cwd=ROOT, capture_output=True)
    if tracked.returncode:
        raise SystemExit('audit/truth_external_v2.json must be committed (and deposited) before scoring')
    dep = open(os.path.join(ROOT, 'audit', 'deposits.txt'), encoding='utf-8').read()
    block = dep.split('truth_external_v2.json', 1)[1] if 'truth_external_v2.json' in dep else ''
    if 'Zenodo DOI' not in block.split('\n\n', 1)[0]:
        raise SystemExit('audit/deposits.txt must list the DOI of the deposit of truth_external_v2.json before scoring')
    dirty = subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', TRUTH], cwd=ROOT).returncode
    if dirty:
        raise SystemExit('audit/truth_external_v2.json differs from the committed (deposited) version')


def step_score():
    import hashlib
    from scipy.stats import t as student_t
    check_deposited()
    refuse_if_exists([SCORE])
    truth = json.load(open(TRUTH, encoding='utf-8'))
    if hashlib.sha256(open(NPZ, 'rb').read()).hexdigest() != truth['predictions_npz_sha256']:
        raise SystemExit('predictions npz does not match the hash recorded in the truth file')
    Z = np.load(NPZ)
    block_n, f_blk = Z['block_rows'], Z['f_blocks']
    starts = np.concatenate([[0], np.cumsum(block_n)[:-1]])
    z95 = 1.959963984540054
    out = dict(record='audit/prediction_external_v2.txt (commit b100bea; doi:10.5281/zenodo.24459864)',
               truth='audit/truth_external_v2.json (commit 6311f78; doi:10.5281/zenodo.24461908)',
               clarifications=SCORE_CLARIFICATIONS, draws=2000, scenarios={})
    t0 = time.time()
    for name, sc in truth['scenarios'].items():
        s = sc['s_u']
        P = {k.split('__', 1)[1]: Z[k] for k in Z.files if k.startswith(name + '__')}
        rng = np.random.default_rng(3002)
        rec = {c: dict(d=[], cr_lo=[], cr_hi=[], dl_lo=[], dl_hi=[], ratio=[]) for c in CONTRASTS}
        n_rows, skipped = [], 0
        for _ in range(2000):
            pick = rng.integers(0, 39, SMALL_K)
            rows = np.concatenate([np.arange(starts[b], starts[b] + block_n[b]) for b in pick])
            cid = np.repeat(np.arange(SMALL_K), block_n[pick])
            y = draw_labels(f_blk[rows], cid, s, rng)
            if y.min() == y.max():
                skipped += 1
                continue
            n_rows.append(int(rows.size))
            for c, (a, b) in CONTRASTS.items():
                cr = cluster_robust_delta(y, P[a][:, rows], P[b][:, rows], cid)
                dl = paired_delong(y, P[a][:, rows], P[b][:, rows])
                h = float(student_t.ppf(0.975, cr.df)) * cr.se_cr2
                r = rec[c]
                r['d'].append(cr.delta); r['cr_lo'].append(cr.delta - h); r['cr_hi'].append(cr.delta + h)
                r['dl_lo'].append(dl.delta - z95 * dl.se); r['dl_hi'].append(dl.delta + z95 * dl.se)
                r['ratio'].append(cr.se_cr2 / dl.se)
        res = {}
        for c in CONTRASTS:
            r = {k: np.array(v) for k, v in rec[c].items()}
            td = sc['truth'][c]['true_difference']
            sign = np.sign(td)
            row = dict(true_difference=td, mean_estimate=float(r['d'].mean()), sd_estimate=float(r['d'].std(ddof=1)),
                       empirical_mde=float(2.80 * r['d'].std(ddof=1)), median_ratio_cr2_delong=float(np.median(r['ratio'])))
            for m in ('cr', 'dl'):
                lo, hi = r[m + '_lo'], r[m + '_hi']
                res_ = (lo > 0) | (hi < 0)
                wrong = (sign > 0) & (hi < 0) | (sign < 0) & (lo > 0)
                key = 'cr2' if m == 'cr' else 'delong'
                row[key] = dict(coverage=float(((lo <= td) & (td <= hi)).mean()), resolved=float(res_.mean()),
                                wrong_sign=float(wrong.mean()), median_halfwidth=float(np.median((hi - lo) / 2)))
            if c in DECLARED:
                apm = sc['a_priori_mde'][c]['mde_a_priori']
                row['a_priori_mde'] = apm
                row['ratio_a_priori_to_empirical_mde'] = apm / row['empirical_mde']
            res[c] = row
        out['scenarios'][name] = dict(s_u=s, n_draws=len(n_rows), skipped_single_class=skipped,
                                      median_rows=float(np.median(n_rows)), contrasts=res)
        print(f'{name}: ' + ' | '.join(f"{c} cov CR2 {v['cr2']['coverage']:.3f} DL {v['delong']['coverage']:.3f} "
                                       f"res {v['cr2']['resolved']:.3f} ratio {v['median_ratio_cr2_delong']:.2f}"
                                       for c, v in res.items()), flush=True)
    S = out['scenarios']
    D = DECLARED
    within = {k: [c for c in D if abs(S[k]['contrasts'][c]['true_difference']) < 0.01] for k in S}
    out['predictions'] = dict(
        P1=dict(holds={k: truth['scenarios'][k]['truth']['C1']['true_difference'] > 0.01 for k in S},
                note='decided by the truth alone'),
        P2=dict(coverage={k: {c: S[k]['contrasts'][c]['cr2']['coverage'] for c in D} for k in S},
                holds=all(0.93 <= S[k]['contrasts'][c]['cr2']['coverage'] <= 0.97 for k in S for c in D)),
        P3=dict(delong_minus_cr2={c: S['S2']['contrasts'][c]['delong']['coverage'] -
                                  S['S2']['contrasts'][c]['cr2']['coverage'] for c in D},
                holds=all(S['S2']['contrasts'][c]['delong']['coverage'] < S['S2']['contrasts'][c]['cr2']['coverage']
                          for c in D)),
        P4=dict(holds=None, note='not evaluable as recorded (C0 is not a null; see clarifications.c0)',
                rejection_descriptive={k: dict(cr2=S[k]['contrasts']['C0']['cr2']['resolved'],
                                               delong=S[k]['contrasts']['C0']['delong']['resolved']) for k in S}),
        P5=dict(within_margin=within,
                resolved_within_margin={k: {c: S[k]['contrasts'][c]['cr2']['resolved'] for c in within[k]} for k in S},
                wrong_sign={k: {c: S[k]['contrasts'][c]['cr2']['wrong_sign'] for c in D} for k in S},
                holds=all(S[k]['contrasts'][c]['cr2']['resolved'] <= 0.10 for k in S for c in within[k]) and
                all(S[k]['contrasts'][c]['cr2']['wrong_sign'] <= 0.025 for k in S for c in D)),
        P6=dict(ratio={k: {c: S[k]['contrasts'][c]['ratio_a_priori_to_empirical_mde'] for c in D} for k in S},
                holds=all(0.7 <= S[k]['contrasts'][c]['ratio_a_priori_to_empirical_mde'] <= 1.4 for k in S for c in D)),
    )
    out['minutes'] = (time.time() - t0) / 60
    json.dump(out, open(SCORE, 'w'), indent=2)
    print(json.dumps({k: v.get('holds') for k, v in out['predictions'].items()}), 'wrote', SCORE)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--step', choices=['truth', 'score'], required=True)
    a = ap.parse_args()
    if a.step == 'truth':
        step_truth()
    else:
        step_score()


if __name__ == '__main__':
    main()
