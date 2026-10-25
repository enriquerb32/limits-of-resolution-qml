"""
code/local_experiments/exp7_noise_budget.py
Local Experiment 7: Noise budget of the 528-month test benchmark.

Question: which AUC differences can this benchmark resolve, and how much of the variance of a
single-run AUC is due to the optimisation seed versus test-set sampling?

Primary data: train-only rerun probabilities (15 seeds x 528 test rows) for RF, XGBoost and the
Standalone LSTM (results/json/results_rerun_train_only_canonical_probs.npz). Quantum hybrids have no
exported multi-seed probabilities (Priority 4 pending), so their rows use per-seed AUCs from
results_master.csv combined with SE proxies derived from the classical models (flagged as proxy).

Two sampling-error regimes are reported everywhere:
  * iid    : DeLong (Sun & Xu 2014) -- optimistic, ignores temporal/spatial correlation.
  * block  : year-block bootstrap (B = 2,000, calendar-year clusters) -- conservative.

Output: results/local_exp7_noise_budget.json
"""

import json
import os
import itertools
import numpy as np
import pandas as pd
from scipy.stats import norm

from noise_utils import (Z975, Z80, load_test_partition, delong_cov, delong_paired,
                         auc_matrix, year_block_bootstrap_aucs)

B = 2000
ALPHA_Z = Z975
POWER_K = Z975 + Z80  # 2.80 -> MDE at alpha = 0.05 (two-sided), power = 0.80


def climatology_probs(ds, y_te, m_te, X_te):
    """Month x Station climatology, fit on train partition only (identical to exp1)."""
    m_tr, y_tr, X_tr = np.array(ds['m_tr']), np.array(ds['y_tr']), np.array(ds['X_tr'])
    st_idx = [31, 32, 33, 34]
    p_uncond = float(y_tr.mean())
    p = np.zeros(len(y_te))
    for i in range(len(y_te)):
        s_col = [c for c in st_idx if X_te[i, c] == 1.0][0]
        mask = (m_tr == m_te[i]) & (X_tr[:, s_col] == 1.0)
        p[i] = y_tr[mask].mean() if mask.sum() > 0 else p_uncond
    return p


def main():
    os.makedirs('results', exist_ok=True)
    ds, y, m_te, X_te, years, max_rpm = load_test_partition()
    d = np.load('results/json/results_rerun_train_only_canonical_probs.npz')
    assert np.array_equal(d['y_te'].astype(int), y)

    regimes = {'train_only': 'train_only_', 'canonical': 'canonical_'}
    models = {'RF': 'RF_probs', 'XGB': 'XGB_probs', 'LSTM': 'LSTM_probs'}
    n_seeds = 15

    p_clim = climatology_probs(ds, y, m_te, X_te)

    out = {
        'meta': {
            'n_test': int(len(y)), 'n_pos': int(y.sum()), 'n_neg': int((1 - y).sum()),
            'n_year_blocks': int(len(np.unique(years))), 'max_rows_per_month_in_a_block': max_rpm,
            'B_bootstrap': B, 'alpha': 0.05, 'power_for_MDE': 0.80, 'mde_multiplier': float(POWER_K),
            'note': 'iid = DeLong (optimistic); block = calendar-year block bootstrap (conservative).'
        }
    }

    for reg, prefix in regimes.items():
        P = {k: d[prefix + v].astype(float) for k, v in models.items()}
        # Stack all 45 probability vectors + climatology for joint analysis.
        names = [(k, s) for k in models for s in range(n_seeds)]
        stack = np.vstack([P[k][s] for k, s in names] + [p_clim[None, :]])
        clim_row = stack.shape[0] - 1
        row = {ns: i for i, ns in enumerate(names)}

        aucs, cov = delong_cov(y, stack)
        boot = year_block_bootstrap_aucs(y, stack, years, B=B, seed=42)

        def se_diff_iid(i, j, c=cov):
            return float(np.sqrt(max(c[i, i] + c[j, j] - 2 * c[i, j], 1e-300)))

        def p_iid(i, j, a=aucs, c=cov):
            dd = a[i] - a[j]
            return float(2 * norm.sf(abs(dd) / se_diff_iid(i, j, c)))

        def block_diff(i, j, b=boot):
            diffs = b[:, i] - b[:, j]
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            return float(np.std(diffs, ddof=1)), bool(lo > 0 or hi < 0)

        res = {'per_model': {}, 'pairs': {}, 'same_architecture': {}, 'vs_climatology': {}}

        # ---- A + C: single-model SE, seed SD, variance share -------------------------------
        for k in models:
            idx = [row[(k, s)] for s in range(n_seeds)]
            a = aucs[idx]
            se_iid = float(np.mean(np.sqrt(np.diag(cov)[idx])))
            se_blk = float(np.mean(np.std(boot[:, idx], axis=0, ddof=1)))
            sd_seed = float(np.std(a, ddof=1))
            res['per_model'][k] = {
                'auc_mean': float(a.mean()), 'auc_sd_seed': sd_seed,
                'se_auc_iid': se_iid, 'se_auc_block': se_blk,
                'seed_variance_share_iid': sd_seed ** 2 / (sd_seed ** 2 + se_iid ** 2),
                'seed_variance_share_block': sd_seed ** 2 / (sd_seed ** 2 + se_blk ** 2),
            }

        # ---- B + D + F: cross-architecture pairs -------------------------------------------
        for A, Bm in [('RF', 'XGB'), ('RF', 'LSTM'), ('XGB', 'LSTM')]:
            # SE of the difference: same-seed-index pairs (models fitted with same seed index).
            se_i = np.mean([se_diff_iid(row[(A, s)], row[(Bm, s)]) for s in range(n_seeds)])
            se_b = np.mean([block_diff(row[(A, s)], row[(Bm, s)])[0] for s in range(n_seeds)])
            sdA = res['per_model'][A]['auc_sd_seed']
            sdB = res['per_model'][Bm]['auc_sd_seed']
            gap = res['per_model'][A]['auc_mean'] - res['per_model'][Bm]['auc_mean']
            # Detection across all 15 x 15 single-seed pairings.
            sig_iid, sig_blk, wrong_iid, wrong_blk = [], [], [], []
            for sa, sb in itertools.product(range(n_seeds), range(n_seeds)):
                i, j = row[(A, sa)], row[(Bm, sb)]
                dd = aucs[i] - aucs[j]
                si = p_iid(i, j) < 0.05
                sbk = block_diff(i, j)[1]
                sig_iid.append(si)
                sig_blk.append(sbk)
                wrong_iid.append(si and np.sign(dd) != np.sign(gap))
                wrong_blk.append(sbk and np.sign(dd) != np.sign(gap))
            res['pairs'][f'{A}-{Bm}'] = {
                'gap_seed_mean': float(gap),
                'se_diff_iid': float(se_i), 'se_diff_block': float(se_b),
                'mde_fixed_seed_iid': float(POWER_K * se_i),
                'mde_fixed_seed_block': float(POWER_K * se_b),
                'mde_single_run_iid': float(POWER_K * np.sqrt(se_i ** 2 + sdA ** 2 + sdB ** 2)),
                'mde_single_run_block': float(POWER_K * np.sqrt(se_b ** 2 + sdA ** 2 + sdB ** 2)),
                'gap_over_mde_iid': float(abs(gap) / (POWER_K * se_i)),
                'detect_rate_iid': float(np.mean(sig_iid)),
                'detect_rate_block': float(np.mean(sig_blk)),
                'wrong_sign_significant_iid': float(np.mean(wrong_iid)),
                'wrong_sign_significant_block': float(np.mean(wrong_blk)),
                'n_pairings': n_seeds * n_seeds,
            }

        # ---- F: same-architecture seed pairs (any "significant" result is a false claim) ---
        for k in models:
            sig_i, sig_b, ses = [], [], []
            for sa, sb in itertools.combinations(range(n_seeds), 2):
                i, j = row[(k, sa)], row[(k, sb)]
                sig_i.append(p_iid(i, j) < 0.05)
                sig_b.append(block_diff(i, j)[1])
                ses.append(se_diff_iid(i, j))
            sd = res['per_model'][k]['auc_sd_seed']
            se_m = float(np.mean(ses))
            res['same_architecture'][k] = {
                'false_claim_rate_iid': float(np.mean(sig_i)),
                'false_claim_rate_block': float(np.mean(sig_b)),
                'mean_se_diff_iid': se_m,
                # Analytic check used for quantum proxies: d ~ N(0, 2 sd^2) on the fixed test.
                'analytic_false_claim_iid': float(2 * norm.sf(ALPHA_Z * se_m / (np.sqrt(2) * sd))),
                'n_pairs': len(sig_i),
            }

        # ---- H: every seed vs Month x Station climatology ----------------------------------
        for k in models:
            dlt, pv, sigb = [], [], []
            for s in range(n_seeds):
                i = row[(k, s)]
                dlt.append(aucs[i] - aucs[clim_row])
                pv.append(p_iid(i, clim_row))
                sigb.append(block_diff(i, clim_row)[1])
            res['vs_climatology'][k] = {
                'clim_auc': float(aucs[clim_row]),
                'delta_mean': float(np.mean(dlt)), 'delta_min': float(np.min(dlt)),
                'delta_max': float(np.max(dlt)),
                'frac_seeds_sig_better_iid': float(np.mean([(p < 0.05 and dd > 0) for p, dd in zip(pv, dlt)])),
                'frac_seeds_sig_worse_iid': float(np.mean([(p < 0.05 and dd < 0) for p, dd in zip(pv, dlt)])),
                'frac_seeds_sig_block': float(np.mean(sigb)),
                'median_p_iid': float(np.median(pv)),
            }
        out[reg] = res

    # ---- G: quantum hybrids (proxy; draft multi-seed AUCs on the fixed canonical test) ------
    m = pd.read_csv('results/results_master.csv')
    ms = m[(m.metric == 'auc') & (m.run_type == 'multi_seed')]
    q_models = ['LSTM+QNN', 'LSTM+PI-QNN', 'LSTM+QTL', 'PI-QNN-LbL']
    tr = out['train_only']
    se_auc_iid = np.mean([tr['per_model'][k]['se_auc_iid'] for k in models])
    se_auc_blk = np.mean([tr['per_model'][k]['se_auc_block'] for k in models])
    se_same = tr['same_architecture']['LSTM']['mean_se_diff_iid']   # neural vs neural, same family
    se_cross = tr['pairs']['XGB-LSTM']['se_diff_iid']                 # weakly correlated models
    blk_ratio = tr['pairs']['XGB-LSTM']['se_diff_block'] / tr['pairs']['XGB-LSTM']['se_diff_iid']
    stats = {k: (ms[ms.model == k].value.mean(), ms[ms.model == k].value.std(ddof=1)) for k in q_models}

    q = {'proxy_inputs': {'se_auc_iid': float(se_auc_iid), 'se_auc_block': float(se_auc_blk),
                          'se_diff_same_family_iid': float(se_same),
                          'se_diff_cross_family_iid': float(se_cross),
                          'block_over_iid_ratio': float(blk_ratio),
                          'warning': 'PROXY: quantum test probabilities per seed are not exported.'},
         'per_model': {}, 'same_architecture': {}, 'cross_pairs': {}}
    for k, (mu, sd) in stats.items():
        q['per_model'][k] = {'auc_mean': float(mu), 'auc_sd_seed': float(sd),
                             'seed_variance_share_iid': float(sd ** 2 / (sd ** 2 + se_auc_iid ** 2)),
                             'seed_variance_share_block': float(sd ** 2 / (sd ** 2 + se_auc_blk ** 2))}
        q['same_architecture'][k] = {
            'false_claim_rate_iid_range': [float(2 * norm.sf(ALPHA_Z * se / (np.sqrt(2) * sd)))
                                           for se in (se_cross, se_same)]}
    for A, Bm in itertools.combinations(q_models, 2):
        (muA, sdA), (muB, sdB) = stats[A], stats[Bm]
        mu, s = muA - muB, np.sqrt(sdA ** 2 + sdB ** 2)
        rates = []
        for se in (se_cross, se_same):
            c = ALPHA_Z * se
            sig = norm.sf((c - mu) / s) + norm.cdf((-c - mu) / s)
            wrong = norm.cdf((-c - abs(mu)) / s)
            rates.append((float(sig), float(wrong)))
        q['cross_pairs'][f'{A} vs {Bm}'] = {
            'gap_seed_mean': float(mu),
            'single_seed_sig_rate_range': [rates[0][0], rates[1][0]],
            'single_seed_wrong_sign_sig_range': [rates[0][1], rates[1][1]],
        }
    out['quantum_proxy'] = q

    with open('results/local_exp7_noise_budget.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)

    # ---- Console summary -------------------------------------------------------------------
    print(f"Test: n={len(y)} (pos={int(y.sum())}), year blocks={len(np.unique(years))}, "
          f"max rows/month/block={max_rpm}")
    for reg in regimes:
        r = out[reg]
        print(f"\n=== {reg} ===")
        print("%-5s %-7s %-8s %-8s %-8s %-10s %-10s" % ('model', 'AUC', 'sd_seed', 'SE_iid', 'SE_blk',
                                                     'seed%_iid', 'seed%_blk'))
        for k, v in r['per_model'].items():
            print("%-5s %.4f  %.4f   %.4f   %.4f   %5.1f%%     %5.1f%%" % (
                k, v['auc_mean'], v['auc_sd_seed'], v['se_auc_iid'], v['se_auc_block'],
                100 * v['seed_variance_share_iid'], 100 * v['seed_variance_share_block']))
        print("%-9s %-8s %-8s %-8s %-9s %-9s %-9s %-9s" % ('pair', 'gap', 'SEd_iid', 'SEd_blk', 'MDE_iid',
                                                          'MDE_blk', 'det_iid', 'det_blk'))
        for k, v in r['pairs'].items():
            print("%-9s %+.4f  %.4f   %.4f   %.4f    %.4f    %5.1f%%    %5.1f%%" % (
                k, v['gap_seed_mean'], v['se_diff_iid'], v['se_diff_block'], v['mde_fixed_seed_iid'],
                v['mde_fixed_seed_block'], 100 * v['detect_rate_iid'], 100 * v['detect_rate_block']))
        for k, v in r['same_architecture'].items():
            print("same-arch %-5s false claims iid=%5.1f%% block=%5.1f%% (analytic iid %5.1f%%)" % (
                k, 100 * v['false_claim_rate_iid'], 100 * v['false_claim_rate_block'],
                100 * v['analytic_false_claim_iid']))
        for k, v in r['vs_climatology'].items():
            print("vs clim %-5s clim=%.4f delta=%+.4f [%+.4f,%+.4f] sig better iid=%3.0f%% worse=%3.0f%% "
                  "block sig=%3.0f%%" % (k, v['clim_auc'], v['delta_mean'], v['delta_min'], v['delta_max'],
                                          100 * v['frac_seeds_sig_better_iid'],
                                          100 * v['frac_seeds_sig_worse_iid'], 100 * v['frac_seeds_sig_block']))
    print("\n=== quantum (PROXY) ===")
    for k, v in q['per_model'].items():
        fc = q['same_architecture'][k]['false_claim_rate_iid_range']
        print("%-12s AUC=%.4f sd=%.4f seed%%iid=%4.1f%% blk=%4.1f%% same-arch false claims %4.1f-%4.1f%%" % (
            k, v['auc_mean'], v['auc_sd_seed'], 100 * v['seed_variance_share_iid'],
            100 * v['seed_variance_share_block'], 100 * fc[0], 100 * fc[1]))
    for k, v in q['cross_pairs'].items():
        print("%-28s gap=%+.4f sig %4.1f-%4.1f%% wrong-sign sig %4.1f-%4.1f%%" % (
            k, v['gap_seed_mean'], 100 * v['single_seed_sig_rate_range'][0],
            100 * v['single_seed_sig_rate_range'][1], 100 * v['single_seed_wrong_sign_sig_range'][0],
            100 * v['single_seed_wrong_sign_sig_range'][1]))
    print("\nSaved results/local_exp7_noise_budget.json")


if __name__ == '__main__':
    main()
