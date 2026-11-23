#!/usr/bin/env python3
"""
exp15_panel_matched_simulation.py - Known-truth simulation whose design parameters are estimated from the real panel.

Same mechanism as exp11 (two correlated scorers, year effects on prevalence and on scores, known population AUC difference), but
instead of round numbers the design is matched to the panel:
  * year sizes      : the actual number of test rows in each of the 19 test years (unequal, fixed in every replication)
  * prevalence      : pooled test prevalence and year-level heterogeneity tau_prev (method of moments on the year prevalences)
  * scorer quality  : AUC_B and the design gap taken from two saved scorers (seed-mean probabilities), binormal mu = sqrt2*Phi^-1(AUC)
  * scorer corr.    : rho = correlation of the within-class normal scores of the two scorers
  * year effects    : shared tau_s and scorer-specific tau_m by moments on the year means of the within-class normal scores,
                      after subtracting the sampling variance 1/n_u (negative estimates are truncated at 0)
These are rough estimates from 19 years, so the script also runs multiples of the year-effect scales (--tau_mult).
Output: type-I error and coverage of DeLong (iid) and of the year-block bootstrap, SD(Delta_hat) against the mean standard errors,
and power at the observed gap.  It does NOT reproduce spatial structure, optimization noise or the real score distribution.

INPUTS   --probs results_rerun_train_only_canonical_probs.npz (--prefix train_only; models RF, LSTM by default)
         --meta  test_meta.csv  (columns ciudad, fecha; same row order as y_te; from export_test_meta.py)
USAGE    python exp15_panel_matched_simulation.py --probs results_rerun_train_only_canonical_probs.npz --meta test_meta.csv \
             --scorer_a RF --scorer_b LSTM --tau_mult 0.5 1 1.5 --reps 500 --B 300 --out results_exp15_panel_sim.json
"""
import argparse, json, os, time
import numpy as np, pandas as pd
from scipy.stats import norm, rankdata
from exp11_simulation_known_truth import comp, auc

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def normal_scores(y, p):
    z = np.empty(len(p))
    for c in (0, 1):
        ix = np.where(y == c)[0]; r = rankdata(p[ix])
        z[ix] = norm.ppf((r - 0.5) / len(ix))
    return z


def estimate(a):
    Z = np.load(a.probs); y = Z['y_te'].astype(int)
    pa = Z[f'{a.prefix}_{a.scorer_a}_probs'].mean(0); pb = Z[f'{a.prefix}_{a.scorer_b}_probs'].mean(0)
    meta = pd.read_csv(a.meta); assert len(meta) == len(y), (len(meta), len(y))
    yr = pd.to_datetime(meta['fecha']).dt.year.values; years = np.unique(yr)
    sizes = np.array([(yr == u).sum() for u in years]); prev_u = np.array([y[yr == u].mean() for u in years])
    auc_a, auc_b = auc(y, pa), auc(y, pb); za, zb = normal_scores(y, pa), normal_scores(y, pb)
    rho = float(np.corrcoef(za, zb)[0, 1]); pi = float(y.mean()); inv = float(np.mean(1 / sizes))
    ma = np.array([za[yr == u].mean() for u in years]); mb = np.array([zb[yr == u].mean() for u in years])
    cov = float(np.cov(ma, mb)[0, 1]); tot = 0.5 * (float(np.var(ma, ddof=1)) + float(np.var(mb, ddof=1))) - inv
    tau_s2 = max(cov - rho * inv, 0.0); tau_m2 = max(tot - tau_s2, 0.0)
    v_prev = float(np.var(prev_u, ddof=1)) - float(np.mean(pi * (1 - pi) / sizes))
    tau_prev = float(np.sqrt(max(v_prev, 0.0)) / (pi * (1 - pi)))
    est = dict(years=[int(u) for u in years], year_sizes=sizes.tolist(), prevalence=pi, auc_a=auc_a, auc_b=auc_b, gap=auc_a - auc_b, rho=rho,
               tau_prev=tau_prev, tau_shared=float(np.sqrt(tau_s2)), tau_specific=float(np.sqrt(tau_m2)),
               raw=dict(cov_year_means=cov, mean_inv_size=inv, total_year_var=tot, var_prev_net=v_prev))
    return est, sizes


def gen_panel(rng, sizes, pi, mu, rho, tau_prev, tau_s, tau_m):
    Y = len(sizes); yr = np.repeat(np.arange(Y), sizes); N = len(yr)
    lg = np.log(pi / (1 - pi)) + tau_prev * rng.standard_normal(Y)
    y = (rng.random(N) < 1 / (1 + np.exp(-lg[yr]))).astype(int)
    w = tau_s * rng.standard_normal(Y)[yr]; z = rng.standard_normal(N); S = []
    for k in range(2):
        e = np.sqrt(rho) * z + np.sqrt(1 - rho) * rng.standard_normal(N)
        v = (tau_m * rng.standard_normal(Y)[yr]) if tau_m > 0 else 0.0
        S.append(y * mu[k] + e + w + v)
    return y, S[0], S[1], yr


def true_delta_panel(rng, sizes, pi, mu, rho, tp, ts, tm, reps=60):
    # population AUCs: many independent panels with the same year sizes (self-contained; no dependence on the exp11 version)
    ta, tb, dd = [], [], []
    for _ in range(4):
        ys, A, B = [], [], []
        for _r in range(reps):
            y, a_, b_, _y = gen_panel(rng, sizes, pi, mu, rho, tp, ts, tm); ys.append(y); A.append(a_); B.append(b_)
        y = np.concatenate(ys); ta.append(auc(y, np.concatenate(A))); tb.append(auc(y, np.concatenate(B))); dd.append(ta[-1] - tb[-1])
    return float(np.mean(ta)), float(np.mean(tb)), float(np.std(dd, ddof=1) / 2)


def run_config(rng, a, est, sizes, mult, delta, abs_tau=None):
    mu_b = np.sqrt(2) * norm.ppf(est['auc_b']); mu_a = np.sqrt(2) * norm.ppf(min(est['auc_b'] + delta, 0.99))
    pi, rho = est['prevalence'], est['rho']
    if abs_tau is None:
        tp, ts, tm = est['tau_prev'] * mult, est['tau_shared'] * mult, est['tau_specific'] * mult
    else:   # absolute year-effect scale t: prevalence logit sd = t, shared score sd = t, scorer-specific sd = t/2
        tp, ts, tm = abs_tau, abs_tau, abs_tau / 2
    tA, tB, tse = true_delta_panel(rng, sizes, pi, (mu_a, mu_b), rho, tp, ts, tm)
    dtrue = 0.0 if delta == 0 else tA - tB
    rej_dl = rej_bb = cov_dl = cov_bb = 0; dh = []; sdl = []; sbb = []; Y = len(sizes)
    for _ in range(a.reps):
        y, sa, sb, yr = gen_panel(rng, sizes, pi, (mu_a, mu_b), rho, tp, ts, tm)
        if y.sum() < 5 or (1 - y).sum() < 5: continue
        A = comp(y, sa); B = comp(y, sb); m, n = A[3], A[4]
        S10 = np.cov(np.vstack([A[1], B[1]])); S01 = np.cov(np.vstack([A[2], B[2]]))
        var = (S10[0, 0] + S10[1, 1] - 2 * S10[0, 1]) / m + (S01[0, 0] + S01[1, 1] - 2 * S01[0, 1]) / n
        d = A[0] - B[0]; se = np.sqrt(max(var, 1e-18)); dh.append(d); sdl.append(se)
        rej_dl += abs(d / se) > 1.96; cov_dl += (d - 1.96 * se <= dtrue <= d + 1.96 * se)
        idx_by = [np.where(yr == u)[0] for u in range(Y)]; bs = []
        for _b in range(a.B):
            pick = rng.integers(0, Y, Y); idx = np.concatenate([idx_by[u] for u in pick]); yy = y[idx]
            if yy.sum() in (0, len(yy)): continue
            bs.append(auc(yy, sa[idx]) - auc(yy, sb[idx]))
        bs = np.array(bs); lo, hi = np.percentile(bs, [2.5, 97.5]); sbb.append(bs.std(ddof=1))
        rej_bb += (lo > 0) or (hi < 0); cov_bb += (lo <= dtrue <= hi)
    k = len(dh)
    return dict(tau_mult=mult, abs_tau=abs_tau, delta_design=delta, delta_true=float(dtrue), reps=k, rejection_delong=rej_dl / k, rejection_block=rej_bb / k,
                coverage_delong=cov_dl / k, coverage_block=cov_bb / k, sd_delta_hat=float(np.std(dh, ddof=1)),
                mean_se_delong=float(np.mean(sdl)), mean_se_block=float(np.mean(sbb)),
                mc_se_rejection=float(np.sqrt(0.05 * 0.95 / k)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--probs', required=True); ap.add_argument('--prefix', default='train_only'); ap.add_argument('--meta', default=os.path.join(ROOT, 'results', 'test_meta.csv'))
    ap.add_argument('--scorer_a', default='RF'); ap.add_argument('--scorer_b', default='LSTM')
    ap.add_argument('--tau_mult', type=float, nargs='+', default=[0.5, 1.0, 1.5])
    ap.add_argument('--tau_abs', type=float, nargs='*', default=[0.0, 0.1, 0.2, 0.3], help='absolute year-effect scales (used because moment estimates can be 0)')
    ap.add_argument('--reps', type=int, default=500); ap.add_argument('--B', type=int, default=300)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--out', default='results_exp15_panel_sim.json')
    a = ap.parse_args(); rng = np.random.default_rng(a.seed)
    est, sizes = estimate(a)
    print('estimated design:', json.dumps({k: v for k, v in est.items() if k != 'raw'}, indent=1)); print('raw moments:', est['raw'])
    rows = []
    grid = [(m, None) for m in a.tau_mult if est['tau_shared'] + est['tau_prev'] + est['tau_specific'] > 0] + [(None, t) for t in a.tau_abs]
    for mult, at in grid:
        for delta in (0.0, max(est['gap'], 0.0)):
            t0 = time.time(); r = run_config(rng, a, est, sizes, mult, delta, at); rows.append(r)
            print(f"tau {'x%g' % mult if at is None else 'abs %g' % at} | delta {delta:.4f} (true {r['delta_true']:.4f}) | reject DL {r['rejection_delong']:.3f} block {r['rejection_block']:.3f} | "
                  f"coverage DL {r['coverage_delong']:.3f} block {r['coverage_block']:.3f} | SD {r['sd_delta_hat']:.4f} vs SE DL {r['mean_se_delong']:.4f} "
                  f"block {r['mean_se_block']:.4f} | {time.time()-t0:.0f}s")
    json.dump(dict(config=vars(a), design=est, rows=rows), open(a.out, 'w'), indent=2); print('saved', a.out)
