#!/usr/bin/env python3
"""
exp11_simulation_known_truth.py  -  Simulation with KNOWN population difference to validate the inferential machinery.

Question: when two scorers have a known population AUC difference Delta, how often do the DeLong (iid) test and the
calendar-year block-bootstrap CI (i) reject Delta=0, (ii) cover the true Delta, and is their SE right?  Answers type-I
error (Delta=0), power (Delta>0), coverage and SE calibration under temporal dependence.

Generative model (all knobs are command-line options)
  rows       : N rows in Y 'years' (blocks) of equal size
  labels     : prevalence pi, plus a year-level random effect on the logit (sd --tau_prev)
  scores     : positives  s_k = mu_k + e_k ,  negatives  s_k = e_k ,  k in {A,B}
               e_k = sqrt(rho) z + sqrt(1-rho) u_k  (z shared by both models  => correlated scorers, rho = --rho)
               plus a year-level shift gamma*w_y shared by both models (sd --tau_score): temporal dependence
  true AUC   : population values computed on a very large sample of the SAME mechanism (pooled AUC), so
               Delta_true includes whatever the year effects do to pooled AUC.
Usage:
  python exp11_simulation_known_truth.py --reps 500 --B 500 --out results_exp11_sim.json
Default design mimics the paper: N=528, 19 years, prevalence .41, AUC_B=.65, rho=.75, deltas 0,.01,.02,.03,.04,.05
"""
import argparse, json, time
import numpy as np
from scipy.stats import rankdata, norm

def comp(y, p):
    pos = y == 1; m = int(pos.sum()); n = len(y) - m
    tz = rankdata(p); tx = rankdata(p[pos]); ty = rankdata(p[~pos])
    return tz[pos].sum() / (m * n) - (m + 1) / (2 * n), (tz[pos] - tx) / n, 1 - (tz[~pos] - ty) / m, m, n

def auc(y, p):
    r = rankdata(p); m = int(y.sum()); n = len(y) - m
    return (r[y == 1].sum() - m * (m + 1) / 2) / (m * n)

def gen(rng, N, Y, pi, mu, rho, tau_prev, tau_score):
    yr = np.repeat(np.arange(Y), int(np.ceil(N / Y)))[:N]
    logit = np.log(pi / (1 - pi)) + tau_prev * rng.standard_normal(Y)
    y = (rng.random(N) < 1 / (1 + np.exp(-logit[yr]))).astype(int)
    w = tau_score * rng.standard_normal(Y)[yr]
    z = rng.standard_normal(N)
    S = []
    for k in range(2):
        e = np.sqrt(rho) * z + np.sqrt(1 - rho) * rng.standard_normal(N)
        S.append(y * mu[k] + e + w)
    return y, S[0], S[1], yr

def true_delta(mu, rho, pi, Y, tau_prev, tau_score, rng, Nbig=400000):
    y, a, b, _ = gen(rng, Nbig, Y * 2000, pi, mu, rho, tau_prev, tau_score)
    return auc(y, a), auc(y, b)

def run(a):
    rng = np.random.default_rng(a.seed)
    mu_B = np.sqrt(2) * norm.ppf(a.auc_b)
    rows = []
    for delta in a.deltas:
        mu_A = np.sqrt(2) * norm.ppf(min(a.auc_b + delta, 0.99))
        tA, tB = true_delta((mu_A, mu_B), a.rho, a.prev, a.years, a.tau_prev, a.tau_score, rng)
        dtrue = tA - tB
        rej_dl = rej_bb = cov_dl = cov_bb = 0; dh = []; se_dl = []; se_bb = []; t0 = time.time()
        for r in range(a.reps):
            y, sa, sb, yr = gen(rng, a.n, a.years, a.prev, (mu_A, mu_B), a.rho, a.tau_prev, a.tau_score)
            if y.sum() < 5 or (1 - y).sum() < 5: continue
            A = comp(y, sa); B = comp(y, sb); m, n = A[3], A[4]
            S10 = np.cov(np.vstack([A[1], B[1]])); S01 = np.cov(np.vstack([A[2], B[2]]))
            var = (S10[0, 0] + S10[1, 1] - 2 * S10[0, 1]) / m + (S01[0, 0] + S01[1, 1] - 2 * S01[0, 1]) / n
            d = A[0] - B[0]; se = np.sqrt(max(var, 1e-18)); dh.append(d); se_dl.append(se)
            rej_dl += abs(d / se) > 1.96; cov_dl += (d - 1.96 * se <= dtrue <= d + 1.96 * se)
            idx_by = [np.where(yr == u)[0] for u in range(a.years)]
            bs = []
            for _ in range(a.B):
                pick = rng.integers(0, a.years, a.years); idx = np.concatenate([idx_by[u] for u in pick]); yy = y[idx]
                if yy.sum() in (0, len(yy)): continue
                bs.append(auc(yy, sa[idx]) - auc(yy, sb[idx]))
            bs = np.array(bs); lo, hi = np.percentile(bs, [2.5, 97.5]); se_bb.append(bs.std(ddof=1))
            rej_bb += (lo > 0) or (hi < 0); cov_bb += (lo <= dtrue <= hi)
        k = len(dh)
        rows.append(dict(delta_design=delta, delta_true=float(dtrue), auc_A=float(tA), auc_B=float(tB), reps=k,
                         rejection_delong=rej_dl / k, rejection_block=rej_bb / k,
                         coverage_delong=cov_dl / k, coverage_block=cov_bb / k,
                         mean_delta_hat=float(np.mean(dh)), sd_delta_hat=float(np.std(dh, ddof=1)),
                         mean_se_delong=float(np.mean(se_dl)), mean_se_block=float(np.mean(se_bb))))
        r_ = rows[-1]
        print(f"Delta_true {dtrue:.4f} | reject DL {r_['rejection_delong']:.2f} block {r_['rejection_block']:.2f} | "
              f"coverage DL {r_['coverage_delong']:.2f} block {r_['coverage_block']:.2f} | "
              f"SD(Delta_hat) {r_['sd_delta_hat']:.4f} vs mean SE DL {r_['mean_se_delong']:.4f} block {r_['mean_se_block']:.4f} | {time.time()-t0:.0f}s")
    json.dump(dict(config=vars(a), rows=rows), open(a.out, 'w'), indent=2)
    print('saved', a.out)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=528); ap.add_argument('--years', type=int, default=19)
    ap.add_argument('--prev', type=float, default=0.41); ap.add_argument('--auc_b', type=float, default=0.65)
    ap.add_argument('--rho', type=float, default=0.75)
    ap.add_argument('--tau_prev', type=float, default=0.3, help='sd of year effect on prevalence logit')
    ap.add_argument('--tau_score', type=float, default=0.3, help='sd of year-level score shift shared by both models')
    ap.add_argument('--deltas', type=float, nargs='+', default=[0, .01, .02, .03, .04, .05])
    ap.add_argument('--reps', type=int, default=500); ap.add_argument('--B', type=int, default=500)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--out', default='results_exp11_sim.json')
    run(ap.parse_args())
