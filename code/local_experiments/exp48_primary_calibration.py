#!/usr/bin/env python3
"""
exp48_primary_calibration.py - which interval procedure is calibrated for this test design? (review 2026-11-18, point 2)
=====================================================================================================================
Estimand: the difference of population AUCs of two scorers for NEW years at the same stations, with the test coverage
mask of the panel (which stations report in which year). Under the binormal mechanism below, with year effects
independent of the prevalence shift, the marginal AUC of scorer k over new, independent years is analytic:
    AUC_k = Phi( mu_k / sqrt(2 (1 + tau_s^2 + tau_m^2 + tau_c^2)) )
(positives and negatives of the population come from different years with probability 1), so coverage is evaluated
against the exact truth, not a Monte Carlo reference.

Mechanism, per replication (rows = the 529 station-months of results/test_meta.csv, or a variant of that mask):
  y      ~ Bernoulli(logit^-1(logit p_station + tau_p a_u))           station base rates from the test set
  S_k    = mu_k y + e_k + w_u + v_{k,u} + tau_c lambda_s eta_{k,u}    k = A, B
  e_A, e_B standard normal with correlation rho; w_u shared year shift (tau_s); v_{k,u} scorer-specific year shift
  (tau_m); lambda_s = +1 (Alicante, San Javier) / -1 (Barcelona, Valencia): scorer-specific year x station effects with
  opposite loadings, which make station contributions cancel within a year (the mechanism proposed for the low
  cluster/iid ratios of the first panel). Year effects are AR(1) over consecutive years with coefficient phi.
Procedures (95 %): DeLong (iid); year-block percentile bootstrap; CR1, CR2 with t_{K-1}; CR2 with Bell-McCaffrey
degrees of freedom; delete-one-year jackknife with t_{K-1}; studentized wild cluster bootstrap (Webb weights, CR1
studentization); and the paper's former unanimity rule (DeLong, block bootstrap and CR1-t_{K-1} all exclude zero).
Single scorers (no seed dimension): the estimand conditions on the fitted models.

  python code/local_experiments/exp48_primary_calibration.py [--reps 1000] [--B 399]
Output: results/local_exp48_primary_calibration.json
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm, rankdata, t as student_t

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
META = os.path.join(ROOT, 'results', 'test_meta.csv')
OUT = os.path.join(ROOT, 'results', 'local_exp48_primary_calibration.json')
LAMBDA = {'Alicante': 1.0, 'San_Javier': 1.0, 'Barcelona': -1.0, 'Valencia': -1.0}
WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])
METHODS = ['delong', 'block_pct', 'cr1_t', 'cr2_t', 'cr2_bm', 'jackknife_t', 'wild_webb', 'unanimity']


# ------------------------------------------------------------------ design (coverage mask)
def real_mask():
    m = pd.read_csv(META, parse_dates=['fecha'])
    return dict(station=m['ciudad'].values, year=m['fecha'].dt.year.values,
                p_station=m.groupby('ciudad')['y_te'].mean().to_dict())


def variant(mask, kind):
    """kind: 'real'; 'K10'/'K38' (real per-year station patterns, fewer/more years); 'm2'/'m4' (19 years, all with
    the two main stations / all four stations, 12 months each)."""
    st, yr = mask['station'], mask['year']
    years = np.unique(yr)
    if kind == 'real':
        return st, yr
    if kind in ('K10', 'K38'):
        pick = np.linspace(0, len(years) - 1, 10).round().astype(int) if kind == 'K10' else np.r_[np.arange(19), np.arange(19)]
        S, Y = [], []
        for j, i in enumerate(pick):
            sel = yr == years[i]
            S.append(st[sel]); Y.append(np.full(sel.sum(), j))
        return np.concatenate(S), np.concatenate(Y)
    stations = ['Alicante', 'Barcelona'] if kind == 'm2' else ['Alicante', 'Barcelona', 'San_Javier', 'Valencia']
    S = np.tile(np.repeat(stations, 12), 19)
    Y = np.repeat(np.arange(19), 12 * len(stations))
    return S, Y


# ------------------------------------------------------------------ statistics
def components(y, s):
    """AUC and DeLong structural components (continuous scores, no ties)."""
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    r_all = rankdata(s)
    r_pos = rankdata(s[pos]); r_neg = rankdata(s[~pos])
    v10 = (r_all[pos] - r_pos) / n0
    v01 = 1.0 - (r_all[~pos] - r_neg) / n1
    return v10.mean(), v10, v01


def auc(y, s):
    pos = y == 1
    n1 = pos.sum(); n0 = y.size - n1
    r = rankdata(s)
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def bm_df(n_g, N):
    """Bell-McCaffrey degrees of freedom of CR2 for a mean (intercept-only working model, iid errors)."""
    h = n_g / N
    C = np.diag(n_g / (1 - h)) - np.outer(n_g / np.sqrt(1 - h), n_g / np.sqrt(1 - h)) / N
    return float(np.trace(C) ** 2 / np.trace(C @ C))


def procedures(y, sa, sb, yr, rng, B, n_wild=999):
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    A, a10, a01 = components(y, sa); Bc, b10, b01 = components(y, sb)
    d = A - Bc
    psi = np.empty(y.size)
    psi[pos] = (a10 - b10 - d) / n1
    psi[~pos] = (a01 - b01 - d) / n0
    lv, inv = np.unique(yr, return_inverse=True)
    K, N = lv.size, y.size
    S = np.bincount(inv, weights=psi); n_g = np.bincount(inv).astype(float)
    se_iid = np.sqrt((psi[pos] ** 2).sum() * n1 / (n1 - 1) + (psi[~pos] ** 2).sum() * n0 / (n0 - 1))
    se_cr1 = np.sqrt(K / (K - 1) * (S ** 2).sum())
    se_cr2 = np.sqrt(((S / np.sqrt(1 - n_g / N)) ** 2).sum())
    tq = student_t.ppf(0.975, K - 1)
    df_bm = bm_df(n_g, N)
    rows = [np.flatnonzero(inv == g) for g in range(K)]
    # jackknife
    th = []
    for g in range(K):
        keep = inv != g
        th.append(auc(y[keep], sa[keep]) - auc(y[keep], sb[keep]))
    th = np.array(th); se_jk = np.sqrt((K - 1) / K * ((th - th.mean()) ** 2).sum())
    # year-block percentile bootstrap
    bs = []
    for _ in range(B):
        idx = np.concatenate([rows[g] for g in rng.integers(0, K, K)])
        yy = y[idx]
        if 0 < yy.sum() < yy.size:
            bs.append(auc(yy, sa[idx]) - auc(yy, sb[idx]))
    lo_b, hi_b = np.percentile(bs, [2.5, 97.5])
    # wild cluster bootstrap-t (Webb), null imposed: contributions S are centred (sum S = 0)
    W = WEBB[rng.integers(0, 6, size=(n_wild, K))]
    WS = W * S
    t_star = WS.sum(1) / np.sqrt(K / (K - 1) * ((WS - WS.mean(1, keepdims=True)) ** 2).sum(1))
    q_w = np.quantile(np.abs(t_star), 0.95)
    ci = dict(delong=(d - 1.96 * se_iid, d + 1.96 * se_iid), block_pct=(lo_b, hi_b),
              cr1_t=(d - tq * se_cr1, d + tq * se_cr1), cr2_t=(d - tq * se_cr2, d + tq * se_cr2),
              cr2_bm=(d - student_t.ppf(0.975, df_bm) * se_cr2, d + student_t.ppf(0.975, df_bm) * se_cr2),
              jackknife_t=(d - tq * se_jk, d + tq * se_jk), wild_webb=(d - q_w * se_cr1, d + q_w * se_cr1))
    return d, ci, dict(se_iid=se_iid, se_cr1=se_cr1, se_cr2=se_cr2, se_jk=se_jk, df_bm=df_bm)


# ------------------------------------------------------------------ mechanism
def ar1(rng, K, phi, size):
    x = np.empty((size, K)); x[:, 0] = rng.standard_normal(size)
    for u in range(1, K):
        x[:, u] = phi * x[:, u - 1] + np.sqrt(1 - phi ** 2) * rng.standard_normal(size)
    return x


def simulate(rng, st, yr, p_station, mu, rho, tp, ts, tm, tc, phi, td=0.0, beta=0.0):
    """td: prevalence dipole (a wet year at Barcelona/Valencia is a dry one at Alicante/San Javier);
    beta: scorer A also scores the station base rate (B does not), so the dipole moves the two scorers differently."""
    K = int(yr.max()) + 1; N = yr.size
    a, w, vA, vB, eA, eB, a2 = ar1(rng, K, phi, 7)
    lam = np.array([LAMBDA[s] for s in st])
    base = logit(np.array([p_station[s] for s in st]))
    lg = base + tp * a[yr] + td * lam * a2[yr]
    y = (rng.random(N) < expit(lg)).astype(int)
    z = rng.standard_normal(N)
    out = []
    for k, (v, eta) in enumerate([(vA, eA), (vB, eB)]):
        e = np.sqrt(rho) * z + np.sqrt(1 - rho) * rng.standard_normal(N)
        s = mu[k] * y + e + ts * w[yr] + tm * v[yr] + tc * lam * eta[yr]
        out.append(s + (beta * (base - base.mean()) if k == 0 else 0.0))
    return y, out[0], out[1]


def true_auc(mu, ts, tm, tc):
    return norm.cdf(mu / np.sqrt(2 * (1 + ts ** 2 + tm ** 2 + tc ** 2)))


def true_delta_mc(rng, st, yr, p_station, mu, sc, copies=300, draws=4):
    """Population difference for new years when no closed form applies: the mask tiled `copies` times, each copy
    with new independent years (AR(1) runs restart per copy), pooled AUC; mean over `draws` and its MC s.e."""
    K = int(yr.max()) + 1
    big_st = np.tile(st, copies); big_yr = np.concatenate([yr + c * K for c in range(copies)])
    out = []
    for _ in range(draws):
        ys, A, Bs = [], [], []
        for c in range(copies):
            sel = slice(c * st.size, (c + 1) * st.size)
            y, a, b = simulate(rng, big_st[sel], big_yr[sel] - c * K, p_station, mu, sc['rho'], sc['tp'], sc['ts'],
                               sc['tm'], sc['tc'], sc['phi'], sc.get('td', 0.0), sc.get('beta', 0.0))
            ys.append(y); A.append(a); Bs.append(b)
        y = np.concatenate(ys); out.append(auc(y, np.concatenate(A)) - auc(y, np.concatenate(Bs)))
    return float(np.mean(out)), float(np.std(out, ddof=1) / np.sqrt(draws))


def run_cell(rng, mask, kind, sc, delta, reps, B, blocks=1):
    st, yr = variant(mask, kind)
    cl = yr // blocks          # clusters of `blocks` consecutive years (the mechanism still runs on years)
    tp, ts, tm, tc, phi = sc['tp'], sc['ts'], sc['tm'], sc['tc'], sc['phi']
    td, beta = sc.get('td', 0.0), sc.get('beta', 0.0)
    scale = np.sqrt(2 * (1 + ts ** 2 + tm ** 2 + tc ** 2))
    mu_b = scale * norm.ppf(0.66); mu_a = scale * norm.ppf(0.66 + delta)
    if td == 0 and beta == 0:
        d_true, d_true_mcse = float(true_auc(mu_a, ts, tm, tc) - true_auc(mu_b, ts, tm, tc)), 0.0   # analytic
    else:
        d_true, d_true_mcse = true_delta_mc(rng, st, yr, mask['p_station'], (mu_a, mu_b), sc)
    rej = {m: 0 for m in METHODS}; cov = {m: 0 for m in METHODS if m != 'unanimity'}
    dh, ses, dfs = [], {k: [] for k in ('se_iid', 'se_cr1', 'se_cr2', 'se_jk')}, []
    for _ in range(reps):
        y, sa, sb = simulate(rng, st, yr, mask['p_station'], (mu_a, mu_b), sc['rho'], tp, ts, tm, tc, phi, td, beta)
        d, ci, aux = procedures(y, sa, sb, cl, rng, B)
        dh.append(d); dfs.append(aux['df_bm'])
        for k in ses:
            ses[k].append(aux[k])
        for m, (lo, hi) in ci.items():
            rej[m] += (lo > 0) or (hi < 0); cov[m] += lo <= d_true <= hi
        rej['unanimity'] += all((ci[m][0] > 0) or (ci[m][1] < 0) for m in ('delong', 'block_pct', 'cr1_t'))
    sd = float(np.std(dh, ddof=1))
    ratio = np.array(ses['se_cr1']) / np.array(ses['se_iid'])
    # 'rejection' is a type-I error only where delta_true == 0; coverage is the calibration measure in every cell
    return dict(mask=kind, scenario=sc['name'], delta=delta, cluster_years=blocks, n_clusters=int(cl.max()) + 1, delta_true=d_true, delta_true_mcse=d_true_mcse,
                ratio_cr1_iid_quantiles={q: float(np.quantile(ratio, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
                p_ratio_le_0_59=float((ratio <= 0.59).mean()), p_ratio_le_0_72=float((ratio <= 0.72).mean()),
                reps=reps, N=int(yr.size),
                K=int(yr.max()) + 1, rejection={m: v / reps for m, v in rej.items()},
                coverage={m: v / reps for m, v in cov.items()}, sd_delta_hat=sd,
                mean_se={k: float(np.mean(v)) for k, v in ses.items()},
                mean_ratio_cr1_iid=float(np.mean(np.array(ses['se_cr1']) / np.array(ses['se_iid']))),
                true_ratio_sd_iid=sd / float(np.mean(ses['se_iid'])), mean_df_bm=float(np.mean(dfs)),
                mc_se=float(np.sqrt(0.05 * 0.95 / reps)))


SCENARIOS = [
    dict(name='iid', tp=0.0, ts=0.0, tm=0.0, tc=0.0, phi=0.0, rho=0.85),
    dict(name='shared', tp=0.3, ts=0.3, tm=0.0, tc=0.0, phi=0.0, rho=0.85),
    dict(name='specific', tp=0.3, ts=0.3, tm=0.15, tc=0.0, phi=0.0, rho=0.85),
    dict(name='station_year', tp=0.3, ts=0.3, tm=0.0, tc=0.25, phi=0.0, rho=0.85),
    dict(name='dipole', tp=0.3, ts=0.3, tm=0.0, tc=0.0, phi=0.0, rho=0.85, td=1.0, beta=0.5),
    dict(name='specific_ar1', tp=0.3, ts=0.3, tm=0.15, tc=0.0, phi=0.5, rho=0.85),
    dict(name='station_year_specific_ar1', tp=0.3, ts=0.3, tm=0.15, tc=0.25, phi=0.5, rho=0.85),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=1000); ap.add_argument('--B', type=int, default=399)
    ap.add_argument('--seed', type=int, default=48); ap.add_argument('--out', default=OUT)
    ap.add_argument('--blocks', type=int, default=1, help='cluster = blocks of this many consecutive years')
    ap.add_argument('--scenarios', nargs='*', default=None, help='restrict to these scenarios on the real mask')
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    mask = real_mask()
    sc_by = {s['name']: s for s in SCENARIOS}
    if a.scenarios:
        cells = [('real', sc_by[n], d) for n in a.scenarios for d in (0.0, 0.02)]
    else:
        cells = [('real', sc, d) for sc in SCENARIOS for d in (0.0, 0.02)]
        cells += [(k, sc_by[n], d) for n in ('specific', 'station_year') for k in ('K10', 'K38', 'm2', 'm4') for d in (0.0, 0.02)]
    rows = []
    for kind, sc, d in cells:
        t0 = time.time()
        r = run_cell(rng, mask, kind, sc, d, a.reps, a.B, a.blocks); rows.append(r)
        print(f"{kind:5s} {sc['name']:20s} d={d:.2f} | reject " + ' '.join(f"{m} {v:.3f}" for m, v in r['rejection'].items())
              + f" | cr1/iid {r['mean_ratio_cr1_iid']:.2f} true sd/iid {r['true_ratio_sd_iid']:.2f} df_BM {r['mean_df_bm']:.1f}"
              + f" | {time.time() - t0:.0f}s", flush=True)
        json.dump(dict(config=vars(a), scenarios=SCENARIOS, rows=rows), open(a.out, 'w'), indent=2)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
