#!/usr/bin/env python3
"""
exp54_power_clusters_rows.py - does resolution come from clusters or from rows? (review of 2026-11-19)
=====================================================================================================
The binormal mechanism of exp48 (same parameters and scenarios) on synthetic test designs that vary, separately,
the number of independent clusters K (years) and the number of rows per cluster m (station-months of a year, spread
evenly over the four stations of the panel, with their test base rates). For each design and dependence pattern:

  * the true resolution of the design: MDE_true = (z_0.975 + z_0.80) * SD of the estimated difference over
    replications (a property of the design, whatever procedure is used);
  * the power of the primary procedure (CR2 with Bell-McCaffrey df) and of DeLong at a true difference of 0.02 AUC,
    and the coverage of both (calibration);
  * the type-I error of both at a true difference of 0.

Single scorers, as in exp48; the truth is the analytic difference for new independent years.

  python code/local_experiments/exp54_power_clusters_rows.py [--reps 1000]
Output: results/local_exp54_power_clusters_rows.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.stats import norm, t as student_t

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from exp48_primary_calibration import SCENARIOS, bm_df, components, real_mask, simulate, true_auc  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'local_exp54_power_clusters_rows.json')
STATIONS = ['Alicante', 'Barcelona', 'San_Javier', 'Valencia']
K_GRID = [10, 19, 38, 76]
M_GRID = [12, 24, 48, 96]
SC_NAMES = ['iid', 'specific', 'station_year', 'specific_ar1']


def design(K, m):
    st = np.tile(np.resize(STATIONS, m), K)
    yr = np.repeat(np.arange(K), m)
    return st, yr


def delong_cr2(y, sa, sb, yr):
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    A, a10, a01 = components(y, sa); B, b10, b01 = components(y, sb)
    d = A - B
    psi = np.empty(y.size); psi[pos] = (a10 - b10 - d) / n1; psi[~pos] = (a01 - b01 - d) / n0
    _, inv = np.unique(yr, return_inverse=True)
    S = np.bincount(inv, weights=psi); n_g = np.bincount(inv).astype(float); N = y.size
    se_iid = np.sqrt((psi[pos] ** 2).sum() * n1 / (n1 - 1) + (psi[~pos] ** 2).sum() * n0 / (n0 - 1))
    se_cr2 = np.sqrt(((S / np.sqrt(1 - n_g / N)) ** 2).sum())
    return d, se_iid, se_cr2


def run_cell(rng, K, m, sc, delta, reps, p_station):
    st, yr = design(K, m)
    ts, tm, tc = sc['ts'], sc['tm'], sc['tc']
    scale = np.sqrt(2 * (1 + ts ** 2 + tm ** 2 + tc ** 2))
    mu_b = scale * norm.ppf(0.66); mu_a = scale * norm.ppf(0.66 + delta)
    d_true = float(true_auc(mu_a, ts, tm, tc) - true_auc(mu_b, ts, tm, tc))
    df = bm_df(np.bincount(yr).astype(float), yr.size)
    q_cr2, q_iid = student_t.ppf(0.975, df), norm.ppf(0.975)
    D, SI, SC = np.empty(reps), np.empty(reps), np.empty(reps)
    for i in range(reps):
        y, sa, sb = simulate(rng, st, yr, p_station, (mu_a, mu_b), sc['rho'], sc['tp'], ts, tm, tc, sc['phi'])
        D[i], SI[i], SC[i] = delong_cr2(y, sa, sb, yr)
    rej = lambda se, q: float(np.mean(np.abs(D) > q * se))                       # noqa: E731
    cov = lambda se, q: float(np.mean(np.abs(D - d_true) <= q * se))             # noqa: E731
    sd = float(np.std(D, ddof=1))
    return dict(K=K, m=m, N=int(yr.size), scenario=sc['name'], delta=delta, delta_true=d_true, reps=reps, df_bm=df,
                sd_delta_hat=sd, mde_true=float((norm.ppf(0.975) + norm.ppf(0.80)) * sd),
                mde_cr2_mean=float((q_cr2 + student_t.ppf(0.80, df)) * SC.mean()),
                mde_delong_mean=float((q_iid + norm.ppf(0.80)) * SI.mean()),
                rejection=dict(cr2_bm=rej(SC, q_cr2), delong=rej(SI, q_iid)),
                coverage=dict(cr2_bm=cov(SC, q_cr2), delong=cov(SI, q_iid)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=1000); ap.add_argument('--seed', type=int, default=54)
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    p_station = real_mask()['p_station']
    scs = [s for s in SCENARIOS if s['name'] in SC_NAMES]
    rows, t0 = [], time.time()
    for sc in scs:
        for K in K_GRID:
            for m in M_GRID:
                for delta in (0.0, 0.02):
                    r = run_cell(rng, K, m, sc, delta, a.reps, p_station)
                    rows.append(r)
                    print(f'{sc["name"]:13s} K={K:3d} m={m:3d} d={delta:.2f} | MDE true {r["mde_true"]:.4f} '
                          f'| power CR2 {r["rejection"]["cr2_bm"]:.3f} DeLong {r["rejection"]["delong"]:.3f} '
                          f'| cov CR2 {r["coverage"]["cr2_bm"]:.3f} DeLong {r["coverage"]["delong"]:.3f} '
                          f'| {time.time() - t0:.0f}s', flush=True)
    json.dump(dict(scenarios=scs, K_grid=K_GRID, m_grid=M_GRID, reps=a.reps, seed=a.seed,
                   mc_se_coverage=float(np.sqrt(0.05 * 0.95 / a.reps)), rows=rows), open(a.out, 'w'), indent=2)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
