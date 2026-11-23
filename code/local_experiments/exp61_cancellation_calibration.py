#!/usr/bin/env python3
"""
exp61_cancellation_calibration.py - is CR2 calibrated where within-year cancellation between stations is strong?
================================================================================================================
The observed cluster/iid ratios of Random Forest against the LSTM and
LSTM+QNN (0.59 and 0.72) are rare under the calibration mechanisms of exp48 (Section 4.1). By the design effect, a ratio
r over clusters of m rows implies an equivalent intraclass correlation (r^2 - 1) / (m - 1) of the influence values,
bounded below by -1 / (m - 1); this script

  A. reports that bound for the test design (mean cluster size of the 19 years);
  B. searches a grid of mechanisms that produce within-year cancellation between stations, on the real test mask,
     for the cluster/iid ratios they generate (lean procedures: DeLong, CR1, CR2 with Bell-McCaffrey df);
  C. in the cells whose ratio distribution reaches the observed values, repeats the full calibration of exp48
     (every procedure, 95 % coverage of the difference for new years, rejection at delta = 0 and 0.02).

Mechanisms (scores S_k = mu_k y + e_k + ..., as in exp48), added to the exp48 terms:
  - station_year (exp48): tc * lambda_s * eta_{k,u}, scorer-specific, opposite loadings for the two station groups;
  - dipole (exp48): prevalence dipole td * lambda_s * a_u, with scorer A also scoring the base rate (beta);
  - zero_sum: scorer-specific station effects centred within each year over the stations present,
    tz * (g_{k,u,s} - mean_s g_{k,u,s}), so the advantage of A at one station is a disadvantage at another;
  - swap: the difference itself is redistributed within the year: A gets +tw * lambda_s * c_u and B gets
    -tw * lambda_s * c_u (perfectly opposite for the two scorers), the strongest form of cancellation;
  - seasonal (stage C only): a fixed station x month pattern of scorer A, the same every year, on its scores ('shift')
    or on its signal ('signal'); every year holds every month, so the years are balanced blocks and the cluster
    variance can legitimately fall below the iid one (the within-station structure of Table S9).
Stage C reports, for DeLong, CR1, CR2 and the floored CR2 interval (SE = max(SE_CR2, SE_iid), Bell-McCaffrey df),
the coverage over all replications and over those whose cluster/iid ratio is at most 0.72 or 0.59.

  python code/local_experiments/exp61_cancellation_calibration.py [--reps-search 300] [--reps 1000] [--B 399]
Output: results/local_exp61_cancellation_calibration.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.special import expit, logit
from scipy.stats import t as student_t

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exp48_primary_calibration as e48  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, 'results', 'local_exp61_cancellation_calibration.json')
OBSERVED = {'RF-PureLSTM': 0.59, 'RF-LSTM_QNN': 0.72}


def simulate(rng, st, yr, p_station, mu, sc):
    """exp48 mechanism plus the zero-sum and swap terms (no AR(1) here: phi = 0 in every cell of this script)."""
    K = int(yr.max()) + 1
    N = yr.size
    lam = np.array([e48.LAMBDA[s] for s in st])
    a, a2, w = (rng.standard_normal(K) for _ in range(3))
    base = logit(np.array([p_station[s] for s in st]))
    y = (rng.random(N) < expit(base + sc['tp'] * a[yr] + sc.get('td', 0.0) * lam * a2[yr])).astype(int)
    z = rng.standard_normal(N)
    stations = np.unique(st)
    s_idx = np.searchsorted(stations, st)
    c = rng.standard_normal(K)
    out = []
    for k in range(2):
        e = np.sqrt(sc['rho']) * z + np.sqrt(1 - sc['rho']) * rng.standard_normal(N)
        s = mu[k] * y + e + sc['ts'] * w[yr] + sc.get('tc', 0.0) * lam * rng.standard_normal(K)[yr]
        if sc.get('tz', 0.0):
            g = rng.standard_normal((K, stations.size))
            present = np.zeros((K, stations.size), bool)
            present[yr, s_idx] = True
            gm = np.where(present, g, 0.0).sum(1) / np.maximum(present.sum(1), 1)
            s = s + sc['tz'] * (g[yr, s_idx] - gm[yr])
        if sc.get('tw', 0.0):
            s = s + (1 if k == 0 else -1) * sc['tw'] * lam * c[yr]
        if k == 0 and sc.get('beta', 0.0):
            s = s + sc['beta'] * (base - base.mean())
        if k == 0 and sc.get('t_seas', 0.0):
            # fixed seasonal pattern of scorer A, the same in every year (drawn once per cell, see main): a shift of
            # its scores in each station x month ('shift') or a change of its signal strength there ('signal')
            h = SEASONAL['h'][s_idx, MONTH - 1]
            s = s + sc['t_seas'] * h * (y if sc.get('seas_kind') == 'signal' else 1.0)
        out.append(s)
    return y, out[0], out[1]


SEASONAL = {}   # filled in main: h[station, month], fixed for a cell
MONTH = None    # calendar month of each test row, aligned with the mask


def lean(y, sa, sb, yr, full=False):
    """Difference, cluster/iid ratio and (full=True) the DeLong, CR1-t_{K-1} and CR2-BM 95 % intervals."""
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    A, a10, a01 = e48.components(y, sa)
    Bc, b10, b01 = e48.components(y, sb)
    d = A - Bc
    psi = np.empty(y.size)
    psi[pos] = (a10 - b10 - d) / n1
    psi[~pos] = (a01 - b01 - d) / n0
    _, inv = np.unique(yr, return_inverse=True)
    K = inv.max() + 1
    S = np.bincount(inv, weights=psi)
    n_g = np.bincount(inv).astype(float)
    se_iid = np.sqrt((psi[pos] ** 2).sum() * n1 / (n1 - 1) + (psi[~pos] ** 2).sum() * n0 / (n0 - 1))
    se_cr1 = np.sqrt(K / (K - 1) * (S ** 2).sum())
    if not full:
        return d, se_cr1 / se_iid
    se_cr2 = np.sqrt(((S / np.sqrt(1 - n_g / y.size)) ** 2).sum())
    q_bm = student_t.ppf(0.975, e48.bm_df(n_g, y.size))
    q_k = student_t.ppf(0.975, K - 1)
    se_fl = max(se_cr2, se_iid)   # floored CR2: the cluster variance may not fall below the iid one
    ci = dict(delong=(d - 1.96 * se_iid, d + 1.96 * se_iid), cr1_t=(d - q_k * se_cr1, d + q_k * se_cr1),
              cr2_bm=(d - q_bm * se_cr2, d + q_bm * se_cr2), cr2_floor=(d - q_bm * se_fl, d + q_bm * se_fl))
    return d, se_cr1 / se_iid, ci


def true_delta(rng, st, yr, p_station, mu, sc, copies=300, draws=4):
    K = int(yr.max()) + 1
    out = []
    for _ in range(draws):
        ys, A, B = [], [], []
        for _c in range(copies):
            y, a, b = simulate(rng, st, yr, p_station, mu, sc)
            ys.append(y); A.append(a); B.append(b)
        y = np.concatenate(ys)
        out.append(e48.auc(y, np.concatenate(A)) - e48.auc(y, np.concatenate(B)))
    return float(np.mean(out)), float(np.std(out, ddof=1) / np.sqrt(draws))


def mus(sc, delta):
    from scipy.stats import norm
    scale = np.sqrt(2 * (1 + sc['ts'] ** 2 + sc.get('tc', 0.0) ** 2 + sc.get('tz', 0.0) ** 2 + sc.get('tw', 0.0) ** 2))
    return scale * norm.ppf(0.66 + delta), scale * norm.ppf(0.66)


GRID = [dict(name=f'station_year_tc{tc}', tp=0.3, ts=0.3, tc=tc, rho=0.85) for tc in (0.25, 0.5, 1.0)]
GRID += [dict(name=f'dipole_td{td}_b{b}', tp=0.3, ts=0.3, td=td, beta=b, rho=0.85) for td in (1.0, 2.0) for b in (0.5, 1.0)]
GRID += [dict(name=f'zero_sum_tz{tz}', tp=0.3, ts=0.3, tz=tz, rho=0.85) for tz in (0.5, 1.0)]
GRID += [dict(name=f'swap_tw{tw}', tp=0.3, ts=0.3, tw=tw, rho=0.85) for tw in (0.1, 0.25, 0.5)]
GRID += [dict(name='dipole_td2_b1_swap_tw0.25', tp=0.3, ts=0.3, td=2.0, beta=1.0, tw=0.25, rho=0.85)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps-search', type=int, default=300)
    ap.add_argument('--reps', type=int, default=1000)
    ap.add_argument('--B', type=int, default=399)
    ap.add_argument('--seed', type=int, default=61)
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--skip-search', action='store_true')
    ap.add_argument('--cells', nargs='*', default=None, help='restrict stage C to these cells')
    a = ap.parse_args()
    global MONTH
    rng = np.random.default_rng(a.seed)
    mask = e48.real_mask()
    st, yr = e48.variant(mask, 'real')
    import pandas as pd
    MONTH = pd.read_csv(e48.META, parse_dates=['fecha'])['fecha'].dt.month.values
    _, inv = np.unique(yr, return_inverse=True)
    m_bar = float(np.bincount(inv).mean())
    cfg = dict(vars(a), out=os.path.relpath(os.path.abspath(a.out), ROOT).replace(os.sep, '/'))
    out = dict(config=cfg, observed=OBSERVED, mean_cluster_size=m_bar,
               icc_lower_bound=-1 / (m_bar - 1),
               icc_implied={k: (r ** 2 - 1) / (m_bar - 1) for k, r in OBSERVED.items()}, search=[], full=[])
    print(f"A. mean rows per year {m_bar:.1f}; ICC bound {out['icc_lower_bound']:.4f}; implied "
          + ', '.join(f'{k} {v:.4f}' for k, v in out['icc_implied'].items()), flush=True)
    yr0 = yr - yr.min()
    for sc in ([] if a.skip_search else GRID):
        mu = mus(sc, 0.02)
        rs = np.array([lean(*simulate(rng, st, yr0, mask['p_station'], mu, sc), yr0)[1] for _ in range(a.reps_search)])
        row = dict(scenario=sc, median_ratio=float(np.median(rs)), q05=float(np.quantile(rs, 0.05)),
                   p_le_0_59=float((rs <= 0.59).mean()), p_le_0_72=float((rs <= 0.72).mean()))
        out['search'].append(row)
        print(f"B. {sc['name']:28s} ratio median {row['median_ratio']:.2f} q05 {row['q05']:.2f} "
              f"P(<=0.59) {row['p_le_0_59']:.3f} P(<=0.72) {row['p_le_0_72']:.3f}", flush=True)
        json.dump(out, open(a.out, 'w'), indent=2)
    # C: no mechanism of the grid makes the observed ratios common, so the question becomes conditional: in the
    # mechanisms under which CR2 is calibrated, how well do the intervals cover when the data show a ratio as low as
    # the observed one? (a low ratio then signals an underestimated cluster variance, not a different mechanism)
    by = {g['name']: g for g in GRID}
    cells = [dict(name='iid', tp=0.0, ts=0.0, rho=0.85), dict(name='shared', tp=0.3, ts=0.3, rho=0.85),
             by['station_year_tc0.25'], by['zero_sum_tz0.5'], by['swap_tw0.1'],
             # a fixed seasonal pattern of the advantage of A, the same every year: each year holds every month, so
             # the years are balanced blocks and the cluster variance can legitimately fall below the iid one
             dict(name='seasonal_shift', tp=0.3, ts=0.3, rho=0.85, t_seas=0.5, seas_kind='shift'),
             dict(name='seasonal_signal', tp=0.3, ts=0.3, rho=0.85, t_seas=0.5, seas_kind='signal'),
             dict(name='seasonal_signal_strong', tp=0.3, ts=0.3, rho=0.85, t_seas=1.0, seas_kind='signal'),
             dict(name='seasonal_signal_2', tp=0.3, ts=0.3, rho=0.85, t_seas=2.0, seas_kind='signal'),
             dict(name='seasonal_shift_2', tp=0.3, ts=0.3, rho=0.85, t_seas=2.0, seas_kind='shift')]
    if a.cells:
        cells = [c for c in cells if c['name'] in a.cells]
    for i, sc in enumerate(cells):
        SEASONAL['h'] = np.random.default_rng(1000 + i).standard_normal((4, 12))
        for delta in (0.0, 0.02):
            t0 = time.time()
            mu = mus(sc, delta)
            d_true, mcse = true_delta(rng, st, yr0, mask['p_station'], mu, sc)
            ratios, cov = [], {m: [] for m in ('delong', 'cr1_t', 'cr2_bm', 'cr2_floor')}
            rej = {m: [] for m in cov}
            for _ in range(a.reps):
                d, r, ci = lean(*simulate(rng, st, yr0, mask['p_station'], mu, sc), yr0, full=True)
                ratios.append(r)
                for m, (lo, hi) in ci.items():
                    cov[m].append(lo <= d_true <= hi)
                    rej[m].append(lo > 0 or hi < 0)
            ratios = np.array(ratios)
            res = dict(scenario=sc['name'], delta=delta, delta_true=d_true, delta_true_mcse=mcse, reps=a.reps,
                       median_ratio=float(np.median(ratios)))
            for lab, sel in (('all', np.ones_like(ratios, bool)), ('ratio_le_0_72', ratios <= 0.72),
                             ('ratio_le_0_59', ratios <= 0.59)):
                n = int(sel.sum())
                res[lab] = dict(n=n, coverage={m: (float(np.mean(np.array(v)[sel])) if n else None)
                                               for m, v in cov.items()},
                                rejection={m: (float(np.mean(np.array(v)[sel])) if n else None)
                                           for m, v in rej.items()})
            out['full'].append(res)
            print(f"C. {sc['name']:22s} d={delta:.2f} true {d_true:+.4f} ratio med {res['median_ratio']:.2f} | " +
                  ' | '.join(f"{lab} n={res[lab]['n']} " + ' '.join(
                      f"{m} {v:.3f}" for m, v in res[lab]['coverage'].items() if v is not None)
                      for lab in ('all', 'ratio_le_0_72', 'ratio_le_0_59')) + f' | {time.time() - t0:.0f}s', flush=True)
            json.dump(out, open(a.out, 'w'), indent=2)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
