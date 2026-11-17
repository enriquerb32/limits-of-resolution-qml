#!/usr/bin/env python3
"""
exp26_within_year_robustness.py - robustness and calibration of the within-year diagnostic of exp25 (no retraining)
===================================================================================================================
exp25 compares the cluster-robust (calendar-year) SE of each paired seed-mean AUC difference with the iid DeLong SE,
and tests the ratio against a permutation null in which the year labels are permuted across single rows. That null
also destroys serial dependence between consecutive months and the dependence between stations on the same date.
This script answers three questions a referee will ask.

  A. Block-permutation nulls. Dates (all stations of a date together) are cut into blocks of b consecutive months
     (b = 1, 3, 6), the blocks are shuffled, and the shuffled sequence is cut into pseudo-years of 12 dates.
     Serial and cross-station dependence up to b months is kept; only the alignment with calendar years is destroyed.
     Also: the 12 phase-shifted 12-month groupings (pseudo-years starting in each calendar month).
  B. Other clusterings and influence: station-year and station clusters; leave-one-year-out range of the ratio.
     Also: share of the iid variance carried by the month x class means of the influence values (a seasonal pattern
     that cancels within complete calendar years) and the ratio after removing it.
  C. Known-truth simulation. Labels and calendar months are taken from the panel; each replicate resamples the 19
     test years with replacement (new years). Scores are binormal with a shared component and model-specific noise,
     plus (i) model-specific year shifts and/or (ii) a fixed seasonal pattern in one model's scores. Reported: SD of
     the estimated difference over replicates, mean DeLong and cluster-robust SEs and their ratio, type-I error of
     DeLong (z) and of the cluster-robust t_{K-1} test around the true difference (mean over replicates).

Usage (from the repository root):  python code/local_experiments/exp26_within_year_robustness.py [--reps 2000]
Output: results/local_exp26_within_year_robustness.json
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from qml_resolution.auc import delong_components  # noqa: E402

NPZ = os.path.join(ROOT, 'results', 'json', 'results_hybrids_trainonly_fixet0_p60_28.1_probs.npz')
META = os.path.join(ROOT, 'results', 'test_meta.csv')
OUT = os.path.join(ROOT, 'results', 'local_exp26_within_year_robustness.json')
PAIRS = [('RF_2273', 'PureLSTM'), ('RF_2273', 'LSTM_QNN'), ('CTRL_LINEAR4D', 'LSTM_QNN'), ('PureLSTM', 'LSTM_QNN')]


def influence_from_components(y, va10, va01, vb10, vb01):
    """psi of the AUC difference; DeLong variance = sum psi^2 (up to (n-1)/n factors)."""
    delta = va10.mean() - vb10.mean()
    psi = np.zeros(y.size)
    psi[y] = (va10 - vb10 - delta) / y.sum()
    psi[~y] = (va01 - vb01 - delta) / (~y).sum()
    return delta, psi


def influence(y, pa, pb):
    ca, cb = delong_components(y, pa).seed_mean(), delong_components(y, pb).seed_mean()
    return influence_from_components(y, ca.v10[0], ca.v01[0], cb.v10[0], cb.v01[0])


def cl_se(psi, g):
    _, inv = np.unique(g, return_inverse=True)
    s = np.bincount(inv, weights=psi)
    k = s.size
    return np.sqrt(k / (k - 1) * (s ** 2).sum()), k


def date_block_groups(dates_idx, n_dates, b, rng):
    """pseudo-year per row: shuffle blocks of b consecutive dates, then cut into consecutive groups of 12 dates."""
    blocks = [np.arange(i, min(i + b, n_dates)) for i in range(0, n_dates, b)]
    order = np.concatenate([blocks[j] for j in rng.permutation(len(blocks))])
    pseudo = np.empty(n_dates, int)
    pseudo[order] = np.arange(n_dates) // 12
    return pseudo[dates_idx]


def fast_auc_components(y, s):
    """DeLong structural components for one score vector (ties counted 1/2)."""
    pos, neg = s[y], s[~y]
    allv = np.concatenate([pos, neg])
    from scipy.stats import rankdata
    r_all = rankdata(allv)
    r_pos = rankdata(pos)
    r_neg = rankdata(neg)
    v10 = (r_all[:pos.size] - r_pos) / neg.size
    v01 = 1.0 - (r_all[pos.size:] - r_neg) / pos.size
    return v10, v01


def simulate(y_by_year, mo_by_year, mu, rho, sigma_year, sigma_season, reps, rng, alpha=0.05):
    """New-year replicates: the 19 panel years are resampled with replacement (labels and calendar months kept).
    Scores: mu*y + shared N(0, rho) + model noise N(0, 1-rho) + model-specific year shift N(0, sigma_year^2)
    + (model A only) a fixed seasonal pattern h(month), h ~ N(0, sigma_season^2) drawn once and kept for all replicates.
    The true difference is estimated as the mean over replicates; type-I error refers to that value."""
    years = list(y_by_year)
    k = len(years)
    h = rng.normal(size=13) * sigma_season
    deltas, se_i, se_c = [], [], []
    for _ in range(reps):
        pick = rng.integers(0, k, k)
        y = np.concatenate([y_by_year[years[j]] for j in pick])
        mo = np.concatenate([mo_by_year[years[j]] for j in pick])
        g = np.concatenate([np.full(y_by_year[years[j]].size, c) for c, j in enumerate(pick)])
        if y.all() or (~y).all():
            continue
        n = y.size
        common = rng.normal(size=n) * np.sqrt(rho)
        sa = mu * y + common + rng.normal(size=n) * np.sqrt(1 - rho) + rng.normal(size=k)[g] * sigma_year + h[mo]
        sb = mu * y + common + rng.normal(size=n) * np.sqrt(1 - rho) + rng.normal(size=k)[g] * sigma_year
        a10, a01 = fast_auc_components(y, sa)
        b10, b01 = fast_auc_components(y, sb)
        delta, psi = influence_from_components(y, a10, a01, b10, b01)
        deltas.append(delta); se_i.append(np.sqrt((psi ** 2).sum())); se_c.append(cl_se(psi, g)[0])
    d, si, sc = np.asarray(deltas), np.asarray(se_i), np.asarray(se_c)
    truth = d.mean()
    rej_i = (np.abs(d - truth) / si > norm.ppf(1 - alpha / 2)).mean()
    rej_c = (np.abs(d - truth) / sc > student_t.ppf(1 - alpha / 2, k - 1)).mean()
    m = d.size
    return dict(reps=m, true_delta=float(truth), sd_delta=float(d.std(ddof=1)), mean_se_iid=float(si.mean()),
                mean_se_cluster=float(sc.mean()), mean_ratio=float((sc / si).mean()),
                typeI_delong=float(rej_i), typeI_cluster_t=float(rej_c), mc_se_typeI=float(np.sqrt(0.05 * 0.95 / m)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--perm', type=int, default=2000)
    ap.add_argument('--reps', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=2026)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    d = np.load(NPZ)
    meta = pd.read_csv(META, parse_dates=['fecha'])
    y = d['y_te'].astype(bool)
    assert (meta['y_te'].values.astype(bool) == y).all()
    year = meta['fecha'].dt.year.values
    station = meta['ciudad'].values
    dates = meta['fecha'].values.astype('datetime64[M]')
    udates = np.unique(dates)
    dates_idx = np.searchsorted(udates, dates)
    month_index = (udates - udates[0]).astype(int)                 # months since first test month
    assert (np.diff(month_index) >= 1).all()
    out = {'pairs': {}}

    for pa, pb in PAIRS:
        delta, psi = influence(y, d[pa + '_probs'], d[pb + '_probs'])
        se_iid = float(np.sqrt((psi ** 2).sum()))
        se_year, k = cl_se(psi, year)
        ratio = se_year / se_iid
        r = {'delta': float(delta), 'se_iid': se_iid, 'se_year': float(se_year), 'ratio_year': float(ratio)}
        # A. block-permutation nulls on dates (stations of a date kept together)
        for b in (1, 3, 6):
            null = np.array([cl_se(psi, date_block_groups(dates_idx, udates.size, b, rng))[0] / se_iid
                             for _ in range(a.perm)])
            r[f'null_dateblock_{b}m'] = dict(q025=float(np.quantile(null, .025)), median=float(np.median(null)),
                                             q975=float(np.quantile(null, .975)), p_lower=float((null <= ratio).mean()))
        # 12 phase-shifted pseudo-years (consecutive 12-month groups starting in each calendar month)
        cal = pd.DatetimeIndex(udates)
        phases = {}
        for start in range(1, 13):
            grp_by_date = np.array([(t.year * 12 + t.month - start) // 12 for t in cal])
            phases[start] = float(cl_se(psi, grp_by_date[dates_idx])[0] / se_iid)
        r['ratio_phase_shifted'] = phases
        # B. other clusterings and leave-one-year-out
        r['ratio_station_year'] = float(cl_se(psi, np.char.add(station.astype(str), year.astype(str)))[0] / se_iid)
        r['ratio_station'] = float(cl_se(psi, station)[0] / se_iid)
        # seasonal component: mean influence by calendar month and class, which cancels within complete years
        mo = meta['fecha'].dt.month.values
        f = np.zeros_like(psi)
        for mm in range(1, 13):
            for c in (True, False):
                kk = (mo == mm) & (y == c)
                f[kk] = psi[kk].mean() if kk.any() else 0.0
        resid = psi - f
        r['season_share_of_iid_var'] = float((f ** 2).sum() / (psi ** 2).sum())
        r['ratio_year_after_removing_season'] = float(cl_se(resid, year)[0] / np.sqrt((resid ** 2).sum()))
        loo = []
        for u in np.unique(year):
            m = year != u
            dl, ps = influence(y[m], d[pa + '_probs'][:, m], d[pb + '_probs'][:, m])
            loo.append(cl_se(ps, year[m])[0] / np.sqrt((ps ** 2).sum()))
        r['ratio_leave_one_year_out'] = [float(min(loo)), float(max(loo))]
        out['pairs'][f'{pa}-{pb}'] = r
        print(f"{pa}-{pb}: ratio(year) {ratio:.3f} | date-block nulls " +
              ' '.join(f"b={b}: [{r[f'null_dateblock_{b}m']['q025']:.2f},{r[f'null_dateblock_{b}m']['q975']:.2f}] "
                       f"p={r[f'null_dateblock_{b}m']['p_lower']:.3f}" for b in (1, 3, 6)) +
              f" | phases {min(phases.values()):.2f}-{max(phases.values()):.2f} | station-year {r['ratio_station_year']:.2f}"
              f" station {r['ratio_station']:.2f} | season share {r['season_share_of_iid_var']:.2f} ratio w/o season "
              f"{r['ratio_year_after_removing_season']:.2f} | LOYO [{loo and min(loo):.2f},{max(loo):.2f}]")

    # C. simulation in the observed regime
    y_by_year = {u: y[year == u] for u in np.unique(year)}
    mo_by_year = {u: meta['fecha'].dt.month.values[year == u] for u in np.unique(year)}
    mu, rho = np.sqrt(2) * norm.ppf(0.68), 0.9          # single-model AUC ~0.68; paired SE close to the panel's
    sims = {}
    for sy, ss in [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (0.0, 0.15), (0.0, 0.3), (0.0, 0.5), (0.5, 0.3)]:
        s_ = simulate(y_by_year, mo_by_year, mu, rho, sy, ss, a.reps, rng)
        sims[f'year{sy}_season{ss}'] = s_
        print(f"sim sigma_year={sy:.2f} sigma_season={ss:.2f}: ratio {s_['mean_ratio']:.2f} | SD(delta) {s_['sd_delta']:.4f} "
              f"mean SE iid {s_['mean_se_iid']:.4f} cluster {s_['mean_se_cluster']:.4f} | type I DeLong "
              f"{s_['typeI_delong']:.3f} cluster-t {s_['typeI_cluster_t']:.3f} (MC SE {s_['mc_se_typeI']:.3f})")
    out['simulation'] = {'mu': float(mu), 'rho': rho, 'scenarios': sims}
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
