#!/usr/bin/env python3
"""
exp63_floor_threshold.py - a concrete trigger for the floored CR2 interval, and the MDE of the procedure that decides
====================================================================================================================
Item 1 of the protocol (Table 6) says: when the cluster standard error falls well below the iid
one, either show a mechanism that produces it or decide with the standard error floored at the iid one. Item 2 says:
report the MDE of the procedure that takes the decision. This script

  A. makes "well below" concrete: the trigger is a cluster/iid ratio (CR1 / DeLong, the "Cluster/iid" column of the
     paper) at or below the 5th percentile of its distribution simulated on the real test mask with independent test
     cases (the 'iid' cell of exp61); it also reports that percentile in the other calibrated mechanisms of exp61 C;
  B. gives the operating characteristics of the triggered rule (floor only when the ratio is at or below the trigger,
     unfloored CR2 otherwise) in those mechanisms: coverage over all replications and given a ratio at or below the
     trigger, against the unfloored CR2 interval and the always-floored one;
  C. applies the rule to the contrasts of exp62 (archived predictions): which contrasts trigger the floor, the p-value
     and verdict of the procedure that then decides, and its 80 %-power MDE, (t_{0.975,nu} + t_{0.80,nu}) SE_used,
     with SE_used = SE_iid when the floor is triggered (floor binding) and SE_CR2 otherwise.

  python code/local_experiments/exp63_floor_threshold.py [--reps 5000] [--reps-q 20000]
Output: results/local_exp63_floor_threshold.json
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import t as student_t

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exp48_primary_calibration as e48  # noqa: E402
import exp61_cancellation_calibration as e61  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, 'results', 'local_exp63_floor_threshold.json')

CELLS = [dict(name='iid', tp=0.0, ts=0.0, rho=0.85), dict(name='shared', tp=0.3, ts=0.3, rho=0.85),
         dict(name='station_year_tc0.25', tp=0.3, ts=0.3, tc=0.25, rho=0.85),
         dict(name='zero_sum_tz0.5', tp=0.3, ts=0.3, tz=0.5, rho=0.85),
         dict(name='swap_tw0.1', tp=0.3, ts=0.3, tw=0.1, rho=0.85),
         dict(name='seasonal_shift', tp=0.3, ts=0.3, rho=0.85, t_seas=0.5, seas_kind='shift'),
         dict(name='seasonal_signal', tp=0.3, ts=0.3, rho=0.85, t_seas=0.5, seas_kind='signal'),
         dict(name='seasonal_signal_strong', tp=0.3, ts=0.3, rho=0.85, t_seas=1.0, seas_kind='signal'),
         dict(name='seasonal_signal_2', tp=0.3, ts=0.3, rho=0.85, t_seas=2.0, seas_kind='signal'),
         dict(name='seasonal_shift_2', tp=0.3, ts=0.3, rho=0.85, t_seas=2.0, seas_kind='shift')]


def setup():
    import pandas as pd
    mask = e48.real_mask()
    st, yr = e48.variant(mask, 'real')
    e61.MONTH = pd.read_csv(e48.META, parse_dates=['fecha'])['fecha'].dt.month.values
    return mask, st, yr - yr.min()


def part_a(rng, mask, st, yr0, reps):
    e61.SEASONAL['h'] = np.random.default_rng(1000).standard_normal((4, 12))
    sc = CELLS[0]
    rs = np.array([e61.lean(*e61.simulate(rng, st, yr0, mask['p_station'], e61.mus(sc, 0.0), sc), yr0)[1]
                   for _ in range(reps)])
    q = float(np.quantile(rs, 0.05))
    print(f'A. iid: ratio median {np.median(rs):.3f}, 5th percentile {q:.4f} ({reps} replications)', flush=True)
    return dict(reps=reps, median=float(np.median(rs)), q05=q, q01=float(np.quantile(rs, 0.01)),
                q10=float(np.quantile(rs, 0.10)))


def part_b(rng, mask, st, yr0, thr, reps):
    out = []
    for i, sc in enumerate(CELLS):
        e61.SEASONAL['h'] = np.random.default_rng(1000 + i).standard_normal((4, 12))
        for delta in (0.0, 0.02):
            mu = e61.mus(sc, delta)
            d_true, _ = e61.true_delta(rng, st, yr0, mask['p_station'], mu, sc)
            ratio, c_cr2, c_fl, c_rule = [], [], [], []
            for _ in range(reps):
                d, r, ci = e61.lean(*e61.simulate(rng, st, yr0, mask['p_station'], mu, sc), yr0, full=True)
                ratio.append(r)
                cov = {m: ci[m][0] <= d_true <= ci[m][1] for m in ('cr2_bm', 'cr2_floor')}
                c_cr2.append(cov['cr2_bm']); c_fl.append(cov['cr2_floor'])
                c_rule.append(cov['cr2_floor'] if r <= thr else cov['cr2_bm'])
            ratio, c_cr2, c_fl, c_rule = map(np.array, (ratio, c_cr2, c_fl, c_rule))
            low = ratio <= thr
            row = dict(scenario=sc['name'], delta=delta, reps=reps, q05_ratio=float(np.quantile(ratio, 0.05)),
                       share_triggered=float(low.mean()),
                       all=dict(cr2=float(c_cr2.mean()), always_floored=float(c_fl.mean()), rule=float(c_rule.mean())),
                       triggered=dict(n=int(low.sum()), cr2=float(c_cr2[low].mean()) if low.any() else None,
                                      rule=float(c_rule[low].mean()) if low.any() else None))
            out.append(row)
            print(f"B. {sc['name']:22s} d={delta:.2f} q05 {row['q05_ratio']:.3f} triggered {row['share_triggered']:.3f}"
                  f" | all: CR2 {row['all']['cr2']:.3f} floor {row['all']['always_floored']:.3f} rule "
                  f"{row['all']['rule']:.3f} | triggered: CR2 {row['triggered']['cr2']:.3f} rule "
                  f"{row['triggered']['rule']:.3f}", flush=True)
    return out


def part_c(thr):
    from exp55_review_closing import YEARS, archived_probs  # noqa: E402
    from exp62_floored_cr2_observed import PAIRS  # noqa: E402
    from qml_resolution import cluster_robust_delta  # noqa: E402
    y, P = archived_probs()
    out = {}
    for a, b in PAIRS:
        r = cluster_robust_delta(y, P[a], P[b], YEARS)
        ratio = r.se_cr1 / r.se_iid
        trig = bool(ratio <= thr)
        se = max(r.se_cr2, r.se_iid) if trig else r.se_cr2
        q, q80 = student_t.ppf(0.975, r.df), student_t.ppf(0.80, r.df)
        p = float(2 * student_t.sf(abs(r.delta) / se, r.df))
        k = f'{a}-{b}'
        out[k] = dict(delta=r.delta, ratio_cr1_iid=ratio, triggered=trig, df=r.df, p_cr2=r.p_value, p_decision=p,
                      resolved_cr2=bool(r.p_value < 0.05), resolved_decision=bool(p < 0.05),
                      mde_cr2=float((q + q80) * r.se_cr2), mde_decision=float((q + q80) * se),
                      mde_delong=float(2.80 * r.se_iid))
        v = out[k]
        print(f"C. {k:28s} ratio {ratio:.3f} {'FLOOR' if trig else '     '} p {r.p_value:.3f} -> {p:.3f} "
              f"MDE {v['mde_cr2']:.4f} -> {v['mde_decision']:.4f} (DeLong {v['mde_delong']:.4f})", flush=True)
    out['triggered'] = [k for k, v in out.items() if isinstance(v, dict) and v['triggered']]
    out['changed'] = [k for k, v in out.items() if isinstance(v, dict) and v['resolved_cr2'] != v['resolved_decision']]
    return out


def part_d(thr):
    """The same trigger on the Brier contrasts of exp58 (S-tab:brier_inference), from their archived SEs. The trigger was
    calibrated for AUC contrasts, so this is a sensitivity check, not a calibrated rule for the Brier score."""
    e58 = json.load(open(os.path.join(ROOT, 'results', 'local_exp58_brier_inference.json'), encoding='utf-8'))
    groups = {'raw/classical': e58['families']['classical'], 'raw/hybrid': e58['families']['hybrid'],
              'raw/interface': e58['interface'], 'raw/climatology': e58['climatology'],
              'recal/classical': e58['recalibrated']['classical'], 'recal/hybrid': e58['recalibrated']['hybrid'],
              'recal/climatology': e58['recalibrated']['climatology']}
    out = {}
    for g, d in groups.items():
        for k, v in d.items():
            trig = bool(v['ratio_cr1_iid'] <= thr)
            se = max(v['se_cr2'], v['se_iid']) if trig else v['se_cr2']
            p = float(2 * student_t.sf(abs(v['delta']) / se, v['df']))
            out[f'{g}/{k}'] = dict(ratio_cr1_iid=v['ratio_cr1_iid'], triggered=trig, p_cr2=v['p_cr2'], p_decision=p,
                                   resolved_cr2=bool(v['resolved']), resolved_decision=bool(p < 0.05))
            if trig:
                print(f"D. {g}/{k:24s} ratio {v['ratio_cr1_iid']:.3f} p {v['p_cr2']:.3f} -> {p:.3f}", flush=True)
    out['triggered'] = [k for k, v in out.items() if isinstance(v, dict) and v['triggered']]
    out['changed'] = [k for k, v in out.items() if isinstance(v, dict) and v['resolved_cr2'] != v['resolved_decision']]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=5000)
    ap.add_argument('--reps-q', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=63)
    ap.add_argument('--only-brier', action='store_true', help='add part D to an existing output')
    a = ap.parse_args()
    if a.only_brier:
        res = json.load(open(OUT, encoding='utf-8'))
        res['brier'] = part_d(res['threshold']['q05'])
        json.dump(res, open(OUT, 'w'), indent=2)
        return
    rng = np.random.default_rng(a.seed)
    mask, st, yr0 = setup()
    res = dict(config=vars(a), threshold=part_a(rng, mask, st, yr0, a.reps_q))
    thr = res['threshold']['q05']
    res['contrasts'] = part_c(thr)
    res['brier'] = part_d(thr)
    json.dump(res, open(OUT, 'w'), indent=2)
    res['operating'] = part_b(rng, mask, st, yr0, thr, a.reps)
    json.dump(res, open(OUT, 'w'), indent=2)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
