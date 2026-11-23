#!/usr/bin/env python3
"""
exp58_brier_inference.py - the inference of the paper with the Brier score (archived predictions only, no training)
===================================================================================================================
The protocol is stated for any metric with per-case influence values,
but every contrast of the paper is an AUC. This script repeats the primary contrasts, the interface controls and the
comparisons with the climatology on the difference of seed-mean Brier scores, A - B (negative: A is better), with the
same primary procedure (CR2, Bell-McCaffrey degrees of freedom, calendar-year clusters), the DeLong-analogue iid
standard error, Holm within the two primary families, and the 80 %-power MDE of the primary procedure.

For the Brier score the influence values are the centred per-case loss differences divided by N, so the CR2 leverage
correction of a mean is exact (for the AUC it is an approximation; Section 3.2).

  python code/local_experiments/exp58_brier_inference.py
Output: results/local_exp58_brier_inference.json
"""
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp55_review_closing import FAMILIES, J, R, YEARS, archived_probs, holm, mde_cr2  # noqa: E402
from qml_resolution import cluster_robust_brier_delta  # noqa: E402

INTERFACE = [('CTRL_LINEAR4D', 'LSTM_QNN'), ('FOURIER2', 'LSTM_QNN'), ('FOURIER_FULL1', 'LSTM_QNN'),
             ('FOURIER_FULL2', 'LSTM_QNN'), ('QTL_FIXED', 'QTL_CLASSICAL'), ('QTL_FIXED', 'QTL_TENSOR1')]
CLIMATOLOGY = [(m, 'CLIM') for m in ('RF', 'XGB', 'PureLSTM', 'CTRL_LINEAR4D', 'LSTM_QNN')]
# mean Brier scores printed in Table 5 of the paper (as trained), reproduced as a check
PAPER_BRIER = {'CLIM': 0.2184, 'RF': 0.2284, 'XGB': 0.2394, 'PureLSTM': 0.2427, 'CTRL_LINEAR4D': 0.2378,
               'LSTM_QNN': 0.2424}
PAPER_BRIER_PLATT = {'CLIM': 0.2155, 'RF': 0.2189, 'XGB': 0.2212, 'PureLSTM': 0.2215, 'CTRL_LINEAR4D': 0.2233,
                     'LSTM_QNN': 0.2245}
EPS = 1e-6


def platt(y_va, p_va, p_te):
    """Platt map fitted on the validation partition, as in exp28b."""
    lg = lambda p: np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS))).reshape(-1, 1)  # noqa: E731
    return LogisticRegression(C=1e6).fit(lg(p_va), y_va).predict_proba(lg(p_te))[:, 1]


def contrast(y, P, a, b):
    r = cluster_robust_brier_delta(y, P[a], P[b], YEARS); lo, hi = r.ci()
    return dict(delta=r.delta, ci_cr2=[lo, hi], p_cr2=r.p_value, df=r.df, se_cr2=r.se_cr2, se_iid=r.se_iid,
                ratio_cr1_iid=r.se_cr1 / r.se_iid, mde_cr2=mde_cr2(r), mde_iid=2.8016 * r.se_iid,
                resolved=bool(lo > 0 or hi < 0))


def recalibrated():
    """Same families and climatology contrasts after a per-seed Platt map fitted on the validation partition.

    Uses the refit of exp28 (the only archive with validation predictions of these models; Table 5, right block).
    """
    d = np.load(os.path.join(J, 'results_exp28_orig_probs.npz'))
    y_va, y = d['y_va'].astype(int), d['y_te'].astype(int)
    assert y.size == YEARS.size
    names = {'RF': 'RF_2273', 'XGB': 'XGB', 'PureLSTM': 'PureLSTM', 'CTRL_LINEAR4D': 'CTRL_LINEAR4D',
             'LSTM_QNN': 'LSTM_QNN'}
    Q = {m: np.array([platt(y_va, pv, pt) for pv, pt in zip(d[f'{k}_val'], d[f'{k}_probs'])])
         for m, k in names.items()}
    Q['CLIM'] = platt(y_va, d['clim_val'], d['clim_te'])[None]
    for m, v in PAPER_BRIER_PLATT.items():
        b = float(((Q[m] - y) ** 2).mean())
        assert abs(b - v) < 5e-5, f'{m}: recalibrated Brier {b:.5f} does not reproduce Table 5 ({v})'
    out = {}
    for fam, pairs in FAMILIES.items():
        res = {f'{a}-{b}': contrast(y, Q, a, b) for a, b in pairs}
        for (k, v), h in zip(res.items(), holm([v['p_cr2'] for v in res.values()])):
            v['resolved_after_holm'] = bool(h)
        out[fam] = res
    out['climatology'] = {f'{a}-{b}': contrast(y, Q, a, b) for a, b in CLIMATOLOGY}
    return out


def main():
    y, P = archived_probs()
    z = np.load(os.path.join(J, 'results_exp33_seas5_orig_probs.npz'))
    assert (z['y_te'].astype(int) == y).all(), 'climatology file is not aligned with the archived predictions'
    P['CLIM'] = z['CLIM_probs']
    for m, v in PAPER_BRIER.items():
        b = float(((P[m] - y) ** 2).mean())
        assert abs(b - v) < 5e-5, f'{m}: Brier {b:.5f} does not reproduce Table 5 ({v})'
    out = {'families': {}, 'interface': {}, 'climatology': {}}
    for fam, pairs in FAMILIES.items():
        res = {f'{a}-{b}': contrast(y, P, a, b) for a, b in pairs}
        rej = holm([v['p_cr2'] for v in res.values()])
        for (k, v), h in zip(res.items(), rej):
            v['resolved_after_holm'] = bool(h)
        out['families'][fam] = res
    out['interface'] = {f'{a}-{b}': contrast(y, P, a, b) for a, b in INTERFACE}
    out['climatology'] = {f'{a}-{b}': contrast(y, P, a, b) for a, b in CLIMATOLOGY}
    out['recalibrated'] = recalibrated()
    for part in ('families', 'interface', 'climatology', 'recalibrated'):
        items = ((k, v) for f in out[part].values() for k, v in f.items()) if part in ('families', 'recalibrated') \
            else out[part].items()
        for k, v in items:
            print(f'  {part:11s} {k:26s} {v["delta"]:+.4f}  CR2 [{v["ci_cr2"][0]:+.4f}, {v["ci_cr2"][1]:+.4f}] '
                  f'p {v["p_cr2"]:.3f}  MDE {v["mde_cr2"]:.4f} (iid {v["mde_iid"]:.4f})  '
                  f'cl/iid {v["ratio_cr1_iid"]:.2f}' + (f'  Holm {v["resolved_after_holm"]}'
                                                        if 'resolved_after_holm' in v else ''))
    path = os.path.join(R, 'local_exp58_brier_inference.json')
    json.dump(out, open(path, 'w'), indent=2)
    print('wrote', path)


if __name__ == '__main__':
    main()
