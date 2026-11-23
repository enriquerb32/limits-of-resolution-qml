#!/usr/bin/env python3
"""
exp62_floored_cr2_observed.py - the floored CR2 interval on the contrasts of the paper (archived predictions only)
=================================================================================================================
exp61 shows that when the cluster/iid ratio of a contrast is well below 1, the
CR2 interval undercovers, because a low ratio mostly signals an underestimated cluster variance. The floored interval
uses SE = max(SE_CR2, SE_iid) with the Bell-McCaffrey degrees of freedom. This script applies it to the primary
families, the interface contrasts, the frozen circuits and LSTM+QTL (QTL_FIXED) against Random Forest, and reports which verdicts
it changes.

  python code/local_experiments/exp62_floored_cr2_observed.py
Output: results/local_exp62_floored_cr2_observed.json
"""
import json
import os
import sys

from scipy.stats import t as student_t

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp55_review_closing import R, YEARS, archived_probs  # noqa: E402
from qml_resolution import cluster_robust_delta  # noqa: E402

PAIRS = [('RF', 'XGB'), ('RF', 'PureLSTM'), ('XGB', 'PureLSTM'), ('RF', 'LSTM_QNN'), ('CTRL_LINEAR4D', 'LSTM_QNN'),
         ('PureLSTM', 'LSTM_QNN'), ('FOURIER2', 'LSTM_QNN'), ('FOURIER_FULL1', 'LSTM_QNN'), ('FOURIER_FULL2', 'LSTM_QNN'),
         ('FROZEN_RANDOM', 'LSTM_QNN'), ('FROZEN_IDENTITY', 'LSTM_QNN'), ('RF', 'QTL_FIXED')]  # QTL_FIXED is LSTM+QTL of the paper


def main():
    y, P = archived_probs()
    out = {}
    for a, b in PAIRS:
        r = cluster_robust_delta(y, P[a], P[b], YEARS)
        q = student_t.ppf(0.975, r.df)
        se_fl = max(r.se_cr2, r.se_iid)
        cr2 = (r.delta - q * r.se_cr2, r.delta + q * r.se_cr2)
        fl = (r.delta - q * se_fl, r.delta + q * se_fl)
        p_fl = float(2 * student_t.sf(abs(r.delta) / se_fl, r.df))
        k = f'{a}-{b}'
        out[k] = dict(delta=r.delta, ratio_cr1_iid=r.se_cr1 / r.se_iid, ratio_cr2_iid=r.se_cr2 / r.se_iid,
                      ci_cr2=list(cr2), p_cr2=r.p_value, ci_floor=list(fl), p_floor=p_fl,
                      resolved_cr2=bool(cr2[0] > 0 or cr2[1] < 0), resolved_floor=bool(fl[0] > 0 or fl[1] < 0))
        v = out[k]
        print(f'{k:28s} {r.delta:+.4f} ratio {v["ratio_cr2_iid"]:.2f} | CR2 p {r.p_value:.3f} '
              f'{"R" if v["resolved_cr2"] else "-"} | floor p {p_fl:.3f} {"R" if v["resolved_floor"] else "-"}')
    out['changed'] = [k for k, v in out.items() if isinstance(v, dict) and v['resolved_cr2'] != v['resolved_floor']]
    print('verdicts changed by the floor:', out['changed'])
    path = os.path.join(R, 'local_exp62_floored_cr2_observed.json')
    json.dump(out, open(path, 'w'), indent=2)
    print('wrote', path)


if __name__ == '__main__':
    main()
