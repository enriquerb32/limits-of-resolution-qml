"""Prints the LaTeX rows of tab:seas5 and the key numbers of the text from results/local_exp33_seas5_<label>.json.
Usage: python scripts/fill_seas5_table.py [--label orig] [--lead 1]"""
import argparse, json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument('--label', default='orig'); ap.add_argument('--lead', type=int, default=1)
ap.add_argument('--json'); a = ap.parse_args()
r = json.load(open(a.json or os.path.join(ROOT, 'results', f'local_exp33_seas5_{a.label}.json')))
L = a.lead
f = lambda x: f'{x:+.4f}'.replace('+-', '-')                                    # noqa: E731
ci = lambda c: f'$[{c[0]:+.3f},{c[1]:+.3f}]$'                                   # noqa: E731
pv = lambda p: '$<0.001$' if p < 0.001 else f'{p:.3f}'                          # noqa: E731
ROWS = [(f'SEAS5_QM_L{L}-CLIM', 'SEAS5 $-$ climatology'), (f'SEAS5_QM_L{3 - L}-CLIM', f'SEAS5 (lead {3 - L}) $-$ climatology'),
        (f'RF_2273-SEAS5_QM_L{L}', 'Random Forest $-$ SEAS5'), (f'LSTM_QNN-SEAS5_QM_L{L}', 'LSTM+QNN $-$ SEAS5'), None,
        (f'BLEND_RF_2273_L{L}-BLEND_CLIM_L{L}', 'Blend RF $-$ blend climatology'),
        (f'BLEND_XGB_L{L}-BLEND_CLIM_L{L}', 'Blend XGBoost $-$ blend climatology'),
        (f'BLEND_PureLSTM_L{L}-BLEND_CLIM_L{L}', 'Blend LSTM $-$ blend climatology'),
        (f'BLEND_CTRL_LINEAR4D_L{L}-BLEND_CLIM_L{L}', 'Blend Linear-4D $-$ blend climatology'),
        (f'BLEND_LSTM_QNN_L{L}-BLEND_CLIM_L{L}', 'Blend LSTM+QNN $-$ blend climatology'),
        (f'BLEND_LSTM_QNN_L{L}-BLEND_CTRL_LINEAR4D_L{L}', 'Blend LSTM+QNN $-$ blend Linear-4D')]
for row in ROWS:
    if row is None:
        print(r'\midrule'); continue
    k, lab = row
    p = r['pairs'].get(k)
    if p is None:
        print(f'% missing {k}'); continue
    print(f"{lab} & ${f(p['delta'])}$ & {pv(p['p_delong'])} & {ci(p['ci_block'])} & {ci(p['ci_two_way'])} & "
          f"{p['mde_delong']:.3f} / {p['mde_block']:.3f} \\\\")
b = r['brier']
g = lambda k: f"${b[k]['bss']:+.3f}$" if k in b else '--'                       # noqa: E731
print(r'\midrule')
print(r'\multicolumn{6}{@{}l}{Brier skill: SEAS5 ' + g(f'SEAS5_QM_L{L}_PLATT') + '; blend climatology ' + g(f'BLEND_CLIM_L{L}')
      + '; blend RF ' + g(f'BLEND_RF_2273_L{L}') + '; blend LSTM+QNN ' + g(f'BLEND_LSTM_QNN_L{L}')
      + r' (year-block intervals in \Cref{S-app:seas5})} \\')
au = r['auc']
print(f"\n% text: SEAS5 AUC lead 1 = {au.get('SEAS5_QM_L1', {}).get('mean', float('nan')):.3f}, "
      f"lead 2 = {au.get('SEAS5_QM_L2', {}).get('mean', float('nan')):.3f}; climatology {au['CLIM']['mean']:.3f}")
bl = [v['delta'] for k, v in r['pairs'].items() if k.startswith('BLEND_') and k.endswith(f'BLEND_CLIM_L{L}')]
md = [v['mde_delong'] for k, v in r['pairs'].items() if k.startswith('BLEND_') and k.endswith(f'BLEND_CLIM_L{L}')]
if bl:
    print(f'% text: blends - blend climatology range {min(bl):+.4f} .. {max(bl):+.4f}; MDE {min(md):.3f}-{max(md):.3f}')
