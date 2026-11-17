import json, sys, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
d = json.load(open(sys.argv[1]))['pairs']
rows = [('RF $-$ XGBoost', 'RF-XGB', 1), ('RF $-$ LSTM', 'RF-PureLSTM', 1), ('XGBoost $-$ LSTM', 'XGB-PureLSTM', 1), None,
        ('RF $-$ LSTM+QNN', 'RF-LSTM_QNN', 1), ('RF $-$ LSTM+QTL', 'RF-LSTM_QTL', 1), ('LSTM $-$ LSTM+QNN', 'PureLSTM-LSTM_QNN', 1),
        ('Linear-4D $-$ LSTM+QNN', 'LSTM_QNN-CTRL_LINEAR4D', -1), None,
        ('Fourier-1 $-$ LSTM+QNN', 'FOURIER1-LSTM_QNN', 1), ('Fourier-2 $-$ LSTM+QNN', 'FOURIER2-LSTM_QNN', 1),
        ('Fourier-3 $-$ LSTM+QNN', 'FOURIER3-LSTM_QNN', 1), ('Fourier-4 $-$ LSTM+QNN', 'FOURIER4-LSTM_QNN', 1),
        ('Full Fourier-1 $-$ LSTM+QNN', 'FOURIER_FULL1-LSTM_QNN', 1), ('Full Fourier-2 $-$ LSTM+QNN', 'FOURIER_FULL2-LSTM_QNN', 1),
        ('Frozen random $-$ LSTM+QNN', 'FROZEN_RANDOM-LSTM_QNN', 1), ('Frozen zero-angle $-$ LSTM+QNN', 'FROZEN_IDENTITY-LSTM_QNN', 1), None,
        ('LSTM+QTL $-$ QTL-classical', 'LSTM_QTL-QTL_CLASSICAL', 1), ('LSTM+QTL $-$ QTL tensor-81', 'LSTM_QTL-QTL_TENSOR1', 1)]
plt.rcParams.update({'font.size': 8, 'font.family': 'serif'})
fig, ax = plt.subplots(figsize=(6.3, 6.0))
y, labels, ys = 0, [], []
style = [('ci_delong', 'o', '#2a5c8a', 'DeLong'), ('ci_block', 's', '#c0622b', 'Year-block bootstrap'), ('ci_two_way', '^', '#5b8c3a', 'Years$\\times$seeds bootstrap')]
for r in rows:
    if r is None: y -= 0.6; continue
    lab, key, sg = r; v = d[key]; g = sg * v['gap']
    for j, (k, m, c, n) in enumerate(style):
        lo, hi = sorted([sg * v[k][0], sg * v[k][1]]); yy = y + (1 - j) * 0.22
        ax.plot([lo, hi], [yy, yy], color=c, lw=1.2); ax.plot(g, yy, m, color=c, ms=3.8, label=n if y == 0 else None)
    labels.append(lab); ys.append(y); y -= 1
ax.axvline(0, color='0.35', lw=0.8, zorder=0)
ax.set_yticks(ys); ax.set_yticklabels(labels); ax.set_xlabel('Difference of 15-seed mean test AUC (529 test rows)')
ax.grid(axis='x', color='0.9', lw=0.6); ax.set_axisbelow(True)
for s in ['top', 'right']: ax.spines[s].set_visible(False)
ax.legend(frameon=False, loc='lower right', fontsize=7)
fig.tight_layout(); fig.savefig(sys.argv[2]); print('ok')
