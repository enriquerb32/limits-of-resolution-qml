#!/usr/bin/env python3
"""make_sensitivity_plot.py - how stable the resolved verdicts are under the robustness checks of the paper
==========================================================================================================
One panel per resolved difference of 15-seed means: the two exploratory verdicts (Random Forest against the LSTM
and against LSTM+QNN) and, for contrast, Random Forest against LSTM+QTL and the frozen random circuit. Each row is
one check, drawn as the CR2 p-value (Bell-McCaffrey degrees of freedom) on a log axis; filled dots exclude zero at
0.05 (resolved), hollow dots do not. Nothing is recomputed; the values come from archived results:

  results/local_cr2_block_sensitivity.json   clusters of 1, 2 and 3 consecutive years
  results/local_cr2_leave_one_year.json      primary procedure without each of the 19 test years
  results/local_two_station_subset.json      Alicante and Barcelona only (441 rows)
  results/local_panel2_cr2.json              second panel (different test set; its difference is printed)

  python scripts/make_sensitivity_plot.py      # -> paper/figs/fig_sensitivity.pdf and .png
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')
OUT = os.path.join(ROOT, 'paper', 'figs', 'fig_sensitivity')
MINUS = '−'
# (panel title, key in the first-panel files, key in the second-panel file or None)
PAIRS = [(f'RF {MINUS} LSTM †', 'RF-PureLSTM', 'RF_2273-PureLSTM'),
         (f'RF {MINUS} LSTM+QNN †', 'RF-LSTM_QNN', 'RF_2273-LSTM_QNN'),
         (f'RF {MINUS} LSTM+QTL', 'RF-LSTM_QTL', None),
         (f'Frozen random circuit {MINUS} LSTM+QNN', 'FROZEN_RANDOM-LSTM_QNN', None)]
ROWS = ['Primary: 1-year clusters', '2-year clusters', '3-year clusters', 'Leave one year out (19)',
        'Alicante + Barcelona only', 'Second panel']
ACCENT, INK, MUTED, GRID, SURFACE = '#2a78d6', '#3d3c39', '#8a8985', '#e4e3df', '#fcfcfb'


def load(name):
    with open(os.path.join(RES, name), encoding='utf-8') as fh:
        return json.load(fh)


def main():
    blk, loo = load('local_cr2_block_sensitivity.json'), load('local_cr2_leave_one_year.json')
    two, p2 = load('local_two_station_subset.json')['pairs'], load('local_panel2_cr2.json')
    for _, k, k2 in PAIRS:   # the archived files must agree with each other on the full-panel verdict
        assert abs(blk[k]['1yr']['p'] - loo[k]['full']['p']) < 1e-9, k
        assert k2 is None or k2 in p2, k2
    plt.rcParams.update({'font.size': 8, 'font.family': 'DejaVu Sans', 'axes.edgecolor': MUTED,
                         'xtick.color': MUTED, 'ytick.color': INK, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.3), sharex=True, sharey=True)
    for ax, (title, k, k2) in zip(axes.flat, PAIRS):
        pts = {0: [blk[k]['1yr']['p']], 1: [blk[k]['2yr']['p']], 2: [blk[k]['3yr']['p']],
               3: [v['p'] for v in loo[k]['by_year'].values()], 4: [two[k]['p_cr2']]}
        notes = {}
        if k2 is not None:
            pts[5] = [p2[k2]['p_cr2']]
            notes[5] = f'Δ = {p2[k2]["delta"]:+.3f}'.replace('-', MINUS)
        ax.axvline(0.05, color=INK, lw=1.0, zorder=1)
        ax.axvline(0.05 / 3, color=MUTED, lw=0.9, ls=(0, (3, 2)), zorder=1)
        for i, ps in pts.items():
            y = len(ROWS) - 1 - i
            col = ACCENT if i == 0 else INK
            for p in ps:
                ax.plot(p, y, 'o', ms=4.6 if len(ps) == 1 else 3.6, mew=1.0, mec=col,
                        mfc=col if p < 0.05 else SURFACE, alpha=0.9 if len(ps) > 1 else 1.0, zorder=3)
            if i in notes:
                ax.annotate(notes[i], (ps[0], y), xytext=(-7, 0), textcoords='offset points', va='center', ha='right',
                            fontsize=7, color=MUTED)
        if k2 is None:
            ax.text(0.9, 0, 'not run', ha='right', va='center', fontsize=7, color=MUTED)
        ax.set_title(title, fontsize=8.5, loc='left', color=INK, fontweight='bold')
        ax.set_xscale('log')
        ax.set_xlim(1e-3, 1.3)
        ax.set_ylim(-0.6, len(ROWS) - 0.4)
        ax.set_yticks(range(len(ROWS)))
        ax.set_yticklabels(ROWS[::-1])
        ax.tick_params(axis='y', length=0)
        for s in ('top', 'right', 'left'):
            ax.spines[s].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel('CR2 $p$-value (log scale)')
        ax.set_xticks([0.001, 0.01, 0.05, 1])
        ax.set_xticklabels(['0.001', '0.01', '0.05', '1'])
        ax.minorticks_off()
    h = [plt.Line2D([], [], ls='', marker='o', ms=4.6, mec=INK, mfc=INK, label='resolved ($p<0.05$)'),
         plt.Line2D([], [], ls='', marker='o', ms=4.6, mec=INK, mfc=SURFACE, label='not resolved'),
         plt.Line2D([], [], color=INK, lw=1.0, label='0.05'),
         plt.Line2D([], [], color=MUTED, lw=0.9, ls=(0, (3, 2)), label='Holm first step (0.017)')]
    fig.legend(handles=h, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(h_pad=1.2, w_pad=0.8)
    fig.savefig(f'{OUT}.pdf', bbox_inches='tight', metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(f'{OUT}.png', dpi=200, bbox_inches='tight', metadata={'Software': None})
    for _, k, k2 in PAIRS:
        flips = loo[k]['years_that_flip_verdict']
        print(k, 'blocks', [round(blk[k][b]['p'], 3) for b in ('1yr', '2yr', '3yr')], 'loo flips', len(flips),
              'AB', round(two[k]['p_cr2'], 3), 'panel2', None if k2 is None else round(p2[k2]['p_cr2'], 3))


if __name__ == '__main__':
    main()
