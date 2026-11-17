#!/usr/bin/env python3
"""
make_forest_plot.py - forest plot of every paired seed-mean comparison of the v2 paper, three intervals per row
=============================================================================================================
Reads the v2 summary (no recomputation, so the figure matches the tables exactly):
  results/local_exp42_v2_summary.json   written by code/local_experiments/exp42_v2_summary.py
The rows are those of Tables 3, 6 and 7 of the manuscript (tab:paired_ci, tab:qnn_paired, tab:controls).
For each pair it draws the DeLong 95 % interval, the calendar-year block-bootstrap interval and the
years x seeds interval, offset vertically, with the point estimate. A missing pair is an error.

Usage (from the repository root):
  python scripts/make_forest_plot.py                      # -> paper/figs/fig_forest.pdf and .png
  python scripts/make_forest_plot.py --out paper/figs/fig_forest
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY = os.path.join(ROOT, 'results', 'local_exp42_v2_summary.json')
MINUS = '−'

# (group, label A - B, key in exp42 'pairs', flipped): flipped means the archive stores B - A
ROWS = [
    ('Classical models', 'RF - XGBoost', 'RF-XGB', False),
    ('Classical models', 'RF - LSTM', 'RF-PureLSTM', False),
    ('Classical models', 'XGBoost - LSTM', 'XGB-PureLSTM', False),
    ('Random Forest vs hybrids', 'RF - LSTM+QNN', 'RF-LSTM_QNN', False),
    ('Random Forest vs hybrids', 'RF - LSTM+QTL', 'RF-LSTM_QTL', False),
    ('Hybrids vs LSTM', 'LSTM - LSTM+QNN', 'PureLSTM-LSTM_QNN', False),
    ('Hybrids vs LSTM', 'LSTM+QTL - its extractor', 'LSTM_QTL-PureLSTM', False),
    ('Circuit vs controls', 'Linear-4D - LSTM+QNN', 'LSTM_QNN-CTRL_LINEAR4D', True),
    ('Circuit vs controls', 'Fourier-1 - LSTM+QNN', 'FOURIER1-LSTM_QNN', False),
    ('Circuit vs controls', 'Fourier-2 - LSTM+QNN', 'FOURIER2-LSTM_QNN', False),
    ('Circuit vs controls', 'Fourier-3 - LSTM+QNN', 'FOURIER3-LSTM_QNN', False),
    ('Circuit vs controls', 'Fourier-4 - LSTM+QNN', 'FOURIER4-LSTM_QNN', False),
    ('Circuit vs controls', 'Full Fourier-1 - LSTM+QNN', 'FOURIER_FULL1-LSTM_QNN', False),
    ('Circuit vs controls', 'Full Fourier-2 - LSTM+QNN', 'FOURIER_FULL2-LSTM_QNN', False),
    ('Circuit vs controls', 'Full Fourier-2 - Fourier-2', 'FOURIER_FULL2-FOURIER2', False),
    ('Circuit vs controls', 'LSTM+QTL - QTL-classical', 'LSTM_QTL-QTL_CLASSICAL', False),
    ('Circuit vs controls', 'LSTM+QTL - QTL tensor-81', 'LSTM_QTL-QTL_TENSOR1', False),
    ('Circuit vs controls', 'Frozen random - LSTM+QNN', 'FROZEN_RANDOM-LSTM_QNN', False),
    ('Circuit vs controls', 'Frozen zero-angle - LSTM+QNN', 'FROZEN_IDENTITY-LSTM_QNN', False),
]
# validated categorical slots 1-3 (dataviz reference palette, light mode); aqua is below 3:1 contrast,
# so every procedure also has its own marker shape and the legend is always shown
PROC = [('DeLong (iid)', 'ci_delong', '#2a78d6', 'o'),
        ('Year-block bootstrap', 'ci_block', '#eb6834', 's'),
        ('Years × seeds bootstrap', 'ci_two_way', '#1baf7a', '^')]
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def entry(pairs, key, flipped):
    """Point estimate and the three intervals of one row, in the A - B orientation of its label."""
    if key not in pairs:
        raise KeyError(f'{key} not in {SUMMARY}; rerun exp42_v2_summary.py')
    e = pairs[key]
    out = {'delta': e['gap']}
    for _, k, _, _ in PROC:
        out[k] = list(e[k])
    if flipped:
        out = {k: ([-v[1], -v[0]] if isinstance(v, list) else -v) for k, v in out.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(ROOT, 'paper', 'figs', 'fig_forest'))
    a = ap.parse_args()
    with open(SUMMARY, encoding='utf-8') as fh:
        pairs = json.load(fh)['pairs']
    rows = [(g, lab.replace(' - ', f' {MINUS} '), entry(pairs, k, fl)) for g, lab, k, fl in ROWS]

    plt.rcParams.update({'font.size': 8.5, 'font.family': 'DejaVu Sans', 'axes.edgecolor': MUTED,
                         'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': INK, 'pdf.fonttype': 42})
    groups = []
    for g, _, _ in rows:
        if g not in groups:
            groups.append(g)
    y, ticks, labels, heads = 0.0, [], [], []
    pos = []
    for g in groups:
        heads.append((y, g))
        y -= 0.9
        for gg, lab, e in rows:
            if gg == g:
                pos.append((y, e))
                ticks.append(y)
                labels.append(lab)
                y -= 1.0
        y -= 0.4
    fig, ax = plt.subplots(figsize=(6.8, 0.27 * (len(rows) + 2 * len(groups)) + 1.2))
    ax.axvline(0, color=MUTED, lw=1.0, zorder=1)
    for yy, e in pos:
        for j, (_, key, col, mk) in enumerate(PROC):
            off = (1 - j) * 0.22
            ax.plot(e[key], [yy + off] * 2, color=col, lw=2.0, solid_capstyle='round', zorder=2)
            ax.plot(e['delta'], yy + off, marker=mk, ms=4.5, color=col, mec='#fcfcfb', mew=0.8, zorder=3)
    ax.set_yticks(ticks + [yy for yy, _ in heads])
    ax.set_yticklabels(labels + [g for _, g in heads])
    for t in ax.get_yticklabels()[len(ticks):]:
        t.set_fontweight('bold')
    ax.tick_params(axis='y', length=0)
    ax.set_ylim(y + 0.4, 0.6)
    ax.grid(axis='x', color=GRID, lw=0.6, zorder=0)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.set_xlabel(f'Difference of 15-seed mean test AUC (A {MINUS} B), 95 % intervals')
    handles = [plt.Line2D([], [], color=c, lw=2, marker=m, ms=4.5, mec='#fcfcfb') for _, _, c, m in PROC]
    ax.legend(handles, [p[0] for p in PROC], loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
              fontsize=8, handlelength=2.2)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(f'{a.out}.pdf', bbox_inches='tight', metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(f'{a.out}.png', dpi=200, bbox_inches='tight', metadata={'Software': None})
    print(f'wrote {a.out}.pdf/.png with {len(rows)} pairs')


if __name__ == '__main__':
    main()
