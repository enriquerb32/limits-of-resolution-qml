#!/usr/bin/env python3
"""
make_forest_plot.py - forest plot of every paired seed-mean comparison of the v2 paper, three intervals per row
=============================================================================================================
Reads the v2 summary (no recomputation, so the figure matches the tables exactly):
  results/local_exp42_v2_summary.json   written by code/local_experiments/exp42_v2_summary.py
The rows are those of Tables 2, 3 and S24 of the manuscript (tab:paired_ci, tab:qnn_paired, tab:controls); the
LSTM+QTL rows are in the supplement (tab:qtl_paired).
For each pair it draws the primary 95 % interval (CR2 with Bell-McCaffrey degrees of freedom, calendar-year
clusters), the DeLong interval and the years x seeds interval, offset vertically, with the point estimate. A missing pair is an error.

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
    ('Hybrid against classical pipelines', 'RF - LSTM+QNN', 'RF-LSTM_QNN', False),
    ('Hybrid against classical pipelines', 'LSTM - LSTM+QNN', 'PureLSTM-LSTM_QNN', False),
    ('Circuit against interface controls', 'Linear-4D - LSTM+QNN', 'LSTM_QNN-CTRL_LINEAR4D', True),
    ('Circuit against interface controls', 'Fourier-2 - LSTM+QNN', 'FOURIER2-LSTM_QNN', False),
    ('Circuit against interface controls', 'Full Fourier-1 - LSTM+QNN', 'FOURIER_FULL1-LSTM_QNN', False),
    ('Circuit against interface controls', 'Full Fourier-2 - LSTM+QNN', 'FOURIER_FULL2-LSTM_QNN', False),
    ('Circuit against interface controls', 'Frozen random - LSTM+QNN', 'FROZEN_RANDOM-LSTM_QNN', False),
    ('Circuit against interface controls', 'Frozen zero-angle - LSTM+QNN', 'FROZEN_IDENTITY-LSTM_QNN', False),
]
# only the primary procedure carries the accent colour (as in the calibration figure and the inference map); the
# two references are neutral greys and also differ by marker shape, so identity never rests on colour alone
PROC = [('CR2, Bell–McCaffrey df (primary)', 'ci_cr2', '#2a78d6', 's'),
        ('DeLong (independence)', 'ci_delong', '#8a8985', 'o'),
        ('Years × seeds bootstrap', 'ci_two_way', '#bdbcb8', '^')]
# resolved by CR2 before but not after Holm and not robust to clusters of two consecutive years (Tables 3 and 6)
EXPLORATORY = {'RF-PureLSTM', 'RF-LSTM_QNN'}
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
    rows = [(g, lab.replace(' - ', f' {MINUS} ') + (' †' if k in EXPLORATORY else ''), entry(pairs, k, fl))
            for g, lab, k, fl in ROWS]

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
    fig, ax = plt.subplots(figsize=(6.8, 0.21 * (len(rows) + 2 * len(groups)) + 1.0))
    ax.axvline(0, color=MUTED, lw=1.0, zorder=1)
    for yy, e in pos:
        for j, (_, key, col, mk) in enumerate(PROC):
            off = (1 - j) * 0.24
            prim = j == 0
            ax.plot(e[key], [yy + off] * 2, color=col, lw=2.4 if prim else 1.6, solid_capstyle='round',
                    zorder=3 if prim else 2)
            ax.plot(e['delta'], yy + off, marker=mk, ms=5.0 if prim else 4.2, color=col, mec='#fcfcfb', mew=0.8,
                    zorder=4 if prim else 3)
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
