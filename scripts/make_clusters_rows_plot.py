#!/usr/bin/env python3
"""make_clusters_rows_plot.py - resolution against rows per cluster and number of clusters (exp54)
====================================================================================================
One panel per dependence pattern; x: rows per cluster m, y: true 80 %-power MDE of the design,
(z_0.975 + z_0.80) x SD of the estimated AUC difference over replications (at a true difference of 0.02);
one line per number of independent clusters K. Dotted: what N^{-1/2} scaling from the smallest m would give, so the
gap between a line and its dotted continuation is the resolution that added rows do not buy.

  python scripts/make_clusters_rows_plot.py      # -> paper/figs/fig_clusters_rows.pdf and .png
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'results', 'local_exp54_power_clusters_rows.json')
OUT = os.path.join(ROOT, 'paper', 'figs', 'fig_clusters_rows')
TITLES = {'iid': 'Independent test cases', 'specific': 'Scorer-specific year effects',
          'station_year': 'Year × station effects', 'specific_ar1': 'Year effects, AR(1) between years'}
# sequential blues, light to dark with K (dataviz sequential ramp)
COLORS = ['#9ec5f0', '#5d9de6', '#2a78d6', '#164b8c']
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def main():
    R = json.load(open(SRC, encoding='utf-8'))
    rows = [r for r in R['rows'] if r['delta'] == 0.02]
    plt.rcParams.update({'font.size': 8, 'font.family': 'DejaVu Sans', 'axes.edgecolor': MUTED,
                         'xtick.color': MUTED, 'ytick.color': MUTED, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.6), sharex=True, sharey=True)
    for ax, (sc, title) in zip(axes.flat, TITLES.items()):
        for K, col in zip(R['K_grid'], COLORS):
            pts = sorted((r['m'], r['mde_true']) for r in rows if r['scenario'] == sc and r['K'] == K)
            m, mde = np.array(pts).T
            ax.plot(m, mde, color=col, lw=1.8, marker='o', ms=3.6, zorder=3, label=f'K = {K}')
            ax.plot(m, mde[0] * np.sqrt(m[0] / m), color=col, lw=0.9, ls=(0, (1, 2)), zorder=2)
        ax.set_title(title, fontsize=8.5, loc='left', color=INK)
        ax.set_xscale('log', base=2); ax.set_yscale('log')
        ax.set_xticks(R['m_grid']); ax.set_xticklabels([str(m) for m in R['m_grid']])
        ax.set_yticks([0.01, 0.02, 0.05, 0.1]); ax.set_yticklabels(['0.01', '0.02', '0.05', '0.1'])
        ax.minorticks_off()
        ax.grid(color=GRID, lw=0.5, zorder=0)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel('Rows per cluster m')
    for ax in axes[:, 0]:
        ax.set_ylabel('True 80 %-power MDE')
    h, lab = axes[0, 0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color=MUTED, lw=0.9, ls=(0, (1, 2)))); lab.append('$N^{-1/2}$ from m = 12')
    fig.legend(h, lab, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False, fontsize=8,
               title='independent clusters', title_fontsize=8)
    fig.tight_layout()
    fig.savefig(f'{OUT}.pdf', bbox_inches='tight', metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(f'{OUT}.png', dpi=200, bbox_inches='tight', metadata={'Software': None})
    print('wrote', OUT)


if __name__ == '__main__':
    main()
