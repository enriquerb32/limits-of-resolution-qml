#!/usr/bin/env python3
"""
make_resolution_plot.py - Figure 2 of the v2 paper: minimum detectable AUC difference against test-set size
==========================================================================================================
Reads the empirical resolution curve written by code/local_experiments/exp43_v2_tables.py
(results/local_exp43_v2_tables.json, key 'resolution_curve': median 80 %-power DeLong MDE over 60 random
subsets of test years per size) and draws it on log-log axes with the Hanley-McNeil approximation
(qml_resolution.paired_mde_hanley_mcneil; prevalence 0.406, AUC 0.68, correlation 0.75) as a reference line.
The exact values are in Table S10 of the Supplement (label tab:resolution).

Usage (from the repository root):
  python scripts/make_resolution_plot.py           # -> paper/figs/fig_resolution.pdf and .png
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from qml_resolution import paired_mde_hanley_mcneil  # noqa: E402

TABLES = os.path.join(ROOT, 'results', 'local_exp43_v2_tables.json')
MINUS = '−'
# same categorical slots and order as Figure 1 (validated palette); each series also has its own marker
SERIES = [('single_RF_QNN', f'Single run, RF {MINUS} LSTM+QNN', '#2a78d6', 'o'),
          ('single_RF_LSTM', f'Single run, RF {MINUS} LSTM', '#eb6834', 's'),
          ('mean_RF_QNN', f'15-seed means, RF {MINUS} LSTM+QNN', '#1baf7a', '^')]
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(ROOT, 'paper', 'figs', 'fig_resolution'))
    a = ap.parse_args()
    with open(TABLES, encoding='utf-8') as fh:
        curve = json.load(fh)['resolution_curve']
    years = sorted(curve, key=int)
    n = np.array([curve[y]['N'] for y in years], dtype=float)

    plt.rcParams.update({'font.size': 8.5, 'font.family': 'DejaVu Sans', 'axes.edgecolor': MUTED,
                         'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED, 'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    grid_n = np.geomspace(120, 2500, 200)
    hm = [paired_mde_hanley_mcneil(x, prevalence=0.406) for x in grid_n]
    ax.plot(grid_n, hm, color=MUTED, lw=1.4, ls=(0, (4, 3)), zorder=2)
    ax.text(900, paired_mde_hanley_mcneil(900, prevalence=0.406) * 1.08, 'Hanley–McNeil, single run',
            color=MUTED, fontsize=8, ha='left', va='bottom', rotation=-17, rotation_mode='anchor')
    for key, label, col, mk in SERIES:
        mde = np.array([curve[y][key] for y in years])
        ax.plot(n, mde, color=col, lw=2.0, marker=mk, ms=6, mec='#fcfcfb', mew=1.0, zorder=3, label=label)
        ax.annotate(f'{mde[-1]:.3f}', (n[-1], mde[-1]), xytext=(6, 0), textcoords='offset points',
                    va='center', fontsize=7.5, color=INK)
    # N^(-1/2) extrapolation of the 15-seed curve from the full test set (Section "Resolution and test-set size")
    m529 = curve[years[-1]]['mean_RF_QNN']
    ext_n = np.geomspace(n[-1], 2500, 50)
    ax.plot(ext_n, m529 * np.sqrt(n[-1] / ext_n), color='#1baf7a', lw=1.4, ls=(0, (1, 2)), zorder=2)
    n_half = n[-1] * 4
    ax.plot([n_half], [m529 / 2], marker='^', ms=6, mfc='#fcfcfb', mec='#1baf7a', mew=1.4, zorder=3)
    ax.annotate(f'{m529 / 2:.3f} at N ≈ {int(round(n_half, -2)):,}: illustrative $N^{{-1/2}}$\n'
                'extrapolation, if added years are independent', (n_half, m529 / 2),
                xytext=(-9, -1), textcoords='offset points', ha='right', va='top', fontsize=7.5, color=INK)
    ax.axvline(n[-1], color=GRID, lw=1.0, zorder=1)
    ax.text(n[-1] * 0.97, 0.115, f'full test set\nN = {int(n[-1])}', color=MUTED, fontsize=7.5, va='top', ha='right')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(120, 2500)
    ax.set_ylim(0.012, 0.13)
    ax.set_xticks([150, 250, 500, 1000, 2000])
    ax.set_xticklabels(['150', '250', '500', '1,000', '2,000'])
    ax.set_yticks([0.015, 0.02, 0.03, 0.05, 0.08, 0.12])
    ax.set_yticklabels(['0.015', '0.02', '0.03', '0.05', '0.08', '0.12'])
    ax.minorticks_off()
    top = ax.secondary_xaxis('top')                  # the test sets are whole years: the clusters behind each point
    top.set_xticks(list(n))
    top.set_xticklabels([str(int(y)) for y in years])
    top.minorticks_off()
    top.tick_params(colors=MUTED, labelsize=7.5)
    top.spines['top'].set_color(MUTED)
    top.set_xlabel('Independent clusters (test years) in the subset', fontsize=8, color=INK)
    ax.grid(color=GRID, lw=0.6, zorder=0)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.set_xlabel('Test rows N (random subsets of 5 to 19 test years)')
    ax.set_ylabel('80 %-power MDE of the AUC difference')
    ax.legend(loc='upper right', frameon=False, fontsize=8, handlelength=2.6)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(f'{a.out}.pdf', bbox_inches='tight', metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(f'{a.out}.png', dpi=200, bbox_inches='tight', metadata={'Software': None})
    print(f'wrote {a.out}.pdf/.png ({len(years)} test sizes)')


if __name__ == '__main__':
    main()
