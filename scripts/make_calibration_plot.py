#!/usr/bin/env python3
"""make_calibration_plot.py - coverage of every candidate procedure in the calibration simulation (exp48)
=========================================================================================================
One row per procedure, one dot per simulation cell (30 cells: scenarios x test-set variants x true difference).
Filled dots: independent years; hollow dots: AR(1) dependence between years. Shaded band: 95 % Monte Carlo interval
around the nominal 0.95 for 600 replications. The primary procedure is drawn in the accent colour, the others in
neutral ink, so identity never rests on colour alone (row labels carry it).

  python scripts/make_calibration_plot.py      # -> paper/figs/fig_calibration.pdf and .png
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'results', 'local_exp48_primary_calibration.json')
OUT = os.path.join(ROOT, 'paper', 'figs', 'fig_calibration')
PROC = [('cr2_bm', 'CR2, Bell–McCaffrey df (primary)'), ('jackknife_t', 'Delete-one-year jackknife'),
        ('cr1_t', 'CR1, $t_{K-1}$'), ('wild_webb', 'Wild cluster bootstrap'),
        ('block_pct', 'Year-block bootstrap (percentile)'), ('delong', 'DeLong (independence)')]
ACCENT, INK, MUTED, GRID, SURFACE = '#2a78d6', '#3d3c39', '#8a8985', '#e4e3df', '#fcfcfb'


def main():
    R = json.load(open(SRC, encoding='utf-8'))
    sc = {s['name']: s for s in R['scenarios']}
    rows = R['rows']
    reps = rows[0]['reps']
    half = 1.96 * np.sqrt(0.95 * 0.05 / reps)
    plt.rcParams.update({'font.size': 8.5, 'font.family': 'DejaVu Sans', 'axes.edgecolor': MUTED,
                         'xtick.color': MUTED, 'ytick.color': INK, 'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    ax.axvspan(0.95 - half, 0.95 + half, color=GRID, zorder=0, lw=0)
    ax.axvline(0.95, color=INK, lw=1.3, zorder=1)
    ax.text(0.95, len(PROC) - 0.45, 'nominal 0.95', ha='center', va='bottom', fontsize=7.5, color=INK)
    rng = np.random.default_rng(0)
    for i, (key, _) in enumerate(PROC):
        y0 = len(PROC) - 1 - i
        col = ACCENT if key == 'cr2_bm' else INK
        for r in rows:
            s = sc[r['scenario']]
            ar = s['phi'] > 0
            worst = ar and s['tc'] > 0          # AR(1) years combined with year x station effects
            yy = y0 + rng.uniform(-0.18, 0.18)
            ax.plot(r['coverage'][key], yy, 'D' if worst else 'o', ms=4.4 if worst else 4.6, mew=1.0, zorder=3,
                    mec=col, mfc=SURFACE if ar else col, alpha=0.95)
    ax.set_yticks(range(len(PROC)))
    ax.set_yticklabels([lab for _, lab in PROC][::-1])
    ax.get_yticklabels()[-1].set_color(ACCENT)
    ax.get_yticklabels()[-1].set_fontweight('bold')
    ax.tick_params(axis='y', length=0)
    ax.set_xlim(0.80, 0.99)
    ax.set_xlabel('Coverage of the 95 % interval for the difference for new years')
    ax.grid(axis='x', color=GRID, lw=0.6, zorder=0)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    h = [plt.Line2D([], [], ls='', marker='o', ms=4.6, mec=INK, mfc=INK, label='independent years'),
         plt.Line2D([], [], ls='', marker='o', ms=4.6, mec=INK, mfc=SURFACE, label='AR(1) years'),
         plt.Line2D([], [], ls='', marker='D', ms=4.4, mec=INK, mfc=SURFACE, label='AR(1) + year×station effects'),
         plt.Rectangle((0, 0), 1, 1, color=GRID, label=f'Monte Carlo band ({reps} replications)')]
    ax.legend(handles=h, loc='lower center', bbox_to_anchor=(0.5, 1.06), ncol=2, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(f'{OUT}.pdf', bbox_inches='tight', metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(f'{OUT}.png', dpi=200, bbox_inches='tight', metadata={'Software': None})
    print(f'wrote {OUT}.pdf/.png ({len(rows)} cells)')


if __name__ == '__main__':
    main()
