#!/usr/bin/env python3
"""make_design_plot.py - study design: eligible windows per station and year, by partition
=========================================================================================
One row per station; one bar per calendar year, height = number of eligible windows (target months, 0-12) in the
archived monthly panel (results/panel_v2_monthly.csv, column win_ok), coloured by partition and labelled directly.
Shows why 83 % of the test windows come from two stations: the covariate records of Valencia and San Javier stop.

  python scripts/make_design_plot.py      # -> paper/figs/fig_design.pdf and .png
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'results', 'panel_v2_monthly.csv')
OUT = os.path.join(ROOT, 'paper', 'figs', 'fig_design')
STATIONS = [('Barcelona', 'Barcelona'), ('Valencia', 'Valencia'), ('Alicante', 'Alicante'), ('San_Javier', 'San Javier')]
PART = [('train', 'training\n1938–1996', 1938, 1996, '#2a78d6'), ('val', 'validation\n1997–2007', 1997, 2007, '#1baf7a'),
        ('test', 'test\n2008–2026', 2008, 2026, '#eb6834')]
ALL4 = (2008, 2011)   # the only test years in which all four stations have eligible windows (checked in main)
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def main():
    p = pd.read_csv(SRC, parse_dates=['fecha'])
    p = p[p['win_ok'].astype(str).str.lower() == 'true']
    p['year'] = p['fecha'].dt.year
    counts = p.groupby(['ciudad', 'year']).size()
    n_test = {c: int(counts.loc[c].loc[2008:].sum()) if c in counts.index.get_level_values(0) else 0 for c, _ in STATIONS}
    test_years = counts[counts.index.get_level_values(1) >= 2008].reset_index().groupby('year')['ciudad'].nunique()
    all4 = sorted(test_years[test_years == len(STATIONS)].index)
    assert (all4[0], all4[-1], len(all4)) == (ALL4[0], ALL4[1], ALL4[1] - ALL4[0] + 1), all4
    assert len(test_years) == 19, len(test_years)
    plt.rcParams.update({'font.size': 8.5, 'font.family': 'DejaVu Sans', 'axes.edgecolor': MUTED,
                         'xtick.color': MUTED, 'ytick.color': MUTED, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(len(STATIONS), 1, figsize=(6.8, 3.6), sharex=True)
    for ax, (key, name) in zip(axes, STATIONS):
        s = counts.loc[key] if key in counts.index.get_level_values(0) else pd.Series(dtype=float)
        for _, _, lo, hi, col in PART:
            yrs = [y for y in s.index if lo <= y <= hi]
            ax.bar(yrs, [s[y] for y in yrs], width=0.82, color=col, linewidth=0)
        ax.add_patch(plt.Rectangle((ALL4[0] - 0.55, 0), ALL4[1] - ALL4[0] + 1.1, 12.7, fill=False, ec=INK, lw=0.9,
                                   zorder=4, clip_on=False))
        ax.set_ylim(0, 13); ax.set_yticks([12]); ax.set_yticklabels([''])
        ax.text(-0.015, 0.58, name, transform=ax.transAxes, ha='right', va='center', color=INK, fontsize=8.5, fontweight='bold')
        ax.text(-0.015, 0.22, f'{n_test[key]} test rows', transform=ax.transAxes, ha='right', va='center', color=MUTED,
                fontsize=7.5)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.grid(axis='y', color=GRID, lw=0.5)
    for _, lab, lo, hi, col in PART:                                  # direct partition labels above the top row
        axes[0].text((lo + hi) / 2, 14.4, lab, ha='center', va='bottom', color=MUTED, fontsize=7.5, linespacing=1.0)
        axes[0].plot([lo - 0.4, hi + 0.4], [13.6, 13.6], color=col, lw=3, solid_capstyle='butt', clip_on=False)
    for ax in axes[1::2]:                                             # Valencia and San Javier: empty after 2012
        ax.text(ALL4[1] + 2.5, 6.5, f'19 test years = 19 clusters;\nboxed: {ALL4[0]}–{ALL4[1]}, the only {len(all4)}\nwith all four stations',
                ha='left', va='center', color=INK, fontsize=7, linespacing=1.1)
        break
    axes[-1].set_xlim(1935.5, 2033)
    axes[-1].set_xlabel('Year of the target month')
    fig.tight_layout(rect=(0.09, 0, 1, 0.93))
    fig.savefig(f'{OUT}.pdf', bbox_inches='tight', metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(f'{OUT}.png', dpi=200, bbox_inches='tight', metadata={'Software': None})
    print('test windows by station:', n_test, 'total', sum(n_test.values()))


if __name__ == '__main__':
    main()
