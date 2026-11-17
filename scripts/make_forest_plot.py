#!/usr/bin/env python3
"""
make_forest_plot.py - forest plot of every paired seed-mean comparison of the paper, three intervals per row
==========================================================================================================
Reads the exp10 outputs (no recomputation, so the figure matches the tables exactly):
  results/json/results_exp10_paired_ci_fixet0.json   classical pairs + LSTM+QNN pairs   (Tables 4 and 7)
  results/json/results_exp10_hybrids_fixet0.json     further hybrids                     (Table 8)
  results/json/results_exp10_controls.json           corrected LSTM+QTL and controls     (Tables 8 and 9; optional)
For each pair it draws the DeLong 95 % interval, the calendar-year block-bootstrap interval and the
years x seeds interval, offset vertically, with the point estimate. Pairs whose JSON is missing are skipped.

Usage (from the repository root):
  python scripts/make_forest_plot.py                      # -> figs/fig_forest.pdf and figs/fig_forest.png
  python scripts/make_forest_plot.py --out figs/fig_forest
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J = lambda f: os.path.join(ROOT, 'results', 'json', f)  # noqa: E731

# (group, label, file, key in 'pairs'); the label states the sign convention A - B
ROWS = [
    ('Classical models', 'RF − XGBoost', 'results_exp10_paired_ci_fixet0.json', 'RF-XGB'),
    ('Classical models', 'RF − LSTM', 'results_exp10_paired_ci_fixet0.json', 'RF-LSTM'),
    ('Classical models', 'XGBoost − LSTM', 'results_exp10_paired_ci_fixet0.json', 'XGB-LSTM'),
    ('Random Forest vs hybrids', 'RF − LSTM+QNN', 'results_exp10_paired_ci_fixet0.json', 'RF_2273-LSTM_QNN'),
    ('Random Forest vs hybrids', 'RF − LSTM+QTL', 'results_exp10_controls.json', 'RF_2273-QTL_FIXED'),
    ('Hybrids vs LSTM', 'LSTM − LSTM+QNN', 'results_exp10_paired_ci_fixet0.json', 'PureLSTM-LSTM_QNN'),
    ('Hybrids vs LSTM', 'Extractor − LSTM+QTL', 'results_exp10_controls.json', 'QTL_EXTRACTOR27-QTL_FIXED'),
    ('Circuit vs controls', 'Linear-4D − LSTM+QNN', 'results_exp10_paired_ci_fixet0.json', 'CTRL_LINEAR4D-LSTM_QNN'),
    ('Circuit vs controls', 'Fourier-1 − LSTM+QNN', 'results_exp10_controls.json', 'FOURIER1-LSTM_QNN'),
    ('Circuit vs controls', 'Fourier-2 − LSTM+QNN', 'results_exp10_controls.json', 'FOURIER2-LSTM_QNN'),
    ('Circuit vs controls', 'Fourier-3 − LSTM+QNN', 'results_exp10_fourier_sep34.json', 'FOURIER3-LSTM_QNN'),
    ('Circuit vs controls', 'Fourier-4 − LSTM+QNN', 'results_exp10_fourier_sep34.json', 'FOURIER4-LSTM_QNN'),
    ('Circuit vs controls', 'Full Fourier-1 − LSTM+QNN', 'results_exp10_fourier_full.json', 'FOURIER_FULL1-LSTM_QNN'),
    ('Circuit vs controls', 'Full Fourier-2 − LSTM+QNN', 'results_exp10_fourier_full.json', 'FOURIER_FULL2-LSTM_QNN'),
    ('Circuit vs controls', 'QTL-classical − LSTM+QTL', 'results_exp10_controls.json', 'QTL_CLASSICAL-QTL_FIXED'),
    ('Circuit vs controls', 'Frozen random − LSTM+QNN', 'results_exp10_controls.json', 'FROZEN_RANDOM-LSTM_QNN'),
    ('Circuit vs controls', 'Frozen zero-angle − LSTM+QNN', 'results_exp10_controls.json', 'FROZEN_IDENTITY-LSTM_QNN'),
]
# validated categorical slots 1-3 (dataviz reference palette, light mode); aqua is below 3:1 contrast,
# so every procedure also has its own marker shape and the legend is always shown
PROC = [('DeLong (iid)', 'ci_delong', '#2a78d6', 'o'),
        ('Year-block bootstrap', 'ci_block', '#eb6834', 's'),
        ('Years × seeds bootstrap', 'ci_two_way', '#1baf7a', '^')]
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def load(f, cache={}):
    if f not in cache:
        p = J(f)
        cache[f] = json.load(open(p, encoding='utf-8')) if os.path.exists(p) else None
    return cache[f]


def ci(entry, key):
    if key in entry and entry[key] is not None:
        return entry[key]
    if key == 'ci_delong':                                    # older outputs: rebuild from the SE
        return [entry['delta'] - 1.959964 * entry['se_delong'], entry['delta'] + 1.959964 * entry['se_delong']]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(ROOT, 'figs', 'fig_forest'))
    a = ap.parse_args()
    rows, missing = [], []
    for g, lab, f, k in ROWS:
        d = load(f)
        e = (d or {}).get('pairs', {}).get(k)
        (rows if e else missing).append((g, lab, e) if e else f'{lab} ({f})')
    if missing:
        print('skipped (output not found):', '; '.join(missing))

    plt.rcParams.update({'font.size': 8.5, 'font.family': 'DejaVu Sans', 'axes.edgecolor': MUTED,
                         'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': INK})
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
        for j, (name, key, col, mk) in enumerate(PROC):
            c = ci(e, key)
            if c is None:
                continue
            off = (1 - j) * 0.22
            ax.plot(c, [yy + off] * 2, color=col, lw=2.0, solid_capstyle='round', zorder=2)
            ax.plot(e['delta'], yy + off, marker=mk, ms=4.5, color=col, mec='#fcfcfb', mew=0.8, zorder=3)
    all_ticks = ticks + [yy for yy, _ in heads]
    all_labels = labels + [g for _, g in heads]
    ax.set_yticks(all_ticks)
    ax.set_yticklabels(all_labels)
    for t in ax.get_yticklabels()[len(ticks):]:
        t.set_fontweight('bold')
    ax.tick_params(axis='y', length=0)
    ax.set_ylim(y + 0.4, 0.6)
    ax.grid(axis='x', color=GRID, lw=0.6, zorder=0)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.set_xlabel('Difference of 15-seed mean test AUC (A − B), 95 % intervals')
    handles = [plt.Line2D([], [], color=c, lw=2, marker=m, ms=4.5, mec='#fcfcfb') for _, _, c, m in PROC]
    ax.legend(handles, [p[0] for p in PROC], loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
              fontsize=8, handlelength=2.2)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{a.out}.{ext}', dpi=200, bbox_inches='tight')
    print(f'wrote {a.out}.pdf/.png with {len(rows)} pairs')


if __name__ == '__main__':
    main()
