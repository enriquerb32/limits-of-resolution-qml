"""
scripts/build_traceability_ext.py - traceability of the claims added in the revision (exp25-exp32)
==================================================================================================
Same conventions as scripts/build_traceability.py: every claim is checked against the archived JSON it comes from,
with an explicit tolerance; a missing file or key gives UNVERIFIED (never a fallback value). Claims whose result does
not exist yet (exp30-exp32 before they are run) are listed as PENDING with the command that produces them.

Output: audit/02_traceability_ext.csv. Exit code 1 if any claim is a MISMATCH (usable in CI / make audit).
Usage:  python scripts/build_traceability_ext.py
"""
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SOURCES = {
    'exp25': 'results/local_exp25_within_year_structure.json',
    'exp26': 'results/local_exp26_within_year_robustness.json',
    'exp29': 'results/local_exp29_phase_intervals.json',
    'exp10_ctrl': 'results/json/results_exp10_controls.json',
    'exp27': 'results/json/results_controls_trainonly_fixet0_p60_28.1.json',
    'exp28_orig': 'results/local_exp28_orig_summary.json',
    'exp28_sm': 'results/local_exp28_stationmonth_summary.json',
    'exp28b_orig': 'results/local_exp28b_orig.json',
    'exp28b_sm': 'results/local_exp28b_stationmonth.json',
    'exp30': 'results/json/results_exp10_fourier_full.json',
    'exp31': 'results/local_exp31_shot_noise.json',
    'exp30_auc': 'results/json/results_fourier_full_trainonly_fixet0_p60_28.1.json',
    'exp29_ff': 'results/local_exp29_fourier_full.json',
    'exp32': 'results/local_exp32_panel2_med2.json',
    'exp34': 'results/local_exp34_circuit_spectrum.json',
    'exp35': 'results/local_exp35_equivalence_fewclusters.json',
    'sep34': 'results/json/results_exp10_fourier_sep34.json',
    'exp25q': 'results/local_exp25_qtl_fixed.json',
    'exp36': 'results/local_exp36_rf_optuna.json',
    'exp33': 'results/local_exp33_seas5_orig.json',
    'exp33_sm': 'results/local_exp33_seas5_stationmonth.json',
}
PENDING_CMD = {
    'exp30': 'python code/local_experiments/exp30_fourier_full.py, then exp10 --extra ... --out results/json/results_exp10_fourier_full.json',
    'exp29_ff': 'exp29_phase_intervals.py --pairs FOURIER_FULL2-LSTM_QNN FOURIER_FULL1-LSTM_QNN --out results/local_exp29_fourier_full.json',
    'exp31': 'python code/local_experiments/exp31_shot_noise.py',
    'exp33': 'python code/local_experiments/exp33_seas5_reference.py --step score / --step analyze',
    'exp33_sm': 'python code/local_experiments/exp33_seas5_reference.py --label stationmonth --step score / analyze',
}


def resolve(key):
    """Path of a source: the expected one, else a unique file of the same name (or the same experiment tag, e.g.
    'exp26') anywhere under results/. The path actually used is written into the CSV."""
    import glob
    p = os.path.join(ROOT, SOURCES[key])
    if os.path.exists(p):
        return p
    base = os.path.basename(SOURCES[key])
    hits = glob.glob(os.path.join(ROOT, 'results', '**', base), recursive=True)
    if not hits:
        tag = base.split('_')[1] if base.startswith('local_') else None          # local_exp26_... -> exp26
        if tag:
            kw = ('fourier', 'panel2', 'seas5', 'stationmonth', 'orig', 'controls')
            want = {k for k in kw if k in base}
            hits = [h for h in glob.glob(os.path.join(ROOT, 'results', '**', f'*{tag}*.json'), recursive=True)
                    if {k for k in kw if k in os.path.basename(h)} == want]
    if len(hits) == 1:
        SOURCES[key] = os.path.relpath(hits[0], ROOT).replace(os.sep, '/')
        print(f'  {key}: using {SOURCES[key]}')
        return hits[0]
    if len(hits) > 1:
        print(f'  {key}: several candidates, set SOURCES[{key!r}] by hand: ' + ', '.join(os.path.relpath(h, ROOT) for h in hits))
    return None


def load(key):
    p = resolve(key)
    if p is None:
        return None
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def get(obj, path):
    for part in path.split('.'):
        if obj is None:
            return None
        if isinstance(obj, dict):
            obj = obj.get(part)
        elif isinstance(obj, list) and part.lstrip('-').isdigit():
            i = int(part)
            obj = obj[i] if -len(obj) <= i < len(obj) else None
        else:
            return None
    return obj


def main():
    ctx = {k: load(k) for k in SOURCES}
    rows = []

    def claim(tid, loc, desc, paper, src, checks):
        """checks: list of (key path in ctx[src], expected, tolerance)."""
        if ctx[src] is None:
            status = 'PENDING' if src in PENDING_CMD else 'UNVERIFIED'
            note = PENDING_CMD.get(src, 'file not found')
            rows.append((tid, loc, desc, paper, SOURCES[src], note, status))
            return
        vals = [(path, get(ctx[src], path), e, t) for path, e, t in checks]
        missing = [p for p, v, _, _ in vals if v is None]
        if missing:
            rows.append((tid, loc, desc, paper, SOURCES[src], 'NOT_FOUND: ' + ', '.join(missing), 'UNVERIFIED'))
            return
        ok = all(abs(float(v) - e) <= t for _, v, e, t in vals)
        sv = '; '.join(f'{p}={float(v):.4f}' for p, v, _, _ in vals)
        rows.append((tid, loc, desc, paper, SOURCES[src], sv if ok else 'MISMATCH: ' + sv, 'VERIFIED' if ok else 'MISMATCH'))

    def claim_range(tid, loc, desc, paper, src, paths, lo, hi, tol=0.005):
        """All values at `paths` lie in [lo, hi] and the extremes match lo and hi."""
        if ctx[src] is None:
            rows.append((tid, loc, desc, paper, SOURCES[src], PENDING_CMD.get(src, 'file not found'),
                         'PENDING' if src in PENDING_CMD else 'UNVERIFIED'))
            return
        v = [get(ctx[src], p) for p in paths]
        if any(x is None for x in v):
            rows.append((tid, loc, desc, paper, SOURCES[src], 'NOT_FOUND', 'UNVERIFIED'))
            return
        ok = abs(min(v) - lo) <= tol and abs(max(v) - hi) <= tol
        sv = f'min={min(v):.4f}; max={max(v):.4f}'
        rows.append((tid, loc, desc, paper, SOURCES[src], sv if ok else 'MISMATCH: ' + sv, 'VERIFIED' if ok else 'MISMATCH'))

    P = 'pairs.'
    # ---- exp25: cluster-robust SE and ratios (main §3 budget, supp S6.3)
    claim('TX-001', 'main §4.1 (budget)', 'Cluster-robust SE RF-LSTM / DeLong', '0.0061 / 0.0112', 'exp25',
          [(P + 'RF_2273-PureLSTM.se_cluster', 0.0061, 0.00006), (P + 'RF_2273-PureLSTM.se_iid', 0.0112, 0.00006)])
    claim('TX-002', 'main §4.1 (budget)', 'Cluster-robust SE RF-LSTM+QNN / DeLong', '0.0062 / 0.0109', 'exp25',
          [(P + 'RF_2273-LSTM_QNN.se_cluster', 0.0062, 0.00006), (P + 'RF_2273-LSTM_QNN.se_iid', 0.0109, 0.00006)])
    claim('TX-003', 'main §4.1 (budget)', 'Within-year cross term and iid variance RF-LSTM', '-1.6e-4 / 1.2e-4', 'exp25',
          [(P + 'RF_2273-PureLSTM.within_cross', -1.6e-4, 0.06e-4), (P + 'RF_2273-PureLSTM.iid_var', 1.2e-4, 0.06e-4)])
    claim_range('TX-004', 'main §4.1 (budget)', 'Cluster/iid ratio, RF vs neural and hybrid models', '0.55--0.71', 'exp25',
                [P + f'RF_2273-{m}.ratio' for m in ['PureLSTM', 'LSTM_QNN', 'LSTM_PIQNN', 'LBL_QNN', 'LSTM_QTL']], 0.55, 0.71)
    claim_range('TX-005', 'main §4.1 (budget)', 'Cluster/iid ratio among neural and hybrid models', '0.85--1.35', 'exp25',
                [P + p + '.ratio' for p in ['CTRL_LINEAR4D-LSTM_QNN', 'PureLSTM-LSTM_QNN', 'CTRL_LINEAR4D-LSTM_PIQNN',
                                            'CTRL_LINEAR4D-LBL_QNN', 'CTRL_LINEAR4D-LSTM_QTL', 'QTL_EXTRACTOR-LSTM_QTL']],
                0.85, 1.35)
    claim('TX-006', 'main §4.3 (hybrids)', 'Cluster-robust t18 interval RF-LSTM+QNN', '[+0.007, +0.033]', 'exp25',
          [(P + 'RF_2273-LSTM_QNN.ci_cluster_t.0', 0.007, 0.0006), (P + 'RF_2273-LSTM_QNN.ci_cluster_t.1', 0.033, 0.0006)])
    # ---- exp26: date-block permutation nulls
    claim('TX-007', 'main §4.1 / supp S6.3', 'Date-block (6 months) permutation p, RF-LSTM', '0.016', 'exp26',
          [(P + 'RF_2273-PureLSTM.null_dateblock_6m.p_lower', 0.016, 0.0006)])
    claim('TX-008', 'main §4.1 / supp S6.3', 'Ratio RF-LSTM at the calendar year', '0.55', 'exp26',
          [(P + 'RF_2273-PureLSTM.ratio_year', 0.55, 0.005)])
    # ---- exp29: phase-shifted pseudo-years
    for tid, pair, k in [('TX-009', 'RF_2273-PureLSTM', 4), ('TX-010', 'RF_2273-LSTM_QNN', 10),
                         ('TX-011', 'RF_2273-LSTM_PIQNN', 12), ('TX-012', 'RF_2273-LBL_QNN', 12)]:
        claim(tid, 'main §4.1 / §4.3', f'Start months with t18 interval excluding zero, {pair}', f'{k} of 12', 'exp29',
              [(pair + '.n_phases_excluding_zero', k, 0)])
    # ---- exp27 + exp10 on the controls (tab:controls)
    for tid, pair, d, se in [('TX-013', 'FOURIER1-LSTM_QNN', 0.0005, 0.0053), ('TX-014', 'FOURIER2-LSTM_QNN', 0.0016, 0.0043),
                             ('TX-015', 'FROZEN_RANDOM-LSTM_QNN', -0.0136, 0.0056),
                             ('TX-016', 'FROZEN_IDENTITY-LSTM_QNN', -0.0238, 0.0066)]:
        claim(tid, 'main tab:controls', f'Paired difference and DeLong SE, {pair}', f'{d:+.4f} / {se:.4f}', 'exp10_ctrl',
              [(P + pair + '.delta', d, 0.00006), (P + pair + '.se_delong', se, 0.00006)])
    claim('TX-017', 'main tab:controls', 'Years x seeds CI, frozen zero-angle - LSTM+QNN', '[-0.048, -0.005]', 'exp10_ctrl',
          [(P + 'FROZEN_IDENTITY-LSTM_QNN.ci_two_way.0', -0.048, 0.0006), (P + 'FROZEN_IDENTITY-LSTM_QNN.ci_two_way.1', -0.005, 0.0006)])
    # ---- exp28 / exp28b: recalibration and station x month labels
    claim('TX-018', 'main tab:recal', 'Platt BSS RF (mean, seeds above climatology)', '+0.016, 15 of 15', 'exp28_orig',
          [('models.RF_2273.bss_platt.mean', 0.016, 0.0006), ('models.RF_2273.bss_platt.n_seeds_above_clim', 15, 0)])
    claim('TX-019', 'main tab:recal', 'ECE of RF before / after Platt', '0.095 / 0.036', 'exp28_orig',
          [('models.RF_2273.ece_raw.mean', 0.095, 0.0006), ('models.RF_2273.ece_platt.mean', 0.036, 0.0006)])
    claim('TX-020', 'main §4.5', 'Climatology AUC / Brier / ECE (pooled label)', '0.6808 / 0.2220 / 0.069', 'exp28_orig',
          [('climatology.auc', 0.6808, 0.00006), ('climatology.brier', 0.2220, 0.00006), ('climatology.ece', 0.069, 0.0006)])
    claim('TX-021', 'main §4.5', 'RF Platt BSS against the Platt-recalibrated climatology', '-0.001, 6 of 15', 'exp28b_orig',
          [('models.RF_2273.bss_vs_platt_clim.mean', -0.001, 0.0006), ('models.RF_2273.bss_vs_platt_clim.n_above', 6, 0)])
    claim('TX-022', 'main §4.6', 'Station x month label: climatology AUC, test rate', '0.504 / 0.347', 'exp28_sm',
          [('climatology.auc', 0.504, 0.0006), ('test_rate', 0.347, 0.0006)])
    # ---- new experiments of this revision (PENDING until run)
    for tid, pair, d, se in [('TX-023', 'FOURIER_FULL1-LSTM_QNN', 0.0127, 0.0043),
                             ('TX-024', 'FOURIER_FULL2-LSTM_QNN', 0.0035, 0.0037),
                             ('TX-024b', 'FOURIER_FULL2-FOURIER2', 0.0019, 0.0037)]:
        claim(tid, 'main tab:controls', f'Paired difference and DeLong SE, {pair}', f'{d:+.4f} / {se:.4f}', 'exp30',
              [(P + pair + '.delta', d, 0.00006), (P + pair + '.se_delong', se, 0.00006)])
    claim('TX-024c', 'main tab:controls', 'Year-block CI, Full Fourier-1 - LSTM+QNN', '[+0.007, +0.018]', 'exp30',
          [(P + 'FOURIER_FULL1-LSTM_QNN.ci_block.0', 0.007, 0.0006), (P + 'FOURIER_FULL1-LSTM_QNN.ci_block.1', 0.018, 0.0006)])
    for tid, pair, k in [('TX-024d', 'FOURIER_FULL1-LSTM_QNN', 12), ('TX-024e', 'FOURIER_FULL2-LSTM_QNN', 0)]:
        claim(tid, 'main §4.3 (controls)', f'Start months with t18 interval excluding zero, {pair}', f'{k} of 12',
              'exp29_ff', [(pair + '.n_phases_excluding_zero', k, 0)])
    for tid, n, loss, sd in [('TX-024f', 32, 0.031, 0.004), ('TX-024g', 100, 0.013, 0.002),
                             ('TX-024h', 1000, 0.002, 0.001), ('TX-024i', 10000, 0.0003, 0.0003)]:
        claim(tid, 'main §6 (shot noise)', f'AUC loss and shot SD of the seed mean, {n} shots', f'{loss} / {sd}', 'exp31',
              [(f'by_shots.{n}.mean_auc_loss', loss, 0.0006), (f'by_shots.{n}.shot_sd_seed_mean', sd, 0.0006)])
    claim('TX-024j', 'main §6 (shot noise)', 'Mean exact AUC of the refitted LSTM+QNN', '0.6654', 'exp31',
          [('auc_exact_mean', 0.6654, 0.00006)])
    # ---- exp33: operational SEAS5 reference (fill the expected values once tab:seas5 is filled; tolerance 0.0006)
    claim('TX-025', 'main tab:seas5', 'SEAS5 (lead 1) - climatology', '+0.0495 [+0.011, +0.081]', 'exp33',
          [('pairs.SEAS5_QM_L1-CLIM.delta', 0.0495, 0.00006), ('pairs.SEAS5_QM_L1-CLIM.ci_block.0', 0.011, 0.0006), ('pairs.SEAS5_QM_L1-CLIM.ci_block.1', 0.081, 0.0006)])
    claim('TX-026', 'main tab:seas5', 'Blend RF - blend climatology (lead 1)', '+0.0045, MDE 0.012', 'exp33',
          [('pairs.BLEND_RF_2273_L1-BLEND_CLIM_L1.delta', 0.0045, 0.00006), ('pairs.BLEND_RF_2273_L1-BLEND_CLIM_L1.mde_delong', 0.012, 0.0006)])
    claim('TX-027', 'main tab:seas5', 'Blend LSTM+QNN - blend Linear-4D (lead 1)', '+0.0003', 'exp33',
          [('pairs.BLEND_LSTM_QNN_L1-BLEND_CTRL_LINEAR4D_L1.delta', 0.0003, 0.00006)])
    claim('TX-028', 'main §4.6', 'SEAS5 AUC and gain over climatology, station x month label (lead 1)', '0.6236 / +0.1198', 'exp33_sm',
          [('auc.SEAS5_QM_L1.mean', 0.6236, 0.00006), ('pairs.SEAS5_QM_L1-CLIM.delta', 0.1198, 0.00006)])

    # ---- exp32: second panel (tag med2)
    claim('TX-029', 'main sec:replication', 'Second panel: test rows and climatology AUC (pooled)', '882 / 0.710', 'exp32',
          [('pooled.n_test', 882, 0), ('pooled.climatology_auc', 0.710, 0.0006)])
    for tid, pair, d, mde, ratio, ph in [('TX-030', 'RF_2273-XGB', 0.031, 0.024, 1.37, 12),
                                         ('TX-031', 'RF_2273-PureLSTM', -0.005, 0.027, 0.93, 0),
                                         ('TX-032', 'RF_2273-LSTM_QNN', 0.009, 0.030, 1.05, 0),
                                         ('TX-033', 'PureLSTM-LSTM_QNN', 0.014, 0.013, 1.23, 10),
                                         ('TX-034', 'CTRL_LINEAR4D-LSTM_QNN', 0.009, 0.015, 1.11, 0)]:
        claim(tid, 'main tab:panel2', f'Second panel {pair}: delta, MDE, ratio, phases', f'{d} / {mde} / {ratio} / {ph}', 'exp32',
              [(f'pooled.pairs.{pair}.delta', d, 0.0006), (f'pooled.pairs.{pair}.mde_delong', mde, 0.0006),
               (f'pooled.pairs.{pair}.ratio', ratio, 0.006), (f'pooled.pairs.{pair}.phases_excluding_zero', ph, 0)])
    claim('TX-035', 'main sec:replication', 'Second panel: standalone LSTM seed-mean AUC (pooled)', '0.712', 'exp32',
          [('pooled.models.PureLSTM.auc_seed_mean', 0.712, 0.0006)])

    # ---- exp34 / exp35 / corrected QTL (second review)
    claim('TX-036', 'supp app:spectrum', 'Variance share of frequencies 1-4, trained circuits', '68.7 / 24.3 / 6.2 / 0.8 %', 'exp34',
          [('summary.trained_variance_share_f1_f5.0', 0.687, 0.0006), ('summary.trained_variance_share_f1_f5.2', 0.0615, 0.0006),
           ('summary.trained_variance_share_f1_f5.3', 0.0079, 0.0006)])
    claim('TX-037', 'main methods / supp', 'R2 (test angles) of the degree-1 tensor basis, mean', '0.985', 'exp34',
          [('summary.r2_test_full_d1.mean', 0.985, 0.0006), ('summary.r2_test_full_d1.min', 0.958, 0.0006)])
    for tid, pair, d, se in [('TX-038', 'FOURIER3-LSTM_QNN', 0.0021, 0.0043), ('TX-039', 'FOURIER4-LSTM_QNN', 0.0033, 0.0040)]:
        claim(tid, 'main tab:controls', f'Paired difference and DeLong SE, {pair}', f'{d} / {se}', 'sep34',
              [(P + pair + '.delta', d, 0.00006), (P + pair + '.se_delong', se, 0.00006)])
    claim('TX-040', 'main sec:budget', 'Wild cluster bootstrap p, RF-LSTM / RF-LSTM+QNN', '0.024 / 0.004', 'exp35',
          [(P + 'RF_2273-PureLSTM.p_wild_cluster', 0.024, 0.0015), (P + 'RF_2273-LSTM_QNN.p_wild_cluster', 0.004, 0.0015)])
    claim('TX-041', 'main controls discussion', 'TOST margin (CR2) Linear-4D - LSTM+QNN', '0.024', 'exp35',
          [(P + 'CTRL_LINEAR4D-LSTM_QNN.equivalence_margin_min.cr2_t', 0.024, 0.0006)])
    claim('TX-042', 'supp tab:within_year', 'Corrected RF - LSTM+QTL: delta / ratio', '+0.019 / 0.67', 'exp25q',
          [(P + 'RF_2273-QTL_FIXED.delta', 0.0185, 0.0006), (P + 'RF_2273-QTL_FIXED.ratio', 0.67, 0.006)])

    claim('TX-043', 'main limitations / supp app:hyperparams', 'Documented Optuna RF: test AUC mean / SD, validation AUC',
          '0.6848 / 0.0044 / 0.707', 'exp36',
          [('test_auc_mean', 0.6848, 0.00006), ('test_auc_sd', 0.0044, 0.00006), ('best_val_auc', 0.707, 0.0006)])

    out = os.path.join(ROOT, 'audit', '02_traceability_ext.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['ID', 'Location', 'Claim_Description', 'Paper_Value', 'Source_File', 'Source_Value', 'Status'])
        w.writerows(rows)
    counts = {s: sum(r[6] == s for r in rows) for s in ['VERIFIED', 'MISMATCH', 'UNVERIFIED', 'PENDING']}
    print(f'{len(rows)} claims: ' + ', '.join(f'{v} {k.lower()}' for k, v in counts.items()) + f'\nwrote {out}')
    for r in rows:
        if r[6] != 'VERIFIED':
            print(f'  {r[0]} {r[6]:10s} {r[2]}: {r[5]}')
    sys.exit(1 if counts['MISMATCH'] else 0)


if __name__ == '__main__':
    main()
