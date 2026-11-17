#!/usr/bin/env python3
"""build_traceability_v2.py - checks the numerical claims of the v2 paper against the archived v2 results.
Run from the repository root of the v2 worktree after exp42_v2_summary.py and exp43_v2_tables.py.
Writes audit/03_traceability_v2.csv and exits with 1 if a claim fails."""
import csv, json, os, sys
R = 'results/'
def L(f): return json.load(open(R + f))
def g(d, path):
    for p in path.split('/'):
        d = d[int(p)] if isinstance(d, list) else d[p]
    return d
C = [  # id, location, description, file, json path, expected, tolerance
 ('V-01', 'sec:dataset_merge', 'test windows', 'local_exp42_v2_summary.json', 'n_test', 529, 0),
 ('V-02', 'sec:dataset_merge', 'test prevalence', 'local_exp42_v2_summary.json', 'prevalence', 0.406, 0.0006),
 ('V-03', 'tab:station_coverage', 'Alicante test windows', 'local_exp43_v2_tables.json', 'station_counts/Alicante/n', 220, 0),
 ('V-04', 'sec:architectures', 'RF tuned depth', 'json/v2_tuned_params.json', 'RF/params/max_depth', 11, 0),
 ('V-05', 'sec:architectures', 'RF tuned trees', 'json/v2_tuned_params.json', 'RF/params/n_estimators', 350, 0),
 ('V-06', 'tab:case', 'climatology AUC', 'local_exp42_v2_summary.json', 'models/CLIM/mean', 0.6851, 0.00006),
 ('V-07', 'tab:case', 'RF AUC', 'local_exp42_v2_summary.json', 'models/RF/mean', 0.6784, 0.00006),
 ('V-08', 'tab:case', 'XGB AUC', 'local_exp42_v2_summary.json', 'models/XGB/mean', 0.6684, 0.00006),
 ('V-09', 'tab:case', 'LSTM AUC', 'local_exp42_v2_summary.json', 'models/PureLSTM/mean', 0.6638, 0.00006),
 ('V-10', 'tab:case', 'LSTM+QNN AUC', 'local_exp42_v2_summary.json', 'models/LSTM_QNN/mean', 0.6587, 0.00006),
 ('V-11', 'tab:case', 'LSTM+QTL AUC', 'local_exp42_v2_summary.json', 'models/LSTM_QTL/mean', 0.6548, 0.00006),
 ('V-12', 'tab:case', 'Full Fourier-1 AUC', 'local_exp42_v2_summary.json', 'models/FOURIER_FULL1/mean', 0.6671, 0.00006),
 ('V-13', 'tab:budget', 'SE one AUC RF iid', 'local_exp43_v2_tables.json', 'noise_budget/RF/se_iid', 0.023, 0.0006),
 ('V-14', 'tab:budget', 'seed share QNN', 'local_exp43_v2_tables.json', 'noise_budget/LSTM_QNN/share_iid', 0.252, 0.0006),
 ('V-15', 'tab:paired_ci', 'RF-XGB gap', 'local_exp42_v2_summary.json', 'pairs/RF-XGB/gap', 0.0100, 0.00006),
 ('V-16', 'tab:paired_ci', 'RF-LSTM p', 'local_exp42_v2_summary.json', 'pairs/RF-PureLSTM/p_delong', 0.186, 0.0006),
 ('V-17', 'tab:paired_ci', 'RF-LSTM cluster/iid', 'local_exp42_v2_summary.json', 'pairs/RF-PureLSTM/ratio_year_iid', 0.59, 0.006),
 ('V-18', 'tab:qnn_paired', 'RF-QNN gap', 'local_exp42_v2_summary.json', 'pairs/RF-LSTM_QNN/gap', 0.0197, 0.00006),
 ('V-19', 'tab:qnn_paired', 'RF-QNN p', 'local_exp42_v2_summary.json', 'pairs/RF-LSTM_QNN/p_delong', 0.088, 0.0006),
 ('V-20', 'tab:qnn_paired', 'RF-QNN MDE', 'local_exp42_v2_summary.json', 'pairs/RF-LSTM_QNN/mde_iid', 0.032, 0.0006),
 ('V-21', 'tab:qnn_paired', 'RF-QTL p', 'local_exp42_v2_summary.json', 'pairs/RF-LSTM_QTL/p_delong', 0.019, 0.0006),
 ('V-22', 'tab:controls', 'Full Fourier-1 - QNN gap', 'local_exp42_v2_summary.json', 'pairs/FOURIER_FULL1-LSTM_QNN/gap', 0.0084, 0.00006),
 ('V-23', 'tab:controls', 'Full Fourier-1 - QNN p', 'local_exp42_v2_summary.json', 'pairs/FOURIER_FULL1-LSTM_QNN/p_delong', 0.064, 0.0006),
 ('V-24', 'tab:controls', 'frozen random p', 'local_exp42_v2_summary.json', 'pairs/FROZEN_RANDOM-LSTM_QNN/p_delong', 0.002, 0.0006),
 ('V-25', 'sec:budget', 'RF-QNN CR2 p', 'local_exp35_equivalence_fewclusters.json', 'pairs/RF_2273-LSTM_QNN/p_cr2_t', 0.029, 0.0006),
 ('V-26', 'sec:budget', 'RF-QNN wild p', 'local_exp35_equivalence_fewclusters.json', 'pairs/RF_2273-LSTM_QNN/p_wild_cluster', 0.022, 0.0006),
 ('V-27', 'sec:budget', 'RF-LSTM cross share', 'local_exp37_cancellation.json', 'RF-LSTM/cross_share', -0.42, 0.006),
 ('V-28', 'sec:budget', 'RF-QNN permutation p', 'local_exp37_cancellation.json', 'RF-QNN/perm_p_lower', 0.12, 0.006),
 ('V-29', 'sec:budget', 'RF-QNN station-month demeaned ratio', 'local_exp37_cancellation.json', 'RF-QNN/ratio_year_demeaned_station_month', 1.06, 0.006),
 ('V-30', 'sec:applying', 'degree-1 R2 trained', 'local_exp34_circuit_spectrum.json', 'summary/r2_test_full_d1/mean', 0.984, 0.0006),
 ('V-31', 'sec:applying', 'degree-1 R2 random weights', 'local_exp34_circuit_spectrum.json', 'summary/r2_test_full_d1_randomW_same_angles/mean', 0.974, 0.0006),
 ('V-32', 'sec:applying', 'teacher oracle AUC', 'local_exp38_teacher_control_interface.json', 'oracle_auc_test', 0.846, 0.0006),
 ('V-33', 'sec:applying', 'teacher circuit AUC', 'local_exp38_teacher_control_interface.json', 'models/CIRCUIT/mean', 0.706, 0.0006),
 ('V-34', 'sec:applying', 'teacher Full Fourier-2 AUC', 'local_exp38_teacher_control_interface.json', 'models/FOURIER_FULL2/mean', 0.810, 0.0006),
 ('V-35', 'sec:applying', 'shots 1000 loss', 'local_exp31_shot_noise.json', 'by_shots/1000/mean_auc_loss', 0.0016, 0.00006),
 ('V-36', 'tab:brier', 'climatology Brier', 'local_exp43_v2_tables.json', 'brier_raw/CLIM/brier', 0.2184, 0.00006),
 ('V-37', 'tab:brier', 'RF BSS as trained', 'local_exp43_v2_tables.json', 'brier_raw/RF/bss', -0.046, 0.0006),
 ('V-38', 'tab:within', 'climatology within-station AUC', 'local_exp43_v2_tables.json', 'within_station/CLIM/within', 0.667, 0.0006),
 ('V-39', 'tab:stationmonth', 'RF station-month AUC', 'local_exp43_v2_tables.json', 'recal_stationmonth/RF_2273/auc', 0.514, 0.0006),
 ('V-40', 'tab:seas5', 'SEAS5 - climatology', 'local_exp33_seas5_orig.json', 'pairs/SEAS5_QM_L1-CLIM/delta', 0.0433, 0.00006),
 ('V-41', 'tab:seas5', 'SEAS5 - climatology p', 'local_exp33_seas5_orig.json', 'pairs/SEAS5_QM_L1-CLIM/p_delong', 0.034, 0.0006),
 ('V-42', 'tab:seas5', 'SEAS5 block CI lower', 'local_exp33_seas5_orig.json', 'pairs/SEAS5_QM_L1-CLIM/ci_block/0', -0.005, 0.0006),
 ('V-43', 'sec:seas5', 'SEAS5 station-month AUC', 'local_exp33_seas5_stationmonth.json', 'auc/SEAS5_QM_L1/mean', 0.639, 0.0006),
 ('V-44', 'tab:panel2', 'panel 2 test windows', 'local_exp32_panel2_med2v2.json', 'pooled/n_test', 884, 0),
 ('V-45', 'tab:panel2', 'panel 2 climatology AUC', 'local_exp32_panel2_med2v2.json', 'pooled/climatology_auc', 0.716, 0.0006),
 ('V-46', 'tab:panel2', 'panel 2 LSTM - QNN', 'local_exp32_panel2_med2v2.json', 'pooled/pairs/PureLSTM-LSTM_QNN/delta', 0.0110, 0.00006),
 ('V-47', 'fig:resolution', 'resolution curve full MDE RF-QNN means', 'local_exp43_v2_tables.json', 'resolution_curve/19/mean_RF_QNN', 0.032, 0.0006),
]
rows, bad = [], 0
for cid, loc, desc, f, path, exp, tol in C:
    try:
        v = g(L(f), path); ok = abs(float(v) - exp) <= tol + 1e-12
    except Exception as e:
        v, ok = f'ERROR {e}', False
    bad += not ok; rows.append([cid, loc, desc, f, path, exp, v, 'verified' if ok else 'FAILED'])
os.makedirs('audit', exist_ok=True)
with open('audit/03_traceability_v2.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['id', 'location', 'claim', 'file', 'path', 'paper', 'archive', 'status']); w.writerows(rows)
for r in rows:
    if r[-1] != 'verified': print(r)
print(f'{len(rows) - bad}/{len(rows)} verified')
sys.exit(1 if bad else 0)
