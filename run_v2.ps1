# run_v2.ps1 - end-to-end v2 pipeline of the paper (Windows PowerShell 5.1+), run from the repository root.
#
#   powershell -ExecutionPolicy Bypass -File .\run_v2.ps1 -RawDir <AEMET daily files> `
#       -Panel2Stations panel2.json -Panel2Dir <second-panel daily files> [-Threads 4] [-SkipLbL]
#
# Needs the AEMET daily records of the first panel (not redistributed; see README, "Data provenance").
# Every step logs to logs_v2\<step>.log and writes logs_v2\<step>.done on success; rerunning the command skips
# finished steps (delete a .done file to force one). Training steps stop the run on error; analysis steps are
# logged and the run continues. The training steps are the ones that produced the archived v2 results
# (markers in logs_v2\); total about 15 h on a 4-core CPU.
#
# Without the raw data, everything after training can be checked from the archives:  make check; make test
param([Parameter(Mandatory = $true)][string]$RawDir, [string]$Panel2Stations = "panel2.json", [string]$Panel2Dir = "data_panel2",
      [int]$Threads = 4, [switch]$SkipLbL)
$ErrorActionPreference = "Stop"
if (-not (Test-Path "requirements-v2.lock")) { throw "run from the repository root" }
$RawDir = (Resolve-Path $RawDir).Path
New-Item -ItemType Directory -Force logs_v2, results\json | Out-Null
$L = "code\local_experiments"

# ---------------------------------------------------------------- pinned environment (requirements-v2.lock)
if (-not (Test-Path ".venv-v2")) {
  uv venv .venv-v2 --python 3.12; if ($LASTEXITCODE) { throw "uv venv failed" }
  uv pip install --python .venv-v2\Scripts\python.exe --require-hashes --index-strategy unsafe-best-match -r requirements-v2.lock
  if ($LASTEXITCODE) { throw "install failed" }
}
$Py = (Resolve-Path ".venv-v2\Scripts\python.exe").Path
& $Py -c "import torch, pennylane, sklearn, xgboost, optuna; print('env OK', torch.__version__, pennylane.__version__, sklearn.__version__, xgboost.__version__, optuna.__version__)"
if ($LASTEXITCODE) { throw "environment smoke test failed" }
uv pip freeze --python $Py | Out-File -Encoding utf8 results_v2_environment.txt

$env:LOR_DATASET = "v2"; $env:PYTHONHASHSEED = "0"; $env:PYTHONUTF8 = "1"
$env:OMP_NUM_THREADS = "$Threads"; $env:MKL_NUM_THREADS = "$Threads"; $env:OPENBLAS_NUM_THREADS = "$Threads"
$th = "--threads $Threads"
function Step($name, $cmd, [switch]$Soft) {
  $mark = "logs_v2\$name.done"
  if (Test-Path $mark) { Write-Host ">> [$name] already done, skipped" -ForegroundColor DarkGray; return }
  $log = "logs_v2\$name.log"; Write-Host "`n>> [$name] $cmd" -ForegroundColor Cyan; $t = Get-Date
  cmd /c "`"$Py`" $cmd > $log 2>&1"; $rc = $LASTEXITCODE
  Write-Host ("   {0:N1} min, exit {1}" -f ((Get-Date) - $t).TotalMinutes, $rc)
  if ($rc -eq 0) { Set-Content $mark (Get-Date -Format s) }
  else { Get-Content $log -Tail 15; if (-not $Soft) { throw "step $name failed (log: $log)" } }
}

# ---------------------------------------------------------------- data and training
Step "00_etl"        "code\etl_v2.py --datos_dir `"$RawDir`" --out $L\datos_limpios_multiestacion.csv --report results\etl_v2_report.json"
Step "01_meta"       "$L\export_test_meta.py --csv $L\datos_limpios_multiestacion.csv --out results\test_meta.csv"
Step "01_panel"      "scripts\export_monthly_panel.py --daily $L\datos_limpios_multiestacion.csv --out results\panel_v2_monthly.csv"
if (-not (Test-Path results\years_te.npy)) { & $Py -c "import pandas as pd, numpy as np; m=pd.read_csv('results/test_meta.csv'); np.save('results/years_te.npy', pd.to_datetime(m.fecha).dt.year.values); print(len(m), 'test rows')" }
Step "02_tune"       "$L\exp40_v2_tune.py --trials 50"                                              # 15-30 min
$models = "RF_2273 PureLSTM CTRL_LINEAR4D LSTM_QNN LSTM_PIQNN LSTM_QTL" + $(if ($SkipLbL) { "" } else { " LBL_QNN" })
Step "03_core"       "$L\exp17_hybrids_multiseed.py --fix-et0 --models $models $th"                 # 3-6 h
# causal-imputation check of the neural models (S1, lookahead audit): same training, V2_CAUSAL=1, separate folder
$env:V2_CAUSAL = "1"; New-Item -ItemType Directory -Force results\causal | Out-Null
Step "03b_core_causal" "$L\exp17_hybrids_multiseed.py --fix-et0 --models PureLSTM LSTM_QNN $th --out-dir results\causal"   # ~2-3 h
$env:V2_CAUSAL = "0"
New-Item -ItemType Directory -Force results\causal_linear4d | Out-Null; $env:V2_CAUSAL = "1"
Step "03c_linear4d_causal" "$L\exp17_hybrids_multiseed.py --fix-et0 --models CTRL_LINEAR4D $th --out-dir results\causal_linear4d"   # ~30 min; completes the second primary family (review 2026-11-19)
$env:V2_CAUSAL = "0"
foreach ($ext in ".json", "_probs.npz") {                                                           # v1 file names, read by the analyses
  Copy-Item "results\json\results_hybrids_trainonly_fixet0$ext" "results\json\results_hybrids_trainonly_fixet0_p60_28.1$ext" -Force }
Step "04_classical"  "$L\exp41_v2_classical.py --seeds 15"                                          # ~5 min
Step "05_controls"   "$L\exp27_controls.py $th"                                                     # ~1.5-2 h
Step "06_full"       "$L\exp30_fourier_full.py $th"                                                 # ~45 min
Step "07_shots"      "$L\exp31_shot_noise.py --skip-training $th"                                   # ~1 h
Step "08_spectrumA"  "$L\exp34_circuit_spectrum.py --part A"
Step "09_spectrumB"  "$L\exp34_circuit_spectrum.py --part B $th"                                    # ~30 min
Step "10_teacherIF"  "$L\exp38_teacher_control.py --mode interface --seeds 5"                       # ~15 min
Step "11_teacherE2E" "$L\exp38_teacher_control.py --mode end-to-end --seeds 5" -Soft                # ~1-1.5 h
foreach ($lab in "orig", "stationmonth") {
  Step "12_exp28_$lab" "$L\exp28_labels_recalibration.py --label $lab $th"                         # ~1 h each
  Step "13_seas5_score_$lab"   "$L\exp33_seas5_reference.py --label $lab --step score" -Soft
  Step "14_seas5_analyze_$lab" "$L\exp33_seas5_reference.py --label $lab --step analyze" -Soft
}
if (Test-Path $Panel2Stations) {
  $Panel2Stations = (Resolve-Path $Panel2Stations).Path; $Panel2Dir = (Resolve-Path $Panel2Dir).Path
  Step "15_panel2_etl"   "$L\exp32_second_panel.py --stations `"$Panel2Stations`" --data-dir `"$Panel2Dir`" --tag med2v2 --step etl"
  Step "15_panel2_monthly" "scripts\export_monthly_panel.py --daily results\panel2\med2v2\datos.csv --out results\panel2\med2v2\panel_monthly.csv"
}
Step "16_panel2_train" "$L\exp32_second_panel.py --tag med2v2 --step train"                         # ~3.5 h
Step "17_panel2_an"    "$L\exp32_second_panel.py --tag med2v2 --step analyze" -Soft

# ---------------------------------------------------------------- analyses from the archived predictions (no training)
$CL = "results\json\results_rerun_train_only_canonical_probs.npz"
Step "a01_paired_classical" "$L\exp10_paired_ci.py --probs $CL --prefix train_only --pairs RF-XGB RF-LSTM XGB-LSTM --B 2000 --out results\json\results_exp10_paired_ci.json" -Soft
Step "a12_within_year_rob"  "$L\exp26_within_year_robustness.py" -Soft
Step "a14_equiv_fewcl"      "$L\exp35_equivalence_fewclusters.py" -Soft
Step "a15_cancellation"     "$L\exp37_cancellation.py --probs results\json\results_hybrids_trainonly_fixet0_p60_28.1_probs.npz --meta results\test_meta.csv --out results\local_exp37_cancellation.json" -Soft
Step "a16_panel_sim"        "$L\exp15_panel_matched_simulation.py --probs $CL --meta results\test_meta.csv --out results\json\results_exp15_panel_sim.json" -Soft
Step "a17b_lookahead"       "$L\exp47_truncation_audit.py"                                          # ~1.5 h; S1 lookahead audit
Step "a17c_primary_sim"     "$L\exp48_primary_calibration.py --reps 600 --B 299"                    # ~2.5 h; calibration of the primary procedure
Step "a17d_minor"           "$L\exp49_review_minor.py"                                              # ~15 min; within-cell AUC, SEAS5+clim, analytic AUC
Step "a17e_teacher_grid"    "$L\exp50_teacher_grid.py --seeds 5" -Soft                              # ~2-3 h; repeated positive control
Step "a17f_primary_sim_2yr" "$L\exp48_primary_calibration.py --reps 600 --B 299 --blocks 2 --scenarios iid specific_ar1 station_year_specific_ar1 --out results\local_exp48_primary_calibration_2yr.json"   # ~40 min
Step "a17g_review_extras"   "$L\exp51_review_extras.py"                                             # ~2 min; causal neural refit, SEAS5 / panel-2 CR2, blocks, leave-one-year-out
Step "a17j_review_closing"  "$L\exp55_review_closing.py"                                            # ~1 min; causal primary families, panel-2 CR2 MDE, SEAS5 under CR2
Step "a17k_clusters_rows"   "$L\exp54_power_clusters_rows.py"                                       # ~10 min; simulation of clusters against rows
Step "a17h_emulation"       "$L\exp52_emulation_substitution.py --part E S --seeds 15 $th"          # ~2.5 h; exact emulation, heads on common frozen angles
Step "a17i_external"        "$L\exp53_external_benchmark.py"                                        # ~10 min; downloads California Housing to data\external
Step "a18_summary"          "$L\exp42_v2_summary.py results"                                        # ~10 min; Tables 3, 5-7
Step "a19_tables"           "$L\exp43_v2_tables.py results"                                         # ~1 min; Tables 2, 9-11, S10
Step "a20_partition"        "scripts\make_partition_table.py"                                       # Table 1
Step "a21_forest"           "scripts\make_forest_plot.py"                                           # Figure 1
Step "a21b_resolution"      "scripts\make_resolution_plot.py"                                       # Figure 2
Step "a21c_design"         "scripts\make_design_plot.py"                                           # study design figure
Step "a21d_calibration"    "scripts\make_calibration_plot.py"                                      # calibration figure
Step "a21f_sensitivity"     "scripts\make_sensitivity_plot.py"                                      # sensitivity figure (supplement)
Step "a21g_clusters_rows"   "scripts\make_clusters_rows_plot.py"                                    # clusters against rows figure
Step "a21e_tables"         "scripts\fill_paired_tables.py"                                         # paired-table rows from the archive
Step "a22_trace_v2"         "scripts\build_traceability_v2.py"
Step "a23_manuscript"       "-m pytest tests\manuscript -q"                                         # paper/main.tex vs archives
Write-Host "`nDONE. Check: git status (results\, paper\figs\, audit\) and make test" -ForegroundColor Green
