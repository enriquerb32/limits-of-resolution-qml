# Phase 0 — Inventory (read-only)

Branch: `release-audit` (created from `main` @ `7c50ec3`). No file other than this one was written.

## 1. Git state at start

- Tracked: 103 files. HEAD = `7c50ec3`.
- Uncommitted modifications (pre-existing, not by audit): `paper/main.tex`, `paper/figs/fig_noise_budget.pdf`, `paper/figs/fig_positive_control.pdf`, `results/json/results_rerun_train_only_canonical.json` (key renamed `train_only_acp_test_n` → `train_only_acp_test_n_positive`), `paper/main.pdf` deleted.
- Untracked and **cited by the paper**: `paper/supp.tex`, `paper/figs/fig_paired_ci.*`, `paper/figs/fig_sim.*`, `code/local_experiments/exp10..exp15*.py`, `export_test_meta.py`, `export_precip_raw.py`, `trainonly_dataset.py`, `results/json/results_exp1[0-4]*.json`, `results/json/results_qml_core_trainonly{.json,_probs.npz}`.
- Untracked, other: `data/datos_limpios_multiestacion.csv` (daily AEMET records), root `precip_raw.csv`, root `month_rows.csv`, `data/years_te.npy`, notebook `notebooks/20260630_Estudio_Completo (5).ipynb` (with outputs, Colab paths), and a full duplicate set of data/results inside `code/local_experiments/`.
- Local-only dirs (not tracked): `.venv/`, `archive/`, `backup_20261016/` (ignored), `__pycache__/`, `solo_lo_indicado.ps1` (ignored).

## 2. Tree (tracked + relevant untracked; `U` = untracked)

```
.gitignore .zenodo.json CITATION.cff Dockerfile LICENSE LICENSE-DATA README.md
environment.yml requirements.txt make_figs.py
month_rows.csv (U)  precip_raw.csv (U)
checkpoints/   01_data.pt … 10_multisemilla_clasico.pt, ablacion_ia_racha_resultados.pt,
               lambda_sensibilidad_resultados.pt, CHECKSUMS.sha256
code/          main.py etl.py fisica.py features.py modelos*.py validacion.py robustez_estadistica.py
               experimento_ruido.py experimentos_control.py control_confusor_ciudad.py
               visualizacion.py q1_style.py
code/local_experiments/
               exp1..exp8_*.py noise_utils.py run_all_local_experiments.py
               exp10..exp15_*.py export_test_meta.py export_precip_raw.py trainonly_dataset.py (U)
               + duplicates (U): datos_limpios_multiestacion.csv panel_checkpoint.csv precip_raw.csv
                 month_rows.csv month_rows_raw.csv test_meta.csv years_te.npy test_probe.txt
                 results_exp10..15*.json results_qml_core_trainonly_probs.npz
                 results_rerun_train_only_canonical{.json,_probs.npz}
data/          panel_checkpoint.csv | datos_limpios_multiestacion.csv (U) years_te.npy (U)
notebooks/     01_full_exploratory_study.ipynb | 20260630_Estudio_Completo (5).ipynb (U)
paper/         main.tex supp.tex(U) figs/{fig_noise_budget,fig_positive_control,fig_paired_ci(U),fig_sim(U),
               figura_auc_violin,figura_single_run_illusion}.{pdf,png}
results/       local_exp1..8_*.json results_master.csv panel_checkpoint.csv
results/json/  01_data.json … 10_multisemilla_clasico.json ablacion_ia_racha_resultados.json
               results_capacity_matched_lstm.json results_circuit_depths.json results_hybrid_perseed_probs.json
               results_rerun_train_only_canonical{.json,_probs.npz} results_wp3_accounting.json
               results_exp10..14*.json (U) results_qml_core_trainonly{.json,_probs.npz} (U)
scripts/       build_results_master.py make_tables.py multiseed_stats.py recompute_metrics.py
               verify_parameter_counts.py export_colab_wp3.py colab_appendix_d_lambda_sweep.py
               colab_appendix_e_circuit_depths.py execute_wp6_restructuring.py validate_todo_code.py
tables/        table1/4/6/7/8/12_*.tex
```

Identical-content duplicates (MD5): `panel_checkpoint.csv` ×3 (data/, results/, code/local_experiments/); `datos_limpios_multiestacion.csv` ×2; `years_te.npy` ×2; `month_rows.csv` ×2; `precip_raw.csv` ×2; all `results_exp10..14*.json` and both `.npz` ×2. **Differs**: `results_rerun_train_only_canonical.json` (results/json vs code/local_experiments). **Only copy outside results/**: `results_exp15_panel_sim.json` (in `code/local_experiments/` only).

## 3. Script inventory

| Script | Purpose | Inputs | Outputs | Feeds (paper) |
|---|---|---|---|---|
| `code/local_experiments/exp7_noise_budget.py` | SE (DeLong, year-block), seed SD/share, MDE | `results/json/results_rerun_train_only_canonical_probs.npz`, `results/json/01_data.json` (years via noise_utils), `results/results_master.csv` | `results/local_exp7_noise_budget.json` | main Tab `budget`; supp Tab `mde`; App. A seed shares |
| `…/exp8_positive_control.py` | Known-δ degradation of RF, detection rates | same npz, 01_data.json | `results/local_exp8_positive_control.json` | supp Tab `positive`, Fig `positive_control` |
| `…/noise_utils.py` | DeLong, AUC matrix, year inference | 01_data.json | — | lib for exp7/8 |
| `…/exp10_paired_ci.py` (U) | Seed-mean Δ, paired DeLong / year / two-way bootstrap, MDE | `--probs`, `--extra`, `--panel`/`--years` | `results/json/results_exp10_paired_ci.json` (pairs RF-XGB, RF-LSTM, XGB-LSTM, RF_2273-LSTM_QNN, CTRL_LINEAR4D-LSTM_QNN, PureLSTM-LSTM_QNN) | main Tab `paired_ci`, `qnn_paired` |
| `…/exp11_simulation_known_truth.py` (U) | Known-truth simulation | args | `results/json/results_exp11_sim.json` (stored config: ρ=0.75, δ∈{0..0.05}, no τ_m, no Dirichlet) | intended: main Tab `sim`, supp Tab `sim_all` — **see F-04** |
| `…/exp12_monthly_coverage_audit.py` (U) | Monthly coverage / label fragility | `datos_limpios_multiestacion.csv`, `--precip_raw` | `results/json/results_exp12_coverage.json`, `month_rows.csv` | main Tab `coverage`; supp §Monthly coverage |
| `…/exp13_label_sensitivity.py` (U) | S0–S7 re-evaluation of saved predictions | npz(s), `test_meta.csv`, `month_rows(_raw).csv` | `results/json/results_exp13_label_sensitivity.json` | main Tab `label_sens`, `label_sens_delta`, `label_sens_qnn` |
| `…/exp14_innerjoin_audit.py` (U) | Rain discarded by inner join | daily csv, `precip_raw.csv`, `month_rows.csv` | `results/json/results_exp14_innerjoin.json`, `month_rows_raw.csv` | S6/S7; main §Label validity |
| `…/exp15_panel_matched_simulation.py` (U) | Panel-matched known-truth sim | npz, `test_meta.csv` | `code/local_experiments/results_exp15_panel_sim.json` | main §Simulation (panel-matched check) |
| `…/export_test_meta.py` (U) | Station/target month of 528 test rows | daily csv via `trainonly_dataset.py` | `test_meta.csv` | exp13, exp15 |
| `…/trainonly_dataset.py` (U) | Train-only dataset builder | daily csv | in-memory | export_test_meta |
| `…/export_precip_raw.py` (U) | Daily precip from raw AEMET files | raw AEMET dir (default `C:/Users/enriq/...`) | `precip_raw.csv` | exp12/exp14 |
| `…/exp1_climatology_baseline.py` | Month×station climatology | 01_data.json | `results/local_exp1_climatology.json` | main Tab `brier` (climatology row) |
| `…/exp2_persistence_baseline.py` | Persistence baselines | 01_data.json | `results/local_exp2_persistence.json` | supp (if cited) |
| `…/exp3_brier_and_reliability.py` | Brier, BSS, reliability (single runs) | 01–04 json | `results/local_exp3_brier_reliability.json` | main Tab `brier` |
| `…/exp4_year_block_bootstrap.py` | Year-block CIs, single runs | 01–04 json | `results/local_exp4_year_block_bootstrap.json` | supp §Probabilistic verification |
| `…/exp5_pca4_controls.py` | Classical PCA-4 controls vs QKSVM | 01, 04 json | `results/local_exp5_pca4_controls.json` | supp Tab `bottleneck_control`(?) |
| `…/exp6_tost_rope_equivalence.py` | TOST/ROPE on 15-seed AUCs | 09, 10 json | `results/local_exp6_tost_rope.json` | supp |
| `…/run_all_local_experiments.py` | Runs exp1–6 | — | — | — |
| **`colab_qml_core_trainonly.py`** | 15-seed core QML experiment | — | `results_qml_core_trainonly{.json,_probs.npz}` | main Tab `case`, `qnn_paired`, `label_sens_qnn`; supp Tab `core_spec` — **NOT IN REPO** |
| `make_figs.py` | Fig noise budget, Fig positive control | **hard-coded numbers** (no file read) | `paper/figs/fig_noise_budget.pdf`, `fig_positive_control.pdf` | supp Fig `positive_control`; `fig_noise_budget.pdf` is **not included** by either .tex |
| (none) | — | — | `paper/figs/fig_paired_ci.pdf`, `fig_sim.pdf` | main Fig `paired_ci`, `sim` — **no generator in repo** |
| `scripts/make_tables.py` | LaTeX tables from results_master.csv | `results/results_master.csv` | `tables/*.tex` | none (`tables/` is not `\input` by either .tex) |
| `scripts/build_results_master.py` | Aggregates results_master.csv | checkpoints / json | `results/results_master.csv` | exp7 (hybrid proxy rows), supp historical tables |
| `scripts/multiseed_stats.py` | Friedman/Wilcoxon/Cliff on 15-seed AUC | 09, 10, ablacion json | stdout | supp historical multiseed |
| `scripts/recompute_metrics.py` | Single-run metrics, Wilson CIs | 01–04, 08 json | stdout | supp Tab `single_run_results`, `recall_wilson_ci` |
| `scripts/verify_parameter_counts.py` | Analytic parameter counts | none | stdout | supp Tab `parameter_accounting`; main 2,613 / 2,609 (to verify, Phase 1) |
| `scripts/export_colab_wp3.py` | state_dict accounting (Colab) | torch models | `results_wp3_accounting.json` | supp Tab `parameter_accounting` |
| `scripts/colab_appendix_d_lambda_sweep.py` | λ_cc sweep (Colab) | 01_data.pt | lambda json | supp §λ sweep |
| `scripts/colab_appendix_e_circuit_depths.py` | Gate counts/depths (Colab) | pennylane | `results_circuit_depths.json` | supp Tab `circuit_gates` |
| `scripts/execute_wp6_restructuring.py` | One-off text surgery on `main_revised.tex` | `main_revised.tex` (absent) | — | none (dead) |
| `scripts/validate_todo_code.py` | AST syntax check of snippets for `todo_experiments.md` | — | stdout | none (dead; README mislabels it as "15-seed classical rerun, ~25 min") |
| `code/main.py` + `etl, fisica, features, modelos*, validacion, robustez_estadistica, experimento_ruido, experimentos_control` | Historical canonical pipeline (Colab) | raw AEMET `./datos` | `checkpoints/01..10*.pt` | supp historical/descriptive tables; supp Tab `noise_degradation` |
| `code/visualizacion.py` | 7 legacy figures | results/json | `figura_*.pdf` | none (not included by either .tex) |
| `code/control_confusor_ciudad.py` | City-confounding diagnostic | ds | stdout | supp §Feature importance(?) |
| `code/q1_style.py` | Matplotlib style | — | — | lib |

## 4. LaTeX build (scratch copy, pdflatex ×3 alternating main/supp, `xr`)

| Doc | Pages | Errors | Undefined refs/cites | Overfull | Underfull |
|---|---|---|---|---|---|
| main.pdf | 22 | 0 | 0 | 1 (42.2 pt, main.tex:330–344, Tab `label_sens`) | 13 (main.tex:156–160, Tab `expmap` p-columns) |
| supp.pdf | 34 | 0 | 0 | 0 | 2 (supp.tex:153, Tab `core_spec` caption) |

Bibliography: inline `thebibliography` in both files (no .bib). `\todo` macro defined in both preambles; one open `\todo` (Zenodo DOI, main.tex:537).

## 5. Environment

- `.venv`: Python 3.12.0; numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1, matplotlib 3.11.2, seaborn 0.13.2 (no torch/pennylane/xgboost installed locally).
- `requirements.txt` / `environment.yml`: lower bounds only (`>=`), no pins; Colab versions undocumented.
