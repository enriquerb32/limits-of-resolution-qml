# Limits of Resolution in Small Machine-Learning Benchmarks

Code, data and reproducibility package for **"Limits of Resolution in Small Machine-Learning Benchmarks: A Noise Budget with a Quantum Case Study"** (Enrique Real, ORCID [0009-0006-3862-9797](https://orcid.org/0009-0006-3862-9797)).

[![CI](https://github.com/enriquerb32/limits-of-resolution-qml/actions/workflows/ci.yml/badge.svg)](https://github.com/enriquerb32/limits-of-resolution-qml/actions/workflows/ci.yml)
[![Reproducibility](https://github.com/enriquerb32/limits-of-resolution-qml/actions/workflows/reproducibility.yml/badge.svg)](https://github.com/enriquerb32/limits-of-resolution-qml/actions/workflows/reproducibility.yml)
[![Coverage: 100%](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](pyproject.toml)
[![Typed: mypy strict](https://img.shields.io/badge/types-mypy%20strict-blue.svg)](pyproject.toml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![PennyLane 0.45](https://img.shields.io/badge/PennyLane-0.45-blueviolet.svg)](https://pennylane.ai/)
[![Code: MIT](https://img.shields.io/badge/Code-MIT-yellow.svg)](LICENSE)
[![Data and text: CC BY 4.0](https://img.shields.io/badge/Data%20%26%20Text-CC%20BY%204.0-lightgrey.svg)](LICENSE-DATA)

> [!IMPORTANT]
> This is **version 2** of the paper and code (corrected data pipeline, tuned baselines, pinned environment; see [What changed in v2](#2-what-changed-in-v2)). The original version is kept unchanged under the git tag [`v1`](https://github.com/enriquerb32/limits-of-resolution-qml/tree/v1), and its archived results are also in `results_v1/`.
>
> **Paper:** [`paper/main.pdf`](paper/main.pdf) · **Supplement:** [`paper/supp.pdf`](paper/supp.pdf)

The repository has two parts:

* **`qml_resolution`** (`src/`): a small, typed and fully tested Python library for the inference the paper relies on: rank AUC, seed-averaged paired DeLong tests, cluster bootstrap and jackknife for clustered test sets, and minimum detectable differences. It depends only on NumPy and SciPy and can be used in any benchmark that stores test-set scores.
* **The research pipeline** (`code/`, `scripts/`, `results/`): data processing, the classical and hybrid quantum models (PennyLane + PyTorch), the 15-seed experiments, and every analysis behind the tables of the paper, with the archived per-seed test probabilities.

---

## 1. What the paper asks and finds

How large an AUC difference can a small benchmark resolve, and how much does the verdict depend on the inferential procedure? The case study is a four-station Western Mediterranean precipitation panel (Alicante, Barcelona, San Javier, Valencia; 1938–2026), with a 1-month-ahead forecast of exceeding a regional 60th percentile and 529 test station-months grouped in 19 test years (2008–2026).

* **Resolution.** The standard error of one AUC (about 0.024) is 3–21 times the seed standard deviation of the classical models. The minimum detectable difference between 15-seed means is 0.022–0.032 AUC and falls only as $N^{-1/2}$; near this limit DeLong and calendar-year cluster procedures disagree.
* **Second panel.** A second four-station panel (same region and test years) matched the predicted resolution, while the model ranking did not replicate. The prediction was written before the analysis: [`audit/prediction_panel2.txt`](audit/prediction_panel2.txt).
* **Quantum controls.** Two hybrid models scored 0.020–0.024 AUC below Random Forest and were not distinguished from classical controls at the same four-angle interface. Fourier surrogates matched the trained circuit, and the same controls detected a circuit-generated signal planted at the interface.
* **References.** No model resolvably improved on a month × station climatology, not even the ECMWF SEAS5 forecast ($+0.043$ AUC, not resolved by the cluster procedures); none discriminated once seasonal and station base rates were removed, and none added information to SEAS5.
* **Protocol.** A stated estimand, a minimum detectable difference computed in advance, separate seed and sampling error, base-rate and operational references, and classical controls at the same interface.

---

## 2. What changed in v2

v2 changes the data pipeline and the environment, not the models or the statistics.

| Issue (v1) | v2 |
|---|---|
| Daily inner join of five files: a missing pressure or wind day removed that day's rain; 23 % of the test months incomplete; Valencia and San Javier almost absent after 2012 | `code/etl_v2.py`: full calendar per station; the label is the monthly sum of the precipitation file alone, defined when at most `V2_MAX_MISSING` (3) days are missing |
| Missing precipitation and sunshine set to 0; covariates forward/backward filled over any gap | missing stays missing; monthly means over observed days (>= 10); gaps of at most `V2_INTERP_MONTHS` (2) months interpolated; windows without features or lags are dropped, not filled |
| Months with < 10 joined days dropped and the rest concatenated (windows could jump over gaps) | every calendar month kept; windows are calendar-contiguous |
| ET0 with a constant wind (u2 = 2 m/s) | ET0 with the observed wind, u2 = 0.748 u10 (FAO-56 eq. 47) |
| JJA ET0 total summed over whatever summer months were present | JJA total requires the three months |
| Tree baselines with manual hyperparameters; XGBoost without search or early stopping | `exp40_v2_tune.py`: Optuna (TPE, 50 trials, declared spaces) for both; XGBoost stops early on the validation AUC |
| Mixed environments (scikit-learn 1.6.1 on Colab, 1.9.1 locally; XGBoost version not recorded) | one environment, `requirements-v2.lock` (hashes), `results_v2_environment.txt`, fixed thread count |

New analyses in v2: `QTL_TENSOR1` in `exp27_controls.py` (the 81-function tensor basis that spans the R_Y-embedding head exactly), the angle distribution and a random-weight R² reference in `exp34_circuit_spectrum.py --part A`, the teacher-circuit positive control `exp38_teacher_control.py`, the cancellation diagnostic `exp37_cancellation.py`, the SEAS5 reference `exp33_seas5_reference.py` and the second panel `exp32_second_panel.py`.

**Switch.** Every script is selected by the environment variable `LOR_DATASET=v2`; without it the scripts behave as v1. The analyses read the v1 archive names (`..._fixet0_p60_28.1_...`); in v2 these files hold v2 results (the v2 $P_{60}$ is stored in the archive JSON as `threshold`). `results/` holds the v2 archives and `results_v1/` the unchanged v1 archives.

---

## 3. Using the library

```bash
pip install -e .            # library only: NumPy + SciPy
pip install -e ".[dev]"     # + test, lint and type-check tools
```

```python
import numpy as np
from qml_resolution import paired_delong, cluster_bootstrap_delta, percentile_ci, jackknife_delta, mde

z = np.load("results/json/results_hybrids_trainonly_fixet0_p60_28.1_probs.npz")
y, years = z["y_te"], np.load("results/years_te.npy")
rf, hybrid = z["RF_2273_probs"], z["LSTM_QNN_probs"]          # (15 seeds, 529 rows) each

test = paired_delong(y, rf, hybrid)                              # seed-mean difference, iid rows
print(f"delta={test.delta:+.4f}  se={test.se:.4f}  p={test.p_value:.3f}  MDE={mde(test.se):.3f}")

block = cluster_bootstrap_delta(y, rf, hybrid, years, n_boot=2000, random_state=0)
two_way = cluster_bootstrap_delta(y, rf, hybrid, years, n_boot=2000, resample_seeds=True)
print("year-block CI", percentile_ci(block), "years x seeds CI", percentile_ci(two_way))
print("jackknife", jackknife_delta(y, rf, hybrid, years).ci())
```

| Module | Contents |
|---|---|
| `qml_resolution.auc` | `rank_auc`, `delong_components`, `delong_covariance`, `paired_delong` (one run or S seeds per model) |
| `qml_resolution.resampling` | `cluster_bootstrap_aucs`, `cluster_bootstrap_delta` (paired; optional seed resampling), `jackknife_delta`, `percentile_ci` |
| `qml_resolution.power` | `mde`, `mde_factor`, `hanley_mcneil_se`, `paired_mde_hanley_mcneil`, `seed_share` |

---

## 4. Installation of the full pipeline

Python 3.12 is the version used for the Docker image and the reported results.

```bash
git clone https://github.com/enriquerb32/limits-of-resolution-qml.git
cd limits-of-resolution-qml

# Option A: uv with the hashed lockfile (recommended, exact versions)
uv venv .venv --python 3.12
source .venv/bin/activate       # Windows: .venv\Scripts\activate
uv pip install --require-hashes -r requirements-v2.lock     # v2 environment (v1: requirements.lock)
uv pip install --no-deps -e .

# Option B: pip with version ranges
python -m venv .venv && source .venv/bin/activate
pip install -e ".[pipeline,dev]"

# Option C: conda
conda env create -f environment.yml && conda activate qml-resolution
```

The lockfile fixes exact versions with SHA-256 hashes (`requirements-v2.lock` from `requirements-v2.in`; the versions used for the v2 results are recorded in `results_v2_environment.txt`). Hardware and BLAS threading can still change floating-point summation order in the last digits; for the closest match set `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`.

---

## 5. Reproducing the paper

Everything except model training runs from the archived results in `results/` and needs no GPU.

The headline v2 quantities and tables are computed from the archives, without training, by:

```bash
python code/local_experiments/exp42_v2_summary.py results   # AUCs, paired DeLong, cluster SEs, bootstraps, MDEs -> results/local_exp42_v2_summary.json
python code/local_experiments/exp43_v2_tables.py results    # noise budget, Brier, within-station AUC, resolution curve -> results/local_exp43_v2_tables.json
```

To rerun any other script on the v2 data, set `LOR_DATASET=v2` first (PowerShell: `$env:LOR_DATASET = "v2"`; bash: `export LOR_DATASET=v2`).

```bash
make check       # figures, partition table, parameter counts, traceability audits and manuscript tables
make manuscript  # every number in the main-text tables of paper/main.tex against the archived v2 results
make test        # full test suite (library, quantum circuit and controls, manuscript, archived-result regressions)
```

How the claims of the manuscript are tied to the archives:

* `tests/manuscript/test_v2_tables.py` reads the numbers **as printed** in the 13 main-text tables and the abstract of `paper/main.tex` and checks that each one is the rounding of the archived value (`results/local_exp42_v2_summary.json`, `local_exp43_v2_tables.json`, `local_exp33_seas5_orig.json`, `local_exp32_panel2_med2v2.json`, `test_meta.csv`). Editing a number in the paper without changing the archive, or the reverse, fails the suite.
* `tests/test_reproducibility.py` regenerates `local_exp42_v2_summary.json` and `local_exp43_v2_tables.json` from the archived per-seed probabilities and requires agreement to $10^{-12}$ (about 10 minutes).
* `tests/regression/` checks the `qml_resolution` library against those archives (its DeLong, AUC and jackknife implementations are independent of the scripts).
* `scripts/build_traceability_v2.py` writes `audit/03_traceability_v2.csv` (47 claims of the v2 text, including Section 2 and the Supplement). `scripts/build_traceability.py` writes `audit/01_traceability_v1.csv` for the v1 manuscript against `results_v1/` (98 claims).

### Retraining (long-running, optional)

```bash
python code/main.py --use-checkpoints --output out         # statistics and figures from stored checkpoints
python code/main.py --skip-qml --force-retrain             # classical baselines only
python code/main.py --force-retrain                        # full canonical pipeline including the circuits
python code/local_experiments/exp17_hybrids_multiseed.py --fix-et0 --p60 28.1 \
    --models RF_2273 PureLSTM CTRL_LINEAR4D LSTM_QNN LSTM_PIQNN LBL_QNN LSTM_QTL   # core experiment (Table 6)
```

Noiseless circuit simulation is CPU-bound and the 15-seed core experiment takes hours on a workstation. Retrained results can differ slightly from the archived ones (hardware, library versions), so the statistics of the paper are computed from the archived probabilities.

### Docker

```bash
docker build -t limits-qml .
docker run --rm -v "$PWD/out:/app/out" limits-qml                                   # figures from results/
docker run --rm -v "$PWD/audit:/app/audit" limits-qml python scripts/build_traceability.py
docker run --rm limits-qml python -m pytest tests/unit tests/regression tests/quantum
```

The image contains `src/`, `code/`, `scripts/`, `results/` and `tests/`; checkpoints and raw data are not baked in.

---

## 6. Mapping paper items to code

Commands run from the repository root. Table numbers refer to `paper/main.pdf`; `run_v2.ps1` runs the whole chain (training needs the raw AEMET records). Each row is checked by `tests/manuscript/test_v2_tables.py` unless stated.

| Paper item | What it is | Archived output | Computed by (no training) | Predictions from (training) |
|---|---|---|---|---|
| **Table 1** | Test windows and prevalence by station | `results/test_meta.csv` | `scripts/make_partition_table.py` | `export_test_meta.py` |
| **Table 2** | Noise budget: SE of one AUC, seed SD, seed share | `results/local_exp43_v2_tables.json` (`noise_budget`) | `exp43_v2_tables.py results` | `exp17`, `exp41`, `exp27` |
| **Table 3** | Paired comparisons of the classical models | `results/local_exp42_v2_summary.json` (`pairs`) | `exp42_v2_summary.py results` | `exp41_v2_classical.py` |
| **Table 4** | 15-seed AUC of every model and control | `results/local_exp42_v2_summary.json` (`models`) | `exp42_v2_summary.py results` | `exp17`, `exp27_controls.py`, `exp30_fourier_full.py` |
| **Tables 5, 6** | Hybrids, Fourier surrogates and frozen circuits against `LSTM+QNN` | `results/local_exp42_v2_summary.json` (`pairs`) | `exp42_v2_summary.py results` | as Table 4 |
| **Figure 1** | Forest plot of Tables 3, 5 and 6 | `paper/figs/fig_forest.pdf` | `scripts/make_forest_plot.py` | — |
| **Table 7** | Brier score, skill and calibration, raw and Platt | `results/local_exp43_v2_tables.json` (`brier_raw`, `recal_orig`) | `exp43_v2_tables.py results` | `exp28_labels_recalibration.py --label orig` |
| **Table 8** | Pooled and within-station AUC | `results/local_exp43_v2_tables.json` (`within_station`) | `exp43_v2_tables.py results` | as Table 4 |
| **Table 9** | Station × month label | `results/local_exp43_v2_tables.json` (`recal_stationmonth`) | `exp43_v2_tables.py results` | `exp28_labels_recalibration.py --label stationmonth` |
| **Table 10** | SEAS5 as an operational reference | `results/local_exp33_seas5_orig.json` | `exp33_seas5_reference.py --label orig --step analyze` | `--step score` (members in `data/seas5/`) |
| **Table 11** | Second panel | `results/local_exp32_panel2_med2v2.json` (`pooled`) | `exp32_second_panel.py --tag med2v2 --step analyze` | `--step etl`, `--step train` |
| **Table 12** | Hanley–McNeil MDE by test size and prevalence | — | `qml_resolution.paired_mde_hanley_mcneil` | — |
| **Table 13** | Empirical resolution curve | `results/local_exp43_v2_tables.json` (`resolution_curve`) | `exp43_v2_tables.py results` | as Table 4 |
| **Supplement** | Tuning (`exp40_v2_tune.py` → `results/json/v2_tuned_params.json`), shot noise (`exp31`), circuit spectrum (`exp34`), few-cluster inference (`exp35`), cancellation (`exp37`), teacher-circuit control (`exp38`), panel-matched simulation (`exp15` → `results/json/results_exp15_panel_sim.json`), known-truth simulation figure (`scripts/make_paper_figs.py --results results_v1 --only sim`) | `results/local_exp3*.json` | see the script names | — (checked by `scripts/build_traceability_v2.py`, not by the table tests) |

### v1 manuscript

The table and figure numbers below refer to the **v1** manuscript (tag [`v1`](https://github.com/enriquerb32/limits-of-resolution-qml/tree/v1)) and its archives, now in `results_v1/` (the commands write to `results/`; pass the `results_v1/` inputs to reproduce v1).

| Paper item | What it is | Command | Output |
|---|---|---|---|
| **Table 1** | Windows by station and partition | `python scripts/make_partition_table.py` | `results/json/partition_counts.json` |
| **Table 3, Table S1** | Noise budget, seed variance, MDEs | `python code/local_experiments/exp7_noise_budget.py` | `results/local_exp7_noise_budget.json` |
| **Table 4, Figure 1** | Paired DeLong and bootstrap CIs, classical models | `python code/local_experiments/exp10_paired_ci.py --probs results/json/results_rerun_train_only_canonical_probs.npz --prefix train_only` | `out/results_exp10_paired_ci.json` (archived: `results/json/results_exp10_paired_ci.json`) |
| **Table 5, Figure 2, Table S4** | Known-truth simulation, scenarios B–E | `python code/local_experiments/exp11_simulation_known_truth.py --reps 500 --B 300` | `results/json/sim*.json` |
| **Supplement, panel-matched simulation** | Simulation with the real year sizes | `python code/local_experiments/exp15_panel_matched_simulation.py --probs results/json/results_rerun_train_only_canonical_probs.npz --meta results/test_meta.csv` | `results/json/results_exp15_panel_sim.json` |
| **Table 6** | Matched 15-seed core experiment, seven models (long-running; RF needs scikit-learn 1.6.1) | `python code/local_experiments/exp17_hybrids_multiseed.py --fix-et0 --p60 28.1 --models RF_2273 PureLSTM CTRL_LINEAR4D LSTM_QNN LSTM_PIQNN LBL_QNN LSTM_QTL` | `results/json/results_hybrids_trainonly_fixet0_p60_28.1.json`, `..._probs.npz` |
| **Table 7** | Paired comparisons with `LSTM+QNN` | `python code/local_experiments/exp10_paired_ci.py --probs results/json/results_rerun_train_only_canonical_probs.npz --prefix train_only --extra results/json/results_hybrids_trainonly_fixet0_p60_28.1_probs.npz --out results/json/results_exp10_paired_ci_fixet0.json` | `results/json/results_exp10_paired_ci_fixet0.json` |
| **Table 8** | Paired comparisons with the other three hybrids | `python code/local_experiments/exp10_paired_ci.py --extra results/json/results_hybrids_trainonly_fixet0_p60_28.1_probs.npz --pairs RF_2273-LSTM_PIQNN RF_2273-LBL_QNN RF_2273-LSTM_QTL CTRL_LINEAR4D-LSTM_PIQNN CTRL_LINEAR4D-LBL_QNN CTRL_LINEAR4D-LSTM_QTL PureLSTM-LSTM_PIQNN PureLSTM-LBL_QNN QTL_EXTRACTOR-LSTM_QTL --out results/json/results_exp10_hybrids_fixet0.json` | `results/json/results_exp10_hybrids_fixet0.json` |
| **Table 9** | 15-seed Brier score, skill against climatology, ECE | `python code/local_experiments/exp21_brier_trainonly.py` | `results/local_exp21_brier_trainonly.json` |
| **Table 10** | Pooled and within-station AUC | `python code/local_experiments/exp23_within_station_auc.py` | `results/local_exp23_within_station_auc.json` |
| **Table 11** | Hanley–McNeil MDE by test size and prevalence | `qml_resolution.paired_mde_hanley_mcneil` (checked in `tests/unit/test_power.py`) | — |
| **Table 12** | Empirical resolution curve from subsets of test years (~15 min) | `python code/local_experiments/exp24_resolution_curve.py` | `results/local_exp24_resolution_curve.json` |
| **Section 4 seed shares, jackknife** | Seed shares of differences, year jackknife, extra climatologies | `python code/local_experiments/exp16_block_c.py --core results/json/results_hybrids_trainonly_fixet0_p60_28.1_probs.npz --out results/local_exp16_block_c_fixet0.json` | `results/local_exp16_block_c_fixet0.json` |
| **Limitations (block length)** | Circular moving-block bootstrap, blocks of 1–4 years | `python code/local_experiments/exp19_block_length.py --core results/json/results_hybrids_trainonly_fixet0_p60_28.1_probs.npz --out results/local_exp19_block_length_fixet0.json` | `results/local_exp19_block_length_fixet0.json` |
| **Table S2, Figure S1** | Positive control (imposed AUC reduction) | `python code/local_experiments/exp8_positive_control.py` | `results/local_exp8_positive_control.json` |
| **Table S10** | Parameter accounting (2,613 vs 2,609) | `python scripts/verify_parameter_counts.py` | `results/json/results_wp3_accounting.json` |
| **Table S14** | Monthly coverage audit | `python code/local_experiments/exp12_monthly_coverage_audit.py --csv code/local_experiments/datos_limpios_multiestacion.csv --out results/json/results_exp12_coverage.json` | `results/json/results_exp12_coverage.json` |
| **Tables S15–S17** | Fixed predictors under label scenarios S0–S7 | `python code/local_experiments/exp13_label_sensitivity.py --probs results/json/results_rerun_train_only_canonical_probs.npz --prefix train_only --extra results/json/results_hybrids_trainonly_fixet0_p60_28.1_probs.npz --rows code/local_experiments/month_rows_raw.csv --pairs RF-XGB RF-LSTM XGB-LSTM RF_2273-LSTM_QNN CTRL_LINEAR4D-LSTM_QNN PureLSTM-LSTM_QNN --out results/json/results_exp13_label_sensitivity_fixet0.json` | `results/json/results_exp13_label_sensitivity_fixet0.json` |
| **Table S20** (historical) | Single-run Brier score and reliability (canonical, seed 42) | `python code/local_experiments/exp3_brier_and_reliability.py` | `results/local_exp3_brier_reliability.json` |
| **Inner-join audit** | Effect of the five-file inner join on the test partition | `python code/local_experiments/exp14_innerjoin_audit.py --csv code/local_experiments/datos_limpios_multiestacion.csv --precip_raw code/local_experiments/precip_raw.csv --rows code/local_experiments/month_rows.csv --out results/json/results_exp14_innerjoin.json` | `results/json/results_exp14_innerjoin.json` |
| **Single-seed pairings** | Same-architecture seed pairs of the core models | `python code/local_experiments/exp22_same_arch_core.py` | `results/local_exp22_same_arch_core.json` |
| **Section 2.2, leakage appendix** | Random Forest by pipeline and fitting rows | `python code/local_experiments/exp20_rf_fitting_rows.py` | `results/local_exp20_rf_fitting_rows.json` |

> [!NOTE]
> `exp12`, `exp13` and `exp14` need the raw daily station records (`datos_limpios_multiestacion.csv`, `precip_raw.csv`, `month_rows*.csv`), which are not redistributed (AEMET Open Data via datosclima.es; see `LICENSE-DATA`). The v2 test partition metadata (`results/test_meta.csv`, `results/years_te.npy`) and the per-seed validation and test probabilities of every model are tracked; the v1 monthly panel is in `results_v1/panel_checkpoint.csv`.

---

## 7. Architecture

```
.
├── src/qml_resolution/      reusable inference library (typed, 100 % test coverage)
│   ├── auc.py               rank AUC, DeLong components, (seed-averaged) paired DeLong
│   ├── resampling.py        cluster bootstrap (one- and two-way), jackknife, percentile CIs
│   └── power.py             MDE, Hanley-McNeil, seed share
├── code/                    research pipeline (frozen; see below)
│   ├── main.py              orchestrator of the canonical pipeline
│   ├── etl.py, fisica.py, features.py              ingestion, physical features, windows, split
│   ├── modelos_clasicos.py, modelos_cuanticos.py   RF / XGBoost / LSTM; variational circuits (PennyLane)
│   └── local_experiments/   one script per analysis; statistics delegate to qml_resolution
├── tests/
│   ├── unit/                library vs independent oracles (scikit-learn AUC, O(mn) DeLong, paper values)
│   ├── regression/          library vs archived results of the paper (DeLong, bootstrap, jackknife)
│   ├── quantum/             circuit physics: bounds, parameter-shift = backprop, Fourier spectrum
│   └── test_*.py            end-to-end reproduction checks (slow)
├── results/                 archived v2 results (JSON, CSV, NPZ) and the v2 monthly panels
├── results_v1/              archived v1 results, unchanged
├── checkpoints/             model weights and preprocessed dataset (checksums in CHECKSUMS.sha256)
├── scripts/                 figures, traceability audit, parameter-count check
├── paper/                   main.tex, supp.tex, compiled PDFs and figures
└── audit/                   traceability matrix and review notes
```

**Why `code/` is frozen.** The checkpoints in `checkpoints/` are PyTorch pickles that reference the module names of `code/` (Spanish names such as `modelos_cuanticos`, matching the feature glossary of the Supplement). Renaming or restructuring those modules would break the archived checkpoints, so new, reusable functionality lives in `src/qml_resolution` and the pipeline calls into it. The refactor of the statistical scripts was checked against the archived outputs: `exp7`, `exp8`, `exp10`, `exp13`, `exp16` and `exp22` reproduce their archived JSON files to within $5 \times 10^{-14}$.

---

## 8. Quality checks

| Check | Command | Scope |
|---|---|---|
| Lint and format | `make lint` (ruff) | whole repository; strict rules for `src/` and `tests/` |
| Static types | `make typecheck` (mypy `--strict`) | `src/` |
| Library tests | `make test-fast` | unit + regression, 100 % line and branch coverage |
| Quantum tests | `make test-quantum` | circuits of the hybrid models and their classical controls |
| Manuscript | `make manuscript` | every number in the main-text tables of `paper/main.tex` against the archives |
| Reproduction | `make check`, `make test` | figures, traceability audits, end-to-end scripts, regeneration of the v2 tables |

The quantum tests check properties that follow from quantum mechanics rather than from the implementation: Pauli-Z expectations lie in $[-1, 1]$; batched and per-sample evaluation agree; backpropagated gradients equal the exact parameter-shift rule; each model output is a truncated Fourier series in every input angle with frequencies up to $2L$ for $L$ re-uploading layers (Schuld, Sweke & Meyer, 2021), and the top frequency 4 of the two-layer circuit is actually reached; and the hybrid has the 2,613 trainable parameters (24 quantum) stated in the paper. For the v2 controls (`tests/quantum/test_controls_v2.py`): the $R_Y$-embedding head of `LSTM+QTL` lies exactly (residual below $10^{-10}$, for arbitrary weights) in the span of the 81-function tensor basis $\prod_j\{1,\cos q_j,\sin q_j\}$ used by `QTL tensor-81` and `Full Fourier-1`, while the separable basis does not span it; and the parameter counts of the Linear-4D, Fourier and frozen-circuit controls equal the sizes printed in Table 4.

Continuous integration (`.github/workflows/`) runs lint, type checks and the library tests on Linux and Windows with Python 3.11 and 3.12, and a separate reproducibility job that installs the hashed lockfile and runs the full suite. Pre-commit hooks are in `.pre-commit-config.yaml` (`make dev` installs them).

---

## 9. Checkpoints

`checkpoints/*.pt` are PyTorch pickles. Load them only from this repository's verified release (`sha256sum -c checkpoints/CHECKSUMS.sha256`) and with `torch.load(..., weights_only=True)` where the file contains only tensors.

---

## 10. Compiling the manuscript

```bash
cd paper
pdflatex main && pdflatex supp && pdflatex main && pdflatex supp
```

`main.tex` and `supp.tex` reference each other through `xr`, so the first pass of each reports undefined cross-references until both have been compiled once.

---

## 11. Data provenance

Meteorological observations are from the historical archives of the Spanish State Meteorological Agency (AEMET), obtained through *datosclima.es* under AEMET's open-data reuse framework. Only aggregated monthly data are redistributed, with attribution to AEMET (CC BY 4.0, see `LICENSE-DATA`); the daily records are not part of the repository.

| File | Content |
|---|---|
| `results/panel_v2_monthly.csv` | v2 monthly panel of the first panel (4 stations, 4,325 station-months, 1924-03 to 2026-05): label (monthly precipitation sum), lag value, the nine monthly covariates, completeness counts and window flags |
| `results/panel2/med2v2/panel_monthly.csv` | the same for the second panel (3,661 station-months) |
| `results_v1/panel_checkpoint.csv` | v1 monthly panel (3,363 station-months, 1938-03 to 2026-02) |

Both v2 panels are written by `scripts/export_monthly_panel.py` from the daily files of `code/etl_v2.py` with the aggregation of the pipeline (`dataset_v2.agregar_mensual_v2`). `tests/manuscript` rebuilds from them the partitions (2,332 / 516 / 529 windows; second panel 1,694 / 528 / 884), the thresholds ($P_{60}$ = 29.9 mm and 28.2 mm) and every test label of the archived predictions.

---

## 12. Citation and licenses

Cite the manuscript and the archived release (DOI to be added on release); see `CITATION.cff`.

* **Code:** [MIT License](LICENSE)
* **Data and text:** [CC BY 4.0](LICENSE-DATA), subject to the AEMET attribution above
