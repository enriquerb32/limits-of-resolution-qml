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

The repository has two parts:

* **`qml_resolution`** (`src/`): a small, typed and fully tested Python library for the inference the paper relies on: rank AUC, seed-averaged paired DeLong tests, cluster bootstrap and jackknife for clustered test sets, and minimum detectable differences. It depends only on NumPy and SciPy and can be used in any benchmark that stores test-set scores.
* **The research pipeline** (`code/`, `scripts/`, `results/`): data processing, the classical and hybrid quantum models (PennyLane + PyTorch), the 15-seed experiments, and every analysis behind the tables of the paper, with the archived per-seed test probabilities.

---

## 1. What the paper asks and finds

How large an AUC difference can a small benchmark resolve, and how much does the verdict depend on the inferential procedure? The case study is a four-station Western Mediterranean precipitation panel (Alicante, Barcelona, San Javier, Valencia; 1938–2026), with a 1-month-ahead forecast of exceeding a regional 60th percentile ($P_{60} = 28.1$ mm) and 528 test station-months grouped in 19 test years.

* **The test set, not the seed, limits resolution for classical models.** One AUC has a standard error of about 0.024, 4–7 times the seed standard deviation of Random Forest, XGBoost and an LSTM. For the variational hybrids the seed share rises to 15–34 %, so both sources must be reported.
* **Resolution.** The minimum detectable difference (80 % power, $\alpha = 0.05$) for 15-seed means is 0.031–0.044 AUC and falls only as $N^{-1/2}$. It is a design sensitivity, not a significance threshold.
* **The procedure matters near the limit.** For Random Forest − LSTM ($+0.015$) and Random Forest − `LSTM+QNN` ($+0.020$) the year-block bootstrap interval excludes zero while DeLong does not ($p = 0.19$ and $0.074$). Known-truth simulations show that each procedure can be miscalibrated under plausible dependence between test years.
* **Quantum case study.** On the same 2,273 training rows with 15 seeds per model, four hybrid models score 0.020–0.026 AUC below Random Forest and are not distinguished from a parameter-matched, circuit-free classical control (2,609 vs 2,613 parameters), the only comparison that isolates the circuit.
* **Reference baselines.** No model, classical or quantum, improves on a month × station climatology (AUC 0.680, Brier score 0.2224), and within stations the climatology discriminates best.

---

## 2. Using the library

```bash
pip install -e .            # library only: NumPy + SciPy
pip install -e ".[dev]"     # + test, lint and type-check tools
```

```python
import numpy as np
from qml_resolution import paired_delong, cluster_bootstrap_delta, percentile_ci, jackknife_delta, mde

z = np.load("results/json/results_hybrids_trainonly_fixet0_p60_28.1_probs.npz")
y, years = z["y_te"], np.load("results/years_te.npy")
rf, hybrid = z["RF_2273_probs"], z["LSTM_QNN_probs"]          # (15 seeds, 528 rows) each

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

## 3. Installation of the full pipeline

Python 3.12 is the version used for the Docker image and the reported results.

```bash
git clone https://github.com/enriquerb32/limits-of-resolution-qml.git
cd limits-of-resolution-qml

# Option A: uv with the hashed lockfile (recommended, exact versions)
uv venv .venv --python 3.12
source .venv/bin/activate       # Windows: .venv\Scripts\activate
uv pip install --require-hashes -r requirements.lock
uv pip install --no-deps -e .

# Option B: pip with version ranges
python -m venv .venv && source .venv/bin/activate
pip install -e ".[pipeline,dev]"

# Option C: conda
conda env create -f environment.yml && conda activate qml-resolution
```

The lockfile fixes exact versions with SHA-256 hashes (generated from `requirements.in`). Hardware and BLAS threading can still change floating-point summation order in the last digits; for the closest match set `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`.

---

## 4. Reproducing the paper

Everything except model training runs from the archived results in `results/` and needs no GPU.

```bash
make check     # figures, partition table, parameter counts and the traceability audit
make test      # full test suite (library, quantum circuit, archived-result regressions)
```

`scripts/build_traceability.py` writes `audit/01_traceability.csv`, which records for each quantitative claim of the manuscript the file it comes from and whether it was recomputed (verified) or read from a stored result; claims that cannot be tied to a file are listed as unverified rather than omitted.

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

## 5. Mapping paper items to code

Commands run from the repository root. Table numbers refer to `paper/main.pdf` ("S" = `paper/supp.pdf`).

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
> `exp12`, `exp13` and `exp14` need the raw daily station records (`datos_limpios_multiestacion.csv`, `precip_raw.csv`, `month_rows*.csv`), which are not redistributed (AEMET Open Data via datosclima.es; see `LICENSE-DATA`). The test partition metadata (`results/test_meta.csv`, `results/years_te.npy`) and the monthly panel (`results/panel_checkpoint.csv`) are tracked. Table 2 of the paper maps each result to its pipeline, fitting rows, seeds and stored probabilities, and is the best starting point for a reviewer.

---

## 6. Architecture

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
├── results/                 archived machine-readable results (JSON, CSV, NPZ) and the monthly panel
├── checkpoints/             model weights and preprocessed dataset (checksums in CHECKSUMS.sha256)
├── scripts/                 figures, traceability audit, parameter-count check
├── paper/                   main.tex, supp.tex, compiled PDFs and figures
└── audit/                   traceability matrix and review notes
```

**Why `code/` is frozen.** The checkpoints in `checkpoints/` are PyTorch pickles that reference the module names of `code/` (Spanish names such as `modelos_cuanticos`, matching the feature glossary of the Supplement). Renaming or restructuring those modules would break the archived checkpoints, so new, reusable functionality lives in `src/qml_resolution` and the pipeline calls into it. The refactor of the statistical scripts was checked against the archived outputs: `exp7`, `exp8`, `exp10`, `exp13`, `exp16` and `exp22` reproduce their archived JSON files to within $5 \times 10^{-14}$.

---

## 7. Quality checks

| Check | Command | Scope |
|---|---|---|
| Lint and format | `make lint` (ruff) | whole repository; strict rules for `src/` and `tests/` |
| Static types | `make typecheck` (mypy `--strict`) | `src/` |
| Library tests | `make test-fast` | unit + regression, 100 % line and branch coverage |
| Quantum tests | `make test-quantum` | circuit used by the hybrid models |
| Reproduction | `make check`, `make test` | figures, traceability audit, end-to-end scripts |

The quantum tests check properties that follow from quantum mechanics rather than from the implementation: Pauli-Z expectations lie in $[-1, 1]$; batched and per-sample evaluation agree; backpropagated gradients equal the exact parameter-shift rule; each model output is a truncated Fourier series in every input angle with frequencies up to $2L$ for $L$ re-uploading layers (Schuld, Sweke & Meyer, 2021); and the hybrid has the 2,613 trainable parameters (24 quantum) stated in the paper.

Continuous integration (`.github/workflows/`) runs lint, type checks and the library tests on Linux and Windows with Python 3.11 and 3.12, and a separate reproducibility job that installs the hashed lockfile and runs the full suite. Pre-commit hooks are in `.pre-commit-config.yaml` (`make dev` installs them).

---

## 8. Checkpoints

`checkpoints/*.pt` are PyTorch pickles. Load them only from this repository's verified release (`sha256sum -c checkpoints/CHECKSUMS.sha256`) and with `torch.load(..., weights_only=True)` where the file contains only tensors.

---

## 9. Compiling the manuscript

```bash
cd paper
pdflatex main && pdflatex supp && pdflatex main && pdflatex supp
```

`main.tex` and `supp.tex` reference each other through `xr`, so the first pass of each reports undefined cross-references until both have been compiled once.

---

## 10. Data provenance

Meteorological observations are from the historical archives of the Spanish State Meteorological Agency (AEMET), obtained through *datosclima.es* under AEMET's open-data reuse framework. Only the processed monthly panel (`results/panel_checkpoint.csv`, 3,363 station-months, 1938-03 to 2026-02) is redistributed, with attribution to AEMET. The raw daily station files are not part of the repository.

---

## 11. Citation and licenses

Cite the manuscript and the archived release (DOI to be added on release); see `CITATION.cff`.

* **Code:** [MIT License](LICENSE)
* **Data and text:** [CC BY 4.0](LICENSE-DATA), subject to the AEMET attribution above
