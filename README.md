# Limits of Resolution in Quantum Machine Learning

Reproducibility repository for the manuscript **"Limits of Resolution in Quantum Machine Learning: A Statistical Noise Budget for Small Benchmarks"** (Enrique Real, ORCID [0009-0006-3862-9797](https://orcid.org/0009-0006-3862-9797)).

[![Code: MIT](https://img.shields.io/badge/Code-MIT-yellow.svg)](LICENSE)
[![Data and text: CC BY 4.0](https://img.shields.io/badge/Data%20%26%20Text-CC%20BY%204.0-lightgrey.svg)](LICENSE-DATA)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](pyproject.toml)
[![uv: fast packaging](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![PennyLane: 0.45+](https://img.shields.io/badge/PennyLane-0.45%2B-blueviolet.svg)](https://pennylane.ai/)
[![Docker: BuildKit](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](Dockerfile)

---

## 1. What the paper asks and finds

How large a difference can a small QML benchmark resolve, and how much does the answer depend on the inferential procedure? The case study is a four-station Western Mediterranean precipitation panel (Alicante, Barcelona, San Javier, Valencia; 1938–2026; 1-month-ahead exceedance of a regional 60th percentile, $P_{60} = 28.1$\,mm; 528 test station-months).

* **Test-set sampling error exceeds optimization-seed variance:** One AUC has a standard error of about 0.024. Across 15 seeds, optimization variance accounts for 2–7% of the combined seed-plus-sampling variance of Random Forest, XGBoost and an LSTM.
* **Resolution:** Estimated paired DeLong minimum detectable differences (80% power, $\alpha = 0.05$, normal approximation) for 15-seed means are 0.031–0.044 AUC for the three primary classical comparisons. The MDE is a planning quantity, not a significance threshold.
* **Matched fitting sets:** Trained on the same 2,273 rows with 15 seeds, Random Forest is 0.029 AUC above the `LSTM+QNN` hybrid (DeLong $p = 0.026$, not surviving correction for multiple comparisons); a circuit-free classical control with the same 4-dimensional bottleneck is within $+0.001$ AUC of the hybrid.
* **Climatology:** A month $\times$ station climatology reaches AUC 0.680 (Brier skill 0.082 against the training base rate). The reported comparisons do not establish improvement over it.
* **Inferential sensitivity:** Conclusions about gaps of one to two hundredths of AUC depend on the procedure (DeLong versus year-block bootstrap). Re-evaluating fixed predictors under alternative label definitions (S0–S7) keeps the sign of every paired difference but changes its nominal significance. No model was retrained under alternative labels.
* **Simulations with known truth** show when each procedure is miscalibrated at this sample size and design, including a check matched to the real year sizes.

---

## 2. Installation

Python 3.12 is the version used for the Docker image and the reported results. All dependencies are in `requirements.txt` (pinned) and `environment.yml`.

```bash
git clone https://github.com/enriquerb32/limits-of-resolution-qml.git
cd limits-of-resolution-qml

# Option A: uv with exact cryptographic lockfile (Recommended)
uv venv .venv --python 3.12
source .venv/bin/activate       # On Windows: .venv\Scripts\activate
uv pip install -r requirements.lock

# Option B: standard pip (flexible)
python -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Option C: conda
conda env create -f environment.yml
conda activate qml-resolution
```

The lockfile `requirements.lock` fixes exact package versions with SHA-256 hashes generated from `requirements.in`. The lockfile guarantees an identical dependency tree across environments; note that differences in hardware architecture, CPU microcode, and BLAS threading can vary floating-point summation order in the least significant digits. For the closest match to the archived results set `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`.

---

## 3. Reproducing the paper

Everything except the model training runs from the archived results in `results/` and needs no GPU.

```bash
python scripts/make_paper_figs.py      # all figures of the paper and supplement -> paper/figs/
python scripts/build_traceability.py   # every quantitative claim -> source file and status -> audit/01_traceability.csv
make check                             # (optional) figures + traceability + statistical scripts on saved results
```

`audit/01_traceability.csv` records for each quantitative claim the file it comes from and whether it was recomputed (verified) or is only read from a stored result; claims that could not be tied to a file are listed as unverified rather than omitted.

### Retraining (long-running, optional)

The pipeline stores model checkpoints and per-seed test probabilities. Re-running it regenerates them:

```bash
python code/main.py --use-checkpoints --output out         # statistics and figures from stored checkpoints
python code/main.py --skip-qml --force-retrain             # classical baselines only
python code/main.py --force-retrain                        # full pipeline including the quantum circuits
python code/local_experiments/qml_core_trainonly.py    # the 15-seed matched QML core experiment (Table 11)
```

Quantum-circuit simulation is CPU-bound and the 15-seed core experiment takes hours on a workstation; runtimes were not benchmarked for this repository. Seeds are fixed in the scripts, but results from retraining are expected to differ slightly from the archived ones (different hardware and library versions), so the statistics reported in the paper are computed from the archived probabilities.

### Docker

```bash
docker build -t limits-qml .
docker run --rm -v "$PWD/out:/app/out" limits-qml                                                      # regenerate figures from results/
docker run --rm -v "$PWD/audit:/app/audit" limits-qml python scripts/build_traceability.py            # audit 82 claims
docker run --rm -v "$PWD/checkpoints:/app/checkpoints" -v "$PWD/out:/app/out" limits-qml \
    python code/main.py --use-checkpoints --output out                                                # needs checkpoints mounted
```

The image contains `code/`, `scripts/` and `results/`; checkpoints and raw data are not baked in.

---

## 4. Mapping paper items to code

Commands run from the repository root. Table numbers refer to the compiled `paper/main.pdf` ("S" = `paper/supp.pdf`); each result is also stored in `results/json/`.

| Paper item | What it is | Command | Output |
|---|---|---|---|
| **Table 1** | Windows by station and partition | `python scripts/make_partition_table.py` | `results/json/partition_counts.json` |
| **Table 3, Table S1** | Noise budget, seed variance, MDEs | `python code/local_experiments/exp7_noise_budget.py` | `results/local_exp7_noise_budget.json` |
| **Table 4, Figure 1** | Paired DeLong and two-way bootstrap CIs, classical models | `python code/local_experiments/exp10_paired_ci.py --probs results/json/results_rerun_train_only_canonical_probs.npz --prefix train_only` | `out/results_exp10_paired_ci.json` (compare with archived `results/json/results_exp10_paired_ci.json`) |
| **Table 5, Figure 2, Table S4** | Known-truth simulation, scenarios B–E | `python code/local_experiments/exp11_simulation_known_truth.py --reps 500 --B 300` | `results/json/sim*.json` |
| **Section 4 (panel-matched check)** | Simulation with the real year sizes | `python code/local_experiments/exp15_panel_matched_simulation.py --probs results/json/results_rerun_train_only_canonical_probs.npz --meta results/test_meta.csv` | `results/json/results_exp15_panel_sim.json` |
| **Table 6** | Monthly coverage audit | `python code/local_experiments/exp12_monthly_coverage_audit.py --csv code/local_experiments/datos_limpios_multiestacion.csv --out results/json/results_exp12_coverage.json` | `results/json/results_exp12_coverage.json` |
| **Tables 7–9** | Fixed predictors under label scenarios S0–S7 | `python code/local_experiments/exp13_label_sensitivity.py --probs results/json/results_rerun_train_only_canonical_probs.npz --prefix train_only --extra results/json/results_qml_core_trainonly_probs.npz` | `results/json/results_exp13_label_sensitivity.json` |
| **(inner-join audit)** | Effect of the five-file inner join on the test partition | `python code/local_experiments/exp14_innerjoin_audit.py --csv code/local_experiments/datos_limpios_multiestacion.csv --precip_raw code/local_experiments/precip_raw.csv --rows code/local_experiments/month_rows.csv --out results/json/results_exp14_innerjoin.json` | `results/json/results_exp14_innerjoin.json` |
| **Table 10** | Brier score, climatology, reliability | `python code/local_experiments/exp3_brier_and_reliability.py` | `results/local_exp3_brier_reliability.json` |
| **Table 11** | Matched 15-seed QML core experiment (long-running) | `python code/local_experiments/qml_core_trainonly.py` | `results/json/results_qml_core_trainonly.json`, `..._probs.npz` |
| **Table 12** | Paired comparisons with the hybrid | `python code/local_experiments/exp10_paired_ci.py --probs results/json/results_rerun_train_only_canonical_probs.npz --prefix train_only --extra results/json/results_qml_core_trainonly_probs.npz` | `out/results_exp10_paired_ci.json` (compare with archived `results/json/results_exp10_paired_ci.json`, which corresponds to this 6-pair command) |
| **Table S2, Figure S1** | Positive control (imposed AUC reduction) | `python code/local_experiments/exp8_positive_control.py` | `results/local_exp8_positive_control.json` |
| **Table S18** | Parameter accounting (2,613 vs 2,609) | `python scripts/verify_parameter_counts.py` | `results/json/results_wp3_accounting.json` |

> [!NOTE]
> **Data Provenance & Derived Files:** The derived spatiotemporal partitions (`results/test_meta.csv`, `results/years_te.npy`), along with the monthly panel (`results/panel_checkpoint.csv`), are tracked in the repository. Full end-to-end re-execution of `exp12` and `exp14` requires the raw daily station records (`datos_limpios_multiestacion.csv`, `precip_raw.csv`), which are not included in this repository; they were obtained from AEMET Open Data via datosclima.es (see `LICENSE-DATA` for provenance and attribution). All downstream statistical comparisons run from the archived outputs in `results/`.
`results/test_meta.csv` (columns `ciudad`, `fecha`, `y_te`, one row per test window) is produced by `python code/local_experiments/export_test_meta.py --csv code/local_experiments/datos_limpios_multiestacion.csv --out results/test_meta.csv`. Table 2 of the paper is a map of experiments (pipeline, fitting rows, seeds and stored probabilities behind each result) and is the best starting point for a reviewer.

---

## 5. Repository structure

```
.
├── paper/                  main.tex, supp.tex (cross-referenced with xr), compiled PDFs, figs/
├── code/
│   ├── main.py             orchestrator of the canonical pipeline (produces checkpoints and per-seed probabilities)
│   ├── etl.py, fisica.py, features.py
│   │                       data ingestion, physical features, windows and chronological split
│   ├── modelos_clasicos.py, modelos_cuanticos.py, modelos_utils.py
│   │                       models (PureLSTM, RF, XGBoost; QNN variants in PennyLane)
│   ├── validacion.py, robustez_estadistica.py
│   │                       DeLong, bootstrap, multi-seed replication
│   ├── visualizacion.py, q1_style.py
│   │                       figure style
│   └── local_experiments/  statistical analyses of the paper, run on stored probabilities (exp7–exp15) and the 15-seed QML core
├── results/                archived machine-readable results (JSON, CSV, NPZ), including the monthly panel
├── checkpoints/            model weights and the preprocessed dataset (checksums in CHECKSUMS.sha256)
├── scripts/                figure generation, traceability audit, parameter-count check
├── Dockerfile, requirements.txt, environment.yml, pyproject.toml
└── LICENSE, LICENSE-DATA, CITATION.cff
```

Two layers: `code/` is the pipeline that trains models and stores probabilities; `code/local_experiments/` takes stored probabilities and does the statistics. Most modules keep Spanish file and variable names (`modelos_*.py`, `choque_clausius`, `potencial_convectivo`, ...), matching the manuscript's feature definitions (Supplement, glossary table); they were not renamed so that checkpoints and the supplement stay consistent.

---

## 6. Checkpoints

`checkpoints/*.pt` are PyTorch files, which are pickles. Load them only from this repository's verified release (`sha256sum -c checkpoints/CHECKSUMS.sha256`), and with `torch.load(..., weights_only=True)` where the file contains only tensors.

---

## 7. Compiling the manuscript

```bash
cd paper
pdflatex main && pdflatex supp && pdflatex main && pdflatex supp
```

`main.tex` and `supp.tex` reference each other through the `xr` package, so the first pass of each reports undefined cross-references until both have been compiled once.

---

## 8. Data provenance

Meteorological observations are from the historical archives of the Spanish State Meteorological Agency (AEMET) obtained through *datosclima.es* under AEMET's open-data reuse framework. Only the processed monthly panel (`results/panel_checkpoint.csv`, 3,363 station-months, 1938-03 to 2026-02) is redistributed, with attribution to AEMET. The raw daily station files are not part of the repository.

---

## 9. Citation and licenses

Cite the manuscript and the archived release (DOI to be added on release); see `CITATION.cff`.

* **Code:** [MIT License](LICENSE)
* **Data and text:** [CC BY 4.0](LICENSE-DATA), subject to the AEMET attribution above
