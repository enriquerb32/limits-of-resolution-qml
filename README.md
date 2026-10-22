# Limits of Resolution in Quantum Machine Learning: A Statistical Noise Budget for Small Benchmarks

[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-yellow.svg)](LICENSE)
[![License: CC BY 4.0](https://img.shields.io/badge/Data%20%26%20Paper%20License-CC%20BY%204.0-lightgrey.svg)](LICENSE-DATA)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](environment.yml)

Official reproducibility repository for the paper:  
**"Limits of Resolution in Quantum Machine Learning: A Statistical Noise Budget for Small Benchmarks"**  
Author: Enrique Real (Independent Researcher, Alicante, Spain — ORCID: [0009-0006-3862-9797](https://orcid.org/0009-0006-3862-9797)).

Repository: [https://github.com/enriquerb32/limits-of-resolution-qml](https://github.com/enriquerb32/limits-of-resolution-qml)

---

## 1. Overview & Core Findings

Quantum Machine Learning (QML) benchmarks on small tabular datasets are frequently interpreted as rankings of architectures based on point estimates or small seed differences. This benchmark demonstrates how large a difference such a benchmark can resolve and how much conclusions depend on the inferential procedure.

Evaluated on a Western Mediterranean precipitation panel (1938–2026; 1-month-ahead exceedance of a regional 60th percentile, $P_{60} = 28.1$\,mm; $N_{\text{test}} = 528$ station-months across Alicante, Barcelona, San Javier, and Valencia):

1. **Sampling Uncertainty Exceeds Seed Variance:** Test-set sampling error ($\text{SE} \approx 0.023$--$0.027$) is substantially larger than across-seed optimization standard deviation ($\sigma_{\text{seed}} \approx 0.0035$--$0.0063$ for classical models; $0.009$--$0.017$ for hybrids). Across 15 seeds, optimization variance accounts for only 2–7% of the combined variance in classical models.
2. **Minimum Detectable Difference ($\text{MDE} \approx 0.031$--$0.044$):** Paired seed-mean comparisons require true differences of 0.03–0.04 AUC for 80% statistical detection power.
3. **Inferential Sensitivity:** Small gaps depend on the inferential procedure: Random Forest vs. XGBoost ($+0.021$) is borderline ($p=0.054$), and Random Forest vs. Standalone LSTM ($+0.015$) is excluded from zero by year-block bootstrap intervals but not by DeLong ($p=0.189$).
4. **QML Case Study (Matched Fitting Rows):** When trained on the identical 2,273 training rows with 15 seeds each, Random Forest exceeds the hybrid `LSTM+QNN` by 0.029 AUC ($p=0.026$, not surviving multiplicity adjustment), whereas a circuit-free classical control on the same 4D bottleneck is within $+0.001$ of the hybrid (intervals extending to $\pm0.012$--$0.017$).
5. **Climatological Reference:** An empirical month$\times$station climatology achieves AUC = 0.680 (Brier Score 0.2224, $\text{BSS} = +0.082$); no evaluated model statistically outperforms the climatology in proper scoring.
6. **Label Fragility:** 23.3% of test months are incomplete (<30/31 days). Re-evaluating fixed predictors under alternative label definitions preserved the direction of differences but altered nominal significance.

---

## 2. Repository Structure

```
.
├── paper/                                  ← Manuscript LaTeX sources & compiled PDFs
│   ├── main.tex                            ← Main paper source (xr cross-references to supp)
│   ├── supp.tex                            ← Supplementary Material source
│   ├── main.pdf                            ← Compiled publication PDF
│   └── figs/                               ← All vector (PDF) and bitmap (PNG) figures
│       ├── fig_paired_ci.{pdf,png}         ← Main Fig. 1: Paired differences with DeLong & bootstrap CIs
│       ├── fig_sim.{pdf,png}               ← Main Fig. 2: Simulation type-I error & SE calibration
│       ├── fig_positive_control.pdf        ← Supp Fig. S1: Positive control power curves
│       └── fig_noise_budget.pdf            ← Noise budget observed gaps vs MDE
│
├── checkpoints/                            ← Model state checkpoints & binary datasets
│   ├── 01_data.pt                          ← Canonical preprocessed dataset (N=3,315 windows)
│   ├── 02_baselines.pt ... 10_*.pt         ← Checkpoints across model tiers
│   └── CHECKSUMS.sha256                    ← Integrity checksums
│
├── code/                                   ← Source code & evaluation pipelines
│   ├── local_experiments/                  ← Core statistical evaluation scripts
│   │   ├── exp7_noise_budget.py            ← Noise budget & MDE calculations (Table budget)
│   │   ├── exp8_positive_control.py        ← Known-loss logit noise degradation (Table positive)
│   │   ├── exp10_paired_ci.py              ← Paired DeLong & 2-way bootstrap CIs (Table paired_ci, qnn_paired)
│   │   ├── exp11_simulation_known_truth.py ← Known-truth simulation engine (Scenarios A-E, Table sim)
│   │   ├── exp12_monthly_coverage_audit.py ← Monthly coverage & label fragility audit (Table coverage)
│   │   ├── exp13_label_sensitivity.py      ← Fixed predictor re-evaluation under S0-S7 (Table label_sens)
│   │   ├── exp14_innerjoin_audit.py        ← Daily precipitation lost in inner join
│   │   ├── exp15_panel_matched_simulation.py ← Panel-matched known-truth check
│   │   ├── export_test_meta.py             ← Station/year test metadata exporter
│   │   ├── trainonly_dataset.py            ← Train-only dataset builder
│   │   └── colab_qml_core_trainonly.py     ← 15-seed matched QML core experiment (Table case)
│   ├── modelos_clasicos.py                 ← Classical baselines (RF, XGBoost, LSTM)
│   ├── modelos_cuanticos.py                ← PennyLane hybrid circuits (QNN, PI-QNN, QTL, LbL, QKSVM)
│   ├── fisica.py                           ← Physical & meteorological feature engineering
│   └── features.py                         ← Sliding window feature generator (35 features)
│
├── scripts/                                ← Table & figure generation utilities
│   ├── make_paper_figs.py                  ← Generates all paper figures directly from results/
│   ├── make_sim_tables.py                  ← Formats simulation LaTeX rows (Table sim, sim_all)
│   ├── make_tables.py                      ← Formats supplementary LaTeX tables from results_master.csv
│   ├── build_traceability.py               ← Verifies all 82 quantitative claims (audit/01_traceability.csv)
│   ├── verify_parameter_counts.py          ← Exact parameter count decomposition (Table parameter_accounting)
│   └── build_results_master.py             ← Compiles tidy metrics table results/results_master.csv
│
├── tables/                                 ← Formatted LaTeX table snippets
│   ├── sim_main_rows.tex                   ← Table 4: Simulation main rows
│   └── sim_supp_rows.tex                   ← Table S13: Simulation all rows
│
├── results/                                ← Archived experimental outputs (JSON & CSV)
│   ├── json/                               ← Machine-readable result archives
│   │   ├── results_exp10_paired_ci.json    ← Paired DeLong, block & two-way bootstrap intervals
│   │   ├── results_exp11_sim.json          ← Scenario A simulation results
│   │   ├── simB_rerun.json ... simE_*.json ← Scenarios B, C, D, E simulation runs
│   │   ├── results_exp12_coverage.json     ← Monthly coverage counts & threshold shifts
│   │   ├── results_exp13_label_sensitivity.json ← S0-S7 label re-evaluation metrics
│   │   ├── results_exp14_innerjoin.json    ← Inner join lost rain quantification
│   │   ├── results_exp15_panel_sim.json    ← Panel-matched simulation results
│   │   ├── results_qml_core_trainonly.json ← 15-seed matched QML core results
│   │   ├── results_qml_core_trainonly_probs.npz ← 528 test probabilities for all core seeds
│   │   └── results_rerun_train_only_canonical_probs.npz ← Probabilities for classical reruns
│   ├── local_exp7_noise_budget.json        ← Precomputed noise budget outputs
│   ├── local_exp8_positive_control.json    ← Positive control detection rates
│   └── panel_checkpoint.csv                ← Canonical monthly panel
│
├── audit/                                  ← Reproducibility & traceability audit
│   ├── 00_inventory.md                     ← Repository asset inventory
│   ├── 01_traceability.csv                 ← Claim-by-claim verification table (82 claims verified)
│   └── 03_manuscript_proposals.md          ← Formatted editorial & typographical proposals
│
├── make_figs.py                            ← Root figure generator entry point
├── Dockerfile                              ← Containerized reproduction environment
├── environment.yml                         ← Conda environment specification
├── requirements.txt                        ← Pinned pip dependencies
├── CITATION.cff                            ← Citation metadata
├── LICENSE                                 ← MIT License (source code)
└── LICENSE-DATA                            ← CC BY 4.0 (data & paper)
```

---

## 3. Installation & Quick Start

### Python Environment (`venv` or Conda)

```bash
# Clone the repository
git clone https://github.com/enriquerb32/limits-of-resolution-qml.git
cd limits-of-resolution-qml

# Using Python venv (Python 3.12 recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Reproducing All Figures (Instantaneous)

All figures in the manuscript can be reproduced directly from the saved experimental results in `< 3 seconds`:

```bash
python make_figs.py
# Outputs:
#   paper/figs/fig_paired_ci.pdf (and .png)
#   paper/figs/fig_sim.pdf (and .png)
#   paper/figs/fig_noise_budget.pdf
#   paper/figs/fig_positive_control.pdf
```

### Reproducing LaTeX Simulation Tables

```bash
python scripts/make_sim_tables.py
# Formats tables/sim_main_rows.tex and tables/sim_supp_rows.tex from results/json/sim*.json
```

### Verifying All 82 Quantitative Claims in the Paper

```bash
python scripts/build_traceability.py
# Generates audit/01_traceability.csv with 100% verified status against results/
```

---

## 4. Reproducing Experimental Analyses

Every table and claim in the paper maps to an executable script in `code/local_experiments/`:

| Paper Item | Description | Script & Command | Output Source |
|---|---|---|---|
| **Table 1** (`station_coverage`) | Station-month coverage by partition | `python -c "import pandas as pd; df=pd.read_csv('results/panel_checkpoint.csv'); print(df.groupby('ciudad')['target'].value_counts())"` | `results/panel_checkpoint.csv` |
| **Table 2** (`budget`) | Noise budget & seed variance share | `python code/local_experiments/exp7_noise_budget.py` | `results/local_exp7_noise_budget.json` |
| **Table 3** (`paired_ci`) | Paired comparisons of 15-seed means | `python code/local_experiments/exp10_paired_ci.py --probs results/json/results_rerun_train_only_canonical_probs.npz --prefix train_only` | `results/json/results_exp10_paired_ci.json` |
| **Table 4** (`sim`) | Known-truth simulation (Scenarios B–E) | `python code/local_experiments/exp11_simulation_known_truth.py --reps 500 --B 300` | `results/json/sim{B,C,D,E}*.json` |
| **Table 5** (`coverage`) | Monthly coverage & label fragility | `python code/local_experiments/exp12_monthly_coverage_audit.py` | `results/json/results_exp12_coverage.json` |
| **Table 6–8** (`label_sens*`) | S0–S7 label sensitivity re-evaluations | `python code/local_experiments/exp13_label_sensitivity.py --probs results/json/results_rerun_train_only_canonical_probs.npz --prefix train_only --extra results/json/results_qml_core_trainonly_probs.npz` | `results/json/results_exp13_label_sensitivity.json` |
| **Table 9** (`brier`) | Climatology & probabilistic skill | `python code/local_experiments/exp3_brier_and_reliability.py` | `results/local_exp3_brier_reliability.json` |
| **Table 10** (`case`) | Matched 15-seed QML core experiment | `python code/local_experiments/colab_qml_core_trainonly.py` (Colab run) | `results/json/results_qml_core_trainonly.json` |
| **Table 11** (`qnn_paired`) | Hybrid paired DeLong & bootstrap CIs | `python code/local_experiments/exp10_paired_ci.py --probs results/json/results_rerun_train_only_canonical_probs.npz --prefix train_only --extra results/json/results_qml_core_trainonly_probs.npz` | `results/json/results_exp10_paired_ci.json` |
| **Supp Table S1** (`mde`) | Single-seed pairings MDE | `python code/local_experiments/exp7_noise_budget.py` | `results/local_exp7_noise_budget.json` |
| **Supp Table S2** (`positive`) | Positive control detection curves | `python code/local_experiments/exp8_positive_control.py` | `results/local_exp8_positive_control.json` |
| **Supp Table S14** (`parameter_accounting`) | Parameter decomposition (2,613 / 2,609) | `python scripts/verify_parameter_counts.py` | `results/json/results_wp3_accounting.json` |

---

## 5. Compiling the Manuscript

The paper uses the `xr` package for bidirectional cross-referencing between `main.tex` and `supp.tex`. To compile cleanly:

```bash
cd paper
pdflatex -interaction=nonstopmode main
pdflatex -interaction=nonstopmode supp
pdflatex -interaction=nonstopmode main
pdflatex -interaction=nonstopmode supp
```

Both documents compile with **0 errors, 0 undefined references, and 0 missing citations**.

---

## 6. Data Provenance & Attribution

The underlying meteorological observations originate from the public historical archives of the **State Meteorological Agency of Spain (AEMET)** via *datosclima.es* under AEMET's Open Data reuse framework.

- **Redistribution scope:** Only the processed, aggregated monthly station panel (`results/panel_checkpoint.csv`, 3,363 station-months, 1938–2026) is redistributed in this repository, with full attribution to AEMET.
- Raw daily station files are excluded from git tracking in accordance with open-data distribution terms.

---

## 7. Citation

If you use this benchmark, code, or methodology in your work, please cite:

```bibtex
@article{real2026limits,
  title={Limits of Resolution in Quantum Machine Learning: A Statistical Noise Budget for Small Benchmarks},
  author={Real, Enrique},
  year={2026},
  url={https://github.com/enriquerb32/limits-of-resolution-qml}
}
```

---

## 8. License

- **Source Code:** [MIT License](LICENSE).
- **Data & Text:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE-DATA).
