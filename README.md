# Limits of Resolution in Quantum Machine Learning: A Noise Budget for Small Benchmarks

[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-yellow.svg)](LICENSE)
[![License: CC BY 4.0](https://img.shields.io/badge/Data%20%26%20Paper%20License-CC%20BY%204.0-lightgrey.svg)](LICENSE-DATA)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](environment.yml)

Official reproducibility repository for the paper:  
**"Limits of Resolution in Quantum Machine Learning: A Noise Budget for Small Benchmarks"** (Enrique Real, 2026).

---

## 1. Overview & Key Findings

Quantum Machine Learning (QML) benchmarks on tabular and environmental datasets are frequently presented as rankings of architectures based on point estimates or small seed differences. This benchmark evaluates a 1-month-ahead probabilistic precipitation exceedance task ($P_{60} = \qty{28.1}{\milli\meter}$) on a multi-decadal Western Mediterranean panel ($N_{\text{test}} = 528$ station-months across Alicante, Barcelona, San Javier, and Valencia; 1938–2026).

**Core Findings:**
1. **Sampling Error Dominates:** On $N_{\text{test}}=528$ observations, test-set sampling error ($\text{SE} \approx 0.023$--$0.027$) is $4\times$ to $7\times$ larger than optimization-seed variance ($\text{SD} \approx 0.0035$--$0.0063$ for classical models). Optimization seeds account for only 2–7% of AUC variance in classical ensembles and 12–34% in hybrid quantum models.
2. **Minimum Detectable Difference ($\text{MDE} \approx 0.036$--$0.052$):** No reported performance gap between classical and hybrid architectures exceeds the MDE.
3. **Climatological Reference is Unbeaten:** An empirical month$\times$station climatology achieves AUC = 0.680 (Brier Score 0.2224) and is not significantly outperformed by any evaluated model in proper scoring.
4. **Positive Control Validation:** Known AUC losses $\delta \in [0.01, 0.05]$ imposed on Random Forest verify that an effect of 0.02 AUC is detected in only 17–27% of runs; 80% statistical power requires $\delta \approx 0.04$.
5. **Matched Classical Controllers:** Classical networks on the identical 4D bottleneck perform statistically equivalently to the quantum circuit ($p_{\text{TOST}} \le 0.015$).

> **Important Note on Archived Quantum Outputs:**  
> In accordance with the empirical protocol documented in Section 4.1 and Section 5.3 of the manuscript, per-seed test probability vectors were archived only for seed 0 for the hybrid quantum models; reference 15-seed test AUC distributions are fully archived in `results/json/`.

---

## 2. Repository Structure

```
.
├── paper/                                  ← Paper manuscript and source assets
│   ├── main.tex                            ← Canonical LaTeX source (renamed from main_noise_budget.tex)
│   ├── main.pdf                            ← Compiled publication PDF (46 pages)
│   └── figs/                               ← All vector (PDF) and high-res (PNG) figures
│
├── checkpoints/                            ← PyTorch model checkpoints & dataset binaries
│   ├── 01_data.pt                          ← Canonical preprocessed spatiotemporal dataset (N=3,315 windows)
│   ├── 02_baselines.pt ... 10_*.pt         ← Model checkpoints across all experimental phases
│   └── CHECKSUMS.sha256                    ← SHA-256 integrity checksums for all checkpoints
│
├── data/                                   ← Tabular spatiotemporal data
│   └── panel_checkpoint.csv                ← Exported monthly station panel (3,363 rows, 1938–2026)
│
├── code/                                   ← Production source code & model architectures
│   ├── local_experiments/                  ← Core experiment scripts (Noise budget, TOST, controls)
│   │   ├── exp7_noise_budget.py            ← Experiments 7 & Table 2/3 (Noise budget & MDE calculation)
│   │   ├── exp8_positive_control.py        ← Experiment 8 & Table 4 (Calibrated logit noise degradation)
│   │   ├── exp1_climatology_baseline.py    ← Climatology baseline evaluation
│   │   ├── exp3_brier_and_reliability.py   ← Probabilistic calibration & Brier scores
│   │   ├── exp4_year_block_bootstrap.py    ← Calendar-year block bootstrap (19 years)
│   │   ├── exp5_pca4_controls.py           ← Classical PCA-4 SVM and tree controls
│   │   └── exp6_tost_rope_equivalence.py   ← Two One-Sided Tests (TOST) on seed pairs
│   ├── modelos_clasicos.py                 ← RF, XGBoost, Gradient Boosting, Standalone LSTM
│   ├── modelos_cuanticos.py                ← PennyLane VQC architectures (QNN, PI-QNN, QTL, LbL, QKSVM)
│   ├── fisica.py                           ← Physical features (FAO-56 ET0, Clausius-Clapeyron proxies)
│   ├── features.py                         ← Sliding window feature generator (35 features)
│   └── visualizacion.py                    ← Paper visualizer & figure generator
│
├── scripts/                                ← Aggregation, table generation & validation utilities
│   ├── build_results_master.py             ← Compiles results/results_master.csv from checkpoints
│   ├── make_tables.py                      ← Generates LaTeX tables for single-run & multi-seed metrics
│   ├── multiseed_stats.py                  ← Multi-seed Friedman, Wilcoxon & Cliff's delta statistics
│   └── verify_parameter_counts.py          ← Exact parameter accounting across hybrid & classical models
│
├── results/                                ← Archived experimental outputs (JSON & CSV)
│   ├── json/                               ← Per-experiment raw outputs
│   │   ├── 01_data.json                    ← Full windowed dataset in JSON
│   │   ├── 02_baselines.json ...           ← Per-model evaluation logs
│   │   ├── results_hybrid_perseed_probs.json ← Seed-0 probabilities & 15-seed ref AUCs
│   │   └── results_rerun_train_only_canonical.json ← Train-only vs canonical rerun
│   ├── local_exp7_noise_budget.json        ← Precomputed noise budget outputs
│   ├── local_exp8_positive_control.json    ← Precomputed positive control detection rates
│   └── results_master.csv                  ← Master tidy table with all reported metrics
│
├── notebooks/                              ← Interactive exploratory notebooks
│   └── 01_full_exploratory_study.ipynb     ← Full exploratory analysis notebook (stripped outputs)
│
├── archive/                                ← Legacy drafts, intermediate logs, and scratch scripts
├── make_figs.py                            ← Generates Fig. 1 (Noise Budget) and Fig. 2 (Positive Control)
├── environment.yml                         ← Conda environment specification
├── requirements.txt                        ← Pinned pip dependencies
├── Dockerfile                              ← Containerized reproducibility setup
├── CITATION.cff                            ← Citation metadata
├── LICENSE                                 ← MIT License for source code
└── LICENSE-DATA                            ← CC BY 4.0 License for dataset and paper
```

---

## 3. Installation & Environment

### Option A: Conda (Recommended)
```bash
conda env create -f environment.yml
conda activate qml_antigravity
```

### Option B: Python Virtual Environment (`venv`)
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

### Option C: Docker Container
```bash
docker build -t qml_benchmark .
docker run -it --rm qml_benchmark
```

---

## 4. Reproducing Paper Tables & Figures

Every result in the manuscript can be verified either by reading the precomputed experimental records in `results/` (instantaneous, seconds) or by re-executing the model training and bootstrap loops.

### Summary of Execution Times

| Component | Execution Command | Estimated Runtime (CPU) |
| :--- | :--- | :--- |
| **Figure 1 & Figure 2** | `python make_figs.py` | **< 2 seconds** |
| **All Table Generation** | `python scripts/make_tables.py` | **< 5 seconds** |
| **Parameter Counts** | `python scripts/verify_parameter_counts.py` | **< 3 seconds** |
| **Noise Budget (Exp 7)** | `python code/local_experiments/exp7_noise_budget.py` | **~ 30 seconds** |
| **Positive Control (Exp 8)** | `python code/local_experiments/exp8_positive_control.py` | **~ 45 seconds** |
| **Brier & Calibration (Exp 3)**| `python code/local_experiments/exp3_brier_and_reliability.py` | **~ 10 seconds** |
| **Block Bootstrap (Exp 4)** | `python code/local_experiments/exp4_year_block_bootstrap.py` | **~ 2 minutes** |
| **PCA-4 Classical Controls (Exp 5)** | `python code/local_experiments/exp5_pca4_controls.py` | **~ 20 seconds** |
| **Paper Figures (Violin, ROC, etc.)** | `python code/visualizacion.py` | **~ 1 minute** |
| **15-Seed Classical Rerun (RF, XGB, LSTM)** | `python scripts/validate_todo_code.py` | **~ 25 minutes** |
| **Full 15-Seed Quantum Training (from scratch)** | `python code/main.py` | **~ 2.5 to 4 hours** |

---

### Step-by-Step Table Reproduction Commands

- **Table 1 (Station Coverage & Class Balance):**
  ```bash
  python -c "import pandas as pd; df=pd.read_csv('data/panel_checkpoint.csv'); print(df.groupby('ciudad')['target'].value_counts())"
  ```
- **Table 2 & Table 3 (Noise Budget, Observed Gaps, and MDE):**
  ```bash
  python code/local_experiments/exp7_noise_budget.py
  # Reads outputs directly: results/local_exp7_noise_budget.json
  ```
- **Table 4 (Positive Control Detection Rates):**
  ```bash
  python code/local_experiments/exp8_positive_control.py
  # Reads outputs directly: results/local_exp8_positive_control.json
  ```
- **Table 5 (Single-Run AUC, Brier Score, and Climatological Skill):**
  ```bash
  python code/local_experiments/exp3_brier_and_reliability.py
  ```
- **Table 6 (15-Seed Test AUC Ranking):**
  ```bash
  python scripts/build_results_master.py
  ```
- **Table 7 (Bottleneck-Controlled Classical Baselines):**
  ```bash
  python code/local_experiments/exp6_tost_rope_equivalence.py
  ```
- **Table 8 (Single-Run Performance & Tail Recalls):**
  ```bash
  python scripts/make_tables.py
  ```
- **Table 9 & Table 10 (Multi-Seed Quantum Distributions & Operational FPR/Recall):**
  ```bash
  python scripts/multiseed_stats.py
  ```
- **Table 12 (Extreme-Tail Recall with Wilson 95% CIs):**
  ```bash
  python scripts/make_tables.py
  ```
- **Table 14 (Inference-Time NISQ Gate Noise Degradation):**
  ```bash
  python code/experimento_ruido.py
  ```
- **Table 17 & Table 18 (Circuit Gate Decomposition & Parameter Accounting):**
  ```bash
  python scripts/verify_parameter_counts.py
  ```

---

### Step-by-Step Figure Reproduction Commands

- **Figure 1 (Observed Gaps vs. MDE Band) & Figure 2 (Positive Control Power Curves):**
  ```bash
  python make_figs.py
  # Output: figs/fig_noise_budget.pdf and figs/fig_positive_control.pdf
  ```
- **Figure 6 (Violin Plots of Quantum Multi-Seed AUC) & Figure 7 (Single-Run Illusion):**
  ```bash
  python code/visualizacion.py
  # Output: figura_auc_violin.pdf and figura_single_run_illusion.pdf
  ```

---

## 5. Compiling the Manuscript

To compile the LaTeX paper to PDF:
```bash
cd paper
pdflatex -interaction=nonstopmode -file-line-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -file-line-error main.tex
pdflatex -interaction=nonstopmode -file-line-error main.tex
```
The resulting document `paper/main.pdf` compiles cleanly with **0 errors, 0 undefined citations, and 0 overfull hboxes**.

---

## 6. Citation

If you use this benchmark, code, or experimental protocol in your work, please cite:

```bibtex
@article{real2026smallbenchmark,
  title={What Can a Small Benchmark Resolve? A Noise Budget for Comparing Quantum and Classical Learning on Mediterranean Precipitation},
  author={Real, Enrique},
  journal={arXiv preprint},
  year={2026},
  url={https://github.com/enriquerb32/physics-inspired-qml-ACP}
}
```

---

## 7. License

- **Source Code:** [MIT License](LICENSE).
- **Data Checkpoints & Manuscript:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE-DATA).
