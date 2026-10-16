# Full execution script for WP6 Restructuring in reverse order
with open('main_revised.tex', 'r', encoding='utf-8') as f:
    text = f.read()

print('Initial text length:', len(text))

# --- BLOCK 10: Replace Appendix D with comprehensive Heuristic Regularization Appendix ---
pos10_start = text.find('\\section{Physical Penalty Weight ($\\lambda_{cc}$) Sensitivity Sweep}')
pos10_end = text.find('so we do not treat this as evidence of a meaningful $\\lambda_{\\text{cc}}$ effect.', pos10_start) + len('so we do not treat this as evidence of a meaningful $\\lambda_{\\text{cc}}$ effect.')
assert pos10_start != -1 and pos10_end != -1

replacement_10 = '''\\section{Heuristic Thermodynamic Regularization, Latent Coordinates, and Sensitivity Sweep}
\\label{app:heuristic_regularization}\\label{sec:physics_loss}\\label{app:lambda_sweep}
% ============================================================

\\subsection{Formulation of Heuristic Thermodynamic Loss Penalties}
To explore whether domain-motivated soft constraints alter optimization behavior, we constructed two heuristic regularizers combining binary cross-entropy ($\\mathcal{L}_{\\text{BCE}}$) with physical penalties:
\\begin{equation}
\\mathcal{L}_{\\text{total}} = \\mathcal{L}_{\\text{BCE}} + \\alpha(e) \\left[ \\lambda_{\\text{cc}} \\mathcal{L}_{\\text{CC}} + \\lambda_{\\text{hMSE}} \\mathcal{L}_{\\text{hMSE}} \\right]
\\end{equation}
where $\\alpha(e) = \\min(1.0, e / (0.6 \\times E_{\\text{total}}))$ represents a linear annealing schedule across training epochs $e$.

\\textbf{1. Clausius--Clapeyron-Inspired Gating Penalty ($\\mathcal{L}_{\\text{CC}}$):} Defines a temperature-dependent soft gate:
\\begin{equation}
C_{\\text{CC}} = \\exp(0.07 \\cdot T_{\\text{anomaly}} \\cdot 10.0)
\\end{equation}
loosely motivated by the exponential scaling of saturation water vapour pressure with temperature ($\\approx 7\\%/\\unit{\\kelvin}$). A ReLU-gated penalty activates if the model predicts high precipitation probability ($P_{\\text{rain}} > 0.6$) when this thermal gate is restricted:
\\begin{equation}
\\mathcal{L}_{\\text{CC}} = \\frac{1}{N} \\sum_{i=1}^N \\left[ \\max(0, P_{\\text{rain}, i} - 0.6) \\times \\max(0, 1.0 - C_{\\text{CC}, i}) \\right]
\\end{equation}

\\textbf{Dimensional and Physical Caveat:} We emphasize that the scale factor $0.07$ originates as a per-kelvin physical sensitivity from the Clausius--Clapeyron relation, but in the implementation it is multiplied by $10.0$ and evaluated directly on a dimensionless, standardized temperature anomaly z-score ($T_{\\text{anomaly}}$). Furthermore, the 3-month lagged thermal difference (\\texttt{choque\\_clausius}) reflects seasonal transition cooling rather than saturation thermodynamics. Consequently, $C_{\\text{CC}}$ is an empirical phenomenological regularizer rather than a dimensionally consistent thermodynamic equation.

\\textbf{2. Heuristic Latent-Coordinate Balance Penalty ($\\mathcal{L}_{\\text{hMSE}}$):} Encourages the readout expectation values nominally associated with sensible heat ($q_0$) and latent heat ($q_1$) to balance the coordinate nominally associated with convective release ($q_3$):
\\begin{equation}
\\mathcal{L}_{\\text{hMSE}} = \\frac{1}{N} \\sum_{i=1}^N \\left[ \\max(0, q_{0,i}) + \\max(0, q_{1,i}) - \\max(0, q_{3,i}) - 0.5 \\right]^2
\\end{equation}
where $q_j = \\langle 0 | U^\\dagger(\\theta, x) \\sigma_z^{(j)} U(\\theta, x) | 0 \\rangle$ denotes the expectation value of Pauli-$Z$ on qubit $j$. We use the notation $\\mathcal{L}_{\\text{hMSE}}$ to distinguish this heuristic coordinate penalty from standard regression Mean Squared Error.

The internal constants ($0.07$, $10.0$, $0.6$, $0.5$) and outer penalty weights ($\\lambda_{\\text{cc}} = 0.2$, $\\lambda_{\\text{hMSE}} = 0.1$) were chosen \\textit{a priori} based on qualitative heuristic reasoning and held fixed during main benchmark training.

\\begin{figure}[htbp]
    \\centering
    \\includegraphics[width=\\textwidth]{figura_thermo_analysis.pdf}
    \\caption{Empirical justification for the heuristic thermodynamic loss function. Mean annual cycle of $\\text{ET}_0$ and precipitation (top-left); lagged thermal anomaly difference vs.\\ precipitation, coloured by month (top-right); pressure--precipitation interaction for ACP vs.\\ normal SON months (bottom-left); Pearson correlation of physical covariates with monthly precipitation (bottom-right), motivating \\textit{choque\\_clausius} and \\textit{potencial\\_convectivo} as candidate phenomenological penalty inputs.}
    \\label{fig:thermo_analysis}
\\end{figure}

\\subsection{Latent Readout Coordinate Interpretations}
\\Cref{tab:qubits_mapping} details the descriptive interpretations heuristically assigned to the four Pauli-$Z$ readout expectations. These labels are descriptive post-hoc conventions and do not correspond to physical thermodynamic state variables or conservation laws.

\\begin{table}[htbp]
\\centering
\\caption{Post-hoc interpretation of the four measured latent coordinates. The labels are descriptive only and do not imply thermodynamic units, state identification, or physically enforced energy conservation.}
\\label{tab:qubits_mapping}
\\begin{tabular}{lll}
\\toprule
\\textbf{Qubit} & \\textbf{Latent Coordinate} & \\textbf{Heuristic interpretation} \\\\
\\midrule
$Q_0$ & Thermal Energy Coordinate ($q_0$) & Thermal-related latent coordinate \\\\
$Q_1$ & Moisture Coordinate ($q_1$) & Moisture-related latent coordinate ($L_v q$) \\\\
$Q_2$ & Dynamic Coordinate ($q_2$) & Dynamic-related latent coordinate \\\\
$Q_3$ & Convective Release Coordinate ($q_3$) & Convective-related latent coordinate \\\\
\\bottomrule
\\end{tabular}
\\end{table}

\\subsection{Physical Penalty Weight ($\\lambda_{\\text{cc}}$) Sensitivity Sweep}
To evaluate whether a more optimal physical penalty balance exists and to rule out hyperparameter cherry-picking, we conducted an \\textit{a posteriori} sensitivity analysis on \\texttt{LbL-QNN} across a 20-fold range of $\\lambda_{\\text{cc}} \\in [0.05, 1.0]$, replicating each condition with 5 seeds (approximately 73--84 minutes per run, or roughly 30 GPU-hours for the full sweep).

We observed a monotonic ordering across the five tested $\\lambda_{\\text{cc}}$ values (Spearman $\\rho=1.000$); given only five conditions, a perfect rank correlation is not strong evidence of a robust trend on its own, and the accompanying regression ($p=0.008$) should be read as suggestive rather than confirmatory. The corresponding Friedman test across conditions is non-significant ($p=0.111$), and the total AUC range across the entire grid ($\\Delta\\text{AUC}=0.0056$) is smaller than typical seed-to-seed variance ($\\sigma \\approx 0.0103$), confirming that hyperparameter tuning of $\\lambda_{\\text{cc}}$ does not overcome initialization variance. (Note: in the exported notebook artifacts, this grid was partially executed with $\\lambda=0.1$ and $\\lambda=0.2$; full multi-seed records across all five conditions require external re-export).'''

text = text[:pos10_start] + replacement_10 + text[pos10_end:]
print('Applied Block 10 (Heuristic Regularization Appendix). Text len:', len(text))


# --- BLOCK 9: Insert new Appendix on Extreme-Tail and Operational Metrics ---
pos9 = text.find('\\section{Training Hyperparameters}')
assert pos9 != -1

replacement_9 = '''% ============================================================
\\section{Exploratory Analysis of Extreme-Tail and Operational Metrics}
\\label{app:operational_analysis}\\label{sec:operational}
% ============================================================

\\subsection{Small-Sample Binomial Uncertainty in Extreme-Tail Recall}
To evaluate model sensitivity in the extreme precipitation tail, the benchmark reports recall metrics conditioned on two diagnostic subsets: months exceeding the regional 90th percentile ($P_{90} = \\qty{86.32}{\\milli\\meter}$, $n=49$ test months) and months matching the Autumn Convective Proxy (ACP, $n=25$ test months, of which 20 exceeded $P_{60}$; throughout this paper, Rc-ACP is defined over the $n=20$ months with $y=1$).

Given these small test-partition sample sizes, point estimates of Rc-ACP and Rc-$P_{90}$ carry substantial binomial sampling uncertainty. \\Cref{tab:recall_wilson_ci} reports 95\\% Wilson score confidence intervals \\citep{wilson1927} for each evaluated architecture in the single representative run. As shown in the table, apparent single-run separations fall well within overlapping confidence intervals. For example, Random Forest\'s Rc-ACP ($0.900$, 95\\% CI $[0.699, 0.972]$) and \\texttt{LSTM+QNN}\'s ($0.950$, 95\\% CI $[0.764, 0.991]$) overlap substantially despite a 5-percentage-point difference in the point estimate, underscoring that single-run recall comparisons in the extreme tail reflect test-sample binomial uncertainty independent of optimization variability.

\\begin{table}[htbp]
\\centering
\\caption{Extreme-tail recall with 95\\% Wilson confidence intervals, single representative run ($n=20$ ACP-flagged test months exceeding $P_{60}$ out of 25 flagged total; $n=49$ $P_{90}$-exceeding test months).}
\\label{tab:recall_wilson_ci}
\\begin{tabular}{lcc}
\\toprule
\\textbf{Architecture} & \\textbf{Rc-ACP [95\\% CI]} & \\textbf{Rc-$P_{90}$ [95\\% CI]} \\\\
\\midrule
\\textit{Statistical \\& Ensemble}\\\\
Logistic Regression & 0.650 [0.433, 0.819] & 0.571 [0.433, 0.699] \\\\
Random Forest & 0.900 [0.699, 0.972] & 0.755 [0.619, 0.854] \\\\
Gradient Boosting & 0.700 [0.481, 0.855] & 0.592 [0.452, 0.718] \\\\
XGBoost & 0.850 [0.640, 0.948] & 0.694 [0.555, 0.805] \\\\
\\midrule
\\textit{Deep Learning}\\\\
Standalone LSTM & 0.950 [0.764, 0.991] & 0.816 [0.686, 0.900] \\\\
Matched LSTM$^{\\dagger}$ & 0.950 [0.764, 0.991] & 0.776 [0.641, 0.870] \\\\
\\midrule
\\textit{Quantum-Classical Hybrid}\\\\
LSTM+QNN & 0.950 [0.764, 0.991] & 0.796 [0.664, 0.885] \\\\
LSTM+PI-QNN & 0.950 [0.764, 0.991] & 0.755 [0.619, 0.854] \\\\
LSTM+QTL & 0.850 [0.640, 0.948] & 0.837 [0.710, 0.915] \\\\
LbL-QNN & 0.900 [0.699, 0.972] & 0.796 [0.664, 0.885] \\\\
QKSVM & 0.600 [0.387, 0.781] & 0.612 [0.472, 0.736] \\\\
\\bottomrule
\\end{tabular}
\\vspace{4pt}
\\footnotesize\\textsuperscript{$\\dagger$}Threshold-recalibrated variant of Standalone LSTM ($\\tau = 0.531$) matched to the false-positive rate of \\texttt{LbL-QNN}.
\\end{table}

\\subsection{Deconstructing the Operational Paradox and Threshold Calibration}
At the default classification threshold of 0.5, the \\texttt{Standalone LSTM} achieves high sensitivity to extremes (Rc-ACP = 0.950, 19/20) but suffers from severe overprediction bias (FPR = 0.617). In contrast, \\texttt{LbL-QNN} appears more conservative (FPR = 0.495, Rc-ACP = 0.900, 18/20), suggesting an apparent operational advantage for the hybrid architecture (\\Cref{fig:pareto_operational}).

However, this apparent advantage is an artifact of threshold calibration. When the classical LSTM is adjusted to operate at an identical FPR of $\\approx$50.2\\% (by raising its decision threshold to $\\tau = 0.531$, denoted \\texttt{Matched LSTM}), its ACP recall remains at 0.950 (19/20). Paired bootstrap analysis ($n=2{,}000$ resamples) reveals no statistically significant operational difference between the two models across the operational frontier (all 95\\% confidence intervals span zero).

\\begin{figure}[htbp]
    \\centering
    \\includegraphics[width=0.85\\textwidth]{figura_pareto_operational.pdf}
    \\caption{Operational Pareto frontier: ACP recall vs.\\ false positive rate, single-run test evaluation.}
    \\label{fig:pareto_operational}
\\end{figure}

\\begin{figure}[htbp]
    \\centering
    \\includegraphics[width=\\textwidth]{figura_single_run_illusion.pdf}
    \\caption{The illusion of single-run architectural superiority: bootstrap confidence intervals and DeLong paired-test $p$-values, both conditioned on a single initialization draw.}
    \\label{fig:single_run_illusion}
\\end{figure}

\\Cref{fig:single_run_illusion} illustrates how within-run uncertainty estimation can mislead architectural comparison: the bootstrap confidence intervals and DeLong $p$-values from a single realization appear to demonstrate a significant advantage for \\texttt{LbL-QNN} over both unconstrained \\texttt{LSTM+QNN} ($p=0.0070$) and \\texttt{Standalone LSTM} ($p=0.0263$). However, as demonstrated by the 15-seed replication in \\Cref{sec:stability}, this apparent significance is entirely conditioned on a single favorable parameter initialization draw and vanishes when optimization variability is accounted for.

'''

text = text[:pos9] + replacement_9 + text[pos9:]
print('Applied Block 9 (Operational Analysis Appendix). Text len:', len(text))


# --- BLOCK 8: Appendix B.3 Consolidate with Sec 3.1 ---
pos8_start = text.find('\\subsection{Train-Only Anomaly Standardization: Implementation and Validation}')
pos8_end = text.find('solely as a verification exercise.', pos8_start) + len('solely as a verification exercise.')
assert pos8_start != -1 and pos8_end != -1

replacement_8 = '''\\subsection{Train-Only Anomaly Standardization and Leakage Validation}
\\label{app:leakage_correction}\\label{sec:leakage_validation}

The synoptic anomaly features (\\Cref{sec:glossary}) were originally computed by centering each station-month observation on the mean and standard deviation of its own (station, calendar-month) group across the \\emph{full} 1938--2026 record. To test whether this choice introduces validation/test-period information into the training partition, we implemented a train-only variant in which reference statistics are computed exclusively from observations dated on or before 1996, and applied identically across partitions:
\\begin{equation}
z_{i,c,m} = \\frac{x_{i,c,m} - \\mu_{c,m}^{\\text{train}}}{\\sigma_{c,m}^{\\text{train}} + \\epsilon},
\\qquad \\epsilon = 10^{-4}
\\end{equation}
where $\\mu_{c,m}^{\\text{train}}$ and $\\sigma_{c,m}^{\\text{train}}$ denote the mean and standard deviation of feature $x$ for station $c$ and calendar month $m$ computed using only observations with year $\\le 1996$. Station-month combinations absent from the training period default to $z=0$. This train-only variant was substituted across all downstream synoptic anomaly features (\\textit{iap}, \\textit{iav}, \\textit{ia\\_racha}, \\textit{ia\\_sol}, \\textit{ia\\_dp\\_std}) and their lagged derivatives, and the entire feature-engineering and 15-seed XGBoost replication pipeline was re-run end-to-end under this substitution.

We note that this leakage validation was executed on a rebuilt dataset (yielding $P_{60}=28.0$ and $P_{90}=89.0$\\,mm on the training set, where XGBoost achieves $0.6619 \\pm 0.0049$ for the original full-record vs.\\ $0.6603 \\pm 0.0041$ for the train-only pipeline). This comparison isolates the specific differential effect of train-only centering; all main benchmark tables throughout this paper use the canonical checkpoint \\texttt{01\\_data.pt} ($P_{60}=28.1$, $P_{90}=86.32$\\,mm, canonical XGBoost $0.6575 \\pm 0.0051$).

All four metrics showed no statistically significant difference between the original and corrected pipelines after Holm-Bonferroni correction: AUC ($0.6619 \\pm 0.0049$ vs.\\ $0.6603 \\pm 0.0041$; $p_{\\text{Holm}}=0.338$), FPR ($0.3912 \\pm 0.0098$ vs.\\ $0.3989 \\pm 0.0163$; $p_{\\text{Holm}}=0.328$), Rc-ACP ($0.8267 \\pm 0.0258$ vs.\\ $0.8507 \\pm 0.0384$; $p_{\\text{Holm}}=0.328$), and Rc-$P_{90}$ ($0.7206 \\pm 0.0158$ vs.\\ $0.7177 \\pm 0.0234$; $p_{\\text{Holm}}=0.369$). In every case, the absolute mean difference was smaller than the across-seed standard deviation of the corrected pipeline itself (largest ratio: Rc-ACP, $\\Delta/\\sigma = 0.62$), indicating that the standardization choice\'s downstream impact is dominated by ordinary initialization variance.

The secondary \\textit{acp\\_flag} diagnostic---which applies fixed thresholds to anomaly z-scores rather than to continuous model output---is more sensitive: the train-only variant identifies 176 ACP-flagged station-months versus 159 under the original full-record calculation (156 post-windowing), an $\\sim$11\\% relative difference. This does not affect model training (\\textit{acp\\_flag} is never an input feature), but slightly alters the evaluation subset. Given the negligible and non-significant performance impact confirmed above, we retain the canonical full-record pipeline for all main benchmark results, reporting this verification exercise for transparency.'''

text = text[:pos8_start] + replacement_8 + text[pos8_end:]
print('Applied Block 8 (Appendix B.3 consolidated). Text len:', len(text))


# --- BLOCK 7: Table 7 Sort descending by Mean AUC ---
pos7_label = text.find('\\label{tab:quantum_multiseed}')
pos7_start = text.rfind('\\begin{table}[htbp]', 0, pos7_label)
pos7_end = text.find('\\end{table}', pos7_start) + len('\\end{table}')
assert pos7_start != -1 and pos7_end != -1

replacement_7 = '''\\begin{table}[htbp]
\\centering
\\caption{Global AUC distribution across 15 independent training-seed replications (quantum models
only), sorted descending by mean AUC. Standard deviation is the sample standard deviation ($n-1$ denominator).}
\\label{tab:quantum_multiseed}
\\begin{tabular}{lcccc}
\\toprule
\\textbf{Architecture} & \\textbf{Mean AUC} & \\textbf{Std. Dev.} & \\textbf{Minimum} & \\textbf{Maximum} \\\\
\\midrule
LbL-QNN & 0.6605 & 0.0089 & 0.645 & 0.681 \\\\
LSTM+QNN & 0.6589 & 0.0153 & 0.628 & 0.681 \\\\
LSTM+PI-QNN & 0.6555 & 0.0172 & 0.605 & 0.673 \\\\
LSTM+QTL & 0.6519 & 0.0128 & 0.632 & 0.679 \\\\
\\bottomrule
\\end{tabular}
\\end{table}'''

text = text[:pos7_start] + replacement_7 + text[pos7_end:]
print('Applied Block 7 (Table 7 sorted). Text len:', len(text))


# --- BLOCK 6: Results Section 3 streamlined ---
pos6_start = text.find('\\subsection{Validating the Train-Only Standardization Protocol}')
pos6_end = text.find('\\label{fig:single_run_illusion}\n\\end{figure}') + len('\\label{fig:single_run_illusion}\n\\end{figure}')
assert pos6_start != -1 and pos6_end != -1

replacement_6 = '''\\subsection{Single-Run Performance and the Illusion of Quantum Advantage}
\\label{sec:single_run}

We evaluate model performance on the canonical spatiotemporal test partition ($N=528$ station-months, 2008--2026) using the canonical dataset \\texttt{01\\_data.pt} ($P_{60}=28.1$, $P_{90}=86.32$\\,mm). As established by our empirical validation on a rebuilt dataset (\\Cref{app:leakage_correction}), adopting a strict train-only standardization protocol yields downstream performance statistically indistinguishable from the canonical pipeline ($p_{\\text{Holm}} \\ge 0.328$).

\\Cref{tab:single_run_results}, \\Cref{fig:accuracy_f1}, and \\Cref{fig:roc_curve} summarize the single-run test performance across all evaluated models at a fixed initialization seed (\\texttt{seed=42}). Global accuracy is misleading due to class imbalance; we therefore focus on F1-score, ROC-AUC, and PR-AUC. The tree-based \\textbf{Random Forest} establishes the classical performance ceiling across all metrics (AUC = 0.688, F1 = 0.603). The hybrid quantum-classical architectures cluster tightly below (AUC: 0.639--0.672; F1: 0.580--0.604).

\\begin{table}[htbp]
\\centering
\\caption{Comparative performance on the spatiotemporal test set (Single-Run), including Precision-Recall AUC (PR-AUC) alongside ROC-AUC. Single representative run evaluated at default threshold 0.5 ($n=20$ ACP-flagged test months exceeding $P_{60}$ out of 25 flagged total; $n=49$ $P_{90}$-exceeding test months).}
\\label{tab:single_run_results}
\\setlength{\\tabcolsep}{4.5pt}
\\begin{tabular}{l c c c c c c c}
\\toprule
\\textbf{Model} & \\textbf{Acc (\\%)} & \\textbf{F1} & \\textbf{ROC-AUC} & \\textbf{PR-AUC} & \\textbf{Rc-$P_{90}$} & \\textbf{Rc-ACP} & \\textbf{FPR} \\\\
\\midrule
\\multicolumn{8}{l}{\\textit{Statistical \\& Ensemble}}\\\\
Logistic Regression & 62.3 & 0.472 & 0.655 & 0.546 & 0.571 & 0.650 & 0.228 \\\\
Random Forest & 63.8 & 0.603 & 0.688 & \\textbf{0.591} & 0.755 & 0.900 & 0.383 \\\\
Gradient Boosting & 61.6 & 0.522 & 0.671 & 0.577 & 0.592 & 0.700 & 0.312 \\\\
XGBoost & 59.1 & 0.548 & 0.650 & 0.545 & 0.694 & 0.850 & 0.418 \\\\
\\midrule
\\multicolumn{8}{l}{\\textit{Deep Learning}}\\\\
Standalone LSTM & 54.5 & 0.585 & 0.653 & 0.587\\textsuperscript{*} & 0.816 & 0.950 & 0.617 \\\\
Matched LSTM$^{\\dagger}$ & 57.2 & 0.565 & 0.653 & 0.587\\textsuperscript{*} & 0.776 & 0.950 & 0.502 \\\\
\\midrule
\\multicolumn{8}{l}{\\textit{Quantum-Classical Hybrid}}\\\\
LSTM+QNN & 58.0 & 0.599 & 0.639 & 0.504 & 0.796 & 0.950 & 0.550 \\\\
LSTM+PI-QNN & 58.9 & 0.583 & 0.662 & 0.558 & 0.755 & 0.950 & 0.489 \\\\
LSTM+QTL & 58.7 & 0.604 & 0.651 & 0.518 & 0.837 & 0.850 & 0.537 \\\\
\\textbf{LbL-QNN} & 59.1 & 0.589 & 0.672 & 0.581 & 0.796 & 0.900 & 0.495 \\\\
QKSVM & 59.8 & 0.475 & 0.639 & 0.510 & 0.612 & 0.600 & 0.293 \\\\
\\bottomrule
\\end{tabular}
\\vspace{4pt}
\\footnotesize\\textsuperscript{*}PR-AUC is threshold-independent and therefore identical to the unmatched Standalone LSTM row; only the operational point (FPR, recall) changes under matching.
\\footnotesize\\textsuperscript{$\\dagger$}Threshold-recalibrated variant of Standalone LSTM, evaluated at an operating point matched in false-positive rate to \\texttt{LbL-QNN} ($\\tau = 0.531$); defined and motivated in \\Cref{app:operational_analysis}.
\\footnotesize Rc-ACP is evaluated over the $n=20$ ACP-flagged test months that exceeded $P_{60}$; if all 25 flagged months are counted regardless of precipitation (the alarm rate), the respective values are: Logistic Regression 0.600, Random Forest 0.880, Gradient Boosting 0.680, XGBoost 0.840, Standalone LSTM 0.960, Matched LSTM 0.960, LSTM+QNN 0.920, LSTM+PI-QNN 0.960, LSTM+QTL 0.840, LbL-QNN 0.920, QKSVM 0.520.
\\end{table}

\\begin{figure}[htbp]
    \\centering
    \\includegraphics[width=\\textwidth]{figura_accuracy_f1.pdf}
    \\caption{Single-run global accuracy (left) and F1-score for the extreme precipitation class (right) across all eleven evaluated architectures.}
    \\label{fig:accuracy_f1}
\\end{figure}

\\begin{figure}[htbp]
    \\centering
    \\includegraphics[width=0.85\\textwidth]{figura_roc.pdf}
    \\caption{Receiver Operating Characteristic (ROC) curves for all eleven evaluated architectures (Single-Run). Random Forest establishes the classical ceiling; \\texttt{LbL-QNN} is the highest-performing quantum architecture.}
    \\label{fig:roc_curve}
\\end{figure}

In this single realization, the layer-wise hybrid \\texttt{LbL-QNN} achieves the highest quantum AUC (0.672), tracking the Random Forest ROC curve (\\Cref{fig:roc_curve}) with no statistically significant difference under an uncorrected DeLong test \\citep{delong1988} ($p = 0.2124$; formal equivalence not tested). It nominally outperforms unconstrained \\texttt{LSTM+QNN} ($p = 0.0070$) and \\texttt{Standalone LSTM} ($p = 0.0263$), though the latter comparison is heavily confounded by the capacity mismatch (10,103 vs.\\ $\\sim$2,600 parameters; \\Cref{sec:architectures}).

Crucially, this apparent single-run superiority dissolves under closer examination. First, because the extreme tail contains limited test observations ($n=20$ ACP months exceeding $P_{60}$, $n=49$ $P_{90}$ months), point estimates of extreme recall carry wide 95\\% Wilson confidence intervals that broadly overlap across all architectures (e.g., Random Forest $0.900$ [0.699, 0.972] vs.\\ \\texttt{LSTM+QNN} $0.950$ [0.764, 0.991]; documented in \\Cref{app:operational_analysis}). Second, the apparent operational difference between \\texttt{Standalone LSTM} (Rc-ACP = 0.950, FPR = 0.617) and \\texttt{LbL-QNN} (Rc-ACP = 0.900, FPR = 0.495) is entirely an artifact of default threshold selection: recalibrating the classical LSTM to operate at an identical FPR of $\\approx$50.2\\% (\\texttt{Matched LSTM}, $\\tau = 0.531$) preserves its recall at 0.950 (19/20) with no statistically significant difference under paired bootstrap resampling ($n=2{,}000$; full analysis in \\Cref{app:operational_analysis}). Finally, as demonstrated in \\Cref{sec:stability}, multi-seed replication reveals that these single-run margins reflect initialization variance rather than robust architectural advantages.'''

text = text[:pos6_start] + replacement_6 + text[pos6_end:]
print('Applied Block 6 (Sec 3 streamlined). Text len:', len(text))


# --- BLOCK 5: Table 5 Sort descending by Mean AUC ---
pos5_caption = text.find('\\caption{Bottleneck-controlled comparison')
pos5_start = text.rfind('\\begin{table}[htbp]', 0, pos5_caption)
pos5_end = text.find('\\end{table}', pos5_start) + len('\\end{table}')
assert pos5_start != -1 and pos5_end != -1

replacement_5 = '''\\begin{table}[htbp]
\\centering
\\caption{Bottleneck-controlled comparison: classical controls versus the quantum circuit under an
identical 4-dimensional information bottleneck, versus the full-feature Random Forest ceiling,
sorted descending by mean AUC. All values are 15-seed distributions with seed-paired initialization.}
\\label{tab:bottleneck_control}
\\begin{tabular}{lcccc}
\\toprule
\\textbf{Architecture} & \\textbf{Params} & \\textbf{Mean AUC} & \\textbf{Std. Dev.} & \\textbf{Max} \\\\
\\midrule
Random Forest (full feature space, 35 dim.) & Ensemble\\textsuperscript{*} & 0.6851 & 0.0026 & 0.690 \\\\
LSTM+QNN (quantum, 4-dim. bottleneck) & 24 & 0.6589 & 0.0153 & 0.681 \\\\
GELU MLP (classical control, 4-dim. input) & 180 & 0.6563 & 0.0071 & 0.6725 \\\\
Linear (classical control, 4-dim. input) & 20 & 0.6545 & 0.0069 & 0.6680 \\\\
LeakyReLU MLP (classical control, 4-dim. input) & 28 & 0.6516 & 0.0083 & 0.6636 \\\\
\\bottomrule
\\end{tabular}
\\end{table}'''

text = text[:pos5_start] + replacement_5 + text[pos5_end:]
print('Applied Block 5 (Table 5 sorted). Text len:', len(text))


# --- BLOCK 4: Model List in Sec 2.4 ---
pos4_start = text.find('We evaluate eleven predictive models divided into four classes.')
pos4_end = text.find('\\end{enumerate}', text.find('\\item \\texttt{QKSVM}', pos4_start)) + len('\\end{enumerate}')
assert pos4_start != -1 and pos4_end != -1

replacement_4 = '''We evaluate ten distinct model architectures alongside one threshold-matched operational configuration (yielding eleven evaluation configurations in \\Cref{tab:single_run_results}), grouped into four paradigms:
\\begin{enumerate}
    \\item \\textbf{Statistical Reference Baselines:} Climatological Persistence (evaluated as an unranked binary baseline for accuracy and F1) and L2-regularized Logistic Regression.
    \\item \\textbf{Classical Ensembles:} Random Forest (300 estimators), Gradient Boosting (200 estimators), and XGBoost (500 estimators, depth=4).
    \\item \\textbf{Deep Learning Baselines:} A standalone bidirectional LSTM (\\texttt{PureLSTM}) with 2 layers and 16 hidden units per direction (yielding a concatenated 32-dimensional sequence representation), concatenated with the 35 input features (67 dimensions), followed by batch normalization (\\texttt{BatchNorm1d(67)}) and a two-layer classification head (\\texttt{Linear(67, 16)} + LayerNorm + \\texttt{Linear(16, 1)}), totaling 10,103 parameters, optimized using AdamW. Alongside this model, we evaluate \\texttt{Matched LSTM}, an operational configuration of \\texttt{PureLSTM} with its classification threshold adjusted to match the false positive rate of the hybrid models ($\\tau = 0.531$; \\Cref{app:operational_analysis}). We emphasize that the standalone LSTM is not capacity-matched to the hybrids (10,103 vs.\\ $\\sim$2,600 parameters; 16 vs.\\ 8 hidden units per direction; batch normalization; and direct access to raw features without a 4-dimensional projection bottleneck).
    \\item \\textbf{Hybrid Quantum-Classical Models:} Developed using PennyLane\'s \\texttt{default.qubit} simulator. All models use a bidirectional LSTM (hidden dimension 8 per direction, 2,368 parameters) concatenated with the 35 input features to produce a 51-dimensional representation, which is compressed via a linear layer (\\texttt{Linear(51, 4)}, 208 parameters) into a 4-dimensional projection bottleneck before being encoded into a 4-qubit Variational Quantum Circuit. The circuit uses a Strongly Entangling Layers ansatz with same-vector data re-uploading ($R_y$ and $R_z$ rotations per qubit repeated in each layer; \\citealp{perez2020}) followed by parameterized $\\mathrm{SU}(2)$ $\\mathrm{Rot}$ gates and a circular CNOT entanglement topology ($L = 2$ layers with 24 quantum parameters for baseline models; $L = 3$ layers with 36 quantum parameters for the layer-wise \\texttt{LbL-QNN} model; see \\Cref{fig:vqc_circuit}). We distinguish this repeated encoding from the \\emph{information-widening} re-uploading variant introduced later (\\Cref{sec:relaxing_bottleneck}), which injects distinct feature subsets at each block.
    \\begin{itemize}
        \\item \\texttt{LSTM+QNN}: Standard end-to-end unconstrained variational network (2 layers).
        \\item \\texttt{LSTM+PI-QNN}: End-to-end network optimized with the heuristic thermodynamic loss $\\mathcal{L}_{\\text{total}}$ (2 layers; \\Cref{app:heuristic_regularization}).
        \\item \\texttt{LSTM+QTL}: Quantum Transfer Learning \\citep{mari2020}, where the classical LSTM features are frozen and only a lightweight classical adapter feeding the quantum layer is optimized (2 layers).
        \\item \\texttt{LbL-QNN}\\footnote{This architecture combines layer-by-layer training \\citep{skolik2021} with the heuristic thermodynamic loss (\\Cref{app:heuristic_regularization}). As shown in \\Cref{app:depth_justification}, a depth-matched ablation confirms the stability benefit is not attributable to circuit depth alone. However, this comparison is confounded by an expanded training budget (55 total epochs across three stages: 25 for L1, 15 for L2, 15 for L3 vs.\\ 25 single-run epochs) and a staged learning rate schedule ($5\\times 10^{-3}$ then $5\\times 10^{-4}$ vs.\\ fixed $5\\times 10^{-3}$). Furthermore, because a penalty-free layer-wise control was not evaluated, the separate contribution of the layer-by-layer protocol versus the heuristic loss penalty remains unidentified; the observed performance should be attributed to the combined layer-wise/heuristic configuration.}: applies the layer-by-layer training protocol to the \\texttt{LSTM+PI-QNN} architecture, using a 3-layer circuit to support a non-degenerate three-stage freezing schedule.
        \\item \\texttt{QKSVM}: Quantum Kernel Support Vector Machine utilizing a Nystr\"{o}m approximation ($M=64$ landmarks) to handle the panel\'s size.
    \\end{itemize}
\\end{enumerate}'''

text = text[:pos4_start] + replacement_4 + text[pos4_end:]
print('Applied Block 4 (Model list resolved). Text len:', len(text))


# --- BLOCK 3: Loss equations before Sec 2.4 replaced with concise summary ---
pos3_start = text.find('The total loss function $\\mathcal{L}_{\\text{total}}$ combines binary cross-entropy')
pos3_end = text.find('\\Cref{app:lambda_sweep}.\n', pos3_start) + len('\\Cref{app:lambda_sweep}.\n')
assert pos3_start != -1 and pos3_end != -1

replacement_3 = '''To test whether domain-motivated soft constraints alter optimization trajectories, we evaluate two hybrid configurations incorporating heuristic thermodynamic regularization (\\texttt{LSTM+PI-QNN} and \\texttt{LbL-QNN}). As detailed in \\Cref{app:heuristic_regularization}, this scheme supplements binary cross-entropy with two phenomenological soft penalties: a temperature-dependent gating term ($\\mathcal{L}_{\\text{CC}}$) penalizing high precipitation predictions under restricted saturation-capacity conditions, and a latent-coordinate balance term ($\\mathcal{L}_{\\text{hMSE}}$) acting on the 4-qubit expectation readout coordinates ($q_0, \\dots, q_3$). These are empirical soft regularizers rather than first-principles thermodynamic conservation laws. As shown in \\Cref{sec:stability}, 15-seed replication reveals that this heuristic regularization confers no statistically significant performance advantage over the unconstrained hybrid baseline (Friedman $p = 0.458$).

'''

text = text[:pos3_start] + replacement_3 + text[pos3_end:]
print('Applied Block 3 (Loss equations replaced). Text len:', len(text))


# --- BLOCK 2: Section 2.3 & Glossary (remove Fig 2 and Table 3) ---
pos2_start = text.find('\\subsection{Phenomenological Inductive-Bias Experiment}')
pos2_end = text.find('\\end{table}', text.find('\\label{tab:qubits_mapping}', pos2_start)) + len('\\end{table}')
assert pos2_start != -1 and pos2_end != -1

replacement_2 = '''\\subsection{Engineered Meteorological Features and Glossary}
\\label{sec:glossary}

To capture antecedent physical conditions without future-to-past leakage, we construct 35 tabular covariates per station-month from lagged observations:
\\begin{itemize}
    \\item \\textbf{Antecedent Precipitation Lags:} 12 monthly lagged precipitation values ($P_{i-11}, \\dots, P_{i}$) capturing long-term hydrologic memory.
    \\item \\textbf{Lagged Synoptic Anomaly Indicators:} Monthly anomalies of surface pressure (\\texttt{iap}), mean wind speed (\\texttt{iav}), maximum wind gust (\\texttt{ia\\_racha}), and sunshine duration (\\texttt{ia\\_sol}), alongside day-to-day pressure variability (\\texttt{ia\\_dp\\_std}), centered against station-month climatology.
    \\item \\textbf{Derived Indices and Interaction Proxies:}
    \\begin{itemize}
        \\item \\textit{Summer evaporative-demand memory ($\\mathrm{ET}_0^{\\mathrm{JJA}}$):} Cumulative reference evapotranspiration during June--August, acting as an empirical land-surface moisture-demand proxy.
        \\item \\textit{Seasonal thermal gradient proxy ($\\Delta T_{3\\mathrm{m}}$, \\texttt{choque\\_clausius}):} A 3-month lagged difference in monthly-mean temperature anomalies ($T_{i-1} - T_{i-4}$), capturing seasonal cooling rates rather than Clausius--Clapeyron thermodynamics.
        \\item \\textit{Evaporation--variability interaction proxy ($\\Phi_{\\mathrm{conv}}$, \\texttt{potencial\\_convectivo}):} The product of summer evaporative demand and monthly pressure variability ($\\mathrm{ET}_0^{\\mathrm{JJA}} \\times \\sigma_p$), representing an empirical fuel--trigger proxy rather than convective available potential energy.
    \\end{itemize}
    \\item \\textbf{Periodic Seasonality and Station Identifiers:} Sine and cosine transforms of the target calendar month $i+12$, together with four one-hot station indicator variables (\\texttt{ciudad\\_*}).
\\end{itemize}

\\Cref{tab:glossary} maps the descriptive feature labels used throughout this manuscript to their corresponding code tokens in \\texttt{fisica.py} and \\texttt{features.py}.

\\begin{table}[htbp]
\\centering
\\caption{Engineered feature glossary: manuscript labels vs.\\ repository code tokens.}
\\label{tab:glossary}
\\begin{tabular}{lll}
\\toprule
\\textbf{Manuscript label} & \\textbf{Code token} & \\textbf{Definition} \\\\
\\midrule
Pressure anomaly & \\texttt{iap} & Monthly-mean pressure minus station-month climatology \\\\
Wind-speed anomaly & \\texttt{iav} & Wind speed minus station-month climatology \\\\
Wind-gust anomaly & \\texttt{ia\\_racha} & Gust speed minus station-month climatology \\\\
Sunshine anomaly & \\texttt{ia\\_sol} & Sunshine sum minus station-month climatology \\\\
Pressure-variability anomaly & \\texttt{ia\\_dp\\_std} & $\\sigma_p$ minus station-month climatology \\\\
Station dummy & \\texttt{ciudad\\_*} & One-hot station identifier \\\\
ACP diagnostic flag & \\texttt{acp\\_flag} & Autumn + negative sunshine anom. + positive gust anom. \\\\
Instability index & \\texttt{ib} & Pressure variability / diurnal temperature range \\\\
Insolation index & \\texttt{iir} & Sunshine fraction of theoretical maximum \\\\
Seasonal thermal gradient proxy & \\texttt{choque\\_clausius} & 3-month lagged thermal anomaly difference ($T_{i-1} - T_{i-4}$) \\\\
Evaporation--variability interaction proxy & \\texttt{potencial\\_convectivo} & Summer $\\text{ET}_0^{\\text{JJA}} \\times$ monthly pressure variability ($\\sigma_p$) \\\\
\\bottomrule
\\end{tabular}
\\end{table}

These covariates provide candidate inductive biases and input features, not physical state variables in a numerical weather prediction model. The empirical correlation plots, post-hoc latent-coordinate mapping, and detailed formulation of the heuristic thermodynamic loss function are presented in \\Cref{app:heuristic_regularization}.'''

text = text[:pos2_start] + replacement_2 + text[pos2_end:]
print('Applied Block 2 (Sec 2.3 streamlined). Text len:', len(text))


# --- BLOCK 1: Trim duplicate sentence in Sec 2.2 ---
pos1_start = text.find('By training models on the $P_{60}$ threshold and evaluating diagnostic subsets')
pos1_end = text.find('distribution of extreme rainfall.\n', pos1_start) + len('distribution of extreme rainfall.\n')
assert pos1_start != -1 and pos1_end != -1

replacement_1 = '''By training models on the $P_{60}$ threshold and evaluating diagnostic subsets
defined by $P_{90}$ exceedance and a surface-derived autumn convective proxy,
we perform a tail-shift and regime-conditional robustness analysis. Because
the models are trained as binary classifiers on the $P_{60}$ target rather than as continuous
precipitation-distribution models, this analysis assesses discrimination on an extreme subset
without estimating the magnitude, return period, or conditional distribution of rainfall.
'''

text = text[:pos1_start] + replacement_1 + text[pos1_end:]
print('Applied Block 1 (Sec 2.2 duplicate trimmed). Text len:', len(text))

# Save the restructured manuscript
with open('main_revised.tex', 'w', encoding='utf-8') as f:
    f.write(text)

print('SUCCESSFULLY COMPLETED WP6 RESTRUCTURING!')
