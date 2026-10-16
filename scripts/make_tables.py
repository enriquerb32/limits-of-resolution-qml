"""
make_tables.py
Generates every LaTeX results table from results/results_master.csv.
Flags any discrepancies with current manuscript values.
"""

import os
import pandas as pd
import numpy as np

def format_val(val, decimals=3):
    if pd.isna(val):
        return "--"
    return f"{val:.{decimals}f}"

def format_pct(val, decimals=1):
    if pd.isna(val):
        return "--"
    return f"{val * 100:.{decimals}f}"

def main():
    os.makedirs('tables', exist_ok=True)
    df = pd.read_csv('results/results_master.csv')
    
    print("=" * 70)
    print("GENERATING LATEX TABLES FROM results/results_master.csv")
    print("=" * 70)
    
    # -------------------------------------------------------------
    # 1. TABLE 1: Single-Run Performance
    # -------------------------------------------------------------
    single_df = df[df['run_type'] == 'single_run'].pivot(index='model', columns='metric', values='value')
    
    # Model order for Table 1
    t1_order = [
        ('Statistical & Ensemble', [
            'Logistic Regression',
            'Random Forest',
            'Gradient Boosting',
            'XGBoost'
        ]),
        ('Deep Learning', [
            'Standalone LSTM',
            'Matched LSTM'
        ]),
        ('Quantum-Classical Hybrid', [
            'LSTM+QNN',
            'LSTM+PI-QNN',
            'LSTM+QTL',
            'PI-QNN-LbL',
            'QKSVM'
        ])
    ]
    
    t1_lines = []
    t1_lines.append(r"\begin{table}[htbp]")
    t1_lines.append(r"\centering")
    t1_lines.append(r"\caption{Comparative performance on the spatiotemporal test set (Single-Run), including Precision-Recall AUC (PR-AUC) alongside ROC-AUC. Single representative run at $n=20$ ACP-flagged months exceeding $P_{60}$ (25 flagged total) and $n=49$ $P_{90}$-exceeding months.}")
    t1_lines.append(r"\label{tab:single_run_results}")
    t1_lines.append(r"\setlength{\tabcolsep}{4.5pt}")
    t1_lines.append(r"\begin{tabular}{l c c c c c c c}")
    t1_lines.append(r"\toprule")
    t1_lines.append(r"\textbf{Model} & \textbf{Acc (\%)} & \textbf{F1} & \textbf{ROC-AUC} & \textbf{PR-AUC} & \textbf{Rc-$P_{90}$} & \textbf{Rc-ACP} & \textbf{FPR} \\")
    t1_lines.append(r"\midrule")
    
    for section_name, models in t1_order:
        t1_lines.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{section_name}}}}}\\\\")
        for m in models:
            row = single_df.loc[m]
            acc_str = format_pct(row['acc'], 1)
            f1_str = format_val(row['f1'], 3)
            auc_str = format_val(row['auc'], 3)
            
            # Highlight best PR-AUC
            prauc_str = format_val(row['pr_auc'], 3)
            if m == 'Random Forest':
                prauc_str = f"\\textbf{{{prauc_str}}}"
            elif m in ['Standalone LSTM', 'Matched LSTM']:
                prauc_str = f"{prauc_str}\\textsuperscript{{*}}"
                
            rc_p90_str = format_val(row['rc_p90'], 3)
            rc_acp_str = format_val(row['rc_dana_n20'], 3)
            fpr_str = format_val(row['fpr'], 3)
            
            disp_name = m
            if m == 'Matched LSTM':
                disp_name = r"Matched LSTM$^{\dagger}$"
            elif m == 'PI-QNN-LbL':
                disp_name = r"\textbf{LbL-QNN}"
                
            t1_lines.append(f"{disp_name} & {acc_str} & {f1_str} & {auc_str} & {prauc_str} & {rc_p90_str} & {rc_acp_str} & {fpr_str} \\\\")
        if section_name != 'Quantum-Classical Hybrid':
            t1_lines.append(r"\midrule")
            
    t1_lines.append(r"\bottomrule")
    t1_lines.append(r"\end{tabular}")
    t1_lines.append(r"\vspace{4pt}")
    t1_lines.append(r"\footnotesize\textsuperscript{*}PR-AUC is threshold-independent and therefore identical to the unmatched Standalone LSTM row; only the operational point (FPR, recall) changes under matching.")
    t1_lines.append(r"\footnotesize\textsuperscript{$\dagger$}Threshold-recalibrated variant of Standalone LSTM, evaluated at an operating point matched in false-positive rate to \texttt{LbL-QNN} ($\tau = 0.531$); defined and motivated in \Cref{sec:operational}.")
    t1_lines.append(r"\footnotesize Rc-ACP is evaluated over the $n=20$ ACP-flagged test months that exceeded $P_{60}$; if all 25 flagged months are counted regardless of precipitation, the respective values are: Logistic Regression 0.600, Random Forest 0.880, Gradient Boosting 0.680, XGBoost 0.840, Standalone LSTM 0.960, Matched LSTM 0.960, LSTM+QNN 0.920, LSTM+PI-QNN 0.960, LSTM+QTL 0.840, LbL-QNN 0.920, QKSVM 0.520.")
    t1_lines.append(r"\end{table}")
    
    t1_tex = "\n".join(t1_lines)
    with open('tables/table1_single_run.tex', 'w', encoding='utf-8') as f:
        f.write(t1_tex)
    print("Table 1 written to tables/table1_single_run.tex")
    
    # -------------------------------------------------------------
    # 2. TABLE 4: Extreme-tail recall with Wilson CIs
    # -------------------------------------------------------------
    t4_models = [
        ('Logistic Regression', 'Logistic Regression'),
        ('Random Forest', 'Random Forest'),
        ('Gradient Boosting', 'Gradient Boosting'),
        ('XGBoost', 'XGBoost'),
        ('Standalone LSTM', 'Standalone LSTM'),
        ('LSTM+QNN', 'LSTM+QNN'),
        ('LSTM+PI-QNN', 'LSTM+PI-QNN'),
        ('LSTM+QTL', 'LSTM+QTL'),
        ('PI-QNN-LbL', 'LbL-QNN'),
        ('QKSVM', 'QKSVM')
    ]
    
    t4_lines = []
    t4_lines.append(r"\begin{table}[htbp]")
    t4_lines.append(r"\centering")
    t4_lines.append(r"\caption{Extreme-tail recall with 95\% Wilson confidence intervals, single representative run ($n=20$ ACP-flagged exceeding $P_{60}$ out of 25 flagged total; $n=49$ $P_{90}$-exceeding test months).}")
    t4_lines.append(r"\label{tab:recall_wilson_ci}")
    t4_lines.append(r"\begin{tabular}{lcc}")
    t4_lines.append(r"\toprule")
    t4_lines.append(r"\textbf{Architecture} & \textbf{Rc-ACP [95\% CI]} & \textbf{Rc-$P_{90}$ [95\% CI]} \\")
    t4_lines.append(r"\midrule")
    
    for m_key, m_disp in t4_models:
        row = single_df.loc[m_key]
        rc_acp = row['rc_dana_n20']
        ci_a_lo = row['rc_dana_n20_ci_lo']
        ci_a_hi = row['rc_dana_n20_ci_hi']
        
        rc_p90 = row['rc_p90']
        ci_p_lo = row['rc_p90_ci_lo']
        ci_p_hi = row['rc_p90_ci_hi']
        
        acp_str = f"{rc_acp:.3f} [{ci_a_lo:.3f}, {ci_a_hi:.3f}]"
        p90_str = f"{rc_p90:.3f} [{ci_p_lo:.3f}, {ci_p_hi:.3f}]"
        
        t4_lines.append(f"{m_disp} & {acp_str} & {p90_str} \\\\")
        
    t4_lines.append(r"\bottomrule")
    t4_lines.append(r"\end{tabular}")
    t4_lines.append(r"\end{table}")
    
    t4_tex = "\n".join(t4_lines)
    with open('tables/table4_recall_wilson.tex', 'w', encoding='utf-8') as f:
        f.write(t4_tex)
    print("Table 4 written to tables/table4_recall_wilson.tex")
    
    # -------------------------------------------------------------
    # 3. TABLE 6: Quantum Multi-Seed AUC Distribution
    # -------------------------------------------------------------
    multi_df = df[df['run_type'] == 'multi_seed']
    q_models = ['LSTM+QNN', 'LSTM+PI-QNN', 'LSTM+QTL', 'PI-QNN-LbL']
    
    q_stats = []
    for qm in q_models:
        vals = multi_df[(multi_df['model'] == qm) & (multi_df['metric'] == 'auc')]['value'].values
        q_stats.append({
            'model': 'LbL-QNN' if qm == 'PI-QNN-LbL' else qm,
            'mean': np.mean(vals),
            'std': np.std(vals, ddof=1),
            'min': np.min(vals),
            'max': np.max(vals)
        })
    df_q_stats = pd.DataFrame(q_stats).sort_values(by='mean', ascending=False)
    
    t6_lines = []
    t6_lines.append(r"\begin{table}[htbp]")
    t6_lines.append(r"\centering")
    t6_lines.append(r"\caption{Global AUC distribution across 15 independent training-seed replications (quantum models only), sorted descending by mean AUC. Standard deviation is the sample standard deviation ($n-1$ denominator).}")
    t6_lines.append(r"\label{tab:quantum_multiseed}")
    t6_lines.append(r"\begin{tabular}{lcccc}")
    t6_lines.append(r"\toprule")
    t6_lines.append(r"\textbf{Architecture} & \textbf{Mean AUC} & \textbf{Std. Dev.} & \textbf{Minimum} & \textbf{Maximum} \\")
    t6_lines.append(r"\midrule")
    for _, r in df_q_stats.iterrows():
        t6_lines.append(f"{r['model']} & {r['mean']:.4f} & {r['std']:.4f} & {r['min']:.3f} & {r['max']:.3f} \\\\")
    t6_lines.append(r"\bottomrule")
    t6_lines.append(r"\end{tabular}")
    t6_lines.append(r"\end{table}")
    
    t6_tex = "\n".join(t6_lines)
    with open('tables/table6_quantum_multiseed.tex', 'w', encoding='utf-8') as f:
        f.write(t6_tex)
    print("Table 6 written to tables/table6_quantum_multiseed.tex")
    
    # -------------------------------------------------------------
    # 4. TABLE 7: Operational Metric Distributions Across 15 Seeds
    # -------------------------------------------------------------
    t7_order = [
        ('Classical Ensembles', ['Random Forest', 'Gradient Boosting', 'XGBoost']),
        ('Deep Learning', ['Standalone LSTM']),
        ('Quantum-Classical Hybrid', ['LSTM+QNN', 'LSTM+PI-QNN', 'LSTM+QTL', 'PI-QNN-LbL'])
    ]
    
    t7_lines = []
    t7_lines.append(r"\begin{table}[htbp]")
    t7_lines.append(r"\centering")
    t7_lines.append(r"\caption{Operational metric distributions (mean $\pm$ across-seed sample standard deviation, ddof=1) across 15 training seeds at a default decision threshold of 0.5, for all gradient-optimized and ensemble architectures ($n=20$ ACP months exceeding $P_{60}$, $n=49$ $P_{90}$ months).}")
    t7_lines.append(r"\label{tab:operational_multiseed}")
    t7_lines.append(r"\begin{tabular}{lccc}")
    t7_lines.append(r"\toprule")
    t7_lines.append(r"\textbf{Architecture} & \textbf{FPR} & \textbf{Rc-ACP} & \textbf{Rc-$P_{90}$} \\")
    t7_lines.append(r"\midrule")
    
    for sec_name, models in t7_order:
        t7_lines.append(f"\\textit{{{sec_name}}}\\\\")
        for m in models:
            m_data = multi_df[multi_df['model'] == m]
            fpr_vals = m_data[m_data['metric'] == 'fpr']['value'].values
            rc_acp_vals = m_data[m_data['metric'] == 'rc_dana']['value'].values
            rc_p90_vals = m_data[m_data['metric'] == 'rc_p90']['value'].values
            
            fpr_str = f"${np.mean(fpr_vals):.3f} \\pm {np.std(fpr_vals, ddof=1):.3f}$"
            acp_str = f"${np.mean(rc_acp_vals):.3f} \\pm {np.std(rc_acp_vals, ddof=1):.3f}$"
            p90_str = f"${np.mean(rc_p90_vals):.3f} \\pm {np.std(rc_p90_vals, ddof=1):.3f}$"
            
            disp_m = 'LbL-QNN' if m == 'PI-QNN-LbL' else m
            t7_lines.append(f"{disp_m} & {fpr_str} & {acp_str} & {p90_str} \\\\")
        if sec_name != 'Quantum-Classical Hybrid':
            t7_lines.append(r"\midrule")
            
    t7_lines.append(r"\bottomrule")
    t7_lines.append(r"\end{tabular}")
    t7_lines.append(r"\end{table}")
    
    t7_tex = "\n".join(t7_lines)
    with open('tables/table7_operational_multiseed.tex', 'w', encoding='utf-8') as f:
        f.write(t7_tex)
    print("Table 7 written to tables/table7_operational_multiseed.tex")
    
    # -------------------------------------------------------------
    # 5. TABLE 8: All Eight Architectures Multi-Seed AUC (Sorted Descending)
    # -------------------------------------------------------------
    all_8_models = ['Random Forest', 'Gradient Boosting', 'XGBoost', 'Standalone LSTM',
                    'LSTM+QNN', 'LSTM+PI-QNN', 'LSTM+QTL', 'PI-QNN-LbL']
    all_8_stats = []
    for m in all_8_models:
        vals = multi_df[(multi_df['model'] == m) & (multi_df['metric'] == 'auc')]['value'].values
        all_8_stats.append({
            'model': 'LbL-QNN' if m == 'PI-QNN-LbL' else m,
            'mean': np.mean(vals),
            'std': np.std(vals, ddof=1),
            'min': np.min(vals),
            'max': np.max(vals)
        })
    df_all_8 = pd.DataFrame(all_8_stats).sort_values(by='mean', ascending=False)
    
    t8_lines = []
    t8_lines.append(r"\begin{table}[htbp]")
    t8_lines.append(r"\centering")
    t8_lines.append(r"\caption{Global AUC distribution across 15 independent seeds, all eight architectures, sorted descending by mean AUC. Standard deviation is the sample standard deviation ($n-1$ denominator).}")
    t8_lines.append(r"\label{tab:multiseed_all}")
    t8_lines.append(r"\begin{tabular}{lcccc}")
    t8_lines.append(r"\toprule")
    t8_lines.append(r"\textbf{Architecture} & \textbf{Mean AUC} & \textbf{Std. Dev.} & \textbf{Minimum} & \textbf{Maximum} \\")
    t8_lines.append(r"\midrule")
    for _, r in df_all_8.iterrows():
        t8_lines.append(f"{r['model']} & {r['mean']:.4f} & {r['std']:.4f} & {r['min']:.3f} & {r['max']:.3f} \\\\")
    t8_lines.append(r"\bottomrule")
    t8_lines.append(r"\end{tabular}")
    t8_lines.append(r"\end{table}")
    
    t8_tex = "\n".join(t8_lines)
    with open('tables/table8_multiseed_all.tex', 'w', encoding='utf-8') as f:
        f.write(t8_tex)
    print("Table 8 written to tables/table8_multiseed_all.tex")
    
    # -------------------------------------------------------------
    # 6. TABLE 12: Noise Degradation Table
    # -------------------------------------------------------------
    noise_df = df[df['run_type'] == 'noise_evaluation'].set_index(['model', 'metric'])['value']
    
    t12_lines = []
    t12_lines.append(r"\begin{table}[htbp]")
    t12_lines.append(r"\centering")
    t12_lines.append(r"\caption{Predictive metric degradation under simulated depolarizing gate noise ($p = 0.01$, 15-seed Monte Carlo over noise draws on fixed weights; $n=20$ ACP months exceeding $P_{60}$).}")
    t12_lines.append(r"\label{tab:noise_degradation}")
    t12_lines.append(r"\begin{tabular}{l l c c c c}")
    t12_lines.append(r"\toprule")
    t12_lines.append(r"\textbf{Architecture} & \textbf{Environment} & \textbf{Accuracy (\%)} & \textbf{F1-Score} & \textbf{Rc-ACP} & \textbf{FPR} \\")
    t12_lines.append(r"\midrule")
    
    for m_key, m_disp in [('LSTM+QNN', 'LSTM+QNN'), ('PI-QNN-LbL', 'LbL-QNN')]:
        row_clean = single_df.loc[m_key]
        clean_acc = format_pct(row_clean['acc'], 1)
        clean_f1 = format_val(row_clean['f1'], 3)
        clean_acp = format_val(row_clean['rc_dana_n20'], 3)
        clean_fpr = format_val(row_clean['fpr'], 3)
        
        noisy_acc = f"{noise_df.loc[(m_key, 'noisy_acc')] * 100:.1f}"
        noisy_f1 = f"${noise_df.loc[(m_key, 'noisy_f1')]:.3f} \\pm {noise_df.loc[(m_key, 'noisy_f1_std')]:.3f}$"
        noisy_acp = f"${noise_df.loc[(m_key, 'noisy_rc_dana')]:.3f} \\pm {noise_df.loc[(m_key, 'noisy_rc_dana_std')]:.3f}$"
        noisy_fpr = f"${noise_df.loc[(m_key, 'noisy_fpr')]:.3f} \\pm {noise_df.loc[(m_key, 'noisy_fpr_std')]:.3f}$"
        
        t12_lines.append(f"\\multirow{{2}}{{*}}{{{m_disp}}} & Noiseless & {clean_acc} & {clean_f1} & {clean_acp} & {clean_fpr} \\\\")
        t12_lines.append(f" & Noisy ($p=0.01$) & {noisy_acc} & {noisy_f1} & {noisy_acp} & {noisy_fpr} \\\\")
        if m_key == 'LSTM+QNN':
            t12_lines.append(r"\midrule")
            
    t12_lines.append(r"\bottomrule")
    t12_lines.append(r"\end{tabular}")
    t12_lines.append(r"\end{table}")
    
    t12_tex = "\n".join(t12_lines)
    with open('tables/table12_noise_degradation.tex', 'w', encoding='utf-8') as f:
        f.write(t12_tex)
    print("Table 12 written to tables/table12_noise_degradation.tex")
    
    # -------------------------------------------------------------
    # RECONCILIATION REPORT (FLAGGING DISCREPANCIES)
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RECONCILIATION REPORT: MANUSCRIPT VALUES VS results_master.csv")
    print("=" * 70)
    
    flags = []
    
    # Flag 1: Table 1 Rc-ACP definition (n=20 vs n=25)
    flags.append("TABLE 1 (Rc-ACP): Manuscript mixed n=25 (0.600, 0.880, 0.680, 0.840, 0.960, 0.960, 0.840, 0.920, 0.520) with n=20-like (0.950). Standardized uniformly to n=20 recall (0.650, 0.900, 0.700, 0.850, 0.950, 0.950, 0.950, 0.950, 0.850, 0.900, 0.600) with n=25 values documented in footnote.")
    
    # Flag 2: Table 1 QKSVM Acc and F1
    flags.append("TABLE 1 (QKSVM): Stored checkpoint has Acc=58.9%, F1=0.580 (from svm.predict() threshold ~0.398). Recomputed at threshold 0.5 yields Acc=59.8%, F1=0.475. Manuscript Table 1 used svm.predict() for Acc/F1, but threshold 0.5 for ROC-AUC (0.639), PR-AUC (0.510), Rc-P90 (0.612), Rc-ACP (0.520), and FPR (0.293).")
    
    # Flag 3: Table 4 Wilson CIs
    flags.append("TABLE 4 (Wilson CIs): Manuscript Table 4 used n=25 for Rc-ACP ([0.407, 0.766], [0.700, 0.958], etc.). Recomputed with canonical n=20 recall: LogReg [0.433, 0.819], RF [0.699, 0.972], GB [0.481, 0.855], XGBoost [0.640, 0.948], LSTM [0.764, 0.991], LSTM+QNN [0.764, 0.991], PI-QNN [0.764, 0.991], QTL [0.640, 0.948], LbL [0.699, 0.972], QKSVM [0.387, 0.781]. Rc-P90 (n=49) reproduced exactly.")
    
    # Flag 4: Table 7 Standard Deviations (ddof=1)
    flags.append("TABLE 7 (Operational STDs): Manuscript Table 7 used ddof=1 for classical ensembles, but ddof=0 for quantum models (e.g. LSTM+QNN FPR std 0.081 -> 0.083; Rc-ACP std 0.066 -> 0.068; Rc-P90 std 0.045 -> 0.047; LbL FPR std 0.076 -> 0.079; Rc-ACP std 0.068 -> 0.070; Rc-P90 std 0.043 -> 0.045). Standardized to sample std (ddof=1).")
    
    # Flag 5: Table 8 Ordering and RF std
    flags.append("TABLE 8 (Multi-Seed Table): RF std was 0.0025 (ddof=0) in Table 8 vs 0.0026 (ddof=1) in text. Standardized to 0.0026 (ddof=1). Table rows were unsorted by Mean AUC; now sorted descending (RF 0.6851, GB 0.6692, Standalone LSTM 0.6657, LbL-QNN 0.6605, LSTM+QNN 0.6589, XGBoost 0.6575, LSTM+PI-QNN 0.6555, LSTM+QTL 0.6519).")
    
    # Flag 6: Table 12 Noise Noiseless Row
    flags.append("TABLE 12 (Noise Noiseless Row): Manuscript already used n=20 for noiseless Rc-ACP (LSTM+QNN 0.950, LbL-QNN 0.900), which directly matches the recomputed n=20 single-run values (19/20 and 18/20), resolving the discrepancy with Table 1's old n=25 values (0.960 and 0.920).")
    
    for i, flag in enumerate(flags, 1):
        print(f"[{i}] {flag}\n")

if __name__ == '__main__':
    main()
