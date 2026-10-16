import json
import numpy as np
import pandas as pd
from scipy import stats

with open('results/json/09_multisemilla.json', 'r', encoding='utf-8') as f:
    d9 = json.load(f)

with open('results/json/10_multisemilla_clasico.json', 'r', encoding='utf-8') as f:
    d10 = json.load(f)

with open('results/json/ablacion_ia_racha_resultados.json', 'r', encoding='utf-8') as f:
    dracha = json.load(f)

# Extract 15 seeds of AUC
models_auc = {}
for m in ['Random Forest', 'Gradient Boosting', 'XGBoost', 'LSTM pura']:
    models_auc[m] = np.array(d10[m]['auc']['valores'])

for m in ['LSTM+QNN', 'LSTM+PI-QNN', 'LSTM+QTL', 'PI-QNN-LbL']:
    models_auc[m] = np.array(d9[m]['auc']['valores'])

print("--- 15-SEED AUC SUMMARY (ddof=1) ---")
for m, vals in models_auc.items():
    mean = np.mean(vals)
    std_ddof1 = np.std(vals, ddof=1)
    std_ddof0 = np.std(vals, ddof=0)
    median = np.median(vals)
    iqr = stats.iqr(vals)
    vmin = np.min(vals)
    vmax = np.max(vals)
    print(f"{m:<18}: Mean={mean:.4f} +/- {std_ddof1:.4f} (ddof0={std_ddof0:.4f}), Med={median:.4f} (IQR={iqr:.4f}), Min={vmin:.4f}, Max={vmax:.4f}")

# Check Table 6: Global AUC distribution across 15 seeds (quantum models)
# Table 7: Operational metric distributions across 15 seeds
# Table 8: Global AUC distribution across 15 seeds, all eight architectures

print("\n--- FRIEDMAN TEST (4 Quantum Models) ---")
q_aucs = [models_auc['LSTM+QNN'], models_auc['LSTM+PI-QNN'], models_auc['LSTM+QTL'], models_auc['PI-QNN-LbL']]
friedman_q = stats.friedmanchisquare(*q_aucs)
print(f"Friedman statistic={friedman_q.statistic:.4f}, p-value={friedman_q.pvalue:.4f}")

print("\n--- LEVENE TEST (LSTM+QNN vs PI-QNN-LbL variance) ---")
levene_q = stats.levene(models_auc['LSTM+QNN'], models_auc['PI-QNN-LbL'], center='mean')
print(f"Levene (center=mean): statistic={levene_q.statistic:.4f}, p-value={levene_q.pvalue:.4f}")
levene_med = stats.levene(models_auc['LSTM+QNN'], models_auc['PI-QNN-LbL'], center='median')
print(f"Levene (center=median): statistic={levene_med.statistic:.4f}, p-value={levene_med.pvalue:.4f}")

print("\n--- WILCOXON SIGNED-RANK / MANN-WHITNEY TESTS ---")
# Between quantum models (paired by seed index)
for m1, m2 in [('LSTM+QNN', 'LSTM+PI-QNN'), ('LSTM+QNN', 'PI-QNN-LbL'), ('LSTM+PI-QNN', 'PI-QNN-LbL'), ('LSTM+QNN', 'LSTM+QTL')]:
    w_res = stats.wilcoxon(models_auc[m1], models_auc[m2])
    print(f"Wilcoxon {m1} vs {m2}: W={w_res.statistic:.1f}, p={w_res.pvalue:.4f}")

# RF vs Quantum models: Probability of superiority & Cliff's delta
# Note: Sklearn seeds and PyTorch seeds are independent (not paired!)
print("\n--- PROBABILITY OF SUPERIORITY & CLIFF'S DELTA (RF vs others) ---")
rf_vals = models_auc['Random Forest']
for m, vals in models_auc.items():
    if m == 'Random Forest': continue
    # All pairwise comparisons
    greater = 0
    less = 0
    equal = 0
    for x in rf_vals:
        for y in vals:
            if x > y: greater += 1
            elif x < y: less += 1
            else: equal += 1
    total = len(rf_vals) * len(vals)
    prob_sup = (greater + 0.5 * equal) / total
    cliffs_d = (greater - less) / total
    # Mann-Whitney U test (unpaired)
    mwu = stats.mannwhitneyu(rf_vals, vals, alternative='two-sided')
    print(f"RF vs {m:<18}: P(RF > m) = {prob_sup:.4f}, Cliff's delta = {cliffs_d:.4f}, Mann-Whitney U = {mwu.statistic:.1f}, p = {mwu.pvalue:.4e}")
