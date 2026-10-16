"""
code/local_experiments/exp2_persistence_baseline.py
Local Experiment 2: Persistence Baseline Analysis
Evaluates:
  1. Continuous precipitation lag-1 score (preceding month i+11)
  2. Binary lag-1 precipitation exceedance (> P60 training threshold)
  3. Author's climatological persistence (ds.precip_media_mes thresholded at 0.5 and continuous)
"""

import json
import os
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    accuracy_score, f1_score
)

def wilson_ci(k, n, z=1.959964):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return max(0.0, float(center - half)), min(1.0, float(center + half))

def main():
    os.makedirs('results', exist_ok=True)
    
    with open('results/json/01_data.json', 'r') as f:
        ds = json.load(f)[1]

    y_tr = np.array(ds['y_tr'])
    p_uncond = float(y_tr.mean())

    m_te = np.array(ds['m_te'])
    y_te = np.array(ds['y_te'])
    X_te = np.array(ds['X_te'])
    y_te_ext = np.array(ds['y_te_ext'])
    dana_te = np.array(ds['dana_te'], dtype=bool)
    dana_pos = dana_te & (y_te == 1)

    bs_uncond_ref = float(brier_score_loss(y_te, np.full_like(y_te, p_uncond)))

    # 1. Physical Precipitation Lag-1 Score (continuous)
    # Feature 11 is precip_lag_1 (scaled to [0, pi])
    lag1_continuous = X_te[:, 11]
    # Normalize to [0, 1] for Brier evaluation
    lag1_norm = lag1_continuous / np.pi
    auc_lag1 = float(roc_auc_score(y_te, lag1_continuous))
    prauc_lag1 = float(average_precision_score(y_te, lag1_continuous))
    bs_lag1 = float(brier_score_loss(y_te, lag1_norm))
    bss_lag1 = float(1.0 - (bs_lag1 / bs_uncond_ref))

    # 2. Binary Lag-1 Exceedance (> P60)
    # Umbral P60 = 28.1 mm scaled to [0, pi]
    scale_ = float(ds['scalers']['precip']['scale_'][0])
    min_ = float(ds['scalers']['precip']['min_'][0])
    p60_scaled = scale_ * float(ds['umbral']) + min_
    lag1_binary = (lag1_continuous > p60_scaled).astype(int)

    acc_bin = float(accuracy_score(y_te, lag1_binary))
    f1_bin = float(f1_score(y_te, lag1_binary))
    auc_bin = float(roc_auc_score(y_te, lag1_binary))
    bs_bin = float(brier_score_loss(y_te, lag1_binary))
    bss_bin = float(1.0 - (bs_bin / bs_uncond_ref))
    fpr_bin = float((lag1_binary[y_te == 0] == 1).sum() / (y_te == 0).sum())
    
    k_p90 = int((lag1_binary[y_te_ext == 1] == 1).sum())
    rc_p90_bin = float(k_p90 / y_te_ext.sum())
    p90_ci_bin = wilson_ci(k_p90, int(y_te_ext.sum()))

    k_acp = int((lag1_binary[dana_pos] == 1).sum())
    rc_acp_bin = float(k_acp / dana_pos.sum())
    acp_ci_bin = wilson_ci(k_acp, int(dana_pos.sum()))

    # 3. Author's Climatological Persistence
    p_month_te = np.array([float(ds['precip_media_mes'][str(m)]) for m in m_te])
    auc_month_cont = float(roc_auc_score(y_te, p_month_te))
    prauc_month_cont = float(average_precision_score(y_te, p_month_te))
    bs_month_cont = float(brier_score_loss(y_te, p_month_te))
    bss_month_cont = float(1.0 - (bs_month_cont / bs_uncond_ref))

    y_pers_author = (p_month_te >= 0.5).astype(int)
    acc_author = float(accuracy_score(y_te, y_pers_author))
    f1_author = float(f1_score(y_te, y_pers_author))
    auc_author = float(roc_auc_score(y_te, y_pers_author))
    bs_author = float(brier_score_loss(y_te, y_pers_author))
    fpr_author = float((y_pers_author[y_te == 0] == 1).sum() / (y_te == 0).sum())

    k_p90_auth = int((y_pers_author[y_te_ext == 1] == 1).sum())
    rc_p90_auth = float(k_p90_auth / y_te_ext.sum())
    p90_ci_auth = wilson_ci(k_p90_auth, int(y_te_ext.sum()))

    k_acp_auth = int((y_pers_author[dana_pos] == 1).sum())
    rc_acp_auth = float(k_acp_auth / dana_pos.sum())
    acp_ci_auth = wilson_ci(k_acp_auth, int(dana_pos.sum()))

    results = {
        'physical_lag1_continuous': {
            'auc': auc_lag1,
            'pr_auc': prauc_lag1,
            'brier_score_norm': bs_lag1,
            'brier_skill_score': bss_lag1
        },
        'physical_lag1_binary_exceedance': {
            'auc': auc_bin,
            'acc': acc_bin,
            'f1': f1_bin,
            'brier_score': bs_bin,
            'brier_skill_score': bss_bin,
            'fpr': fpr_bin,
            'rc_p90': rc_p90_bin,
            'rc_p90_ci_95': list(p90_ci_bin),
            'rc_acp_n20': rc_acp_bin,
            'rc_acp_ci_95': list(acp_ci_bin)
        },
        'author_climatological_persistence_continuous': {
            'auc': auc_month_cont,
            'pr_auc': prauc_month_cont,
            'brier_score': bs_month_cont,
            'brier_skill_score': bss_month_cont
        },
        'author_climatological_persistence_binary': {
            'acc': acc_author,
            'f1': f1_author,
            'auc_binary': auc_author,
            'brier_score': bs_author,
            'fpr': fpr_author,
            'rc_p90': rc_p90_auth,
            'rc_p90_ci_95': list(p90_ci_auth),
            'rc_acp_n20': rc_acp_auth,
            'rc_acp_ci_95': list(acp_ci_auth)
        }
    }

    out_path = 'results/local_exp2_persistence.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"Exp 2 Persistence completed successfully. Output saved to {out_path}.")
    print(f"Physical Lag-1 Continuous:       AUC = {auc_lag1:.4f}")
    print(f"Physical Lag-1 Binary (> P60):   AUC = {auc_bin:.4f}, Acc = {acc_bin*100:.2f}%, F1 = {f1_bin:.4f}")
    print(f"Author Climatological Binary:    AUC = {auc_author:.4f}, Acc = {acc_author*100:.2f}%, F1 = {f1_author:.4f}")
    print(f"Author Climatological Cont. P:   AUC = {auc_month_cont:.4f}, Brier = {bs_month_cont:.4f}")

if __name__ == '__main__':
    main()
