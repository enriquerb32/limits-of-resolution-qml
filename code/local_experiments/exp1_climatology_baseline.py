"""
code/local_experiments/exp1_climatology_baseline.py
Local Experiment 1: Month x Station Climatology Baseline
Strictly fit on training partition (N=2,273; <= 1996) and evaluated on test partition (N=528).
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

    m_tr = np.array(ds['m_tr'])
    y_tr = np.array(ds['y_tr'])
    X_tr = np.array(ds['X_tr'])

    m_te = np.array(ds['m_te'])
    y_te = np.array(ds['y_te'])
    X_te = np.array(ds['X_te'])
    y_te_ext = np.array(ds['y_te_ext'])
    dana_te = np.array(ds['dana_te'], dtype=bool)
    dana_pos = dana_te & (y_te == 1)

    st_idx = [31, 32, 33, 34]
    stations = ['Alicante', 'Barcelona', 'San_Javier', 'Valencia']

    # 1. Unconditional base rate
    p_uncond = float(y_tr.mean())
    bs_uncond_ref = float(brier_score_loss(y_te, np.full_like(y_te, p_uncond)))

    # 2. Month x Station climatology lookup table
    clim_table = {}
    for s_name, s_col in zip(stations, st_idx):
        clim_table[s_name] = {}
        for m in range(1, 13):
            mask = (m_tr == m) & (X_tr[:, s_col] == 1.0)
            n_obs = int(mask.sum())
            k_pos = int(y_tr[mask].sum())
            p_val = float(k_pos / n_obs) if n_obs > 0 else p_uncond
            clim_table[s_name][m] = {
                'n_train': n_obs,
                'pos_train': k_pos,
                'freq_train': p_val
            }

    # Map to test partition
    p_clim_te = np.zeros(len(y_te))
    for i in range(len(y_te)):
        s_col = [c for c in st_idx if X_te[i, c] == 1.0][0]
        s_name = stations[st_idx.index(s_col)]
        p_clim_te[i] = clim_table[s_name][int(m_te[i])]['freq_train']

    # 3. Month-only climatology (across all stations)
    month_table = {}
    for m in range(1, 13):
        mask = (m_tr == m)
        n_obs = int(mask.sum())
        k_pos = int(y_tr[mask].sum())
        month_table[m] = float(k_pos / n_obs) if n_obs > 0 else p_uncond
    p_month_te = np.array([month_table[int(m)] for m in m_te])

    # Evaluate Month x Station Climatology
    auc_clim = float(roc_auc_score(y_te, p_clim_te))
    prauc_clim = float(average_precision_score(y_te, p_clim_te))
    bs_clim = float(brier_score_loss(y_te, p_clim_te))
    bss_clim = float(1.0 - (bs_clim / bs_uncond_ref))

    pred_clim_05 = (p_clim_te >= 0.5).astype(int)
    acc_clim_05 = float(accuracy_score(y_te, pred_clim_05))
    f1_clim_05 = float(f1_score(y_te, pred_clim_05))
    fpr_clim_05 = float((pred_clim_05[y_te == 0] == 1).sum() / (y_te == 0).sum())
    
    k_p90 = int((pred_clim_05[y_te_ext == 1] == 1).sum())
    rc_p90_clim = float(k_p90 / y_te_ext.sum())
    p90_ci = wilson_ci(k_p90, int(y_te_ext.sum()))

    k_acp = int((pred_clim_05[dana_pos] == 1).sum())
    rc_acp_clim = float(k_acp / dana_pos.sum())
    acp_ci = wilson_ci(k_acp, int(dana_pos.sum()))

    # Evaluate Month-only Climatology
    auc_month = float(roc_auc_score(y_te, p_month_te))
    prauc_month = float(average_precision_score(y_te, p_month_te))
    bs_month = float(brier_score_loss(y_te, p_month_te))
    bss_month = float(1.0 - (bs_month / bs_uncond_ref))

    results = {
        'unconditional_base_rate_train': p_uncond,
        'brier_score_unconditional_ref': bs_uncond_ref,
        'month_x_station_climatology': {
            'auc': auc_clim,
            'pr_auc': prauc_clim,
            'brier_score': bs_clim,
            'brier_skill_score': bss_clim,
            'acc_th05': acc_clim_05,
            'f1_th05': f1_clim_05,
            'fpr_th05': fpr_clim_05,
            'rc_p90_th05': rc_p90_clim,
            'rc_p90_ci_95': list(p90_ci),
            'rc_acp_n20_th05': rc_acp_clim,
            'rc_acp_ci_95': list(acp_ci)
        },
        'month_only_climatology': {
            'auc': auc_month,
            'pr_auc': prauc_month,
            'brier_score': bs_month,
            'brier_skill_score': bss_month
        },
        'lookup_table': clim_table
    }

    out_path = 'results/local_exp1_climatology.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"Exp 1 Climatology completed successfully. Output saved to {out_path}.")
    print(f"Month x Station Climatology: AUC = {auc_clim:.4f}, Brier = {bs_clim:.4f}, BSS = {bss_clim:+.4f}")
    print(f"Month-only Climatology:      AUC = {auc_month:.4f}, Brier = {bs_month:.4f}, BSS = {bss_month:+.4f}")

if __name__ == '__main__':
    main()
