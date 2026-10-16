"""
code/local_experiments/exp3_brier_and_reliability.py
Local Experiment 3: Brier Score, Brier Skill Score, and Calibration/Reliability Analysis
Evaluates the 10 single-run models on the test partition (N=528).
"""

import json
import os
import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression

def compute_calibration_metrics(y_true, y_prob, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    bin_counts = []
    bin_accs = []
    bin_confs = []
    ece = 0.0
    mce = 0.0
    
    n_total = len(y_true)
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i+1])
        else:
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i+1])
            
        n_b = int(np.sum(mask))
        bin_counts.append(n_b)
        if n_b > 0:
            acc_b = float(np.mean(y_true[mask]))
            conf_b = float(np.mean(y_prob[mask]))
            bin_accs.append(acc_b)
            bin_confs.append(conf_b)
            gap = abs(acc_b - conf_b)
            ece += (n_b / n_total) * gap
            if gap > mce:
                mce = gap
        else:
            bin_accs.append(None)
            bin_confs.append(None)
            
    # Logistic calibration fit: logit(y) ~ a + b * logit(p)
    # Clip probs to avoid infs
    eps = 1e-6
    p_clipped = np.clip(y_prob, eps, 1 - eps)
    logit_p = np.log(p_clipped / (1 - p_clipped)).reshape(-1, 1)
    try:
        lr_cal = LogisticRegression(C=1e5, solver='lbfgs')
        lr_cal.fit(logit_p, y_true)
        cal_intercept = float(lr_cal.intercept_[0])
        cal_slope = float(lr_cal.coef_[0][0])
    except Exception:
        cal_intercept = None
        cal_slope = None

    return {
        'ece_10': float(ece),
        'mce_10': float(mce),
        'cal_intercept': cal_intercept,
        'cal_slope': cal_slope,
        'bins': {
            'edges': list(bin_edges),
            'counts': bin_counts,
            'observed_frequency': bin_accs,
            'mean_predicted_prob': bin_confs
        }
    }

def main():
    os.makedirs('results', exist_ok=True)
    
    with open('results/json/01_data.json', 'r') as f:
        ds = json.load(f)[1]
    with open('results/json/02_baselines.json', 'r') as f:
        b = json.load(f)
    with open('results/json/03_qml.json', 'r') as f:
        q = json.load(f)
    with open('results/json/04_qksvm.json', 'r') as f:
        qksvm = json.load(f)

    y_tr = np.array(ds['y_tr'])
    p_uncond = float(y_tr.mean())

    m_tr = np.array(ds['m_tr'])
    X_tr = np.array(ds['X_tr'])
    m_te = np.array(ds['m_te'])
    y_te = np.array(ds['y_te'])
    X_te = np.array(ds['X_te'])

    st_idx = [31, 32, 33, 34]
    clim_map = {}
    for s_col in st_idx:
        for m in range(1, 13):
            mask = (m_tr == m) & (X_tr[:, s_col] == 1.0)
            clim_map[(m, s_col)] = float(y_tr[mask].mean())
    p_clim_te = np.array([clim_map[(m_te[i], [c for c in st_idx if X_te[i, c] == 1.0][0])] for i in range(len(y_te))])

    bs_uncond_ref = float(brier_score_loss(y_te, np.full_like(y_te, p_uncond)))
    bs_clim_ref = float(brier_score_loss(y_te, p_clim_te))

    logreg_key = [k for k in b['_probs'] if 'Log' in k][0]
    models_probs = {
        'Logistic Regression': np.array(b['_probs'][logreg_key]),
        'Random Forest': np.array(b['_probs']['Random Forest']),
        'Gradient Boosting': np.array(b['_probs']['Gradient Boosting']),
        'XGBoost': np.array(b['_probs']['XGBoost']),
        'Standalone LSTM': np.array(b['_probs']['LSTM pura']),
        'LSTM+QNN': np.array(q['_probs']['LSTM+QNN']),
        'LSTM+PI-QNN': np.array(q['_probs']['LSTM+PI-QNN']),
        'LSTM+QTL': np.array(q['_probs']['LSTM+QTL']),
        'PI-QNN-LbL': np.array(q['_probs']['PI-QNN-LbL']),
        'QKSVM': np.array(qksvm['QKSVM']['_prob']),
        'Month x Station Climatology': p_clim_te
    }

    results = {
        'reference_base_rate_train': p_uncond,
        'brier_score_unconditional_ref': bs_uncond_ref,
        'brier_score_climatology_ref': bs_clim_ref,
        'models': {}
    }

    print(f"Reference BS (Unconditional): {bs_uncond_ref:.4f} | Reference BS (Climatology): {bs_clim_ref:.4f}")
    print('%-28s %-7s %-7s %-10s %-10s %-8s %-8s' % ('Model', 'AUC', 'Brier', 'BSS_uncond', 'BSS_clim', 'ECE(10)', 'Slope'))
    print('-'*85)

    for m_name, probs in models_probs.items():
        auc = float(roc_auc_score(y_te, probs))
        bs = float(brier_score_loss(y_te, probs))
        bss_uncond = float(1.0 - (bs / bs_uncond_ref))
        bss_clim = float(1.0 - (bs / bs_clim_ref))
        
        cal_dict = compute_calibration_metrics(y_te, probs, n_bins=10)
        
        results['models'][m_name] = {
            'auc': auc,
            'brier_score': bs,
            'brier_skill_score_unconditional': bss_uncond,
            'brier_skill_score_climatology': bss_clim,
            'ece_10': cal_dict['ece_10'],
            'mce_10': cal_dict['mce_10'],
            'calibration_intercept': cal_dict['cal_intercept'],
            'calibration_slope': cal_dict['cal_slope'],
            'calibration_bins': cal_dict['bins']
        }
        
        slope_str = f"{cal_dict['cal_slope']:.2f}" if cal_dict['cal_slope'] is not None else 'N/A'
        print('%-28s %-7.4f %-7.4f %-+10.4f %-+10.4f %-8.4f %-8s' % (
            m_name, auc, bs, bss_uncond, bss_clim, cal_dict['ece_10'], slope_str
        ))

    out_path = 'results/local_exp3_brier_reliability.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Exp 3 completed successfully. Output saved to {out_path}.")

if __name__ == '__main__':
    main()
