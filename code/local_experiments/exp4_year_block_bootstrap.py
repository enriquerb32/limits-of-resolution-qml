"""
code/local_experiments/exp4_year_block_bootstrap.py
Local Experiment 4: Year-Block Bootstrap Analysis for Single-Run Models
Resamples 19 calendar years (2008--2026) with replacement (B=2,000) to account for
spatial-temporal autocorrelation across stations within calendar years.
"""

import json
import os
import numpy as np
from sklearn.metrics import roc_auc_score

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

    y_te = np.array(ds['y_te'])
    m_te = np.array(ds['m_te'])

    # Infer calendar years for test samples
    year = 2008
    years = []
    for i in range(len(m_te)):
        if i > 0 and m_te[i] < m_te[i-1]:
            year += 1
        years.append(year)
    years = np.array(years)
    unique_years = np.unique(years)

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
        'QKSVM': np.array(qksvm['QKSVM']['_prob'])
    }

    # Point estimates
    point_aucs = {m: float(roc_auc_score(y_te, probs)) for m, probs in models_probs.items()}

    # Resampling
    np.random.seed(42)
    B = 2000
    boot_aucs = {m: [] for m in models_probs}
    
    # Store indices per year for fast lookup
    year_indices = {yr: np.where(years == yr)[0] for yr in unique_years}

    for _ in range(B):
        sampled_years = np.random.choice(unique_years, size=len(unique_years), replace=True)
        idx = np.concatenate([year_indices[yr] for yr in sampled_years])
        y_b = y_te[idx]
        if len(np.unique(y_b)) < 2:
            continue
        for m, probs in models_probs.items():
            boot_aucs[m].append(float(roc_auc_score(y_b, probs[idx])))

    # Compute individual model CIs
    model_results = {}
    for m in models_probs:
        vals = np.array(boot_aucs[m])
        ci_lo, ci_hi = np.percentile(vals, [2.5, 97.5])
        model_results[m] = {
            'point_auc': point_aucs[m],
            'boot_mean': float(np.mean(vals)),
            'boot_se': float(np.std(vals, ddof=1)),
            'ci_95': [float(ci_lo), float(ci_hi)]
        }

    # Key pairwise differences
    pairs_to_test = [
        ('Random Forest', 'PI-QNN-LbL'),
        ('Random Forest', 'Standalone LSTM'),
        ('Random Forest', 'LSTM+QNN'),
        ('PI-QNN-LbL', 'Standalone LSTM'),
        ('PI-QNN-LbL', 'LSTM+QNN'),
        ('PI-QNN-LbL', 'LSTM+PI-QNN'),
        ('PI-QNN-LbL', 'LSTM+QTL'),
        ('LSTM+QNN', 'Standalone LSTM'),
        ('Gradient Boosting', 'PI-QNN-LbL'),
        ('XGBoost', 'LSTM+QNN'),
        ('PI-QNN-LbL', 'QKSVM')
    ]

    pairwise_results = {}
    print(f"Year-Block Bootstrap (B={B}, 19 years: 2008--2026):")
    print('%-32s %-10s %-8s %-18s %-8s' % ('Comparison (A - B)', 'Point Diff', 'SE', '95% Percentile CI', 'p-value'))
    print('-'*80)

    for m1, m2 in pairs_to_test:
        diff_pt = point_aucs[m1] - point_aucs[m2]
        diffs = np.array(boot_aucs[m1]) - np.array(boot_aucs[m2])
        ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
        p_val = 2.0 * min(float(np.mean(diffs <= 0)), float(np.mean(diffs >= 0)))
        p_val = min(1.0, max(1.0 / B, p_val))
        
        label = f"{m1} vs {m2}"
        pairwise_results[label] = {
            'model_A': m1,
            'model_B': m2,
            'point_diff': float(diff_pt),
            'boot_mean_diff': float(np.mean(diffs)),
            'boot_se': float(np.std(diffs, ddof=1)),
            'ci_95': [float(ci_lo), float(ci_hi)],
            'p_value': float(p_val)
        }
        print('%-32s %-+10.4f %-8.4f [%-+7.4f, %-+7.4f]  %-8.4f' % (
            label, diff_pt, np.std(diffs, ddof=1), ci_lo, ci_hi, p_val
        ))

    out_path = 'results/local_exp4_year_block_bootstrap.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'num_bootstrap_samples': B,
            'num_clusters_years': len(unique_years),
            'years': list(map(int, unique_years)),
            'models': model_results,
            'pairwise_differences': pairwise_results
        }, f, indent=2)

    print(f"Exp 4 completed successfully. Output saved to {out_path}.")

if __name__ == '__main__':
    main()
