"""
code/local_experiments/exp6_tost_rope_equivalence.py
Local Experiment 6: Two One-Sided Tests (TOST) and ROPE Equivalence Testing
Evaluates 15-seed replication data under prespecified equivalence margins:
  - Primary margin: Delta = +/- 0.0100 AUC (1.0 percentage points)
  - Sensitivity margins: Delta = +/- 0.0150 and +/- 0.0200 AUC
"""

import json
import os
import numpy as np
from scipy import stats

def compute_tost_rope(x, y, delta=0.010, B_boot=10000, seed=42):
    d = np.array(x) - np.array(y)
    n = len(d)
    m = float(np.mean(d))
    s = float(np.std(d, ddof=1))
    se = s / np.sqrt(n)
    
    # 1. Student's t TOST
    # H01: mu <= -delta  vs  H11: mu > -delta
    t1 = (m - (-delta)) / se
    p1 = float(1.0 - stats.t.cdf(t1, df=n-1))
    
    # H02: mu >= +delta  vs  H12: mu < +delta
    t2 = (m - delta) / se
    p2 = float(stats.t.cdf(t2, df=n-1))
    
    p_tost = float(max(p1, p2))
    ci90 = [float(m - stats.t.ppf(0.95, df=n-1) * se), float(m + stats.t.ppf(0.95, df=n-1) * se)]
    ci95 = [float(m - stats.t.ppf(0.975, df=n-1) * se), float(m + stats.t.ppf(0.975, df=n-1) * se)]
    
    # 2. Wilcoxon Signed-Rank TOST
    # Subtract -delta: d - (-delta) = d + delta. Test if median > 0 (greater)
    try:
        w1_res = stats.wilcoxon(d + delta, alternative='greater')
        p_w1 = float(w1_res.pvalue)
    except Exception:
        p_w1 = 1.0
    try:
        w2_res = stats.wilcoxon(d - delta, alternative='less')
        p_w2 = float(w2_res.pvalue)
    except Exception:
        p_w2 = 1.0
    p_wilcoxon_tost = float(max(p_w1, p_w2))
    
    # 3. Frequentist t-based ROPE coverage (descriptive; not Bayesian posterior mass)
    prob_rope_analytic = float(
        stats.t.cdf((delta - m) / se, df=n - 1)
        - stats.t.cdf((-delta - m) / se, df=n - 1)
    )
    
    # 4. Bootstrap ROPE
    np.random.seed(seed)
    boot_means = [np.mean(np.random.choice(d, size=n, replace=True)) for _ in range(B_boot)]
    prob_rope_boot = float(np.mean((np.array(boot_means) >= -delta) & (np.array(boot_means) <= delta)))
    
    return {
        'delta': delta,
        'mean_diff': m,
        'std_diff': s,
        'se_diff': se,
        'ci_90': ci90,
        'ci_95': ci95,
        'p_tost_t': p_tost,
        'p_tost_wilcoxon': p_wilcoxon_tost,
        'rope_t_coverage': prob_rope_analytic,
        'rope_bootstrap_coverage': prob_rope_boot,
        'is_equivalent_alpha05': bool(p_tost < 0.05)
    }

def main():
    os.makedirs('results', exist_ok=True)
    
    with open('results/json/09_multisemilla.json', 'r', encoding='utf-8') as f:
        d9 = json.load(f)
    with open('results/json/10_multisemilla_clasico.json', 'r', encoding='utf-8') as f:
        d10 = json.load(f)

    models_data = {
        'LSTM+QNN': np.array(d9['LSTM+QNN']['auc']['valores']),
        'LSTM+PI-QNN': np.array(d9['LSTM+PI-QNN']['auc']['valores']),
        'LSTM+QTL': np.array(d9['LSTM+QTL']['auc']['valores']),
        'PI-QNN-LbL': np.array(d9['PI-QNN-LbL']['auc']['valores']),
        'Random Forest': np.array(d10['Random Forest']['auc']['valores']),
        'Gradient Boosting': np.array(d10['Gradient Boosting']['auc']['valores']),
        'XGBoost': np.array(d10['XGBoost']['auc']['valores']),
        'Standalone LSTM': np.array(d10['LSTM pura']['auc']['valores'])
    }

    pairs = [
        ('LSTM+QNN', 'PI-QNN-LbL'),
        ('LSTM+QNN', 'LSTM+PI-QNN'),
        ('LSTM+QNN', 'LSTM+QTL'),
        ('PI-QNN-LbL', 'LSTM+PI-QNN'),
        ('PI-QNN-LbL', 'LSTM+QTL'),
        ('LSTM+PI-QNN', 'LSTM+QTL'),
        ('LSTM+QNN', 'Standalone LSTM'),
        ('PI-QNN-LbL', 'Standalone LSTM'),
        ('Random Forest', 'PI-QNN-LbL')
    ]

    deltas = [0.010, 0.015, 0.020]
    all_results = {}

    print("=== Local Experiment 6: TOST and ROPE Equivalence Analysis (15 Seeds) ===")
    
    for delta in deltas:
        delta_key = f"delta_{delta:.3f}"
        all_results[delta_key] = {}
        print(f"\n--- Equivalence Margin Delta = +/- {delta:.4f} AUC (alpha = 0.05, 90% CI) ---")
        print('%-28s %-8s %-7s %-18s %-9s %-9s %-9s %-10s' % (
            'Comparison (A vs B)', 'MeanDiff', 'SE', '90% CI', 'p_TOST(t)', 'p_Wilcox', 'ROPE(%)', 'Equivalent?'
        ))
        print('-'*98)
        
        for m1, m2 in pairs:
            res = compute_tost_rope(models_data[m1], models_data[m2], delta=delta)
            label = f"{m1} vs {m2}"
            all_results[delta_key][label] = res
            
            eq_str = 'YES' if res['is_equivalent_alpha05'] else 'NO'
            ci_str = f"[{res['ci_90'][0]:+.4f}, {res['ci_90'][1]:+.4f}]"
            print('%-28s %-+8.4f %-7.4f %-18s %-9.4f %-9.4f %-9.1f %-10s' % (
                label, res['mean_diff'], res['se_diff'], ci_str,
                res['p_tost_t'], res['p_tost_wilcoxon'], res['rope_analytic']*100, eq_str
            ))

    out_path = 'results/local_exp6_tost_rope.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nExp 6 completed successfully. Output saved to {out_path}.")

if __name__ == '__main__':
    main()
