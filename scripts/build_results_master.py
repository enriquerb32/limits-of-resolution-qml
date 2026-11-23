"""
build_results_master.py
Generates results/results_master.csv and associated summaries from saved exports.
All calculations follow the canonical dataset (01_data.pt) and settled audit definitions.
"""

import json
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

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
    
    # 1. Load canonical test ground truth
    with open('results/json/01_data.json', 'r') as f:
        ds = json.load(f)[1]
    
    y_te = np.array(ds['y_te'])
    y_te_ext = np.array(ds['y_te_ext'])
    dana_te = np.array(ds['dana_te'], dtype=bool)
    dana_pos = dana_te & (y_te == 1)
    
    n_test = len(y_te)
    n_pos = int(y_te.sum())
    n_p90 = int(y_te_ext.sum())
    n_dana = int(dana_te.sum())
    n_dana_pos = int(dana_pos.sum())
    
    print(f"Dataset facts: N={n_test}, Pos={n_pos}, P90={n_p90}, ACP={n_dana}, ACP(y=1)={n_dana_pos}")
    
    # 2. Load single-run probabilities
    with open('results/json/02_baselines.json', 'r') as f:
        b = json.load(f)
    with open('results/json/03_qml.json', 'r') as f:
        q = json.load(f)
    with open('results/json/04_qksvm.json', 'r') as f:
        qksvm = json.load(f)
    with open('results/json/08_validacion.json', 'r', encoding='utf-8') as f:
        val = json.load(f)
        
    umbral_lstm = float(val['bootstrap_pareado_operativo']['_umbral_lstm'])
    
    logreg_key = 'Reg. Logística' if 'Reg. Logística' in b['_probs'] else [k for k in b['_probs'] if 'Log' in k][0]
    
    single_models_probs = {
        'Logistic Regression': (np.array(b['_probs'][logreg_key]), 0.5),
        'Random Forest': (np.array(b['_probs']['Random Forest']), 0.5),
        'Gradient Boosting': (np.array(b['_probs']['Gradient Boosting']), 0.5),
        'XGBoost': (np.array(b['_probs']['XGBoost']), 0.5),
        'Standalone LSTM': (np.array(b['_probs']['LSTM pura']), 0.5),
        'Matched LSTM': (np.array(b['_probs']['LSTM pura']), umbral_lstm),
        'LSTM+QNN': (np.array(q['_probs']['LSTM+QNN']), 0.5),
        'LSTM+PI-QNN': (np.array(q['_probs']['LSTM+PI-QNN']), 0.5),
        'LSTM+QTL': (np.array(q['_probs']['LSTM+QTL']), 0.5),
        'PI-QNN-LbL': (np.array(q['_probs']['PI-QNN-LbL']), 0.5),
        'QKSVM': (np.array(qksvm['QKSVM']['_prob']), 0.5)
    }
    
    master_records = []
    
    # Recompute single-run metrics
    for model_name, (probs, th) in single_models_probs.items():
        pred = (probs >= th).astype(int)
        acc = float(accuracy_score(y_te, pred))
        f1 = float(f1_score(y_te, pred))
        auc = float(roc_auc_score(y_te, probs))
        prauc = float(average_precision_score(y_te, probs))
        
        k_p90 = int((pred[y_te_ext == 1] == 1).sum())
        rc_p90 = float(k_p90 / n_p90)
        ci_p90_lo, ci_p90_hi = wilson_ci(k_p90, n_p90)
        
        k_dana20 = int((pred[dana_pos] == 1).sum())
        rc_dana20 = float(k_dana20 / n_dana_pos)
        ci_dana20_lo, ci_dana20_hi = wilson_ci(k_dana20, n_dana_pos)
        
        k_dana25 = int((pred[dana_te] == 1).sum())
        rc_dana25 = float(k_dana25 / n_dana)
        ci_dana25_lo, ci_dana25_hi = wilson_ci(k_dana25, n_dana)
        
        fpr = float((pred[y_te == 0] == 1).sum() / (y_te == 0).sum())
        
        # Add to master records
        for metric, val_m in [
            ('acc', acc),
            ('f1', f1),
            ('auc', auc),
            ('pr_auc', prauc),
            ('rc_p90', rc_p90),
            ('rc_p90_ci_lo', ci_p90_lo),
            ('rc_p90_ci_hi', ci_p90_hi),
            ('rc_dana_n20', rc_dana20),
            ('rc_dana_n20_ci_lo', ci_dana20_lo),
            ('rc_dana_n20_ci_hi', ci_dana20_hi),
            ('rc_dana_n25', rc_dana25),
            ('rc_dana_n25_ci_lo', ci_dana25_lo),
            ('rc_dana_n25_ci_hi', ci_dana25_hi),
            ('fpr', fpr),
            ('threshold', th)
        ]:
            master_records.append({
                'run_type': 'single_run',
                'model': model_name,
                'seed': -1,
                'metric': metric,
                'value': val_m,
                'details': f"threshold={th:.4f}"
            })
            
    # Add Persistence (single run)
    master_records.append({
        'run_type': 'single_run',
        'model': 'Persistence',
        'seed': -1,
        'metric': 'acc',
        'value': float(b['Persistencia']['acc']),
        'details': 'no probabilities'
    })
    master_records.append({
        'run_type': 'single_run',
        'model': 'Persistence',
        'seed': -1,
        'metric': 'f1',
        'value': float(b['Persistencia']['f1']),
        'details': 'no probabilities'
    })
    
    # 3. Load multi-seed data
    with open('results/json/09_multisemilla.json', 'r', encoding='utf-8') as f:
        d9 = json.load(f)
    with open('results/json/10_multisemilla_clasico.json', 'r', encoding='utf-8') as f:
        d10 = json.load(f)
    with open('results/json/ablacion_ia_racha_resultados.json', 'r', encoding='utf-8') as f:
        dracha = json.load(f)
        
    # Quantum multi-seed models (15 seeds: acc, f1, auc, fpr, rc_dana, rc_p90)
    for qm in ['LSTM+QNN', 'LSTM+PI-QNN', 'LSTM+QTL', 'PI-QNN-LbL']:
        for metric in ['acc', 'f1', 'auc', 'fpr', 'rc_dana', 'rc_p90']:
            valores = d9[qm][metric]['valores']
            for s_idx, val_s in enumerate(valores):
                master_records.append({
                    'run_type': 'multi_seed',
                    'model': qm,
                    'seed': s_idx,
                    'metric': metric,
                    'value': float(val_s),
                    'details': '15 training seeds'
                })
                
    # Classical multi-seed models (15 seeds: auc, fpr, rc_dana, rc_p90)
    for cm, cm_name in [('Random Forest', 'Random Forest'), ('Gradient Boosting', 'Gradient Boosting'), 
                        ('XGBoost', 'XGBoost'), ('LSTM pura', 'Standalone LSTM')]:
        for metric in ['auc', 'fpr', 'rc_dana', 'rc_p90']:
            valores = d10[cm][metric]['valores']
            for s_idx, val_s in enumerate(valores):
                master_records.append({
                    'run_type': 'multi_seed',
                    'model': cm_name,
                    'seed': s_idx,
                    'metric': metric,
                    'value': float(val_s),
                    'details': '15 training seeds'
                })
                
    # ia_racha ablation (XGBoost 15 seeds: with vs without ia_racha)
    for cond in ['con', 'sin']:
        cond_label = 'XGBoost_with_ia_racha' if cond == 'con' else 'XGBoost_without_ia_racha'
        for metric in ['auc', 'fpr', 'rc_dana', 'rc_p90']:
            valores = dracha[cond][metric]
            for s_idx, val_s in enumerate(valores):
                master_records.append({
                    'run_type': 'ia_racha_ablation',
                    'model': cond_label,
                    'seed': s_idx,
                    'metric': metric,
                    'value': float(val_s),
                    'details': f'ia_racha_{cond}'
                })
                
    # Noise data (07_metricas_ruido)
    with open('results/json/07_metricas_ruido.json', 'r', encoding='utf-8') as f:
        dn = json.load(f)
    for nm in ['LSTM+QNN', 'PI-QNN-LbL']:
        for k, v in dn[nm].items():
            master_records.append({
                'run_type': 'noise_evaluation',
                'model': nm,
                'seed': -1,
                'metric': f'noisy_{k}',
                'value': float(v),
                'details': 'depolarizing noise p=0.01, 15 MC noise draws on clean weights'
            })
            
    # Save results_master.csv
    df_master = pd.DataFrame(master_records)
    df_master.to_csv('results/results_master.csv', index=False)
    print(f"Saved results/results_master.csv successfully ({len(df_master)} rows).")

if __name__ == '__main__':
    main()
