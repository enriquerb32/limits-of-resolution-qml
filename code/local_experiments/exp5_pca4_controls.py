"""
code/local_experiments/exp5_pca4_controls.py
Local Experiment 5: Classical Classifiers on PCA-4 Bottleneck vs QKSVM
Fits PCA(n_components=4) exclusively on training partition (N=2,273) and evaluates:
  1. Random Forest on PCA-4 (15 seeds: mean +/- std)
  2. Gradient Boosting on PCA-4 (15 seeds: mean +/- std)
  3. SVM-RBF on PCA-4 (probabilistically calibrated)
  4. Benchmark comparison against QKSVM (4-qubit quantum kernel)
"""

import json
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
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

def evaluate_predictions(y_true, y_prob, y_p90, dana_pos, th=0.5):
    pred = (y_prob >= th).astype(int)
    auc = float(roc_auc_score(y_true, y_prob))
    prauc = float(average_precision_score(y_true, y_prob))
    bs = float(brier_score_loss(y_true, y_prob))
    acc = float(accuracy_score(y_true, pred))
    f1 = float(f1_score(y_true, pred))
    fpr = float((pred[y_true == 0] == 1).sum() / (y_true == 0).sum())
    
    k_p90 = int((pred[y_p90 == 1] == 1).sum())
    rc_p90 = float(k_p90 / y_p90.sum())
    ci_p90 = wilson_ci(k_p90, int(y_p90.sum()))
    
    k_acp = int((pred[dana_pos] == 1).sum())
    rc_acp = float(k_acp / dana_pos.sum())
    ci_acp = wilson_ci(k_acp, int(dana_pos.sum()))
    
    return {
        'auc': auc,
        'pr_auc': prauc,
        'brier_score': bs,
        'acc': acc,
        'f1': f1,
        'fpr': fpr,
        'rc_p90': rc_p90,
        'rc_p90_ci_95': list(ci_p90),
        'rc_acp_n20': rc_acp,
        'rc_acp_ci_95': list(ci_acp)
    }

def main():
    os.makedirs('results', exist_ok=True)
    
    with open('results/json/01_data.json', 'r') as f:
        ds = json.load(f)[1]
    with open('results/json/04_qksvm.json', 'r') as f:
        qksvm = json.load(f)

    X_tr = np.array(ds['X_tr'])
    y_tr = np.array(ds['y_tr'])
    X_te = np.array(ds['X_te'])
    y_te = np.array(ds['y_te'])
    y_te_ext = np.array(ds['y_te_ext'])
    dana_te = np.array(ds['dana_te'], dtype=bool)
    dana_pos = dana_te & (y_te == 1)

    # 1. Fit PCA-4 strictly on training data
    pca = PCA(n_components=4, random_state=42)
    X_tr_pca = pca.fit_transform(X_tr)
    X_te_pca = pca.transform(X_te)
    exp_var = [float(v) for v in pca.explained_variance_ratio_]
    tot_var = float(sum(pca.explained_variance_ratio_))

    # 2. RF on PCA-4 across 15 seeds
    rf_seeds = []
    for seed in range(15):
        rf = RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight='balanced',
            random_state=seed, n_jobs=-1
        )
        rf.fit(X_tr_pca, y_tr)
        p_rf = rf.predict_proba(X_te_pca)[:, 1]
        m_rf = evaluate_predictions(y_te, p_rf, y_te_ext, dana_pos)
        m_rf['seed'] = seed
        rf_seeds.append(m_rf)

    rf_aucs = [m['auc'] for m in rf_seeds]
    rf_summary = {
        'auc_mean': float(np.mean(rf_aucs)),
        'auc_std': float(np.std(rf_aucs, ddof=1)),
        'brier_mean': float(np.mean([m['brier_score'] for m in rf_seeds])),
        'f1_mean': float(np.mean([m['f1'] for m in rf_seeds])),
        'acc_mean': float(np.mean([m['acc'] for m in rf_seeds])),
        'fpr_mean': float(np.mean([m['fpr'] for m in rf_seeds])),
        'rc_p90_mean': float(np.mean([m['rc_p90'] for m in rf_seeds])),
        'rc_acp_mean': float(np.mean([m['rc_acp_n20'] for m in rf_seeds])),
        'seeds': rf_seeds
    }

    # 3. GB on PCA-4 across 15 seeds
    gb_seeds = []
    for seed in range(15):
        gb = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=seed
        )
        gb.fit(X_tr_pca, y_tr)
        p_gb = gb.predict_proba(X_te_pca)[:, 1]
        m_gb = evaluate_predictions(y_te, p_gb, y_te_ext, dana_pos)
        m_gb['seed'] = seed
        gb_seeds.append(m_gb)

    gb_aucs = [m['auc'] for m in gb_seeds]
    gb_summary = {
        'auc_mean': float(np.mean(gb_aucs)),
        'auc_std': float(np.std(gb_aucs, ddof=1)),
        'brier_mean': float(np.mean([m['brier_score'] for m in gb_seeds])),
        'f1_mean': float(np.mean([m['f1'] for m in gb_seeds])),
        'acc_mean': float(np.mean([m['acc'] for m in gb_seeds])),
        'fpr_mean': float(np.mean([m['fpr'] for m in gb_seeds])),
        'rc_p90_mean': float(np.mean([m['rc_p90'] for m in gb_seeds])),
        'rc_acp_mean': float(np.mean([m['rc_acp_n20'] for m in gb_seeds])),
        'seeds': gb_seeds
    }

    # 4. SVM-RBF on PCA-4 (single run, deterministic given seed)
    svm = SVC(C=1.0, kernel='rbf', probability=True, class_weight='balanced', random_state=42)
    svm.fit(X_tr_pca, y_tr)
    p_svm = svm.predict_proba(X_te_pca)[:, 1]
    svm_summary = evaluate_predictions(y_te, p_svm, y_te_ext, dana_pos)

    # 5. QKSVM reference
    p_qksvm = np.array(qksvm['QKSVM']['_prob'])
    qksvm_summary = evaluate_predictions(y_te, p_qksvm, y_te_ext, dana_pos)

    results = {
        'pca_explained_variance_ratio': exp_var,
        'pca_total_variance_ratio': tot_var,
        'models': {
            'SVM_RBF_PCA4': svm_summary,
            'RF_PCA4': rf_summary,
            'GB_PCA4': gb_summary,
            'QKSVM_benchmark': qksvm_summary
        }
    }

    print("=== Local Experiment 5: PCA-4 Bottleneck Controls vs QKSVM ===")
    print(f"PCA-4 Explained Variance: {tot_var*100:.2f}% (Components: {exp_var})")
    print('%-22s %-18s %-7s %-7s %-7s %-7s %-7s' % (
        'Model', 'AUC (mean +/- std)', 'Brier', 'Acc(%)', 'F1', 'Rc-P90', 'Rc-ACP'
    ))
    print('-'*80)
    print('%-22s %-18.4f %-7.4f %-7.2f %-7.4f %-7.4f %-7.4f' % (
        'SVM-RBF (PCA-4)', svm_summary['auc'], svm_summary['brier_score'],
        svm_summary['acc']*100, svm_summary['f1'], svm_summary['rc_p90'], svm_summary['rc_acp_n20']
    ))
    rf_str = f"{rf_summary['auc_mean']:.4f} +/- {rf_summary['auc_std']:.4f}"
    print('%-22s %-18s %-7.4f %-7.2f %-7.4f %-7.4f %-7.4f' % (
        'Random Forest (PCA-4)', rf_str, rf_summary['brier_mean'],
        rf_summary['acc_mean']*100, rf_summary['f1_mean'], rf_summary['rc_p90_mean'], rf_summary['rc_acp_mean']
    ))
    gb_str = f"{gb_summary['auc_mean']:.4f} +/- {gb_summary['auc_std']:.4f}"
    print('%-22s %-18s %-7.4f %-7.2f %-7.4f %-7.4f %-7.4f' % (
        'GradBoosting (PCA-4)', gb_str, gb_summary['brier_mean'],
        gb_summary['acc_mean']*100, gb_summary['f1_mean'], gb_summary['rc_p90_mean'], gb_summary['rc_acp_mean']
    ))
    print('%-22s %-18.4f %-7.4f %-7.2f %-7.4f %-7.4f %-7.4f' % (
        'QKSVM (4-qubit)', qksvm_summary['auc'], qksvm_summary['brier_score'],
        qksvm_summary['acc']*100, qksvm_summary['f1'], qksvm_summary['rc_p90'], qksvm_summary['rc_acp_n20']
    ))

    out_path = 'results/local_exp5_pca4_controls.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Exp 5 completed successfully. Output saved to {out_path}.")

if __name__ == '__main__':
    main()
