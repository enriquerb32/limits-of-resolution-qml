import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

def wilson_ci(k, n, z=1.959964):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, float(center - half)), min(1.0, float(center + half)))

# Load canonical dataset
with open('results/json/01_data.json', 'r') as f:
    ds = json.load(f)[1]

y_te = np.array(ds['y_te'])
y_te_ext = np.array(ds['y_te_ext'])
dana_te = np.array(ds['dana_te'], dtype=bool)
dana_pos = dana_te & (y_te == 1)

print(f"Test size: {len(y_te)}, Positives: {y_te.sum()}, P90: {y_te_ext.sum()}, ACP: {dana_te.sum()}, ACP(y=1): {dana_pos.sum()}")

with open('results/json/02_baselines.json', 'r') as f:
    b = json.load(f)
with open('results/json/03_qml.json', 'r') as f:
    q = json.load(f)
with open('results/json/04_qksvm.json', 'r') as f:
    qksvm = json.load(f)
with open('results/json/08_validacion.json', 'r', encoding='utf-8') as f:
    val = json.load(f)
umbral_lstm = val['bootstrap_pareado_operativo']['_umbral_lstm']

probs_dict = {
    'Logistic Regression': np.array(b['_probs']['Reg. Log\u00edstica' if 'Reg. Log\u00edstica' in b['_probs'] else [k for k in b['_probs'] if 'Log' in k][0]]),
    'Random Forest': np.array(b['_probs']['Random Forest']),
    'Gradient Boosting': np.array(b['_probs']['Gradient Boosting']),
    'XGBoost': np.array(b['_probs']['XGBoost']),
    'Standalone LSTM': np.array(b['_probs']['LSTM pura']),
    'Matched LSTM': np.array(b['_probs']['LSTM pura']),
    'LSTM+QNN': np.array(q['_probs']['LSTM+QNN']),
    'LSTM+PI-QNN': np.array(q['_probs']['LSTM+PI-QNN']),
    'LSTM+QTL': np.array(q['_probs']['LSTM+QTL']),
    'PI-QNN-LbL': np.array(q['_probs']['PI-QNN-LbL']),
    'QKSVM': np.array(qksvm['QKSVM']['_prob'])
}

rows = []
for m, probs in probs_dict.items():
    th = umbral_lstm if m == 'Matched LSTM' else 0.5
    pred = (probs >= th).astype(int)
    acc = accuracy_score(y_te, pred)
    f1 = f1_score(y_te, pred)
    auc = roc_auc_score(y_te, probs)
    prauc = average_precision_score(y_te, probs)
    k_p90 = int((pred[y_te_ext == 1] == 1).sum())
    rc_p90 = k_p90 / float((y_te_ext == 1).sum())
    k_dana20 = int((pred[dana_pos] == 1).sum())
    rc_dana20 = k_dana20 / 20.0
    k_dana25 = int((pred[dana_te] == 1).sum())
    rc_dana25 = k_dana25 / 25.0
    fpr = float((pred[y_te == 0] == 1).sum()) / float((y_te == 0).sum())
    
    ci_p90 = wilson_ci(k_p90, 49)
    ci_dana20 = wilson_ci(k_dana20, 20)
    ci_dana25 = wilson_ci(k_dana25, 25)
    
    rows.append({
        'model': m,
        'threshold': th,
        'acc': acc,
        'f1': f1,
        'auc': auc,
        'pr_auc': prauc,
        'rc_p90': rc_p90,
        'ci_p90_lo': ci_p90[0],
        'ci_p90_hi': ci_p90[1],
        'rc_dana_n20': rc_dana20,
        'ci_dana20_lo': ci_dana20[0],
        'ci_dana20_hi': ci_dana20[1],
        'rc_dana_n25': rc_dana25,
        'ci_dana25_lo': ci_dana25[0],
        'ci_dana25_hi': ci_dana25[1],
        'fpr': fpr
    })

# Persistence
pers_acc = b['Persistencia']['acc']
pers_f1 = b['Persistencia']['f1']
print(f"Persistencia from 02_baselines: acc={pers_acc:.4f}, f1={pers_f1:.4f}")

df_single = pd.DataFrame(rows)
print("\n--- SINGLE RUN RECOMPUTED TABLE ---")
print(df_single[['model', 'acc', 'f1', 'auc', 'pr_auc', 'rc_p90', 'rc_dana_n20', 'rc_dana_n25', 'fpr']].to_string())

print("\n--- WILSON CONFIDENCE INTERVALS ---")
for r in rows:
    print(f"{r['model']:<20} | Rc-ACP (n=20): {r['rc_dana_n20']:.3f} [{r['ci_dana20_lo']:.3f}, {r['ci_dana20_hi']:.3f}] | Rc-ACP (n=25): {r['rc_dana_n25']:.3f} [{r['ci_dana25_lo']:.3f}, {r['ci_dana25_hi']:.3f}] | Rc-P90 (n=49): {r['rc_p90']:.3f} [{r['ci_p90_lo']:.3f}, {r['ci_p90_hi']:.3f}]")

