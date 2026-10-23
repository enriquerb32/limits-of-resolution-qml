"""
validacion.py  —  Rigorous Statistical Validation Module (MULTI-STATION)
========================================================================
Addresses reviewer critique with formal statistical testing:

  1. DeLong Test (1988): statistical significance test for AUC differences.
  2. 95% Confidence Intervals via non-parametric bootstrapping (n=2000).
  3. Nystrom sensitivity experiment M={32,64,128} with real PCA.
  4. Mathematical justification of Soft Thermodynamic Constraints.

Public function:
  ejecutar_validacion(ds, res_base, res_qml, res_qksvm, skip_nystrom) / run_validation(...) -> dict
"""

import time
import numpy as np
from scipy import stats
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                              recall_score, confusion_matrix, roc_curve)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import pennylane as qml

from features import DatasetResult


# ══════════════════════════════════════════════════════════════════
# 1. DELONG TEST — Statistical significance of AUC differences
# ══════════════════════════════════════════════════════════════════

def _delong_roc_variance(ground_truth, predictions):
    order = np.argsort(-predictions)
    label_1 = ground_truth[order]
    n_pos = int(label_1.sum())
    n_neg = len(label_1) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.25

    pos_idx = np.where(label_1 == 1)[0]
    neg_idx = np.where(label_1 == 0)[0]

    V10 = np.zeros(n_pos)
    for k, i in enumerate(pos_idx):
        V10[k] = (label_1[i + 1:] == 0).sum() / n_neg

    V01 = np.zeros(n_neg)
    for k, j in enumerate(neg_idx):
        V01[k] = (label_1[:j] == 1).sum() / n_pos

    var = (np.var(V10, ddof=1) / n_pos + np.var(V01, ddof=1) / n_neg)
    return max(var, 1e-10)


def test_delong(y_true, prob_a, prob_b, nombre_a="A", nombre_b="B") -> dict:
    auc_a = roc_auc_score(y_true, prob_a)
    auc_b = roc_auc_score(y_true, prob_b)

    var_a = _delong_roc_variance(y_true, prob_a)
    var_b = _delong_roc_variance(y_true, prob_b)

    corr = np.corrcoef(prob_a, prob_b)[0, 1]
    cov = corr * np.sqrt(var_a * var_b)

    denom = np.sqrt(var_a + var_b - 2 * cov)
    if denom < 1e-10:
        z, p = 0.0, 1.0
    else:
        z = (auc_a - auc_b) / denom
        p = 2 * (1 - stats.norm.cdf(abs(z)))

    significativo = p < 0.05
    return {
        f'auc_{nombre_a}': round(auc_a, 4),
        f'auc_{nombre_b}': round(auc_b, 4),
        'delta_auc': round(auc_a - auc_b, 4),
        'Z': round(z, 4),
        'p_valor': round(p, 4),
        'significativo_95': significativo,
        'conclusion': (
            f"Difference AUC({nombre_a})={auc_a:.3f} vs "
            f"AUC({nombre_b})={auc_b:.3f} is "
            f"{'STATISTICALLY SIGNIFICANT' if significativo else 'NOT significant'} "
            f"(Z={z:.3f}, p={p:.4f})"
        )
    }


# ══════════════════════════════════════════════════════════════════
# 2. BOOTSTRAPPING — 95% Confidence Intervals
# ══════════════════════════════════════════════════════════════════

def bootstrap_ci(y_true, y_prob, n_bootstrap=2000, ci=0.95, threshold=0.5, seed=42) -> dict:
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(float)

    aucs, f1s, accs = [], [], []

    for _ in range(n_bootstrap):
        idx = rng.randint(0, len(y_true), len(y_true))
        yt = y_true[idx]
        yp = y_prob[idx]
        yd = y_pred[idx]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        aucs.append(roc_auc_score(yt, yp))
        f1s.append(f1_score(yt, yd, zero_division=0))
        accs.append(accuracy_score(yt, yd))

    alpha = 1 - ci
    lo, hi = alpha / 2, 1 - alpha / 2

    def _ic(arr):
        if not arr:
            return (float('nan'), float('nan'))
        return (float(np.percentile(arr, lo * 100)), float(np.percentile(arr, hi * 100)))

    return {
        'n_bootstrap': n_bootstrap,
        'ci_nivel': ci,
        'auc': {'estimado': float(roc_auc_score(y_true, y_prob)), 'ic': _ic(aucs)},
        'f1': {'estimado': float(f1_score(y_true, y_pred, zero_division=0)), 'ic': _ic(f1s)},
        'acc': {'estimado': float(accuracy_score(y_true, y_pred)), 'ic': _ic(accs)},
    }


# ══════════════════════════════════════════════════════════════════
# 3. NYSTRÖM SENSITIVITY — Justification of Landmark Count M
# ══════════════════════════════════════════════════════════════════

def experimento_sensibilidad_nystrom(ds: DatasetResult, M_valores=(32, 64, 128), n_semillas=3) -> dict:
    N_Q = 4
    dev = qml.device('default.qubit', wires=N_Q)

    @qml.qnode(dev, interface='autograd')
    def _qcirc(x1, x2):
        qml.AngleEmbedding(x1, wires=range(N_Q), rotation='Y')
        qml.adjoint(qml.AngleEmbedding)(x2, wires=range(N_Q), rotation='Y')
        return qml.probs(wires=range(N_Q))

    def _krow(xi, Xref):
        return np.array([_qcirc(xi, xj)[0] for xj in Xref])

    std_sc = StandardScaler()
    X_tr_std = std_sc.fit_transform(ds.X_sk)
    X_te_std = std_sc.transform(ds.X_all[ds.s2:])

    pca = PCA(n_components=N_Q, random_state=42)
    X_tr_pca = pca.fit_transform(X_tr_std)
    X_te_pca = pca.transform(X_te_std)

    sc = MinMaxScaler(feature_range=(0, np.pi))
    X_tr_q = sc.fit_transform(X_tr_pca)
    X_te_q = sc.transform(X_te_pca)

    n_neg = (ds.y_sk == 0).sum()
    n_pos = (ds.y_sk == 1).sum()
    cw = {0: 1.0, 1: float(n_neg / max(n_pos, 1))}

    rng_err = np.random.RandomState(0)
    idx_err = rng_err.choice(len(X_tr_q), min(20, len(X_tr_q)), replace=False)
    X_err = X_tr_q[idx_err]
    K_exacta = np.array([[_qcirc(xi, xj)[0] for xj in X_err] for xi in X_err])

    resultados = {}

    for M in M_valores:
        aucs_m, f1s_m, tiempos_m, errores_m = [], [], [], []

        for semilla in range(n_semillas):
            rng = np.random.RandomState(semilla * 7 + 13)
            idx_pos = np.where(ds.y_sk == 1)[0]
            idx_neg = np.where(ds.y_sk == 0)[0]
            n_pos_lm = min(M // 2, len(idx_pos))
            n_neg_lm = min(M - n_pos_lm, len(idx_neg))
            lm = np.concatenate([
                rng.choice(idx_pos, n_pos_lm, replace=False),
                rng.choice(idx_neg, n_neg_lm, replace=False),
            ])
            X_lm = X_tr_q[lm]

            t0 = time.time()
            K_lm_lm = np.vstack([_krow(x, X_lm) for x in X_lm])
            K_tr_lm = np.vstack([_krow(x, X_lm) for x in X_tr_q])
            K_te_lm = np.vstack([_krow(x, X_lm) for x in X_te_q])
            dt = time.time() - t0

            K_lm_inv = np.linalg.pinv(K_lm_lm + 1e-6 * np.eye(M))
            K_tr = K_tr_lm @ K_lm_inv @ K_tr_lm.T
            K_te = K_te_lm @ K_lm_inv @ K_tr_lm.T

            svm = SVC(kernel='precomputed', C=1.0, class_weight=cw, probability=True, random_state=semilla)
            svm.fit(K_tr, ds.y_sk)
            y_prob = svm.predict_proba(K_te)[:, 1]
            y_pred = svm.predict(K_te)

            y_te = ds.y_te.numpy()
            aucs_m.append(roc_auc_score(y_te, y_prob))
            f1s_m.append(f1_score(y_te, y_pred, zero_division=0))
            tiempos_m.append(dt)

            K_err_lm = np.vstack([_krow(x, X_lm) for x in X_err])
            K_approx_err = K_err_lm @ K_lm_inv @ K_err_lm.T
            error_fro = (np.linalg.norm(K_exacta - K_approx_err, 'fro') /
                         (np.linalg.norm(K_exacta, 'fro') + 1e-8))
            errores_m.append(float(error_fro))

        resultados[M] = {
            'auc_media': float(np.mean(aucs_m)),
            'auc_std': float(np.std(aucs_m)),
            'f1_media': float(np.mean(f1s_m)),
            'f1_std': float(np.std(f1s_m)),
            'tiempo_medio': float(np.mean(tiempos_m)),
            'error_fro': float(np.mean(errores_m)) if errores_m else float('nan'),
        }
        print(f"  [Nystrom M={M:3d}] AUC={resultados[M]['auc_media']:.3f}+-{resultados[M]['auc_std']:.3f}  "
              f"F1={resultados[M]['f1_media']:.3f}+-{resultados[M]['f1_std']:.3f}  "
              f"eps_Fro={resultados[M]['error_fro']:.3f}  t={resultados[M]['tiempo_medio']:.0f}s")

    aucs_por_M = [(M, resultados[M]['auc_media']) for M in M_valores]
    mejora_32_64 = aucs_por_M[1][1] - aucs_por_M[0][1] if len(M_valores) >= 2 else 0
    mejora_64_128 = aucs_por_M[2][1] - aucs_por_M[1][1] if len(M_valores) >= 3 else 0
    knee_point = 64 if mejora_64_128 < mejora_32_64 * 0.5 else 128

    resultados['_conclusion'] = {
        'knee_point_M': knee_point,
        'texto': f"Knee-point identified at M={knee_point}. Marginal gain M=32->64: {mejora_32_64:+.4f}, M=64->128: {mejora_64_128:+.4f}."
    }
    return resultados


# ══════════════════════════════════════════════════════════════════
# 4. PHYSICAL JUSTIFICATION
# ══════════════════════════════════════════════════════════════════

JUSTIFICACION_FISICA_TERMODINAMICA = """
Mathematical Justification: Soft Thermodynamic Constraints vs Boolean Rules
=============================================================================
Potential critique: If physical laws dictate that extreme precipitation cannot 
occur without sufficient thermal energy (Clausius-Clapeyron), why not apply an 
if/then post-processing filter instead of penalizing the loss function?

Mathematical and Physical Response:
1. BOOLEAN RULE (Post-processing): Introduces a discontinuity on the decision 
   surface. The gradient dL/dTheta is strictly zero across the low-energy region. 
   The parameterized circuit receives zero feedback and fails to learn the 
   low-energy regime; it simply suppresses outputs post-hoc.
2. DIFFERENTIABLE CONSTRAINT (Clausius-Clapeyron Loss):
   L_cc = E[ ReLU(P_pred - 0.6) * ReLU(1.0 - CC_Capacity(T_anomaly)) ]
   The gradient is continuous and well-defined across the thermodynamic parameter 
   manifold. When the ansatz assigns high probability to extreme rainfall without 
   adequate latent heat support, it receives an informative gradient penalty.
   The circuit actively sculpts its Hilbert space representation to reflect atmospheric 
   stability bounds.
3. ENERGY CONSERVATION (Moist Static Energy):
   L_mse = E[ ReLU(E_thermal + E_moisture - E_kinetic - 0.5)^2 ]
   Enforces conservation of energy across entangled registers. If qubits Q0 and Q1 
   exhibit high expectation values, qubit Q3 must co-activate accordingly.
Conclusion: Differentiable thermodynamic constraints (CC + MSE) are vastly superior 
for guiding NISQ optimization trajectories, where meaningful gradient signals 
are essential to avoid barren plateaus and train effective representations.
"""


# ══════════════════════════════════════════════════════════════════
# 5. PAIRED OPERATIONAL BOOTSTRAP — Fixed-Threshold Evaluation
# ══════════════════════════════════════════════════════════════════

def bootstrap_pareado_operativo(y_true, y_ext, ACP, prob_a, prob_b,
                                 umbral_a, umbral_b,
                                 nombre_a="Model A", nombre_b="Model B",
                                 n_iterations=2000, seed=42) -> dict:
    """
    95% CI (PAIRED bootstrap) for the performance delta (B - A) in Rc-p90, Rc-ACP,
    and FPR, evaluating each model at a FIXED pre-computed threshold.

    We deliberately avoid recalibrating the optimal threshold within each iteration:
    doing so would answer "if each model recalibrated on each resample, which is best?",
    which reintroduces optimistic bias. Here, thresholds represent operational deployment.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)

    pred_a_full = (np.asarray(prob_a) >= umbral_a).astype(float)
    pred_b_full = (np.asarray(prob_b) >= umbral_b).astype(float)

    diffs = {'rc_p90': [], 'rc_ACP': [], 'fpr': []}

    print(f"  [>] Paired bootstrap (n={n_iterations}), fixed thresholds: "
          f"{nombre_a}={umbral_a:.3f}, {nombre_b}={umbral_b:.3f}...")

    for _ in range(n_iterations):
        idx = rng.randint(0, n, n)
        yb, yxb, db = y_true[idx], y_ext[idx], ACP[idx]
        pa, pb = pred_a_full[idx], pred_b_full[idx]

        if yb.sum() == 0 or yb.sum() == n or yxb.sum() == 0:
            continue

        diffs['rc_p90'].append(recall_score(yxb, pb, zero_division=0) -
                                recall_score(yxb, pa, zero_division=0))

        if db.sum() > 0:
            diffs['rc_ACP'].append(
                recall_score(yb[db == 1], pb[db == 1], zero_division=0) -
                recall_score(yb[db == 1], pa[db == 1], zero_division=0))

        tn_a, fp_a, fn_a, tp_a = confusion_matrix(yb, pa, labels=[0, 1]).ravel()
        tn_b, fp_b, fn_b, tp_b = confusion_matrix(yb, pb, labels=[0, 1]).ravel()
        fpr_a = fp_a / (fp_a + tn_a) if (fp_a + tn_a) > 0 else np.nan
        fpr_b = fp_b / (fp_b + tn_b) if (fp_b + tn_b) > 0 else np.nan
        diffs['fpr'].append(fpr_b - fpr_a)

    resultados = {}
    for k, vals in diffs.items():
        vals = np.asarray(vals, dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            resultados[k] = {'delta_medio': float('nan'), 'ic_lo': float('nan'),
                              'ic_hi': float('nan'), 'significativo': False, 'n_validos': 0}
            continue
        lo, hi = np.percentile(vals, [2.5, 97.5])
        resultados[k] = {
            'delta_medio': float(vals.mean()),
            'ic_lo': float(lo), 'ic_hi': float(hi),
            'significativo': not (lo <= 0.0 <= hi),
            'n_validos': int(len(vals)),
        }
    return resultados


# ══════════════════════════════════════════════════════════════════
# 6. MAIN PUBLIC VALIDATION FUNCTION
# ══════════════════════════════════════════════════════════════════

def ejecutar_validacion(ds: DatasetResult,
                        res_base: dict,
                        res_qml: dict,
                        res_qksvm: dict = None,
                        skip_nystrom: bool = False) -> dict:

    y_te = ds.y_te.numpy()

    probs_todos = {}
    if '_probs' in res_base:
        probs_todos.update(res_base['_probs'])
    if '_probs' in res_qml:
        probs_todos.update(res_qml['_probs'])
    if res_qksvm and 'QKSVM' in res_qksvm and '_prob' in res_qksvm['QKSVM']:
        probs_todos['QKSVM'] = res_qksvm['QKSVM']['_prob']

    print("\n" + "=" * 65)
    print("RIGOROUS STATISTICAL VALIDATION (MULTI-STATION)")
    print("=" * 65)

    # 1. Bootstrap CIs
    print(f"\n  [1] 95% Confidence Intervals (bootstrap n=2000, samples={len(y_te)})")
    print(f"  {'Model':<22} {'AUC':>7}  {'95% CI':>18}  {'F1':>6}  {'95% CI':>16}")
    print("  " + "-" * 75)

    ic_resultados = {}
    for nm, prob in probs_todos.items():
        if prob is None or len(prob) == 0:
            continue
        ic = bootstrap_ci(y_te, prob, n_bootstrap=2000)
        ic_resultados[nm] = ic
        auc_lo, auc_hi = ic['auc']['ic']
        f1_lo, f1_hi = ic['f1']['ic']
        print(f"  {nm:<22} {ic['auc']['estimado']:>6.3f}  [{auc_lo:.3f}, {auc_hi:.3f}]  "
              f"{ic['f1']['estimado']:>5.3f}  [{f1_lo:.3f}, {f1_hi:.3f}]")

    # 2. DeLong Tests
    print("\n  [2] DeLong Tests (H0: AUC_A = AUC_B)")
    print("  " + "-" * 65)

    tests_delong = {}
    pares_clave = [
        ('LSTM+QNN', 'LSTM pura', 'QNN_vs_LSTMpura'),
        ('LSTM+PI-QNN', 'LSTM+QNN', 'PIQNN_vs_QNN'),
        ('LSTM+QNN', 'Random Forest', 'QNN_vs_RF'),
        ('LSTM+PI-QNN', 'Random Forest', 'PIQNN_vs_RF'),
        ('QKSVM', 'LSTM+QNN', 'QKSVM_vs_QNN'),
        ('LbL-QNN', 'LSTM+PI-QNN', 'LbL_vs_PIQNN'),
        ('LbL-QNN', 'LSTM pura', 'LbL_vs_LSTMpura'),
        ('LSTM+QTL', 'LSTM pura', 'QTL_vs_LSTMpura'),
        ('LSTM+QTL', 'LbL-QNN', 'QTL_vs_LbL'),
        ('LbL-QNN', 'LSTM+QNN', 'LbL_vs_QNN'),
        ('LbL-QNN', 'Random Forest', 'LbL_vs_RF'),
    ]

    for nm_a, nm_b, etiq in pares_clave:
        if nm_a not in probs_todos or nm_b not in probs_todos:
            continue
        resultado = test_delong(y_te, probs_todos[nm_a], probs_todos[nm_b], nombre_a=nm_a, nombre_b=nm_b)
        tests_delong[etiq] = resultado
        sig = "sig. (p<0.05)" if resultado['significativo_95'] else "not sig. (p>=0.05)"
        print(f"  {nm_a:<18} vs {nm_b:<18}  Delta-AUC={resultado['delta_auc']:+.4f}  p={resultado['p_valor']:.4f}  [{sig}]")

    # 3. Nystrom Sensitivity
    res_nystrom = None
    if not skip_nystrom and 'QKSVM' in probs_todos:
        print("\n  [3] Nystrom Sensitivity Experiment M={32,64,128}")
        print("  " + "-" * 65)
        try:
            res_nystrom = experimento_sensibilidad_nystrom(ds)
            print(f"\n  Conclusion: {res_nystrom['_conclusion']['texto']}")
        except Exception as e:
            print(f"  [!] Nystrom execution failed: {e}")
            res_nystrom = {'error': str(e)}

    # 4. AUC CI Overlap Check
    print("\n  [4] Verification: do AUC 95% CIs overlap between key models?")
    print("  " + "-" * 65)
    modelos_a_mostrar = [m for m in ['Random Forest', 'XGBoost', 'LSTM pura', 'LSTM+QNN',
                                     'LSTM+PI-QNN', 'LSTM+QTL', 'LbL-QNN', 'QKSVM']
                         if m in ic_resultados]
    for nm in modelos_a_mostrar:
        lo, hi = ic_resultados[nm]['auc']['ic']
        print(f"  {nm:<22} AUC CI=[{lo:.3f}, {hi:.3f}]")

    # 5. Paired operational bootstrap
    res_bootstrap_pareado = {}
    if 'LbL-QNN' in probs_todos and 'LSTM pura' in probs_todos:
        print("\n  [5] Paired Bootstrap (fixed threshold): LbL-QNN vs Pure LSTM (FPR-Matched)")
        print("  " + "-" * 65)

        prob_lbl = probs_todos['LbL-QNN']
        prob_lstm = probs_todos['LSTM pura']
        ACP_te = ds.ACP_te.numpy() if hasattr(ds.ACP_te, 'numpy') else ds.ACP_te
        y_te_ext = ds.y_te_ext.numpy()

        umbral_qml = 0.5
        pred_lbl_05 = (prob_lbl >= umbral_qml).astype(float)
        tn, fp, fn, tp = confusion_matrix(y_te, pred_lbl_05, labels=[0, 1]).ravel()
        fpr_objetivo = fp / (fp + tn) if (fp + tn) > 0 else float('nan')

        fprs, _, thresholds = roc_curve(y_te, prob_lstm)
        idx = np.argmin(np.abs(fprs - fpr_objetivo))
        umbral_lstm = thresholds[idx]

        print(f"  Target FPR (LbL-QNN @ 0.5) = {fpr_objetivo:.3f} -> "
              f"adjusted LSTM threshold = {umbral_lstm:.3f}")

        res_bootstrap_pareado = bootstrap_pareado_operativo(
            y_te, y_te_ext, ACP_te, prob_lstm, prob_lbl,
            umbral_lstm, umbral_qml,
            nombre_a="Pure LSTM (FPR Matched)", nombre_b="LbL-QNN",
        )
        res_bootstrap_pareado['_fpr_objetivo'] = float(fpr_objetivo)
        res_bootstrap_pareado['_umbral_lstm'] = float(umbral_lstm)

        for k, etiqueta in (('rc_p90', 'Delta-Rc-p90 '), ('rc_ACP', 'Delta-Rc-ACP'), ('fpr', 'Delta-FPR    ')):
            r = res_bootstrap_pareado[k]
            sig = "significant (CI excludes 0)" if r['significativo'] else "NOT significant (CI spans 0)"
            print(f"  {etiqueta} (LbL - LSTM) = {r['delta_medio']:+.4f}  "
                  f"95% CI=[{r['ic_lo']:+.4f}, {r['ic_hi']:+.4f}]  [{sig}]")
    else:
        print("\n  [5] Paired bootstrap skipped: 'LbL-QNN' or 'LSTM pura' missing in probs_todos.")

    return {
        'ic_bootstrap': ic_resultados,
        'tests_delong': tests_delong,
        'sensibilidad_nystrom': res_nystrom,
        'bootstrap_pareado_operativo': res_bootstrap_pareado,
        'fisica_justificacion': JUSTIFICACION_FISICA_TERMODINAMICA,
        'n_test': int(len(y_te)),
        'n_positivos_test': int(y_te.sum()),
    }


# English alias for public API
run_validation = ejecutar_validacion