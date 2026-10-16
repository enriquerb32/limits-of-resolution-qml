"""
validacion.py  —  Módulo de validación estadística rigurosa (MULTI-ESTACIÓN)
============================================================================
Responde directamente a las críticas del revisor:

  1. Test de DeLong (1988): prueba de significancia para diferencias AUC.
  2. Bootstrapping con IC al 95%: intervalos de confianza (n=2000).
  3. Experimento de sensibilidad Nyström M={32,64,128} con PCA real.
  4. Justificación matemática de las Restricciones Termodinámicas Suaves.

Función pública:
  ejecutar_validacion(ds, res_base, res_qml, res_qksvm, skip_nystrom) → dict
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
# 1. TEST DE DELONG — Significancia estadística de diferencias AUC
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
            f"La diferencia AUC({nombre_a})={auc_a:.3f} vs "
            f"AUC({nombre_b})={auc_b:.3f} es "
            f"{'ESTADÍSTICAMENTE SIGNIFICATIVA' if significativo else 'NO significativa'} "
            f"(Z={z:.3f}, p={p:.4f})"
        )
    }


# ══════════════════════════════════════════════════════════════════
# 2. BOOTSTRAPPING — Intervalos de confianza al 95%
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
# 3. SENSIBILIDAD NYSTRÖM — Justifica M=64 (CON PCA REAL)
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
        print(f"  [Nyström M={M:3d}] AUC={resultados[M]['auc_media']:.3f}±{resultados[M]['auc_std']:.3f}  "
              f"F1={resultados[M]['f1_media']:.3f}±{resultados[M]['f1_std']:.3f}  "
              f"ε_Fro={resultados[M]['error_fro']:.3f}  t={resultados[M]['tiempo_medio']:.0f}s")

    aucs_por_M = [(M, resultados[M]['auc_media']) for M in M_valores]
    mejora_32_64 = aucs_por_M[1][1] - aucs_por_M[0][1] if len(M_valores) >= 2 else 0
    mejora_64_128 = aucs_por_M[2][1] - aucs_por_M[1][1] if len(M_valores) >= 3 else 0
    knee_point = 64 if mejora_64_128 < mejora_32_64 * 0.5 else 128

    resultados['_conclusion'] = {
        'knee_point_M': knee_point,
        'texto': f"Knee-point identificado en M={knee_point}. Rendimiento marginal M=32→64: {mejora_32_64:+.4f}, M=64→128: {mejora_64_128:+.4f}."
    }
    return resultados


# ══════════════════════════════════════════════════════════════════
# 4. JUSTIFICACIÓN FÍSICA
# ══════════════════════════════════════════════════════════════════

JUSTIFICACION_FISICA_TERMODINAMICA = """
Justificación matemática: Restricciones termodinámicas suaves vs reglas booleanas
==================================================================================
Crítica potencial: Si la física dicta que no puede haber lluvia extrema sin 
carga térmica (Clausius-Clapeyron), ¿por qué no usar un filtro post-proceso 
if/then en lugar de una penalización en la función de pérdida?

Respuesta matemática y física:
1. REGLA BOOLEANA (post-proceso): Introduce una discontinuidad en la superficie 
   de decisión. El gradiente ∂L/∂θ es exactamente cero en la región de baja 
   energía térmica. El circuito no aprende a representar el régimen de baja 
   energía; simplemente lo censura.
2. RESTRICCIÓN DIFERENCIABLE (Clausius-Clapeyron Loss):
   L_cc = E[ ReLU(P_pred - 0.6) * ReLU(1.0 - CC_Capacity(T_anomaly)) ]
   El gradiente existe en TODA la región termodinámica. Cuando el modelo 
   predice lluvia extrema sin suficiente calor latente, recibe una señal 
   de gradiente suave pero constante. El circuito aprende a representar 
   ACTIVAMENTE la física de la estabilidad atmosférica.
3. CONSERVACIÓN DE ENERGÍA (Moist Static Energy):
   L_mse = E[ ReLU(E_thermal + E_moisture - E_kinetic - 0.5)^2 ]
   Forzamos la conservación de la energía. Si los qubits Q0 y Q1 están 
   altamente excitados, el qubit Q3 DEBE excitarse.
Conclusión: Las restricciones termodinámicas diferenciables (CC + MSE) son 
infinitamente más informativas para el optimizador cuántico, especialmente 
en el régimen NISQ donde cada señal de gradiente es crucial.
"""


# ══════════════════════════════════════════════════════════════════
# 5. BOOTSTRAP PAREADO OPERATIVO — Resuelve la "paradoja" a umbral fijo
# ══════════════════════════════════════════════════════════════════

def bootstrap_pareado_operativo(y_true, y_ext, ACP, prob_a, prob_b,
                                 umbral_a, umbral_b,
                                 nombre_a="Modelo A", nombre_b="Modelo B",
                                 n_iterations=2000, seed=42) -> dict:
    """
    IC 95% (bootstrap PAREADO) de la diferencia (B - A) en Rc-p90, Rc-ACP y
    FPR, evaluando cada modelo en un umbral FIJO precalculado una sola vez
    fuera de esta función (p.ej. el umbral que analisis_extremos ya imprimió
    en el Paso 8).

    A propósito NO se recalcula el umbral óptimo dentro de cada iteración:
    hacerlo respondería a "si cada modelo pudiera reajustar su umbral en cada
    muestra, ¿cuál sería mejor en promedio?", que reintroduce el mismo
    problema que el threshold-matching quería eliminar (dejar que el modelo
    elija su punto de operación más favorable). Aquí los umbrales son los
    que de verdad se despliegan, y solo se remuestrean los resultados.

    Es "pareado" porque cada iteración usa los MISMOS índices remuestreados
    para ambos modelos, preservando la correlación entre ellos.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)

    pred_a_full = (np.asarray(prob_a) >= umbral_a).astype(float)
    pred_b_full = (np.asarray(prob_b) >= umbral_b).astype(float)

    diffs = {'rc_p90': [], 'rc_ACP': [], 'fpr': []}

    print(f"  [>] Bootstrap pareado (n={n_iterations}), umbrales fijos: "
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
# 6. FUNCIÓN PÚBLICA PRINCIPAL
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

    print("\n" + "═" * 65)
    print("VALIDACIÓN ESTADÍSTICA RIGUROSA (MULTI-ESTACIÓN)")
    print("═" * 65)

    # 1. ICs Bootstrap
    print(f"\n  [1] Intervalos de confianza al 95% (bootstrap n=2000, muestras={len(y_te)})")
    print(f"  {'Modelo':<22} {'AUC':>7}  {'IC 95%':>18}  {'F1':>6}  {'IC 95%':>16}")
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

    # 2. Tests de DeLong
    print("\n  [2] Tests de DeLong (H0: AUC_A = AUC_B)")
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
        ('LbL-QNN', 'Random Forest', 'LbL_vs_RF'),  # 🎯 FIX: faltaba la comparación vs el mejor baseline clásico
    ]

    for nm_a, nm_b, etiq in pares_clave:
        if nm_a not in probs_todos or nm_b not in probs_todos:
            continue
        resultado = test_delong(y_te, probs_todos[nm_a], probs_todos[nm_b], nombre_a=nm_a, nombre_b=nm_b)
        tests_delong[etiq] = resultado
        sig = "✅ p<0.05" if resultado['significativo_95'] else "⚠️  p≥0.05 (no sig.)"
        print(f"  {nm_a:<18} vs {nm_b:<18}  ΔAUC={resultado['delta_auc']:+.4f}  p={resultado['p_valor']:.4f}  {sig}")

    # 3. Sensibilidad Nyström
    res_nystrom = None
    if not skip_nystrom and 'QKSVM' in probs_todos:
        print("\n  [3] Experimento de sensibilidad Nyström M={32,64,128}")
        print("  " + "-" * 65)
        try:
            res_nystrom = experimento_sensibilidad_nystrom(ds)
            print(f"\n  Conclusión: {res_nystrom['_conclusion']['texto']}")
        except Exception as e:
            print(f"  [!] Nyström no ejecutable: {e}")
            res_nystrom = {'error': str(e)}

    # 4. Verificación de solapamiento de ICs
    print("\n  [4] Verificación: ¿se solapan los ICs de AUC entre modelos clave?")
    print("  " + "-" * 65)
    modelos_a_mostrar = [m for m in ['Random Forest', 'XGBoost', 'LSTM pura', 'LSTM+QNN',
                                     'LSTM+PI-QNN', 'LSTM+QTL', 'LbL-QNN', 'QKSVM']
                         if m in ic_resultados]
    for nm in modelos_a_mostrar:
        lo, hi = ic_resultados[nm]['auc']['ic']
        print(f"  {nm:<22} AUC IC=[{lo:.3f}, {hi:.3f}]")

    # 5. Bootstrap pareado operativo (resuelve la "paradoja" del threshold-matching)
    res_bootstrap_pareado = {}
    if 'LbL-QNN' in probs_todos and 'LSTM pura' in probs_todos:
        print("\n  [5] Bootstrap pareado (umbral fijo): LbL-QNN vs LSTM pura (FPR-Matched)")
        print("  " + "-" * 65)

        prob_lbl = probs_todos['LbL-QNN']
        prob_lstm = probs_todos['LSTM pura']
        ACP_te = ds.ACP_te.numpy() if hasattr(ds.ACP_te, 'numpy') else ds.ACP_te
        y_te_ext = ds.y_te_ext.numpy()

        # 🎯 FIX: el umbral objetivo se calcula aquí, no se hardcodea. Es el
        # mismo FPR que ya reporta 'LbL-QNN' en res_qml (umbral real 0.5,
        # tal como se despliega) — no un valor de una corrida anterior que
        # quedaría obsoleto en cuanto cambien los resultados.
        umbral_qml = 0.5
        pred_lbl_05 = (prob_lbl >= umbral_qml).astype(float)
        tn, fp, fn, tp = confusion_matrix(y_te, pred_lbl_05, labels=[0, 1]).ravel()
        fpr_objetivo = fp / (fp + tn) if (fp + tn) > 0 else float('nan')

        fprs, _, thresholds = roc_curve(y_te, prob_lstm)
        idx = np.argmin(np.abs(fprs - fpr_objetivo))
        umbral_lstm = thresholds[idx]

        print(f"  FPR objetivo (LbL-QNN @ 0.5) = {fpr_objetivo:.3f}  →  "
              f"umbral LSTM ajustado = {umbral_lstm:.3f}")

        res_bootstrap_pareado = bootstrap_pareado_operativo(
            y_te, y_te_ext, ACP_te, prob_lstm, prob_lbl,
            umbral_lstm, umbral_qml,
            nombre_a="LSTM pura (FPR Matched)", nombre_b="LbL-QNN",
        )
        res_bootstrap_pareado['_fpr_objetivo'] = float(fpr_objetivo)
        res_bootstrap_pareado['_umbral_lstm'] = float(umbral_lstm)

        for k, etiqueta in (('rc_p90', 'ΔRc-p90 '), ('rc_ACP', 'ΔRc-ACP'), ('fpr', 'ΔFPR    ')):
            r = res_bootstrap_pareado[k]
            sig = "✅ significativo (IC no cruza 0)" if r['significativo'] else "⚠️  NO significativo (IC cruza 0)"
            print(f"  {etiqueta} (LbL − LSTM) = {r['delta_medio']:+.4f}  "
                  f"IC95%=[{r['ic_lo']:+.4f}, {r['ic_hi']:+.4f}]  {sig}")
    else:
        print("\n  [5] Bootstrap pareado omitido: falta 'LbL-QNN' o 'LSTM pura' en probs_todos.")

    return {
        'ic_bootstrap': ic_resultados,
        'tests_delong': tests_delong,
        'sensibilidad_nystrom': res_nystrom,
        'bootstrap_pareado_operativo': res_bootstrap_pareado,
        'fisica_justificacion': JUSTIFICACION_FISICA_TERMODINAMICA,
        'n_test': int(len(y_te)),
        'n_positivos_test': int(y_te.sum()),
    }