"""
experimento_ruido.py — Evaluación en hardware ruidoso (NISQ)
============================================================
Inyecta ruido despolarizante (decoherencia y errores de puerta)
para evaluar la resiliencia de la PI-QNN termodinámica frente
a modelos cuánticos no restringidos por física.

Métricas reportadas:
  - Acc, F1, AUC     : discriminación global
  - Rc-p90           : recall sobre eventos de precipitación extrema
  - Rc-ACP          : recall específico sobre eventos ACP (n pequeño)
  - FPR              : tasa de falsa alarma (fp / (fp + tn))
  
Ahora ejecuta un bucle de Monte Carlo con `n_seeds` para calcular la 
media y desviación estándar, validando rigurosamente el impacto del ruido.
"""

import math
import torch
import numpy as np
import pennylane as qml
from sklearn.metrics import recall_score, confusion_matrix
from modelos import LstmQNN, _evaluar_pytorch


def _metricas_completas(y_t, y_t_x, ACP_t, prob):
    """
    Calcula Rc-p90, Rc-ACP y FPR a partir de las probabilidades predichas.
    """
    pred = (prob >= 0.5).astype(float)

    rc_p90 = recall_score(y_t_x, pred, zero_division=0)

    if ACP_t.sum() > 0:
        rc_ACP = recall_score(y_t[ACP_t == 1], pred[ACP_t == 1], zero_division=0)
    else:
        rc_ACP = float('nan')

    tn, fp, fn, tp = confusion_matrix(y_t, pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float('nan')

    return rc_p90, rc_ACP, fpr


def ejecutar_evaluacion_ruido(ds, modelo_qnn_limpio, modelo_lbl_limpio, p_error=0.01, n_seeds=15):
    print("\n" + "═" * 65)
    print(f"PASO EXTRA · Evaluación NISQ bajo ruido despolarizante (p={p_error})")
    print("═" * 65)

    try:
        from qiskit_aer.noise import NoiseModel, depolarizing_error
    except ImportError:
        print("  [!] Librerías de ruido no encontradas. Saltando experimento.")
        print("      Para ejecutarlo: pip install qiskit-aer pennylane-qiskit")
        return {}, {}

    # 1. Configurar Qiskit Aer con el modelo de ruido
    noise_model = NoiseModel()
    error_1q = depolarizing_error(p_error, 1)
    error_2q = depolarizing_error(p_error, 2)

    noise_model.add_all_qubit_quantum_error(error_1q, ['ry', 'rz', 'rx', 'u1', 'u2', 'u3'])
    noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cz'])

    # Añadimos shots=1024 para obligar al device a emular el ruido estocástico del hardware
    try:
        dev_noisy = qml.device('qiskit.aer', wires=4, noise_model=noise_model, shots=1024)
    except qml.DeviceError:
        print("  [!] Plugin pennylane-qiskit no instalado correctamente. Saltando experimento.")
        return {}, {}

    def inyectar_ruido(modelo_entrenado, n_l):
        modelo_ruidoso = LstmQNN(ds.in_dim, n_q=4, n_l=n_l, physics=modelo_entrenado.physics).double()
        modelo_ruidoso.load_state_dict(modelo_entrenado.state_dict())
        modelo_ruidoso.eval()

        @qml.qnode(dev_noisy, interface='torch')
        def circ_ruidoso(W, x):
            for l in range(n_l):
                for i in range(4):
                    qml.RY(x[i % x.shape[0]], wires=i)
                    qml.RZ(x[i % x.shape[0]], wires=i)
                qml.StronglyEntanglingLayers(W[l:l + 1], wires=range(4))
            return [qml.expval(qml.PauliZ(i)) for i in range(4)]

        modelo_ruidoso.circ = circ_ruidoso
        return modelo_ruidoso

    y_te = ds.y_te.numpy()
    y_te_x = ds.y_te_ext.numpy()
    ACP_te = ds.ACP_te.numpy() if hasattr(ds.ACP_te, 'numpy') else np.asarray(ds.ACP_te)

    # 2. Inicializar modelos
    qnn_ruidoso = inyectar_ruido(modelo_qnn_limpio, n_l=2)
    lbl_ruidoso = inyectar_ruido(modelo_lbl_limpio, n_l=3)

    print(f"  [NISQ Sim] Ejecutando simulación Monte Carlo con {n_seeds} semillas...")
    
    res_qnn = {'acc': [], 'f1': [], 'auc': [], 'rc_p90': [], 'rc_ACP': [], 'fpr': []}
    res_lbl = {'acc': [], 'f1': [], 'auc': [], 'rc_p90': [], 'rc_ACP': [], 'fpr': []}

    # 3. Bucle de semillas para significancia estadística
    for i in range(n_seeds):
        # Evaluación QNN
        met_qnn, prob_qnn = _evaluar_pytorch(qnn_ruidoso, ds.X_te.double(), y_te)
        rc_p90_qnn, rc_ACP_qnn, fpr_qnn = _metricas_completas(y_te, y_te_x, ACP_te, prob_qnn)
        
        res_qnn['acc'].append(met_qnn['acc'])
        res_qnn['f1'].append(met_qnn['f1'])
        res_qnn['auc'].append(met_qnn['auc'])
        res_qnn['rc_p90'].append(rc_p90_qnn)
        res_qnn['rc_ACP'].append(rc_ACP_qnn)
        res_qnn['fpr'].append(fpr_qnn)
        
        # Evaluación LbL
        met_lbl, prob_lbl = _evaluar_pytorch(lbl_ruidoso, ds.X_te.double(), y_te)
        rc_p90_lbl, rc_ACP_lbl, fpr_lbl = _metricas_completas(y_te, y_te_x, ACP_te, prob_lbl)
        
        res_lbl['acc'].append(met_lbl['acc'])
        res_lbl['f1'].append(met_lbl['f1'])
        res_lbl['auc'].append(met_lbl['auc'])
        res_lbl['rc_p90'].append(rc_p90_lbl)
        res_lbl['rc_ACP'].append(rc_ACP_lbl)
        res_lbl['fpr'].append(fpr_lbl)
        
        if (i + 1) % 5 == 0:
            print(f"    Evaluación {i+1}/{n_seeds} completada.")

    # 4. Agregación Estadística
    def _aggregate(res_dict):
        return {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} for k, v in res_dict.items()}

    agg_qnn = _aggregate(res_qnn)
    agg_lbl = _aggregate(res_lbl)

    print(f"\n  [Resultados NISQ Robustos - Media ± Desviación Estándar ({n_seeds} runs)]:")
    print(f"  -> LSTM+QNN   | Acc: {agg_qnn['acc']['mean']*100:.2f}% | F1: {agg_qnn['f1']['mean']:.3f} ± {agg_qnn['f1']['std']:.3f} "
          f"| AUC: {agg_qnn['auc']['mean']:.3f} ± {agg_qnn['auc']['std']:.3f}")
    print(f"                | Rc-ACP: {agg_qnn['rc_ACP']['mean']:.3f} ± {agg_qnn['rc_ACP']['std']:.3f} "
          f"| FPR: {agg_qnn['fpr']['mean']:.3f} ± {agg_qnn['fpr']['std']:.3f}")

    print(f"\n  -> LbL-QNN | Acc: {agg_lbl['acc']['mean']*100:.2f}% | F1: {agg_lbl['f1']['mean']:.3f} ± {agg_lbl['f1']['std']:.3f} "
          f"| AUC: {agg_lbl['auc']['mean']:.3f} ± {agg_lbl['auc']['std']:.3f}")
    print(f"                | Rc-ACP: {agg_lbl['rc_ACP']['mean']:.3f} ± {agg_lbl['rc_ACP']['std']:.3f} "
          f"| FPR: {agg_lbl['fpr']['mean']:.3f} ± {agg_lbl['fpr']['std']:.3f}")

    # 5. Formatear para compatibilidad hacia atrás con el JSON original en main.py
    resultado_qnn_flat, resultado_lbl_flat = {}, {}
    
    for k, v in agg_qnn.items():
        resultado_qnn_flat[k] = v['mean']
        resultado_qnn_flat[f"{k}_std"] = v['std']
        
    for k, v in agg_lbl.items():
        resultado_lbl_flat[k] = v['mean']
        resultado_lbl_flat[f"{k}_std"] = v['std']

    return {
        'LSTM+QNN': resultado_qnn_flat,
        'LbL-QNN': resultado_lbl_flat
    }