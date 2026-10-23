"""
experimento_ruido.py — Noisy Quantum Hardware Evaluation (NISQ)
===============================================================
Injects depolarizing noise (decoherence and gate errors) to assess
the resilience of the thermodynamic PI-QNN compared to unconstrained
quantum models.

Reported metrics:
  - Acc, F1, AUC     : Global discrimination
  - Rc-p90           : Recall on extreme precipitation events
  - Rc-ACP           : Specific recall on ACP events (small sample)
  - FPR              : False positive rate (fp / (fp + tn))

Executes a Monte Carlo loop over `n_seeds` to compute mean and standard
deviation, guaranteeing statistical reproducibility via deterministic seed isolation.
"""

from dataclasses import dataclass
import logging
from typing import Any, Dict, Optional, Tuple
import math
import torch
import numpy as np
import pennylane as qml
from sklearn.metrics import recall_score, confusion_matrix

from modelos import LstmQNN, _evaluar_pytorch
from modelos_utils import deterministic_seed, MetricAccumulator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NoiseConfig:
    """Immutable, typed configuration for NISQ noise simulation."""
    p_error: float = 0.01
    n_seeds: int = 15
    shots: int = 1024
    wires: int = 4
    single_qubit_gates: Tuple[str, ...] = ('ry', 'rz', 'rx', 'u1', 'u2', 'u3')
    two_qubit_gates: Tuple[str, ...] = ('cx', 'cz')


def _metricas_completas(
    y_true: np.ndarray,
    y_extreme: np.ndarray,
    acp_mask: np.ndarray,
    prob: np.ndarray
) -> Tuple[float, float, float]:
    """
    Computes Rc-p90, Rc-ACP, and FPR from predicted probabilities.
    """
    pred = (prob >= 0.5).astype(float)
    rc_p90 = float(recall_score(y_extreme, pred, zero_division=0))

    if acp_mask.sum() > 0:
        rc_acp = float(recall_score(y_true[acp_mask == 1], pred[acp_mask == 1], zero_division=0))
    else:
        rc_acp = float('nan')

    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else float('nan')

    return rc_p90, rc_acp, fpr


def ejecutar_evaluacion_ruido(
    ds: Any,
    modelo_qnn_limpio: LstmQNN,
    modelo_lbl_limpio: LstmQNN,
    p_error: float = 0.01,
    n_seeds: int = 15,
    config: Optional[NoiseConfig] = None
) -> Dict[str, Dict[str, float]]:
    """
    Runs NISQ evaluation under depolarizing noise with Monte Carlo statistical aggregation.
    Guarantees strict return type contract across all execution or failure paths.
    """
    cfg = config or NoiseConfig(p_error=p_error, n_seeds=n_seeds)

    print("\n" + "=" * 65)
    print(f"EXTRA STEP · NISQ Evaluation under Depolarizing Noise (p={cfg.p_error})")
    print("=" * 65)

    try:
        from qiskit_aer.noise import NoiseModel, depolarizing_error
    except ImportError:
        print("  [!] Noise simulation libraries not found. Skipping experiment.")
        print("      To execute: pip install qiskit-aer pennylane-qiskit")
        return {'LSTM+QNN': {}, 'LbL-QNN': {}}

    # 1. Configure Qiskit Aer with noise model
    noise_model = NoiseModel()
    error_1q = depolarizing_error(cfg.p_error, 1)
    error_2q = depolarizing_error(cfg.p_error, 2)

    noise_model.add_all_qubit_quantum_error(error_1q, list(cfg.single_qubit_gates))
    noise_model.add_all_qubit_quantum_error(error_2q, list(cfg.two_qubit_gates))

    try:
        dev_noisy = qml.device('qiskit.aer', wires=cfg.wires, noise_model=noise_model, shots=cfg.shots)
    except qml.DeviceError:
        print("  [!] Plugin pennylane-qiskit not installed properly. Skipping experiment.")
        return {'LSTM+QNN': {}, 'LbL-QNN': {}}

    def inyectar_ruido(modelo_entrenado: LstmQNN, n_l: int) -> LstmQNN:
        modelo_ruidoso = LstmQNN(ds.in_dim, n_q=cfg.wires, n_l=n_l, physics=modelo_entrenado.physics).double()
        modelo_ruidoso.load_state_dict(modelo_entrenado.state_dict())
        modelo_ruidoso.eval()

        @qml.qnode(dev_noisy, interface='torch')
        def circ_ruidoso(W, x):
            is_batched = (x.ndim == 2)
            for l in range(n_l):
                for i in range(cfg.wires):
                    val = x[:, i % x.shape[1]] if is_batched else x[i % x.shape[0]]
                    qml.RY(val, wires=i)
                    qml.RZ(val, wires=i)
                qml.StronglyEntanglingLayers(W[l:l + 1], wires=range(cfg.wires))
            return [qml.expval(qml.PauliZ(i)) for i in range(cfg.wires)]

        modelo_ruidoso.circ = circ_ruidoso
        return modelo_ruidoso

    y_te = ds.y_te.numpy()
    y_te_x = ds.y_te_ext.numpy()
    ACP_te = ds.ACP_te.numpy() if hasattr(ds.ACP_te, 'numpy') else np.asarray(ds.ACP_te)

    # 2. Initialize noisy models
    qnn_ruidoso = inyectar_ruido(modelo_qnn_limpio, n_l=2)
    lbl_ruidoso = inyectar_ruido(modelo_lbl_limpio, n_l=3)

    print(f"  [NISQ Sim] Running Monte Carlo simulation with {cfg.n_seeds} deterministic seeds...")

    acc_qnn = MetricAccumulator()
    acc_lbl = MetricAccumulator()

    # 3. Monte Carlo loop with isolated deterministic seeds
    for seed_idx in range(cfg.n_seeds):
        with deterministic_seed(seed_idx):
            # QNN evaluation
            met_qnn, prob_qnn = _evaluar_pytorch(qnn_ruidoso, ds.X_te.double(), y_te)
            rc_p90_qnn, rc_acp_qnn, fpr_qnn = _metricas_completas(y_te, y_te_x, ACP_te, prob_qnn)
            acc_qnn.record(
                acc=met_qnn['acc'], f1=met_qnn['f1'], auc=met_qnn['auc'],
                rc_p90=rc_p90_qnn, rc_ACP=rc_acp_qnn, fpr=fpr_qnn
            )

            # LbL evaluation
            met_lbl, prob_lbl = _evaluar_pytorch(lbl_ruidoso, ds.X_te.double(), y_te)
            rc_p90_lbl, rc_acp_lbl, fpr_lbl = _metricas_completas(y_te, y_te_x, ACP_te, prob_lbl)
            acc_lbl.record(
                acc=met_lbl['acc'], f1=met_lbl['f1'], auc=met_lbl['auc'],
                rc_p90=rc_p90_lbl, rc_ACP=rc_acp_lbl, fpr=fpr_lbl
            )

        if (seed_idx + 1) % 5 == 0:
            print(f"    Evaluation {seed_idx + 1}/{cfg.n_seeds} completed.")

    # 4. Aggregation and reporting
    agg_qnn = acc_qnn.summary()
    agg_lbl = acc_lbl.summary()

    print(f"\n  [Robust NISQ Results - Mean ± Standard Deviation ({cfg.n_seeds} runs)]:")
    print(f"  -> LSTM+QNN   | Acc: {agg_qnn['acc']['mean']*100:.2f}% | F1: {agg_qnn['f1']['mean']:.3f} ± {agg_qnn['f1']['std']:.3f} "
          f"| AUC: {agg_qnn['auc']['mean']:.3f} ± {agg_qnn['auc']['std']:.3f}")
    print(f"                | Rc-ACP: {agg_qnn['rc_ACP']['mean']:.3f} ± {agg_qnn['rc_ACP']['std']:.3f} "
          f"| FPR: {agg_qnn['fpr']['mean']:.3f} ± {agg_qnn['fpr']['std']:.3f}")

    print(f"\n  -> LbL-QNN | Acc: {agg_lbl['acc']['mean']*100:.2f}% | F1: {agg_lbl['f1']['mean']:.3f} ± {agg_lbl['f1']['std']:.3f} "
          f"| AUC: {agg_lbl['auc']['mean']:.3f} ± {agg_lbl['auc']['std']:.3f}")
    print(f"                | Rc-ACP: {agg_lbl['rc_ACP']['mean']:.3f} ± {agg_lbl['rc_ACP']['std']:.3f} "
          f"| FPR: {agg_lbl['fpr']['mean']:.3f} ± {agg_lbl['fpr']['std']:.3f}")

    # 5. Format for strict backward compatibility with main.py
    return {
        'LSTM+QNN': acc_qnn.to_flat_dict(),
        'LbL-QNN': acc_lbl.to_flat_dict()
    }

# English alias for modern pipeline invocations
run_noise_evaluation = ejecutar_evaluacion_ruido