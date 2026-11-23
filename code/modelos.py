"""
modelos.py — Backward-compatible re-export facade
===================================================
All implementations now live in:
  - modelos_utils.py      (shared utilities: metrics, training loops, evaluation)
  - modelos_clasicos.py   (PureLSTM, sklearn baselines, multi-seed classical)
  - modelos_cuanticos.py  (LstmQNN, PI-QNN, QTL, LbL, QKSVM, ablation)

This file re-exports everything so that existing code doing
``from modelos import LstmQNN`` or ``from modelos import entrenar_baselines``
continues to work without changes.
"""

# Shared utilities
from modelos_utils import (                                  # noqa: F401
    _metricas,
    _safe_fit_sklearn,
    _safe_predict_sklearn,
    _evaluar_pytorch,
    _loop,
    _extraer_probs,
    analisis_extremos,
    extreme_analysis,
    deterministic_seed,
    MetricAccumulator,
)

# Classical baselines
from modelos_clasicos import (                               # noqa: F401
    PureLSTM,
    entrenar_baselines,
    train_classical_baselines,
    entrenar_baselines_multisemilla,
    train_classical_baselines_multiseed,
    entrenar_lstm_standalone_semilla,
    train_lstm_standalone_seed,
)

# Quantum models
from modelos_cuanticos import (                              # noqa: F401
    _make_qnode,
    LstmQNN,
    make_thermodynamic_physics_loss,
    _loop_with_physics,
    entrenar_qml,
    train_qml,
    entrenar_qksvm,
    train_qksvm,
    entrenar_ablacion,
    train_ablation,
    QuantumTransferLearning,
    entrenar_qtl,
    train_qtl,
    entrenar_qnn_layerbylayer,
    train_qnn_layerbylayer,
)