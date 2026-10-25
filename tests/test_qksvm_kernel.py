"""
tests/test_qksvm_kernel.py - Verify equivalence between PennyLane circuit and analytical QKSVM kernel.
"""
import numpy as np

try:
    import pytest
except ImportError:
    pytest = None

try:
    import pennylane as qml
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False


def test_analytical_kernel_matches_circuit():
    """Verify that prod(cos((x1 - x2)/2)^2) equals |<psi(x2)|psi(x1)>|^2."""
    if not HAS_PENNYLANE:
        if pytest:
            pytest.skip("PennyLane not installed")
        else:
            return

    N_Q = 4
    dev = qml.device("default.qubit", wires=N_Q)

    @qml.qnode(dev)
    def circuit(x1, x2):
        qml.AngleEmbedding(x1, wires=range(N_Q), rotation="Y")
        qml.adjoint(qml.AngleEmbedding)(x2, wires=range(N_Q), rotation="Y")
        return qml.probs(wires=range(N_Q))

    def analytical_kernel(x1, x2):
        return np.prod(np.cos((x1 - x2) * 0.5) ** 2)

    rng = np.random.default_rng(42)
    for _ in range(5):
        x1 = rng.uniform(0, np.pi, size=N_Q)
        x2 = rng.uniform(0, np.pi, size=N_Q)

        prob_state_zero = circuit(x1, x2)[0]
        analytic_val = analytical_kernel(x1, x2)

        np.testing.assert_allclose(analytic_val, prob_state_zero, atol=1e-6)


if __name__ == "__main__":
    test_analytical_kernel_matches_circuit()
    print("QKSVM analytical kernel test passed successfully.")
