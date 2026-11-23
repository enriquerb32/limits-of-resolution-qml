"""Quantum claims of the v2 manuscript about the circuits and their classical controls.

* The re-uploading circuit of ``LSTM+QNN`` (``RY(x) RZ(x)`` per layer, two layers) has per-coordinate
  frequencies exactly ``{0, ±1, ..., ±4}``: the top frequency is reached, not only bounded.
* The ``R_Y``-embedding head of ``LSTM+QTL`` (``AngleEmbedding(Y)`` + ``StronglyEntanglingLayers``)
  lies *exactly* in the span of the 81-function tensor basis ``prod_j {1, cos q_j, sin q_j}`` used by
  ``QTL tensor-81`` and ``Full Fourier-1``, whatever the weights; the separable basis does not span it.
* Parameter counts of every control equal the sizes printed in the manuscript (Table 5 and Section 2).

The checks use the model classes of ``code/`` that produced the archived results.
"""

from __future__ import annotations

import math
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.quantum
torch = pytest.importorskip("torch")
qml = pytest.importorskip("pennylane")

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "paper" / "main.tex"
SUPP = ROOT / "paper" / "supp.tex"
N_Q = 4


@pytest.fixture(scope="module")
def code() -> Iterator[SimpleNamespace]:
    """Import the v2 model code; it sets float64 as the default dtype, which is restored afterwards."""
    dtype = torch.get_default_dtype()
    for p in (ROOT / "code", ROOT / "code" / "local_experiments"):
        sys.path.insert(0, str(p))
    import exp27_controls
    import exp30_fourier_full
    import qml_core_trainonly
    from modelos_cuanticos import _make_qnode

    yield SimpleNamespace(e27=exp27_controls, e30=exp30_fourier_full, core=qml_core_trainonly, qnode=_make_qnode)
    torch.set_default_dtype(dtype)


def n_trainable(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def tensor_basis(q: np.ndarray, degree: int) -> np.ndarray:
    """``prod_j {1, cos k q_j, sin k q_j}_{k <= degree}``, built independently of the code under test."""
    cols = [np.ones_like(q)]
    for k in range(1, degree + 1):
        cols += [np.cos(k * q), np.sin(k * q)]
    per_coord = np.stack(cols, axis=-1)  # (n, N_Q, 2d+1)
    out = per_coord[:, 0]
    for j in range(1, q.shape[1]):
        out = np.einsum("na,nb->nab", out, per_coord[:, j]).reshape(len(q), -1)
    return out


def relative_residual(features: np.ndarray, target: np.ndarray) -> float:
    coef, *_ = np.linalg.lstsq(features, target, rcond=None)
    return float(np.linalg.norm(features @ coef - target) / np.linalg.norm(target - target.mean(0)))


# ---------------------------------------------------------------- LSTM+QNN: spectrum is exactly {0, ..., 4}
@pytest.mark.parametrize("wire", range(N_Q))
def test_reuploading_circuit_reaches_frequency_four(code: SimpleNamespace, wire: int) -> None:
    circuit = code.qnode(N_Q, 2)
    gen = torch.Generator().manual_seed(1)
    weights = (torch.rand(2, N_Q, 3, generator=gen, dtype=torch.float64) * 2 - 1) * math.pi
    n_grid = 32
    x = torch.tensor([0.4, 1.3, 2.2, 0.9], dtype=torch.float64).repeat(n_grid, 1)
    x[:, wire] = torch.arange(n_grid, dtype=torch.float64) * 2 * math.pi / n_grid
    out = torch.stack(list(circuit(weights, x)), dim=-1).detach().numpy()
    spectrum = np.abs(np.fft.rfft(out, axis=0)) / n_grid
    assert spectrum[5:].max() < 1e-12  # nothing above 4
    assert spectrum[4].max() > 1e-4  # the top frequency of "four rotation gates per angle" is present


# ---------------------------------------------------------------- LSTM+QTL head: exact 81-function span
@pytest.fixture(scope="module")
def qtl_readouts(code: SimpleNamespace) -> tuple[np.ndarray, np.ndarray]:
    encoder = code.core.PureLSTM(35)
    head = code.e27.TransferHead(encoder, in_embed=16 + 35, head="quantum", n_q=N_Q, n_l=2)
    rng = np.random.default_rng(0)
    q = rng.uniform(0, math.pi, size=(400, N_Q))  # the adapter maps into [0, pi]
    with torch.no_grad():
        head.W.uniform_(-math.pi, math.pi, generator=torch.Generator().manual_seed(2))  # far from the init
        z = head.circ(head.W, torch.tensor(q))
    return q, torch.stack(list(z), dim=-1).numpy()


def test_qtl_head_is_spanned_by_81_tensor_functions(qtl_readouts: tuple[np.ndarray, np.ndarray]) -> None:
    q, z = qtl_readouts
    basis = tensor_basis(q, degree=1)
    assert basis.shape[1] == 3**N_Q == 81
    assert relative_residual(basis, z) < 1e-10


def test_separable_basis_does_not_span_qtl_head(qtl_readouts: tuple[np.ndarray, np.ndarray]) -> None:
    q, z = qtl_readouts
    separable = np.column_stack([np.ones(len(q)), np.cos(q), np.sin(q)])  # 9 functions, no products
    assert relative_residual(separable, z) > 1e-2


def test_tensor1_control_uses_the_same_basis(code: SimpleNamespace) -> None:
    """The QTL tensor-81 control and Full Fourier-1 build the basis tested above (up to column order)."""
    rng = np.random.default_rng(3)
    q = rng.uniform(0, math.pi, size=(50, N_Q))
    ref = tensor_basis(q, degree=1)
    full1 = code.e30.FourierFull4D(in_dim=35, n_q=N_Q, degree=1).basis(torch.tensor(q)).numpy()
    np.testing.assert_allclose(np.sort(full1, axis=1), np.sort(ref, axis=1), atol=1e-12)
    assert relative_residual(full1, ref) < 1e-12 and relative_residual(ref, full1) < 1e-12


# ---------------------------------------------------------------- parameter counts against the manuscript
def _printed_sizes(path: Path, label: str) -> dict[str, int]:
    """Map sizes printed in a model table, e.g. ``Fourier-2 (68)`` or ``Full Fourier-1 (81 features)``."""
    text = path.read_text(encoding="utf-8")
    table = text[text.index(f"\\label{{{label}}}") :]
    table = table[: table.index("\\end{table")]
    return {
        m[0].strip(): int(m[1]) for m in re.findall(r"^\s*([A-Za-z0-9\- ]+?) \((\d+)(?: features)?\) &", table, re.M)
    }


def test_control_sizes_match_table(code: SimpleNamespace) -> None:
    # every control is in the complete table of the supplement (the former main-text subset, tab:case, is there);
    # the main text quotes the parameter count of Linear-4D
    sizes = _printed_sizes(SUPP, "tab:case_all")
    assert "\\label{tab:case}" not in TEX.read_text(encoding="utf-8")
    assert "2{,}609 parameters against 2{,}613" in TEX.read_text(encoding="utf-8")
    lin = code.core.LinearControl4D(in_dim=35)
    backbone = n_trainable(lin) - 20
    assert n_trainable(lin) == 2609  # Section 2: "2,609 parameters against 2,613"
    assert sizes["Linear-4D control"] == 20
    for d in (1, 2, 3, 4):
        model = code.e27.FourierControl4D(in_dim=35, n_q=N_Q, degree=d)
        assert n_trainable(model) - backbone == sizes[f"Fourier-{d}"] == n_trainable(model.ctrl)
    for d, n_feat in ((1, 81), (2, 625)):
        model = code.e30.FourierFull4D(in_dim=35, n_q=N_Q, degree=d)
        assert model.ctrl.in_features == sizes[f"Full Fourier-{d}"] == n_feat == (2 * d + 1) ** N_Q


def test_hybrid_and_frozen_circuits(code: SimpleNamespace) -> None:
    trained = code.core.HybridQNNControl(in_dim=35, mode="trained")
    assert n_trainable(trained) == 2613 and trained.W.numel() == 24
    for mode in ("frozen_random", "frozen_identity"):
        frozen = code.core.HybridQNNControl(in_dim=35, mode=mode)
        assert n_trainable(frozen) == 2613 - 24 and not frozen.W.requires_grad
    assert torch.count_nonzero(code.core.HybridQNNControl(in_dim=35, mode="frozen_identity").W) == 0
