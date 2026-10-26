import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
pytest.importorskip("pennylane")
torch = pytest.importorskip("torch")


def test_entrenar_qksvm_runs_end_to_end():
    from modelos_cuanticos import entrenar_qksvm

    rng = np.random.default_rng(0)
    n_tr, n_te, d = 200, 60, 8
    X_all = rng.normal(size=(n_tr + n_te, d))
    y_all = (X_all[:, 0] + 0.5 * rng.normal(size=n_tr + n_te) > 0.8).astype(int)
    ds = SimpleNamespace(X_sk=X_all[:n_tr], y_sk=y_all[:n_tr], X_all=X_all, s2=n_tr,
                         y_te=torch.tensor(y_all[n_tr:]))
    res = entrenar_qksvm(ds)["QKSVM"]
    assert res["_prob"].shape == (n_te,)
    assert 0.0 <= res["auc"] <= 1.0
