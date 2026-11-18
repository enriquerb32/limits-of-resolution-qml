"""Unit tests for the protocol report (qml_resolution.report)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from qml_resolution import report as rp

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def data() -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(11)
    n, k = 380, 19
    g = np.repeat(np.arange(k), n // k)
    y = rng.integers(0, 2, n)
    base = rng.normal(size=(5, n))
    scores = {"A": 0.9 * y + base, "B": 0.6 * y + rng.normal(size=(5, n)),
              "A2": 0.9 * y + base + 0.1 * rng.normal(size=n)}
    return y, scores, g


def test_holm_matches_definition() -> None:
    assert rp.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert rp.holm([0.5]) == [0.5]


def test_power_and_mde_are_consistent() -> None:
    se, df = 0.01, 15.9
    m = rp.mde_t(se, df)
    assert rp.power_t(m, se, df) == pytest.approx(0.80, abs=0.01)
    assert rp.power_t(0.0, se, df) == pytest.approx(0.05, abs=1e-9)
    with pytest.raises(ValueError):
        rp.power_t(0.01, 0.0, df)


def test_report_structure_holm_and_shared(data: tuple) -> None:
    y, scores, g = data
    a = rp.Assumptions(n_boot=200, power_deltas=(0.02,), n_target=4 * y.size)
    res = rp.report(y, scores, g, [("A", "B"), ("A", "A2")], shared=[("A", "A2")], family=[("A", "B"), ("A", "A2")],
                    assumptions=a)
    ab, aa = res["pairs"]["A-B"], res["pairs"]["A-A2"]
    assert ab["seeds_resampled"] == "independent" and aa["seeds_resampled"] == "joint"
    assert ab["p_holm"] >= ab["p_primary"] and ab["resolved_after_holm"] in (True, False)
    assert ab["ci_primary"][0] < ab["delta"] < ab["ci_primary"][1]
    pw = ab["power"]["0.02"]
    assert pw[f"at_n_{4 * y.size}"] > pw["at_this_test_set"]          # a larger test set has more power
    assert res["n_clusters"] == 19
    with pytest.raises(KeyError):
        rp.report(y, scores, g, [("A", "Z")])
    with pytest.raises(KeyError):
        rp.report(y, scores, g, [("A", "B")], family=[("A", "A2")])


def test_cli_writes_json(tmp_path: Path, data: tuple) -> None:
    y, scores, g = data
    f = tmp_path / "s.npz"
    np.savez(f, y=y, years=g, **scores)
    out = tmp_path / "r.json"
    assert rp.main([str(f), "--clusters", "years", "--pair", "A:B", "--n-boot", "50", "--out", str(out)]) == 0
    assert "A-B" in json.loads(out.read_text())["pairs"]


def test_report_reproduces_the_paper_on_the_archive() -> None:
    """Table 6, RF - LSTM+QNN: the tool gives the primary values printed in the paper."""
    h = ROOT / "results" / "json" / "results_hybrids_trainonly_fixet0_p60_28.1_probs.npz"
    c = ROOT / "results" / "json" / "results_rerun_train_only_canonical_probs.npz"
    meta = ROOT / "results" / "test_meta.csv"
    if not (h.exists() and c.exists() and meta.exists()):
        pytest.skip("archived predictions not available")
    import pandas as pd

    zh, zc = np.load(h), np.load(c)
    years = pd.to_datetime(pd.read_csv(meta)["fecha"]).dt.year.values
    y = zh["y_te"].astype(int)
    r = rp.compare_pair(y, zc["train_only_RF_probs"], zh["LSTM_QNN_probs"], years, "RF-LSTM_QNN",
                        assumptions=rp.Assumptions(n_boot=50))
    summary = json.loads((ROOT / "results" / "local_exp42_v2_summary.json").read_text())["pairs"]["RF-LSTM_QNN"]
    assert r.delta == pytest.approx(summary["gap"], abs=1e-12)
    assert r.p_primary == pytest.approx(summary["p_cr2"], abs=1e-9)
    assert r.ci_primary == pytest.approx(tuple(summary["ci_cr2"]), abs=1e-9)
    assert r.mde_delong == pytest.approx(summary["mde_iid"], rel=1e-3)
