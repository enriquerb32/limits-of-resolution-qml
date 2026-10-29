"""The library must reproduce the archived statistics of the paper from the archived test probabilities.

These tests tie `qml_resolution` to the numbers in the manuscript: if a refactor changes a
DeLong standard error, a jackknife or a bootstrap interval, they fail.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from qml_resolution import cluster_bootstrap_delta, jackknife_delta, paired_delong, percentile_ci

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
CORE = RESULTS / "json" / "results_hybrids_trainonly_fixet0_p60_28.1_probs.npz"
RERUN = RESULTS / "json" / "results_rerun_train_only_canonical_probs.npz"


@pytest.fixture(scope="module")
def archive() -> dict[str, np.ndarray]:
    core, rerun = np.load(CORE), np.load(RERUN)
    assert np.array_equal(core["y_te"], rerun["y_te"])
    models = {k[: -len("_probs")]: core[k] for k in core.files if k.endswith("_probs")}
    models |= {k[len("train_only_") : -len("_probs")]: rerun[k] for k in rerun.files if k.startswith("train_only_")}
    models["y"] = core["y_te"].astype(int)
    models["years"] = np.load(RESULTS / "years_te.npy")
    return models


@pytest.fixture(scope="module")
def exp10() -> dict[str, dict[str, float]]:
    pairs: dict[str, dict[str, float]] = json.loads(
        (RESULTS / "json" / "results_exp10_paired_ci_fixet0.json").read_text()
    )["pairs"]
    return pairs


PAIRS = ["RF-XGB", "RF-LSTM", "XGB-LSTM", "RF_2273-LSTM_QNN", "CTRL_LINEAR4D-LSTM_QNN", "PureLSTM-LSTM_QNN"]


def test_test_set_shape(archive: dict[str, np.ndarray]) -> None:
    assert archive["y"].shape == (528,)
    assert archive["y"].sum() == 217
    assert np.unique(archive["years"]).size == 19
    assert archive["LSTM_QNN"].shape == (15, 528)


@pytest.mark.parametrize("pair", PAIRS)
def test_paired_delong_reproduces_table(pair: str, archive: dict[str, np.ndarray], exp10: dict) -> None:
    a, b = pair.split("-")
    test = paired_delong(archive["y"], archive[a], archive[b])
    ref = exp10[pair]
    assert test.delta == pytest.approx(ref["delta"], abs=1e-12)
    assert test.se == pytest.approx(ref["se_delong"], rel=1e-10)
    assert test.p_value == pytest.approx(ref["p_delong"], rel=1e-8)


def test_block_bootstrap_reproduces_first_pair_exactly(archive: dict[str, np.ndarray], exp10: dict) -> None:
    """exp10 draws the RF-XGB year-block bootstrap first, from default_rng(0) with B=2000."""
    reps = cluster_bootstrap_delta(archive["y"], archive["RF"], archive["XGB"], archive["years"], 2000, 0)
    np.testing.assert_allclose(percentile_ci(reps), exp10["RF-XGB"]["ci_block"], rtol=0, atol=1e-15)
    assert reps.std(ddof=1) == pytest.approx(exp10["RF-XGB"]["se_block"], rel=1e-12)


@pytest.mark.parametrize(
    ("pair", "key"),
    [
        ("RF-LSTM", "C4_year_jackknife"),
        ("RF_2273-LSTM_QNN", "C4_year_jackknife"),
        ("CTRL_LINEAR4D-LSTM_QNN", "C4_year_jackknife"),
        ("RF_2273-LSTM_QTL", "C4_year_jackknife_extra_hybrids"),
    ],
)
def test_jackknife_reproduces_archive(pair: str, key: str, archive: dict[str, np.ndarray]) -> None:
    ref = json.loads((RESULTS / "local_exp16_block_c_fixet0.json").read_text())[key][pair]
    a, b = pair.split("-")
    res = jackknife_delta(archive["y"], archive[a], archive[b], archive["years"])
    assert res.delta == pytest.approx(ref["delta"], abs=1e-12)
    assert res.se == pytest.approx(ref["se_jackknife"], rel=1e-10)
    # exp16 used the rounded quantile t_{0.975,18} = 2.101; the library uses the exact 2.10092.
    np.testing.assert_allclose(res.ci(), ref["ci95_t18"], atol=1e-6)
