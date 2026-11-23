"""The library must reproduce the archived statistics of both versions of the paper.

These tests tie `qml_resolution` to the archived numbers: if a refactor changes a DeLong standard
error, a jackknife or a bootstrap interval, they fail.

* v2 (``results/``, the current manuscript): references written by ``exp42_v2_summary.py`` and
  ``exp43_v2_tables.py``, which implement DeLong and the jackknife independently of the library.
* v1 (``results_v1/``, tag ``v1``): references written by the v1 scripts ``exp10`` and ``exp16``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from qml_resolution import cluster_bootstrap_delta, jackknife_delta, mde, paired_delong, percentile_ci, rank_auc

ROOT = Path(__file__).resolve().parents[2]


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# ====================================================================== v2
V2 = ROOT / "results"


@pytest.fixture(scope="module")
def v2() -> dict[str, np.ndarray]:
    core = np.load(V2 / "json" / "results_core_hybrids_controls_fixet0_p60_28.1_probs.npz")
    rerun = np.load(V2 / "json" / "results_rerun_train_only_canonical_probs.npz")
    clim = np.load(V2 / "json" / "results_exp28_orig_probs.npz")
    assert np.array_equal(core["y_te"], rerun["y_te"]) and np.array_equal(core["y_te"], clim["y_te"])
    models = {k[: -len("_probs")]: core[k] for k in core.files if k.endswith("_probs")}
    # same model names as exp42: tuned trees from the rerun archive, corrected transfer model (exp27)
    models |= {"RF": rerun["train_only_RF_probs"], "XGB": rerun["train_only_XGB_probs"]}
    models |= {"RF_untuned": rerun["train_only_RF_untuned_probs"], "XGB_untuned": rerun["train_only_XGB_untuned_probs"]}
    models |= {"LSTM_QTL": core["QTL_FIXED_probs"], "CLIM": clim["clim_te"][None]}
    models["y"] = core["y_te"].astype(int)
    models["years"] = np.load(V2 / "years_te.npy")
    return models


V2_SUMMARY = _json(V2 / "local_exp42_v2_summary.json")
V2_PAIRS = sorted(k for k, v in V2_SUMMARY["pairs"].items() if v["se_iid"] > 0)


def test_v2_test_set(v2: dict[str, np.ndarray]) -> None:
    with (V2 / "test_meta.csv").open(encoding="utf-8", newline="") as fh:
        meta = list(csv.DictReader(fh))  # columns: ciudad, fecha (YYYY-MM-DD), y_te
    assert v2["y"].shape == (529,)
    assert v2["y"].sum() == 215
    assert np.array_equal(v2["y"], [int(r["y_te"]) for r in meta])
    assert np.array_equal(v2["years"], [int(r["fecha"][:4]) for r in meta])
    assert np.unique(v2["years"]).size == 19
    assert v2["LSTM_QNN"].shape == (15, 529)


@pytest.mark.parametrize("pair", V2_PAIRS)
def test_v2_paired_delong_reproduces_exp42(pair: str, v2: dict[str, np.ndarray]) -> None:
    a, b = pair.split("-")
    ref = V2_SUMMARY["pairs"][pair]
    test = paired_delong(v2["y"], v2[a], v2[b])
    assert test.delta == pytest.approx(ref["gap"], abs=1e-12)
    assert test.se == pytest.approx(ref["se_iid"], rel=1e-10)
    assert test.p_value == pytest.approx(ref["p_delong"], rel=1e-8)
    # exp42 uses the rounded factor 2.80; the library uses z_{0.975} + z_{0.80} = 2.8016
    assert mde(test.se) == pytest.approx(ref["mde_iid"], rel=1e-3)


@pytest.mark.parametrize("model", sorted(V2_SUMMARY["models"]))
def test_v2_seed_mean_auc(model: str, v2: dict[str, np.ndarray]) -> None:
    if model in ("QTL_EXTRACTOR", "QTL_EXTRACTOR27", "LSTM_QTL_first", "RF_2273"):
        pytest.skip("alias of another archived model")
    ref = V2_SUMMARY["models"][model]
    aucs = np.array([rank_auc(v2["y"], seed) for seed in v2[model]])
    assert aucs.size == ref["n_seeds"]
    assert aucs.mean() == pytest.approx(ref["mean"], abs=1e-12)
    assert (aucs.std(ddof=1) if aucs.size > 1 else 0.0) == pytest.approx(ref["sd"], abs=1e-12)


@pytest.mark.parametrize("pair", ["RF-LSTM_QNN", "RF-PureLSTM"])
def test_v2_jackknife_reproduces_exp43(pair: str, v2: dict[str, np.ndarray]) -> None:
    ref = _json(V2 / "local_exp43_v2_tables.json")["jackknife"][pair]
    a, b = pair.split("-")
    res = jackknife_delta(v2["y"], v2[a], v2[b], v2["years"])
    assert res.se == pytest.approx(ref, rel=1e-10)


# ====================================================================== v1
V1 = ROOT / "results_v1"


@pytest.fixture(scope="module")
def v1() -> dict[str, np.ndarray]:
    core = np.load(V1 / "json" / "results_hybrids_trainonly_fixet0_p60_28.1_probs.npz")
    rerun = np.load(V1 / "json" / "results_rerun_train_only_canonical_probs.npz")
    assert np.array_equal(core["y_te"], rerun["y_te"])
    models = {k[: -len("_probs")]: core[k] for k in core.files if k.endswith("_probs")}
    models |= {k[len("train_only_") : -len("_probs")]: rerun[k] for k in rerun.files if k.startswith("train_only_")}
    models["y"] = core["y_te"].astype(int)
    models["years"] = np.load(V1 / "years_te.npy")
    return models


@pytest.fixture(scope="module")
def exp10() -> dict[str, dict[str, float]]:
    pairs: dict[str, dict[str, float]] = _json(V1 / "json" / "results_exp10_paired_ci_fixet0.json")["pairs"]
    return pairs


V1_PAIRS = ["RF-XGB", "RF-LSTM", "XGB-LSTM", "RF_2273-LSTM_QNN", "CTRL_LINEAR4D-LSTM_QNN", "PureLSTM-LSTM_QNN"]


def test_v1_test_set(v1: dict[str, np.ndarray]) -> None:
    assert v1["y"].shape == (528,)
    assert v1["y"].sum() == 217
    assert np.unique(v1["years"]).size == 19
    assert v1["LSTM_QNN"].shape == (15, 528)


@pytest.mark.parametrize("pair", V1_PAIRS)
def test_v1_paired_delong_reproduces_table(pair: str, v1: dict[str, np.ndarray], exp10: dict) -> None:
    a, b = pair.split("-")
    test = paired_delong(v1["y"], v1[a], v1[b])
    ref = exp10[pair]
    assert test.delta == pytest.approx(ref["delta"], abs=1e-12)
    assert test.se == pytest.approx(ref["se_delong"], rel=1e-10)
    assert test.p_value == pytest.approx(ref["p_delong"], rel=1e-8)


def test_v1_block_bootstrap_reproduces_first_pair_exactly(v1: dict[str, np.ndarray], exp10: dict) -> None:
    """exp10 draws the RF-XGB year-block bootstrap first, from default_rng(0) with B=2000."""
    reps = cluster_bootstrap_delta(v1["y"], v1["RF"], v1["XGB"], v1["years"], 2000, 0)
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
def test_v1_jackknife_reproduces_archive(pair: str, key: str, v1: dict[str, np.ndarray]) -> None:
    ref = _json(V1 / "local_exp16_block_c_fixet0.json")[key][pair]
    a, b = pair.split("-")
    res = jackknife_delta(v1["y"], v1[a], v1[b], v1["years"])
    assert res.delta == pytest.approx(ref["delta"], abs=1e-12)
    assert res.se == pytest.approx(ref["se_jackknife"], rel=1e-10)
    # exp16 used the rounded quantile t_{0.975,18} = 2.101; the library uses the exact 2.10092.
    np.testing.assert_allclose(res.ci(), ref["ci95_t18"], atol=1e-6)
