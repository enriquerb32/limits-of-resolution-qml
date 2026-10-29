"""Unit tests for rank AUC and DeLong inference, checked against independent oracles."""

from __future__ import annotations

import numpy as np
import pytest

from qml_resolution import delong_components, delong_covariance, paired_delong, rank_auc

roc_auc_score = pytest.importorskip("sklearn.metrics").roc_auc_score


def _toy(n: int = 300, seeds: int = 1, seed: int = 0, ties: bool = False) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    y[:2] = [0, 1]
    s = 0.8 * y + rng.normal(size=(seeds, n))
    if ties:
        s = np.round(s, 1)
    return y, s


def _brute_force_delong(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """O(m n) DeLong from the Heaviside kernel, written independently of the rank form."""
    pos, neg = s[:, y == 1], s[:, y == 0]
    psi = (pos[:, :, None] > neg[:, None, :]) + 0.5 * (pos[:, :, None] == neg[:, None, :])
    v10, v01 = psi.mean(axis=2), psi.mean(axis=1)
    aucs = psi.mean(axis=(1, 2))
    cov = np.atleast_2d(np.cov(v10)) / pos.shape[1] + np.atleast_2d(np.cov(v01)) / neg.shape[1]
    return aucs, cov


@pytest.mark.parametrize("ties", [False, True])
def test_rank_auc_matches_sklearn(ties: bool) -> None:
    y, s = _toy(seeds=5, ties=ties)
    expected = [roc_auc_score(y, row) for row in s]
    np.testing.assert_allclose(rank_auc(y, s), expected, rtol=0, atol=1e-12)


def test_rank_auc_accepts_vector() -> None:
    y, s = _toy()
    assert rank_auc(y, s[0]).shape == (1,)


@pytest.mark.parametrize("ties", [False, True])
def test_delong_covariance_matches_brute_force(ties: bool) -> None:
    y, s = _toy(n=150, seeds=3, ties=ties)
    aucs, cov = delong_covariance(y, s)
    ref_aucs, ref_cov = _brute_force_delong(y, s)
    np.testing.assert_allclose(aucs, ref_aucs, atol=1e-12)
    np.testing.assert_allclose(cov, ref_cov, atol=1e-14)


def test_components_average_to_auc() -> None:
    y, s = _toy(seeds=4)
    comp = delong_components(y, s)
    np.testing.assert_allclose(comp.v10.mean(axis=1), comp.auc, atol=1e-12)
    np.testing.assert_allclose(comp.v01.mean(axis=1), comp.auc, atol=1e-12)


def test_seed_mean_components_give_mean_auc() -> None:
    """The mean of per-seed AUCs equals the AUC of the seed-averaged components."""
    y, s = _toy(seeds=15)
    comp = delong_components(y, s).seed_mean()
    assert comp.v10.mean() == pytest.approx(rank_auc(y, s).mean(), abs=1e-12)


def test_paired_single_seed_equals_covariance_formula() -> None:
    y, s = _toy(seeds=2)
    aucs, cov = delong_covariance(y, s)
    test = paired_delong(y, s[0], s[1])
    assert test.delta == pytest.approx(aucs[0] - aucs[1], abs=1e-14)
    assert test.se == pytest.approx(np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]), rel=1e-12)


def test_paired_is_antisymmetric() -> None:
    y, a = _toy(seeds=15, seed=1)
    _, b = _toy(seeds=10, seed=2)
    ab, ba = paired_delong(y, a, b), paired_delong(y, b, a)
    assert ab.delta == pytest.approx(-ba.delta, abs=1e-15)
    assert ab.se == pytest.approx(ba.se, rel=1e-12)
    assert ab.p_value == pytest.approx(ba.p_value, rel=1e-9)


def test_paired_identical_models() -> None:
    y, s = _toy(seeds=3)
    test = paired_delong(y, s, s)
    assert test.delta == 0.0
    assert test.p_value == pytest.approx(1.0)


def test_ci_contains_delta_and_widens_with_level() -> None:
    y, a = _toy(seeds=5, seed=3)
    _, b = _toy(seeds=5, seed=4)
    test = paired_delong(y, a, b)
    lo95, hi95 = test.ci(0.95)
    lo99, hi99 = test.ci(0.99)
    assert lo99 < lo95 < test.delta < hi95 < hi99


@pytest.mark.parametrize(
    ("y", "s", "match"),
    [
        (np.zeros(10), np.arange(10.0), "both classes"),
        (np.array([0, 1, 2]), np.arange(3.0), "binary"),
        (np.array([0, 1, 1]), np.arange(4.0), "shape"),
        (np.array([0, 1, 1]), np.array([0.0, np.nan, 1.0]), "non-finite"),
    ],
)
def test_invalid_inputs(y: np.ndarray, s: np.ndarray, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        rank_auc(y, s)
