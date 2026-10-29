"""Unit tests for cluster bootstrap and jackknife."""

from __future__ import annotations

import numpy as np
import pytest

from qml_resolution import (
    cluster_bootstrap_aucs,
    cluster_bootstrap_delta,
    jackknife_delta,
    percentile_ci,
    rank_auc,
)


@pytest.fixture
def clustered() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    n, n_clusters = 400, 19
    clusters = np.repeat(np.arange(n_clusters), int(np.ceil(n / n_clusters)))[:n]
    y = rng.integers(0, 2, n)
    a = 0.9 * y + rng.normal(size=(15, n))
    b = 0.7 * y + rng.normal(size=(10, n))
    return y, a, b, clusters


def test_bootstrap_is_reproducible(clustered: tuple[np.ndarray, ...]) -> None:
    y, a, b, g = clustered
    r1 = cluster_bootstrap_delta(y, a, b, g, n_boot=50, random_state=3)
    r2 = cluster_bootstrap_delta(y, a, b, g, n_boot=50, random_state=np.random.default_rng(3))
    np.testing.assert_array_equal(r1, r2)


def test_bootstrap_of_identical_models_is_zero(clustered: tuple[np.ndarray, ...]) -> None:
    y, a, _, g = clustered
    np.testing.assert_array_equal(cluster_bootstrap_delta(y, a, a, g, n_boot=30), 0.0)


def test_bootstrap_centres_on_estimate(clustered: tuple[np.ndarray, ...]) -> None:
    y, a, b, g = clustered
    delta = rank_auc(y, a).mean() - rank_auc(y, b).mean()
    reps = cluster_bootstrap_delta(y, a, b, g, n_boot=400, random_state=1)
    assert abs(reps.mean() - delta) < 2 * reps.std() / np.sqrt(reps.size) + 0.005


def test_two_way_bootstrap_adds_seed_variance(clustered: tuple[np.ndarray, ...]) -> None:
    y, a, b, g = clustered
    one_way = cluster_bootstrap_delta(y, a, b, g, n_boot=400, random_state=1)
    two_way = cluster_bootstrap_delta(y, a, b, g, n_boot=400, random_state=1, resample_seeds=True)
    assert two_way.std() > one_way.std()


def test_bootstrap_aucs_shape_and_single_class_discard() -> None:
    y = np.array([0, 0, 0, 0, 1, 1])
    g = np.array([0, 0, 1, 1, 2, 2])  # resamples drawing only clusters 0-1 or only 2 are single-class
    s = np.arange(6.0)
    reps = cluster_bootstrap_aucs(y, s, g, n_boot=200, random_state=0)
    assert reps.shape[1] == 1
    assert 0 < reps.shape[0] < 200


def test_jackknife_matches_naive_loop(clustered: tuple[np.ndarray, ...]) -> None:
    y, a, b, g = clustered
    res = jackknife_delta(y, a, b, g)
    theta = []
    for c in np.unique(g):
        keep = g != c
        theta.append(rank_auc(y[keep], a[:, keep]).mean() - rank_auc(y[keep], b[:, keep]).mean())
    theta_arr = np.array(theta)
    k = theta_arr.size
    expected = np.sqrt((k - 1) / k * ((theta_arr - theta_arr.mean()) ** 2).sum())
    assert res.n_clusters == 19
    assert res.se == pytest.approx(expected, rel=1e-12)
    lo, hi = res.ci()
    assert lo < res.delta < hi


def test_percentile_ci() -> None:
    lo, hi = percentile_ci(np.arange(1001.0), level=0.9)
    assert (lo, hi) == pytest.approx((50.0, 950.0))
    with pytest.raises(ValueError):
        percentile_ci([], 0.95)


def test_requires_two_clusters() -> None:
    with pytest.raises(ValueError, match="two clusters"):
        jackknife_delta([0, 1, 0, 1], np.arange(4.0), np.arange(4.0), [0, 0, 0, 0])
