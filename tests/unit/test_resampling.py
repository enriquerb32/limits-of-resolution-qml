"""Unit tests for cluster bootstrap and jackknife."""

from __future__ import annotations

import numpy as np
import pytest

from qml_resolution import (
    bell_mccaffrey_df,
    cluster_bootstrap_aucs,
    cluster_bootstrap_delta,
    cluster_robust_delta,
    jackknife_delta,
    paired_delong,
    percentile_ci,
    rank_auc,
)


def test_bell_mccaffrey_df_equal_and_unequal_clusters() -> None:
    assert bell_mccaffrey_df(np.full(19, 28)) == pytest.approx(18.0)
    assert bell_mccaffrey_df([48] * 5 + [24] * 14) < 18.0
    with pytest.raises(ValueError):
        bell_mccaffrey_df([10])


def test_cluster_robust_delta_matches_delong_and_orders_cr1_cr2(clustered: tuple[np.ndarray, ...]) -> None:
    y, a, b, g = clustered
    cr = cluster_robust_delta(y, a, b, g)
    dl = paired_delong(y, a, b)
    assert cr.delta == pytest.approx(dl.delta)
    assert cr.se_iid == pytest.approx(dl.se, rel=1e-6)
    assert cr.se_cr2 > 0 and cr.se_cr1 > 0
    lo, hi = cr.ci()
    assert lo < cr.delta < hi
    assert 0.0 <= cr.p_value <= 1.0
    assert cr.df == pytest.approx(bell_mccaffrey_df(np.bincount(g)))


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


def test_shared_seeds_resamples_the_common_component_once() -> None:
    # B = A + a small seed-independent shift: the seed effect is shared, so drawing the same seeds for both
    # cancels it and the joint interval is narrower than the one with independent seed draws.
    rng = np.random.default_rng(3)
    n, s = 400, 10
    y = rng.integers(0, 2, n)
    g = np.repeat(np.arange(20), n // 20)
    seed_effect = rng.normal(scale=1.0, size=(s, n))
    a = y + seed_effect
    b = a + 0.05 * rng.normal(size=n)
    indep = cluster_bootstrap_delta(y, a, b, g, n_boot=300, random_state=0, resample_seeds=True)
    joint = cluster_bootstrap_delta(y, a, b, g, n_boot=300, random_state=0, resample_seeds=True, shared_seeds=True)
    assert joint.std() < 0.5 * indep.std()
    with pytest.raises(ValueError, match="same number of seeds"):
        cluster_bootstrap_delta(y, a, b[:5], g, resample_seeds=True, shared_seeds=True)


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
