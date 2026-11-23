"""Error paths of the public API: invalid inputs must fail loudly, not return wrong numbers."""

from __future__ import annotations

import numpy as np
import pytest

from qml_resolution import (
    cluster_bootstrap_delta,
    hanley_mcneil_se,
    jackknife_delta,
    mde,
    paired_delong,
    percentile_ci,
    rank_auc,
)

Y = np.array([0, 1, 0, 1, 0, 1])
S = np.arange(6.0)
G = np.array([0, 0, 1, 1, 2, 2])


def test_labels_must_be_1d() -> None:
    with pytest.raises(ValueError, match="1-D"):
        rank_auc(Y.reshape(2, 3), S)


def test_clusters_must_match_rows() -> None:
    with pytest.raises(ValueError, match="clusters must have shape"):
        jackknife_delta(Y, S, S, G[:-1])


@pytest.mark.parametrize("level", [0.0, 1.0, 1.5])
def test_invalid_confidence_levels(level: float) -> None:
    with pytest.raises(ValueError, match="level"):
        paired_delong(Y, S, S[::-1]).ci(level)
    with pytest.raises(ValueError, match="level"):
        percentile_ci([0.1, 0.2], level)
    with pytest.raises(ValueError, match="level"):
        jackknife_delta(Y, S, S[::-1], G).ci(level)


def test_negative_standard_error() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        mde(-0.01)


def test_hanley_mcneil_needs_both_classes() -> None:
    with pytest.raises(ValueError, match="positive"):
        hanley_mcneil_se(0.7, 0, 10)


def test_jackknife_rejects_cluster_holding_a_whole_class() -> None:
    y = np.array([1, 1, 0, 0, 0, 0])
    with pytest.raises(ValueError, match="single class"):
        jackknife_delta(y, S, S[::-1], G)


def test_bootstrap_discards_single_class_resamples() -> None:
    y = np.array([1, 1, 0, 0, 0, 0])
    reps = cluster_bootstrap_delta(y, S, S[::-1], G, n_boot=200, random_state=0)
    assert 0 < reps.size < 200
