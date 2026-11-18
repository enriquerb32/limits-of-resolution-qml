"""Cluster resampling for test sets whose rows are grouped (e.g. station-months by year).

All functions resample whole clusters with replacement and keep every row of a drawn
cluster, as in a block bootstrap with natural blocks. With few clusters (here 19
calendar years) cluster-based variance estimates are biased downward and unstable
(Cameron, Gelbach & Miller, 2008); report them next to DeLong rather than instead of it.

Random draws are made in a fixed order so that results are reproducible for a given
seed: per replicate, first the clusters, then (optionally) the seeds of model A and of
model B. Replicates whose resample contains only one class are discarded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import t as student_t

from qml_resolution._validation import as_clusters, as_labels, as_score_matrix
from qml_resolution.auc import _rank_auc_unchecked

RandomState = np.random.Generator | int | None


def _rng(random_state: RandomState) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def _cluster_rows(clusters: NDArray[np.generic]) -> list[NDArray[np.intp]]:
    return [np.flatnonzero(clusters == c) for c in np.unique(clusters)]


def cluster_bootstrap_aucs(
    y: ArrayLike,
    scores: ArrayLike,
    clusters: ArrayLike,
    n_boot: int = 2000,
    random_state: RandomState = 42,
) -> NDArray[np.float64]:
    """Bootstrap the AUC of each score vector by resampling clusters.

    Returns
    -------
    numpy.ndarray
        Matrix of shape ``(n_valid, k)``; ``n_valid <= n_boot`` after discarding
        single-class resamples.
    """
    labels = as_labels(y)
    mat = as_score_matrix(scores, labels.size)
    rows = _cluster_rows(as_clusters(clusters, labels.size))
    rng = _rng(random_state)
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(rows), size=len(rows))
        idx = np.concatenate([rows[j] for j in pick])
        yb = labels[idx]
        if yb.all() or not yb.any():
            continue
        out.append(_rank_auc_unchecked(yb, mat[:, idx]))
    return np.array(out).reshape(-1, mat.shape[0])


def cluster_bootstrap_delta(
    y: ArrayLike,
    scores_a: ArrayLike,
    scores_b: ArrayLike,
    clusters: ArrayLike,
    n_boot: int = 2000,
    random_state: RandomState = 0,
    resample_seeds: bool = False,
    shared_seeds: bool = False,
) -> NDArray[np.float64]:
    """Bootstrap replicates of the difference of seed-mean AUCs, ``A - B``.

    Both models are evaluated on the same resampled clusters (a paired bootstrap).
    With ``resample_seeds=True`` the seeds are also resampled (a two-way bootstrap over
    clusters and seeds), which adds seed variance to the interval. By default the seeds
    of the two models are drawn independently. Use ``shared_seeds=True`` when seed ``s``
    of A and seed ``s`` of B share a fitted component (for example, a head trained on the
    frozen network of the same run): the same seed indices are then drawn for both, so the
    shared component is resampled once. A common seed *number* in two unrelated
    frameworks does not by itself make the runs paired.

    Parameters
    ----------
    scores_a, scores_b:
        Scores of shape ``(n,)`` or ``(S, n)``; with ``shared_seeds`` both need the same ``S``.
    """
    labels = as_labels(y)
    mat_a = as_score_matrix(scores_a, labels.size, "scores_a")
    mat_b = as_score_matrix(scores_b, labels.size, "scores_b")
    rows = _cluster_rows(as_clusters(clusters, labels.size))
    rng = _rng(random_state)
    s_a, s_b = mat_a.shape[0], mat_b.shape[0]
    if shared_seeds and s_a != s_b:
        raise ValueError(f"shared_seeds needs the same number of seeds, got {s_a} and {s_b}")
    seeds_a, seeds_b = np.arange(s_a), np.arange(s_b)
    out = np.full(n_boot, np.nan)
    for b in range(n_boot):
        pick = rng.integers(0, len(rows), size=len(rows))
        idx = np.concatenate([rows[j] for j in pick])
        yb = labels[idx]
        if yb.all() or not yb.any():
            continue
        if resample_seeds:
            seeds_a = rng.integers(0, s_a, size=(s_a,))
            seeds_b = seeds_a if shared_seeds else rng.integers(0, s_b, size=(s_b,))
        auc_a = _rank_auc_unchecked(yb, mat_a[seeds_a][:, idx]).mean()
        auc_b = _rank_auc_unchecked(yb, mat_b[seeds_b][:, idx]).mean()
        out[b] = auc_a - auc_b
    result: NDArray[np.float64] = out[~np.isnan(out)]
    return result


def percentile_ci(replicates: ArrayLike, level: float = 0.95) -> tuple[float, float]:
    """Percentile interval of bootstrap replicates."""
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1)")
    reps = np.asarray(replicates, dtype=np.float64)
    if reps.size == 0:
        raise ValueError("no replicates")
    alpha = 100.0 * (1.0 - level) / 2.0
    lo, hi = np.percentile(reps, [alpha, 100.0 - alpha])
    return float(lo), float(hi)


def bell_mccaffrey_df(cluster_sizes: ArrayLike) -> float:
    """Bell-McCaffrey degrees of freedom of the CR2 variance of a mean.

    Working model: independent, equal-variance rows; design: intercept only. With equal
    cluster sizes the result is exactly ``G - 1``; unequal sizes give fewer.
    """
    n_g = np.asarray(cluster_sizes, dtype=np.float64)
    if n_g.ndim != 1 or n_g.size < 2 or np.any(n_g <= 0):
        raise ValueError("need at least two positive cluster sizes")
    h = n_g / n_g.sum()
    root = n_g / np.sqrt(1.0 - h)
    c = np.diag(n_g / (1.0 - h)) - np.outer(root, root) / n_g.sum()
    return float(np.trace(c) ** 2 / np.trace(c @ c))


@dataclass(frozen=True)
class ClusterRobustResult:
    """Analytic cluster-robust inference for a difference of seed-mean AUCs.

    The influence values of the paired DeLong statistic are summed within clusters.
    ``se_cr1`` uses the ``G/(G-1)`` correction; ``se_cr2`` the Bell-McCaffrey (CR2)
    adjustment ``S_g / sqrt(1 - n_g/N)``, with ``df`` Bell-McCaffrey degrees of freedom.
    """

    delta: float
    se_iid: float
    se_cr1: float
    se_cr2: float
    df: float

    def ci(self, level: float = 0.95) -> tuple[float, float]:
        """CR2 interval ``delta +/- t_df * se_cr2``."""
        if not 0.0 < level < 1.0:
            raise ValueError("level must be in (0, 1)")
        half = float(student_t.ppf(0.5 + level / 2.0, self.df)) * self.se_cr2
        return self.delta - half, self.delta + half

    @property
    def p_value(self) -> float:
        """Two-sided p-value of the CR2 t statistic with ``df`` degrees of freedom."""
        return float(2.0 * student_t.sf(abs(self.delta) / self.se_cr2, self.df))


def cluster_robust_delta(y: ArrayLike, scores_a: ArrayLike, scores_b: ArrayLike, clusters: ArrayLike) -> ClusterRobustResult:
    """CR1 and CR2 (Bell-McCaffrey) inference for the difference of seed-mean AUCs, ``A - B``."""
    from qml_resolution.auc import delong_components

    labels = as_labels(y)
    mat_a = as_score_matrix(scores_a, labels.size, "scores_a")
    mat_b = as_score_matrix(scores_b, labels.size, "scores_b")
    groups = as_clusters(clusters, labels.size)
    ca, cb = delong_components(labels, mat_a).seed_mean(), delong_components(labels, mat_b).seed_mean()
    delta = float(ca.auc[0] - cb.auc[0])
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    psi = np.empty(labels.size)
    psi[labels] = (ca.v10[0] - cb.v10[0] - delta) / n_pos
    psi[~labels] = (ca.v01[0] - cb.v01[0] - delta) / n_neg
    _, inv = np.unique(groups, return_inverse=True)
    s = np.bincount(inv, weights=psi)
    n_g = np.bincount(inv).astype(np.float64)
    k = s.size
    se_iid = float(np.sqrt((psi[labels] ** 2).sum() * n_pos / (n_pos - 1) + (psi[~labels] ** 2).sum() * n_neg / (n_neg - 1)))
    se_cr1 = float(np.sqrt(k / (k - 1) * (s ** 2).sum()))
    se_cr2 = float(np.sqrt(((s / np.sqrt(1.0 - n_g / labels.size)) ** 2).sum()))
    return ClusterRobustResult(delta=delta, se_iid=se_iid, se_cr1=se_cr1, se_cr2=se_cr2, df=bell_mccaffrey_df(n_g))


@dataclass(frozen=True)
class JackknifeResult:
    """Delete-one-cluster jackknife of a difference of seed-mean AUCs.

    Attributes
    ----------
    delta:
        Full-sample difference.
    se:
        Jackknife standard error, ``sqrt((G-1)/G * sum_g (theta_g - theta_bar)^2)``.
    replicates:
        Leave-one-cluster-out differences ``theta_g``, one per cluster.
    """

    delta: float
    se: float
    replicates: NDArray[np.float64]

    @property
    def n_clusters(self) -> int:
        """Number of clusters (jackknife replicates)."""
        return int(self.replicates.size)

    def ci(self, level: float = 0.95) -> tuple[float, float]:
        """Interval ``delta +/- t_{G-1} * se``."""
        if not 0.0 < level < 1.0:
            raise ValueError("level must be in (0, 1)")
        half = float(student_t.ppf(0.5 + level / 2.0, self.n_clusters - 1)) * self.se
        return self.delta - half, self.delta + half


def jackknife_delta(y: ArrayLike, scores_a: ArrayLike, scores_b: ArrayLike, clusters: ArrayLike) -> JackknifeResult:
    """Delete-one-cluster jackknife of the difference of seed-mean AUCs, ``A - B``."""
    labels = as_labels(y)
    mat_a = as_score_matrix(scores_a, labels.size, "scores_a")
    mat_b = as_score_matrix(scores_b, labels.size, "scores_b")
    groups = as_clusters(clusters, labels.size)
    reps = []
    for g in np.unique(groups):
        keep = groups != g
        yk = labels[keep]
        if yk.all() or not yk.any():
            raise ValueError(f"removing cluster {g!r} leaves a single class")
        reps.append(_rank_auc_unchecked(yk, mat_a[:, keep]).mean() - _rank_auc_unchecked(yk, mat_b[:, keep]).mean())
    theta = np.array(reps)
    n_groups = theta.size
    se = float(np.sqrt((n_groups - 1) / n_groups * np.sum((theta - theta.mean()) ** 2)))
    delta = float(_rank_auc_unchecked(labels, mat_a).mean() - _rank_auc_unchecked(labels, mat_b).mean())
    return JackknifeResult(delta=delta, se=se, replicates=theta)
