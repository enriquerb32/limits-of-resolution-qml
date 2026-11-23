"""Cluster resampling for test sets whose rows are grouped (e.g. station-months by year).

All functions resample whole clusters with replacement and keep every row of a drawn
cluster, as in a block bootstrap with natural blocks. With few clusters (here 19
calendar years) cluster-based variance estimates are biased downward and unstable
(Cameron, Gelbach & Miller, 2008); report them next to DeLong rather than instead of it.

Random draws are made in a fixed order so that results are reproducible for a given
seed: per replicate, first the clusters, then (optionally) the seeds of model A and of
model B. Replicates whose resample contains only one class are discarded.

Implementation note: the AUC of a resample depends on the scores only through the
Mann-Whitney counts between clusters (see :class:`_ClusterAUC`), which are computed once.
Each replicate then costs ``O(k G^2)`` for ``G`` clusters instead of a rank of every
resampled row, ``O(k n log n)``, with bit-identical results.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import t as student_t

from qml_resolution._validation import as_clusters, as_labels, as_paired, as_score_matrix, check_open_unit
from qml_resolution.auc import _rank_auc_unchecked, delong_components

RandomState = np.random.Generator | int | None

# The pairwise-count path stores k * G^2 numbers and costs O(k G^2) per replicate; ranking the
# resampled rows costs O(k n log n). Use counts while G^2 stays within this multiple of n.
_MAX_G2_PER_ROW = 16


def _rng(random_state: RandomState) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


class _Clusters:
    """Labels of a test set indexed by cluster, for repeated resampling of whole clusters."""

    def __init__(self, labels: NDArray[np.bool_], clusters: NDArray[np.generic]) -> None:
        self.labels = labels
        self.names, self.index = np.unique(clusters, return_inverse=True)
        self.size = int(self.names.size)
        self.n_pos = np.bincount(self.index, weights=labels, minlength=self.size)
        self.n_neg = np.bincount(self.index, minlength=self.size) - self.n_pos
        self._rows: list[NDArray[np.intp]] | None = None

    @property
    def use_pair_counts(self) -> bool:
        return self.size * self.size <= _MAX_G2_PER_ROW * self.labels.size

    def rows(self, pick: NDArray[np.intp]) -> NDArray[np.intp]:
        """Row indices of the drawn clusters, concatenated in draw order."""
        if self._rows is None:
            self._rows = [np.flatnonzero(self.index == g) for g in range(self.size)]
        return np.concatenate([self._rows[j] for j in pick])

    def counts(self, pick: NDArray[np.intp]) -> NDArray[np.float64]:
        """Count how many times each cluster is drawn."""
        return np.bincount(pick, minlength=self.size).astype(np.float64)

    def draws(self, rng: np.random.Generator, n_boot: int) -> Iterator[NDArray[np.intp]]:
        """Yield the clusters drawn in each replicate that contains both classes.

        One draw of ``G`` cluster indices is made per replicate, in order, whether or not it is kept.
        """
        for _ in range(n_boot):
            pick = rng.integers(0, self.size, size=self.size)
            c = self.counts(pick)
            if c @ self.n_pos == 0 or c @ self.n_neg == 0:
                continue
            yield pick


class _ClusterAUC:
    """AUCs of the rows of a score matrix on any multiset of whole clusters.

    A resample that draws cluster ``g`` ``c_g`` times has the Mann-Whitney statistic
    ``U = sum_{g,h} c_g c_h U[g, h]``, where ``U[g, h]`` counts the (positive of ``g``,
    negative of ``h``) pairs that are ordered correctly, ties counting 1/2. Every term is a
    multiple of 1/2, so the sums are exact in floating point and the AUC ``U / (n_pos n_neg)``
    is bit-identical to ranking the resampled rows. When ``G^2`` is large relative to ``n``
    the resampled rows are ranked directly instead.
    """

    def __init__(self, design: _Clusters, mat: NDArray[np.float64]) -> None:
        self.design = design
        self.mat = mat
        self.u = self._pair_counts(design, mat) if design.use_pair_counts else None

    @staticmethod
    def _pair_counts(design: _Clusters, mat: NDArray[np.float64]) -> NDArray[np.float64]:
        pos, neg = design.labels, ~design.labels
        pos_cluster, neg_cluster = design.index[pos], design.index[neg]
        mat_pos, mat_neg = mat[:, pos], mat[:, neg]
        out = np.empty((mat.shape[0], design.size, design.size))
        for h in range(design.size):
            neg_h = np.sort(mat_neg[:, neg_cluster == h], axis=1)
            for r, (row_pos, row_neg) in enumerate(zip(mat_pos, neg_h, strict=True)):
                # 2 * (#negatives below) + (#ties) = left + right insertion points
                twice = np.searchsorted(row_neg, row_pos, "left") + np.searchsorted(row_neg, row_pos, "right")
                out[r, :, h] = 0.5 * np.bincount(pos_cluster, weights=twice, minlength=design.size)
        return out

    def __call__(self, pick: NDArray[np.intp]) -> NDArray[np.float64]:
        """AUC of each row on the clusters ``pick`` (indices into the sorted cluster labels)."""
        if self.u is None:
            idx = self.design.rows(pick)
            return _rank_auc_unchecked(self.design.labels[idx], self.mat[:, idx])
        c = self.design.counts(pick)
        auc: NDArray[np.float64] = (self.u @ c @ c) / ((c @ self.design.n_pos) * (c @ self.design.n_neg))
        return auc


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
    design = _Clusters(labels, as_clusters(clusters, labels.size))
    auc = _ClusterAUC(design, mat)
    out = [auc(pick) for pick in design.draws(_rng(random_state), n_boot)]
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
    labels, mat_a, mat_b, groups = as_paired(y, scores_a, scores_b, clusters)
    s_a, s_b = mat_a.shape[0], mat_b.shape[0]
    if shared_seeds and s_a != s_b:
        raise ValueError(f"shared_seeds needs the same number of seeds, got {s_a} and {s_b}")
    design = _Clusters(labels, groups)
    auc_a, auc_b = _ClusterAUC(design, mat_a), _ClusterAUC(design, mat_b)
    rng = _rng(random_state)
    seeds_a, seeds_b = np.arange(s_a), np.arange(s_b)
    out = []
    for pick in design.draws(rng, n_boot):
        if resample_seeds:
            seeds_a = rng.integers(0, s_a, size=(s_a,))
            seeds_b = seeds_a if shared_seeds else rng.integers(0, s_b, size=(s_b,))
        out.append(auc_a(pick)[seeds_a].mean() - auc_b(pick)[seeds_b].mean())
    return np.array(out, dtype=np.float64)


def percentile_ci(replicates: ArrayLike, level: float = 0.95) -> tuple[float, float]:
    """Percentile interval of bootstrap replicates."""
    check_open_unit(level, "level")
    reps = np.asarray(replicates, dtype=np.float64)
    if reps.size == 0:
        raise ValueError("no replicates")
    alpha = 100.0 * (1.0 - level) / 2.0
    lo, hi = np.percentile(reps, [alpha, 100.0 - alpha])
    return float(lo), float(hi)


def _t_interval(center: float, se: float, df: float, level: float) -> tuple[float, float]:
    check_open_unit(level, "level")
    half = float(student_t.ppf(0.5 + level / 2.0, df)) * se
    return center - half, center + half


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
        return _t_interval(self.delta, self.se_cr2, self.df, level)

    @property
    def p_value(self) -> float:
        """Two-sided p-value of the CR2 t statistic with ``df`` degrees of freedom."""
        return float(2.0 * student_t.sf(abs(self.delta) / self.se_cr2, self.df))


def _cluster_robust(
    delta: float, psi: NDArray[np.float64], groups: NDArray[np.generic], se_iid: float
) -> ClusterRobustResult:
    """CR1 and CR2 standard errors from per-row influence values ``psi`` summed within clusters."""
    _, inv = np.unique(groups, return_inverse=True)
    s = np.bincount(inv, weights=psi)
    n_g = np.bincount(inv).astype(np.float64)
    k = s.size
    se_cr1 = float(np.sqrt(k / (k - 1) * (s**2).sum()))
    se_cr2 = float(np.sqrt(((s / np.sqrt(1.0 - n_g / psi.size)) ** 2).sum()))
    return ClusterRobustResult(delta=delta, se_iid=se_iid, se_cr1=se_cr1, se_cr2=se_cr2, df=bell_mccaffrey_df(n_g))


def cluster_robust_delta(
    y: ArrayLike, scores_a: ArrayLike, scores_b: ArrayLike, clusters: ArrayLike
) -> ClusterRobustResult:
    """CR1 and CR2 (Bell-McCaffrey) inference for the difference of seed-mean AUCs, ``A - B``."""
    labels, mat_a, mat_b, groups = as_paired(y, scores_a, scores_b, clusters)
    ca, cb = delong_components(labels, mat_a).seed_mean(), delong_components(labels, mat_b).seed_mean()
    delta = float(ca.auc[0] - cb.auc[0])
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    psi = np.empty(labels.size)
    psi[labels] = (ca.v10[0] - cb.v10[0] - delta) / n_pos
    psi[~labels] = (ca.v01[0] - cb.v01[0] - delta) / n_neg
    var_iid = (psi[labels] ** 2).sum() * n_pos / (n_pos - 1) + (psi[~labels] ** 2).sum() * n_neg / (n_neg - 1)
    return _cluster_robust(delta, psi, groups, float(np.sqrt(var_iid)))


def cluster_robust_brier_delta(
    y: ArrayLike, probs_a: ArrayLike, probs_b: ArrayLike, clusters: ArrayLike
) -> ClusterRobustResult:
    """CR1 and CR2 (Bell-McCaffrey) inference for the difference of seed-mean Brier scores, ``A - B``.

    The Brier score is a mean of per-case losses, so its influence values are the centred
    per-case loss differences divided by ``N`` and the CR2 adjustment of a mean is exact.
    Negative ``delta``: ``A`` has the lower (better) Brier score.
    """
    labels, mat_a, mat_b, groups = as_paired(y, probs_a, probs_b, clusters, ("probs_a", "probs_b"))
    if np.min(mat_a) < 0 or np.max(mat_a) > 1 or np.min(mat_b) < 0 or np.max(mat_b) > 1:
        raise ValueError("Brier inference needs probabilities in [0, 1]")
    yf = labels.astype(np.float64)
    d = ((mat_a - yf) ** 2).mean(axis=0) - ((mat_b - yf) ** 2).mean(axis=0)
    n = labels.size
    delta = float(d.mean())
    psi = (d - delta) / n
    return _cluster_robust(delta, psi, groups, float(np.sqrt((psi**2).sum() * n / (n - 1))))


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
        return _t_interval(self.delta, self.se, self.n_clusters - 1, level)


def jackknife_delta(y: ArrayLike, scores_a: ArrayLike, scores_b: ArrayLike, clusters: ArrayLike) -> JackknifeResult:
    """Delete-one-cluster jackknife of the difference of seed-mean AUCs, ``A - B``."""
    labels, mat_a, mat_b, groups = as_paired(y, scores_a, scores_b, clusters)
    design = _Clusters(labels, groups)
    auc_a, auc_b = _ClusterAUC(design, mat_a), _ClusterAUC(design, mat_b)
    everything = np.arange(design.size)
    reps = []
    for g in everything:
        if design.n_pos[g] == design.n_pos.sum() or design.n_neg[g] == design.n_neg.sum():
            raise ValueError(f"removing cluster {design.names[g]!r} leaves a single class")
        keep = np.delete(everything, g)
        reps.append(auc_a(keep).mean() - auc_b(keep).mean())
    theta = np.array(reps)
    n_groups = theta.size
    se = float(np.sqrt((n_groups - 1) / n_groups * np.sum((theta - theta.mean()) ** 2)))
    delta = float(_rank_auc_unchecked(labels, mat_a).mean() - _rank_auc_unchecked(labels, mat_b).mean())
    return JackknifeResult(delta=delta, se=se, replicates=theta)
