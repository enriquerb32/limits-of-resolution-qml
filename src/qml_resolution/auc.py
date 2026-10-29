"""Rank AUC and DeLong inference, including the seed-averaged paired test.

The DeLong estimator (DeLong, DeLong & Clarke-Pearson, 1988) writes the empirical AUC
as the mean of *structural components*: one value per positive (``v10``) and one per
negative (``v01``). The fast rank form used here follows Sun & Xu (2014).

Because the AUC is the mean of its components, the mean of S per-seed AUCs equals the
AUC statistic built from components averaged over the seeds. :func:`paired_delong`
uses this to give the paired sampling variance of a difference of seed-mean AUCs.
That variance conditions on the fitted seeds; it does not include seed variance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm, rankdata

from qml_resolution._validation import as_labels, as_score_matrix

_MIN_VARIANCE = 1e-18


def rank_auc(y: ArrayLike, scores: ArrayLike) -> NDArray[np.float64]:
    """Return the ROC AUC of each row of ``scores`` (Mann-Whitney form, ties count 1/2).

    Parameters
    ----------
    y:
        Binary labels, shape ``(n,)``.
    scores:
        Scores of shape ``(n,)`` or ``(k, n)``; higher means more likely positive.

    Returns
    -------
    numpy.ndarray
        AUCs of shape ``(k,)`` (``(1,)`` for a single score vector).
    """
    labels = as_labels(y)
    mat = as_score_matrix(scores, labels.size)
    return _rank_auc_unchecked(labels, mat)


def _rank_auc_unchecked(labels: NDArray[np.bool_], mat: NDArray[np.float64]) -> NDArray[np.float64]:
    n_pos = int(labels.sum())
    n_neg = labels.size - n_pos
    ranks = rankdata(mat, axis=1)
    out: NDArray[np.float64] = (ranks[:, labels].sum(axis=1) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return out


@dataclass(frozen=True)
class DeLongComponents:
    """Structural components of the AUCs of k score vectors on the same labels.

    Attributes
    ----------
    auc:
        AUC of each score vector, shape ``(k,)``.
    v10:
        Components of the positives, shape ``(k, n_pos)``.
    v01:
        Components of the negatives, shape ``(k, n_neg)``.
    """

    auc: NDArray[np.float64]
    v10: NDArray[np.float64]
    v01: NDArray[np.float64]

    @property
    def n_pos(self) -> int:
        """Number of positives."""
        return int(self.v10.shape[1])

    @property
    def n_neg(self) -> int:
        """Number of negatives."""
        return int(self.v01.shape[1])

    def seed_mean(self) -> DeLongComponents:
        """Average the components over the rows (seeds); the result has one row."""
        return DeLongComponents(
            auc=np.array([self.auc.mean()]),
            v10=self.v10.mean(axis=0, keepdims=True),
            v01=self.v01.mean(axis=0, keepdims=True),
        )


def delong_components(y: ArrayLike, scores: ArrayLike) -> DeLongComponents:
    """Compute DeLong structural components for each row of ``scores``."""
    labels = as_labels(y)
    mat = as_score_matrix(scores, labels.size)
    n_pos = int(labels.sum())
    n_neg = labels.size - n_pos
    rank_all = rankdata(mat, axis=1)
    rank_pos = rankdata(mat[:, labels], axis=1)
    rank_neg = rankdata(mat[:, ~labels], axis=1)
    v10 = (rank_all[:, labels] - rank_pos) / n_neg
    v01 = 1.0 - (rank_all[:, ~labels] - rank_neg) / n_pos
    return DeLongComponents(auc=_rank_auc_unchecked(labels, mat), v10=v10, v01=v01)


def _covariance(comp: DeLongComponents) -> NDArray[np.float64]:
    s10 = np.atleast_2d(np.cov(comp.v10))
    s01 = np.atleast_2d(np.cov(comp.v01))
    cov: NDArray[np.float64] = s10 / comp.n_pos + s01 / comp.n_neg
    return cov


def delong_covariance(y: ArrayLike, scores: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the AUCs ``(k,)`` and their DeLong covariance matrix ``(k, k)``.

    Test rows are treated as independent; with clustered test data (for example,
    station-months grouped by year) compare with :func:`~qml_resolution.cluster_bootstrap_delta`.
    """
    comp = delong_components(y, scores)
    return comp.auc, _covariance(comp)


@dataclass(frozen=True)
class PairedTest:
    """Result of a paired DeLong comparison ``A - B``.

    Attributes
    ----------
    delta:
        Difference of mean AUCs (mean over the seeds of each model).
    se:
        DeLong standard error of ``delta``, conditional on the fitted seeds.
    auc_a, auc_b:
        Per-seed AUCs of each model.
    """

    delta: float
    se: float
    auc_a: NDArray[np.float64]
    auc_b: NDArray[np.float64]

    @property
    def z(self) -> float:
        """Wald statistic ``delta / se``."""
        return self.delta / self.se

    @property
    def p_value(self) -> float:
        """Two-sided p-value under the normal approximation."""
        return float(2.0 * norm.sf(abs(self.z)))

    def ci(self, level: float = 0.95) -> tuple[float, float]:
        """Wald confidence interval for ``delta``."""
        if not 0.0 < level < 1.0:
            raise ValueError("level must be in (0, 1)")
        half = float(norm.ppf(0.5 + level / 2.0)) * self.se
        return self.delta - half, self.delta + half


def paired_delong(y: ArrayLike, scores_a: ArrayLike, scores_b: ArrayLike) -> PairedTest:
    """Paired DeLong test of the difference between the seed-mean AUCs of two models.

    Parameters
    ----------
    y:
        Binary labels, shape ``(n,)``.
    scores_a, scores_b:
        Scores of each model, shape ``(n,)`` for one run or ``(S, n)`` for S seeds.
        The two models may have different numbers of seeds.

    Notes
    -----
    With one row per model this is the classical paired DeLong test. With several
    seeds the components are averaged within each model before the paired variance
    is computed, so the seeds are treated as fixed.
    """
    labels = as_labels(y)
    comp_a = delong_components(labels, scores_a)
    comp_b = delong_components(labels, scores_b)
    mean_a, mean_b = comp_a.seed_mean(), comp_b.seed_mean()
    s10 = np.cov(np.vstack([mean_a.v10, mean_b.v10]))
    s01 = np.cov(np.vstack([mean_a.v01, mean_b.v01]))
    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / comp_a.n_pos + (
        s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]
    ) / comp_a.n_neg
    delta = float(comp_a.auc.mean() - comp_b.auc.mean())
    return PairedTest(
        delta=delta,
        se=float(np.sqrt(max(var, _MIN_VARIANCE))),
        auc_a=comp_a.auc,
        auc_b=comp_b.auc,
    )
