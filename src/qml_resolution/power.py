"""Design sensitivity: minimum detectable differences and the seed share.

The minimum detectable difference (MDE) is the true difference that a two-sided test at
level ``alpha`` detects with probability ``power``, under a normal approximation with a
locally constant standard error. It describes the sensitivity of a design; it is not a
threshold that an observed difference must exceed.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm


def _check_probability(value: float, name: str) -> None:
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be in (0, 1), got {value}")


def mde_factor(alpha: float = 0.05, power: float = 0.80) -> float:
    """Return ``z_{1-alpha/2} + z_{power}`` (about 2.80 for alpha=0.05, power=0.80)."""
    _check_probability(alpha, "alpha")
    _check_probability(power, "power")
    return float(norm.ppf(1.0 - alpha / 2.0) + norm.ppf(power))


def mde(se: ArrayLike, alpha: float = 0.05, power: float = 0.80) -> NDArray[np.float64] | float:
    """Minimum detectable difference for a statistic with standard error ``se``."""
    se_arr = np.asarray(se, dtype=np.float64)
    if np.any(se_arr < 0):
        raise ValueError("standard errors must be non-negative")
    out = mde_factor(alpha, power) * se_arr
    return float(out) if out.ndim == 0 else out


def hanley_mcneil_se(auc: float, n_pos: float, n_neg: float) -> float:
    """Return the standard error of one AUC from Hanley & McNeil (1982).

    Uses ``Q1 = A / (2 - A)`` and ``Q2 = 2 A^2 / (1 + A)``, which assume exponential score
    distributions; in practice the approximation is close to DeLong for moderate AUCs.
    """
    if not 0.0 < auc < 1.0:
        raise ValueError("auc must be in (0, 1)")
    if n_pos < 1 or n_neg < 1:
        raise ValueError("need at least one positive and one negative")
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc**2 / (1.0 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc**2) + (n_neg - 1) * (q2 - auc**2)) / (n_pos * n_neg)
    return float(np.sqrt(var))


def paired_mde_hanley_mcneil(
    n: float,
    auc: float = 0.68,
    prevalence: float = 0.41,
    rho: float = 0.75,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Approximate MDE for the AUC difference of two models scored on the same ``n`` rows.

    Both AUCs are assumed to have the Hanley-McNeil standard error and correlation ``rho``,
    so the paired standard error is ``se * sqrt(2 (1 - rho))``. Rows are treated as
    independent; clustered test data can move the result in either direction.
    """
    _check_probability(prevalence, "prevalence")
    if not -1.0 <= rho < 1.0:
        raise ValueError("rho must be in [-1, 1)")
    se = hanley_mcneil_se(auc, prevalence * n, (1.0 - prevalence) * n)
    return mde_factor(alpha, power) * se * float(np.sqrt(2.0 * (1.0 - rho)))


def seed_share(sigma_seed: ArrayLike, se: ArrayLike) -> NDArray[np.float64] | float:
    """Descriptive ratio ``sigma_seed^2 / (sigma_seed^2 + se^2)``.

    It compares the scale of seed variance (on a fixed test set) with test-sampling variance
    (for a fixed model). It is not a decomposition of total uncertainty.
    """
    s2 = np.asarray(sigma_seed, dtype=np.float64) ** 2
    e2 = np.asarray(se, dtype=np.float64) ** 2
    if np.any(s2 + e2 == 0):
        raise ValueError("sigma_seed and se cannot both be zero")
    out = s2 / (s2 + e2)
    return float(out) if out.ndim == 0 else out
