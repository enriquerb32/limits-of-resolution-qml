"""Input checks shared by the public functions."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def as_labels(y: ArrayLike) -> NDArray[np.bool_]:
    """Return binary labels as a boolean vector; raise if they are not 0/1 or one class is missing."""
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError(f"labels must be 1-D, got shape {arr.shape}")
    values = np.unique(arr)
    if not np.all(np.isin(values, (0, 1))):
        raise ValueError(f"labels must be binary 0/1, got values {values[:5]}")
    if values.size < 2:
        raise ValueError("labels must contain both classes")
    return arr.astype(bool)


def as_score_matrix(scores: ArrayLike, n: int, name: str = "scores") -> NDArray[np.float64]:
    """Return scores as a (k, n) float matrix; a 1-D vector becomes a single row."""
    arr = np.atleast_2d(np.asarray(scores, dtype=np.float64))
    if arr.ndim != 2 or arr.shape[1] != n:
        raise ValueError(f"{name} must have shape (n,) or (k, n) with n={n}, got {np.shape(scores)}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def as_clusters(clusters: ArrayLike, n: int) -> NDArray[np.generic]:
    """Return cluster labels as a 1-D array of length n with at least two clusters."""
    arr = np.asarray(clusters)
    if arr.shape != (n,):
        raise ValueError(f"clusters must have shape ({n},), got {arr.shape}")
    if np.unique(arr).size < 2:
        raise ValueError("at least two clusters are required")
    return arr
