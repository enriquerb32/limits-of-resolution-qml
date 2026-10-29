"""Statistical resolution tools for small machine-learning benchmarks.

This package holds the inference used in the paper *Limits of Resolution in Small
Machine-Learning Benchmarks*: rank AUC, (seed-averaged) paired DeLong tests,
cluster bootstrap and jackknife for clustered test sets, and minimum detectable
differences. It depends only on NumPy and SciPy, so it can be reused in any
benchmark that stores test-set scores.

Example
-------
>>> import numpy as np
>>> from qml_resolution import paired_delong, mde
>>> rng = np.random.default_rng(0)
>>> y = rng.integers(0, 2, 500)
>>> a = y + rng.normal(size=(15, 500))   # 15 seeds of model A
>>> b = y + rng.normal(size=(15, 500))   # 15 seeds of model B
>>> test = paired_delong(y, a, b)
>>> round(mde(test.se) / test.se, 2)
2.8
"""

from qml_resolution.auc import (
    DeLongComponents,
    PairedTest,
    delong_components,
    delong_covariance,
    paired_delong,
    rank_auc,
)
from qml_resolution.power import (
    hanley_mcneil_se,
    mde,
    mde_factor,
    paired_mde_hanley_mcneil,
    seed_share,
)
from qml_resolution.resampling import (
    JackknifeResult,
    cluster_bootstrap_aucs,
    cluster_bootstrap_delta,
    jackknife_delta,
    percentile_ci,
)

__version__ = "1.1.0"

__all__ = [
    "DeLongComponents",
    "JackknifeResult",
    "PairedTest",
    "__version__",
    "cluster_bootstrap_aucs",
    "cluster_bootstrap_delta",
    "delong_components",
    "delong_covariance",
    "hanley_mcneil_se",
    "jackknife_delta",
    "mde",
    "mde_factor",
    "paired_delong",
    "paired_mde_hanley_mcneil",
    "percentile_ci",
    "rank_auc",
    "seed_share",
]
