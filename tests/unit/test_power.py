"""Unit tests for MDE, Hanley-McNeil and the seed share, including values printed in the paper."""

from __future__ import annotations

import numpy as np
import pytest

from qml_resolution import hanley_mcneil_se, mde, mde_factor, paired_mde_hanley_mcneil, seed_share


def test_mde_factor_standard_design() -> None:
    assert mde_factor() == pytest.approx(1.959964 + 0.841621, abs=1e-6)
    assert mde(0.01) == pytest.approx(0.0280, abs=1e-4)
    np.testing.assert_allclose(mde([0.01, 0.02]), [mde(0.01), mde(0.02)])


def test_mde_grows_with_power_and_stricter_alpha() -> None:
    assert mde_factor(power=0.9) > mde_factor() > mde_factor(alpha=0.10)


@pytest.mark.parametrize(
    ("n", "prev", "expected"),
    [  # main text, Table "hm_mde"
        (250, 0.41, 0.069),
        (500, 0.41, 0.049),
        (1000, 0.41, 0.035),
        (2500, 0.41, 0.022),
        (5000, 0.41, 0.015),
        (250, 0.10, 0.122),
        (500, 0.10, 0.086),
        (1000, 0.10, 0.061),
        (2500, 0.10, 0.038),
        (5000, 0.10, 0.027),
    ],
)
def test_hanley_mcneil_table_of_paper(n: int, prev: float, expected: float) -> None:
    assert round(paired_mde_hanley_mcneil(n, prevalence=prev), 3) == expected


def test_hanley_mcneil_se_of_test_set() -> None:
    """Paper: 'for the present test set it returns 0.024' (217 positives, 311 negatives, AUC 0.68)."""
    assert round(hanley_mcneil_se(0.68, 217, 311), 3) == 0.024


@pytest.mark.parametrize(
    ("sd", "se", "share"),
    [  # Appendix A of the paper
        (0.0035, 0.0232, 0.023),
        (0.0054, 0.0236, 0.049),
        (0.0063, 0.0237, 0.066),
        (0.0098, 0.0237, 0.147),
        (0.0170, 0.0238, 0.339),
    ],
)
def test_seed_share_values_of_paper(sd: float, se: float, share: float) -> None:
    # The paper computes the shares from unrounded SDs and SEs; the inputs here are rounded.
    assert seed_share(sd, se) == pytest.approx(share, abs=2e-3)


def test_seed_share_limits() -> None:
    assert seed_share(0.0, 0.02) == 0.0
    assert seed_share(0.02, 0.02) == 0.5
    with pytest.raises(ValueError):
        seed_share(0.0, 0.0)


@pytest.mark.parametrize(
    ("func", "kwargs"),
    [
        (mde_factor, {"alpha": 0.0}),
        (mde_factor, {"power": 1.0}),
        (hanley_mcneil_se, {"auc": 1.0, "n_pos": 10, "n_neg": 10}),
        (paired_mde_hanley_mcneil, {"n": 500, "rho": 1.0}),
    ],
)
def test_invalid_arguments(func, kwargs) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        func(**kwargs)
