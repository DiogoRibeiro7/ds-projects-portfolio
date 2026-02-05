"""Regression tests guarding against historical numerical bugs.
"""

import math

import pytest

from src.statistics.core import bootstrap_ci_diff, two_prop_ztest

pytestmark = pytest.mark.regression


def test_two_prop_handles_all_zero_variance():
    """Historically, zero-variance cohorts caused division-by-zero errors.

    Ensure we return a neutral statistic and p-value of 1.0 instead of
    crashing when both cohorts recorded zero conversions.
    """
    z_stat, p_value = two_prop_ztest(0, 128, 0, 256)
    assert z_stat == pytest.approx(0.0)
    assert p_value == pytest.approx(1.0)


def test_two_prop_handles_degenerate_all_success_branch():
    """When one cohort hits 100% conversion, make sure we surface +/- inf.
    """
    z_stat, p_value = two_prop_ztest(0, 100, 100, 100)
    assert math.isinf(z_stat)
    assert p_value == pytest.approx(0.0)


def test_bootstrap_ci_diff_snapshot():
    """Bootstrap intervals should remain stable for fixed seeds/data.
    """
    ci = bootstrap_ci_diff(
        0.12,
        0.18,
        1000,
        1000,
        B=500,
        alpha=0.05,
        random_state=7,
    )
    assert ci == pytest.approx((0.027475, 0.09205), rel=1e-3)
