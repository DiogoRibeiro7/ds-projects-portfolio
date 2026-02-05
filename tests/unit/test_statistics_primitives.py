"""Fast unit tests for statistical primitives in src.statistics.core.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from src.statistics.core import (
    bootstrap_ci_diff,
    calculate_power,
    calculate_sample_size,
    two_prop_ztest,
)

pytestmark = pytest.mark.unit


def test_two_prop_ztest_rejects_invalid_counts():
    with pytest.raises(ValueError, match="exceed sample sizes"):
        two_prop_ztest(5, 4, 1, 10)


def test_two_prop_ztest_symmetric_flip_changes_sign():
    z1, _ = two_prop_ztest(30, 100, 40, 100)
    z2, _ = two_prop_ztest(40, 100, 30, 100)
    assert math.isclose(z1, -z2, rel_tol=1e-9)


def test_bootstrap_ci_diff_is_deterministic_with_seed():
    ci_1 = bootstrap_ci_diff(0.2, 0.25, 1000, 1000, random_state=123, B=200)
    ci_2 = bootstrap_ci_diff(0.2, 0.25, 1000, 1000, random_state=123, B=200)
    assert ci_1 == ci_2


def test_bootstrap_ci_diff_rejects_invalid_parameters():
    with pytest.raises(ValueError, match="Proportions must be between 0 and 1"):
        bootstrap_ci_diff(-0.1, 0.1, 100, 100)
    with pytest.raises(ValueError, match="Sample sizes must be positive"):
        bootstrap_ci_diff(0.1, 0.2, 0, 100)
    with pytest.raises(ValueError, match="Number of bootstrap samples"):
        bootstrap_ci_diff(0.1, 0.2, 100, 100, B=0)


def test_sample_size_mde_monotonicity():
    small_mde = calculate_sample_size(0.2, 0.01)
    large_mde = calculate_sample_size(0.2, 0.03)
    assert small_mde > large_mde


def test_sample_size_invalid_inputs_raise():
    with pytest.raises(ValueError, match="between 0 and 1"):
        calculate_sample_size(-0.1, 0.02)
    with pytest.raises(ValueError, match="positive"):
        calculate_sample_size(0.1, 0.0)


def test_calculate_power_monotone_with_sample_size():
    low_power = calculate_power(500, 500, 0.1, 0.02)
    high_power = calculate_power(2000, 2000, 0.1, 0.02)
    assert high_power > low_power


@given(
    baseline=st.floats(min_value=0.05, max_value=0.6),
    mde=st.floats(min_value=0.01, max_value=0.2),
    ratio=st.floats(min_value=0.5, max_value=3.0),
)
@settings(max_examples=50)
def test_sample_size_respects_mde_property(baseline, mde, ratio):
    assume(baseline + mde * 0.5 < 0.999)
    assume(baseline + mde < 0.999)
    n_small = calculate_sample_size(baseline, mde * 0.5, ratio=ratio)
    n_large = calculate_sample_size(baseline, mde, ratio=ratio)
    assert n_small >= n_large


@given(
    baseline=st.floats(min_value=0.05, max_value=0.8),
    effect=st.floats(min_value=0.005, max_value=0.15),
)
@settings(max_examples=50)
def test_power_increases_with_more_users(baseline, effect):
    assume(baseline + effect < 0.99)
    small = calculate_power(400, 400, baseline, effect)
    large = calculate_power(1200, 1200, baseline, effect)
    assert large >= small
