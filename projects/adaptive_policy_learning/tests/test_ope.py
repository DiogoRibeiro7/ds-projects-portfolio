from __future__ import annotations

import numpy as np
import pytest

from adaptive_policy_learning.ope import (
    direct_method,
    doubly_robust,
    importance_weights,
    ips,
    overlap_diagnostics,
    promotion_decision,
    snips,
)


def test_importance_weights_and_estimators_match_hand_calculation() -> None:
    reward = np.array([1.0, 0.0, 1.0])
    target = np.array([0.5, 0.25, 0.5])
    logging = np.array([0.25, 0.5, 0.5])
    weights = importance_weights(target, logging)

    np.testing.assert_allclose(weights, [2.0, 0.5, 1.0])
    assert ips(reward, weights) == pytest.approx(1.0)
    assert snips(reward, weights) == pytest.approx(3.0 / 3.5)


def test_direct_and_doubly_robust_follow_definition() -> None:
    reward = np.array([1.0, 0.0])
    weights = np.array([2.0, 0.5])
    q_logged = np.array([0.6, 0.4])
    dm_row = np.array([0.55, 0.45])

    assert direct_method(dm_row) == pytest.approx(0.5)
    expected = np.mean(dm_row + weights * (reward - q_logged))
    assert doubly_robust(reward, weights, q_logged, dm_row) == pytest.approx(expected)


def test_invalid_logging_support_is_hard_failure() -> None:
    with pytest.raises(ValueError, match="logging propensities"):
        importance_weights(np.array([0.5]), np.array([0.0]))


def test_weight_cap_is_explicit_sensitivity_only() -> None:
    weights = importance_weights(
        np.array([0.5, 0.5]),
        np.array([0.01, 0.25]),
        cap=10.0,
    )
    np.testing.assert_allclose(weights, [10.0, 2.0])


def test_overlap_diagnostics_report_effective_sample_size() -> None:
    logging = np.array([0.5, 0.5, 0.5, 0.5])
    weights = np.array([1.0, 1.0, 1.0, 1.0])
    diagnostics = overlap_diagnostics(logging, weights)

    assert diagnostics.n == 4
    assert diagnostics.ess == pytest.approx(4.0)
    assert diagnostics.ess_fraction == pytest.approx(1.0)
    assert diagnostics.weight_max == pytest.approx(1.0)


def test_zero_weight_vector_has_no_effective_sample_size() -> None:
    with pytest.raises(ValueError, match="ESS is undefined"):
        overlap_diagnostics(np.array([0.5, 0.5]), np.array([0.0, 0.0]))


def test_promotion_requires_positive_lower_bound_and_overlap() -> None:
    assert promotion_decision(0.001, 0.20) == "promote_challenger"
    assert promotion_decision(0.0, 0.20) == "do_not_promote"
    assert promotion_decision(0.001, 0.099) == "do_not_promote"
    assert promotion_decision(-0.001, 0.50) == "do_not_promote"
