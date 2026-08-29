"""Tests for mobility operational decision metrics."""

from __future__ import annotations

import numpy as np
import pytest

from mobility_optimization.decision import decision_regret, realised_operational_cost


def test_realised_operational_cost_decomposes_loss() -> None:
    """Unmet and idle costs should be charged to the appropriate zones."""
    result = realised_operational_cost(
        demand=np.array([10.0, 4.0]),
        supply=np.array([7.0, 6.0]),
        unmet_penalty=np.array([5.0, 5.0]),
        idle_penalty=np.array([1.0, 1.0]),
        relocation_cost=3.0,
    )

    assert result.unmet_demand == pytest.approx(15.0)
    assert result.idle_capacity == pytest.approx(2.0)
    assert result.total == pytest.approx(20.0)


def test_decision_regret_is_relative_to_oracle() -> None:
    """Regret should equal policy cost minus oracle cost."""
    assert decision_regret(policy_cost=120.0, oracle_cost=90.0) == pytest.approx(30.0)


def test_negative_supply_is_rejected() -> None:
    """Fleet supply cannot be negative."""
    with pytest.raises(ValueError, match="supply must be non-negative"):
        realised_operational_cost(
            demand=[2.0],
            supply=[-1.0],
            unmet_penalty=[5.0],
            idle_penalty=[1.0],
        )
