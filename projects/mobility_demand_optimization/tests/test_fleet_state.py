"""Tests for rolling mobility fleet-state transitions."""

from __future__ import annotations

import numpy as np
import pytest

from mobility_optimization.fleet_state import transition_fleet_state


def test_transition_conserves_fleet_and_moves_served_vehicles() -> None:
    """Served vehicles should follow observed destination shares."""
    result = transition_fleet_state(
        supply=np.array([10.0, 0.0]),
        demand=np.array([6.0, 0.0]),
        od_counts=np.array([[1.0, 2.0], [0.0, 0.0]]),
    )
    assert result.sum() == pytest.approx(10.0)
    assert result[0] == pytest.approx(6.0)
    assert result[1] == pytest.approx(4.0)


def test_zero_od_row_keeps_served_vehicles_at_origin() -> None:
    """Missing within-region destinations should not destroy fleet mass."""
    result = transition_fleet_state(
        supply=[4.0, 1.0],
        demand=[3.0, 0.0],
        od_counts=[[0.0, 0.0], [0.0, 0.0]],
    )
    assert np.allclose(result, [4.0, 1.0])


def test_invalid_od_shape_is_rejected() -> None:
    """OD matrices must align with the fleet-state dimension."""
    with pytest.raises(ValueError, match="square"):
        transition_fleet_state(supply=[1.0, 2.0], demand=[1.0, 1.0], od_counts=[[1.0]])
