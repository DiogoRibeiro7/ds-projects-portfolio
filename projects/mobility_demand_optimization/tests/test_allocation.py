"""Tests for mobility fleet-allocation optimization."""

from __future__ import annotations

import numpy as np
import pytest

from mobility_optimization.allocation import allocate_fleet


def test_allocation_moves_vehicle_when_service_gain_exceeds_cost() -> None:
    """A two-zone solution should move one vehicle to cover expensive unmet demand."""
    result = allocate_fleet(
        initial_supply=[2.0, 0.0],
        target_demand=[1.0, 1.0],
        relocation_cost_matrix=[[0.0, 0.25], [0.25, 0.0]],
        unmet_penalty=[5.0, 5.0],
        idle_penalty=[1.0, 1.0],
    )

    np.testing.assert_allclose(result.supply, [1.0, 1.0], atol=1e-8)
    assert result.relocation_matrix[0, 1] == pytest.approx(1.0)
    assert result.relocation_cost == pytest.approx(0.25)


def test_allocation_preserves_total_fleet() -> None:
    """Relocation must never create or destroy vehicles."""
    result = allocate_fleet(
        initial_supply=[3.0, 2.0, 1.0],
        target_demand=[0.0, 4.0, 4.0],
        relocation_cost_matrix=[
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ],
        unmet_penalty=[5.0, 5.0, 5.0],
        idle_penalty=[1.0, 1.0, 1.0],
    )

    assert result.supply.sum() == pytest.approx(6.0)
    assert np.all(result.supply >= -1e-10)


def test_allocation_leaves_fleet_when_relocation_is_too_expensive() -> None:
    """Relocation should not occur when moving costs more than the avoided loss."""
    result = allocate_fleet(
        initial_supply=[2.0, 0.0],
        target_demand=[1.0, 1.0],
        relocation_cost_matrix=[[0.0, 10.0], [10.0, 0.0]],
        unmet_penalty=[5.0, 5.0],
        idle_penalty=[1.0, 1.0],
    )

    np.testing.assert_allclose(result.supply, [2.0, 0.0], atol=1e-8)
    assert result.relocation_cost == pytest.approx(0.0)
