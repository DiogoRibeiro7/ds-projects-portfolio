"""Tests for the preregistered v2 stochastic allocation primitives."""

from __future__ import annotations

import numpy as np

from mobility_optimization.stochastic import (
    allocate_fleet_stochastic_saa,
    sample_nb2_scenarios,
    timestamp_seed,
)


def test_nb2_scenarios_are_reproducible() -> None:
    mean = np.array([5.0, 10.0, 20.0])
    first = sample_nb2_scenarios(mean, alpha=0.05, scenario_count=128, seed=123)
    second = sample_nb2_scenarios(mean, alpha=0.05, scenario_count=128, seed=123)
    assert first.shape == (128, 3)
    assert np.array_equal(first, second)
    assert (first >= 0.0).all()


def test_timestamp_seed_is_deterministic_and_hour_specific() -> None:
    hour_ns = 3_600_000_000_000
    first = timestamp_seed(100 * hour_ns, base_seed=20260830)
    repeat = timestamp_seed(100 * hour_ns, base_seed=20260830)
    next_hour = timestamp_seed(101 * hour_ns, base_seed=20260830)
    assert first == repeat
    assert first != next_hour


def test_one_zone_stochastic_allocation_keeps_fleet_conserved() -> None:
    result = allocate_fleet_stochastic_saa(
        initial_supply=np.array([10.0]),
        demand_scenarios=np.array([[5.0], [10.0], [15.0]]),
        relocation_cost_matrix=np.array([[0.0]]),
        unmet_penalty=np.array([5.0]),
        idle_penalty=np.array([1.0]),
    )
    assert np.allclose(result.supply, [10.0])
    assert np.isclose(result.supply.sum(), 10.0)
    assert np.isclose(result.relocation_cost, 0.0)
    # At supply 10: scenario losses are 5, 0, and 25; average = 10.
    assert np.isclose(result.objective, 10.0)


def test_two_zone_stochastic_allocation_respects_origin_capacity() -> None:
    initial = np.array([10.0, 0.0])
    scenarios = np.array([[0.0, 10.0], [0.0, 10.0], [10.0, 0.0]])
    result = allocate_fleet_stochastic_saa(
        initial_supply=initial,
        demand_scenarios=scenarios,
        relocation_cost_matrix=np.array([[0.0, 0.25], [0.25, 0.0]]),
        unmet_penalty=np.array([5.0, 5.0]),
        idle_penalty=np.array([1.0, 1.0]),
    )
    assert np.isclose(result.supply.sum(), initial.sum())
    assert (result.supply >= 0.0).all()
    assert result.relocation_matrix[0].sum() <= initial[0] + 1e-9
    assert result.relocation_matrix[1].sum() <= initial[1] + 1e-9
