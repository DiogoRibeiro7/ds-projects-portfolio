"""Tests for duration-aware rolling fleet transitions."""

from __future__ import annotations

import numpy as np
import pytest

from mobility_optimization.fleet_state import dispatch_with_trip_durations


def test_duration_dispatch_conserves_fleet() -> None:
    """Idle plus all future arrivals must equal the available fleet."""
    supply = np.array([8.0, 2.0])
    demand = np.array([5.0, 1.0])
    profiles = np.zeros((2, 2, 2))
    profiles[0, 0, 1] = 3.0
    profiles[1, 0, 0] = 2.0
    profiles[0, 1, 1] = 1.0

    result = dispatch_with_trip_durations(
        supply=supply,
        demand=demand,
        trip_counts_by_lag=profiles,
    )

    assert result.idle_next_hour.sum() + result.arrivals_by_lag.sum() == pytest.approx(10.0)


def test_longer_trip_arrives_later() -> None:
    """Observed trip-duration shares should preserve availability lag."""
    profiles = np.zeros((3, 1, 1))
    profiles[2, 0, 0] = 4.0

    result = dispatch_with_trip_durations(
        supply=[4.0],
        demand=[4.0],
        trip_counts_by_lag=profiles,
    )

    assert np.allclose(result.arrivals_by_lag[:2], 0.0)
    assert result.arrivals_by_lag[2, 0] == pytest.approx(4.0)


def test_missing_trip_profile_returns_next_hour() -> None:
    """A missing realised profile must not make served fleet disappear."""
    result = dispatch_with_trip_durations(
        supply=[3.0, 2.0],
        demand=[2.0, 0.0],
        trip_counts_by_lag=np.zeros((2, 2, 2)),
    )

    assert result.arrivals_by_lag[0, 0] == pytest.approx(2.0)
    assert result.idle_next_hour[1] == pytest.approx(2.0)


def test_invalid_duration_profile_shape_is_rejected() -> None:
    """Duration profiles must carry lag, origin and destination axes."""
    with pytest.raises(ValueError, match="shape"):
        dispatch_with_trip_durations(
            supply=[1.0, 1.0],
            demand=[1.0, 1.0],
            trip_counts_by_lag=np.zeros((2, 2)),
        )
