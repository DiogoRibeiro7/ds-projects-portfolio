"""Rolling fleet-state transitions driven by realised trip movements."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class DurationTransition:
    """Available idle fleet plus future arrivals after one service epoch.

    ``arrivals_by_lag[k]`` contains vehicles that become available at the
    ``k + 1``-th hourly decision epoch after the current one.
    """

    idle_next_hour: FloatArray
    arrivals_by_lag: FloatArray


def transition_fleet_state(
    *,
    supply: npt.ArrayLike,
    demand: npt.ArrayLike,
    od_counts: npt.ArrayLike,
) -> FloatArray:
    """Advance fleet supply one hour using realised OD composition.

    Vehicles not used during the hour remain in their current zone. Served vehicles
    are redistributed according to the realised destination shares for trips whose
    pickup and dropoff are both inside the frozen analysis-zone set. If an origin
    has no within-set OD observations for the hour, its served vehicles remain in
    that origin. Total fleet size is conserved exactly.

    This function intentionally represents the earlier one-hour transition
    sensitivity. Use :func:`dispatch_with_trip_durations` when passenger trip
    duration must keep vehicles unavailable for multiple decision epochs.
    """
    available = np.asarray(supply, dtype=np.float64)
    observed = np.asarray(demand, dtype=np.float64)
    flows = np.asarray(od_counts, dtype=np.float64)
    if available.ndim != 1 or available.size == 0:
        raise ValueError("supply must be a non-empty one-dimensional array.")
    if observed.shape != available.shape:
        raise ValueError("demand must have the same shape as supply.")
    if flows.shape != (available.size, available.size):
        raise ValueError("od_counts must be square with one row per fleet zone.")
    for name, values in (("supply", available), ("demand", observed), ("od_counts", flows)):
        if not np.isfinite(values).all() or (values < 0.0).any():
            raise ValueError(f"{name} must contain finite non-negative values.")

    served = np.minimum(available, observed)
    idle = available - served
    next_state = idle.copy()

    row_totals = flows.sum(axis=1)
    for origin in range(available.size):
        if served[origin] == 0.0:
            continue
        if row_totals[origin] == 0.0:
            next_state[origin] += served[origin]
            continue
        shares = flows[origin] / row_totals[origin]
        next_state += served[origin] * shares

    if not np.isclose(next_state.sum(), available.sum()):
        raise RuntimeError("Fleet conservation failed during state transition.")
    return next_state.astype(np.float64)


def dispatch_with_trip_durations(
    *,
    supply: npt.ArrayLike,
    demand: npt.ArrayLike,
    trip_counts_by_lag: npt.ArrayLike,
) -> DurationTransition:
    """Dispatch served vehicles into duration-aware future arrival buckets.

    Args:
        supply: Available vehicles by origin zone after rebalancing.
        demand: Realised pickup demand by origin zone.
        trip_counts_by_lag: Realised within-region trip counts with shape
            ``(n_lags, n_origins, n_destinations)``. Lag index zero means the
            vehicle is available at the next hourly decision epoch, lag index one
            means two epochs later, and so on.

    Returns:
        Idle vehicles that remain available next hour and destination-specific
        future arrivals for served vehicles.

    Notes:
        Destination and duration shares are conditional on realised trips whose
        pickup and dropoff both lie inside the frozen analysis region. If an origin
        has no usable realised trip profile in an hour, served vehicles from that
        origin conservatively return to the same origin at the next decision epoch.

    Raises:
        ValueError: If dimensions or values are invalid.
    """
    available = np.asarray(supply, dtype=np.float64)
    observed = np.asarray(demand, dtype=np.float64)
    profiles = np.asarray(trip_counts_by_lag, dtype=np.float64)

    if available.ndim != 1 or available.size == 0:
        raise ValueError("supply must be a non-empty one-dimensional array.")
    if observed.shape != available.shape:
        raise ValueError("demand must have the same shape as supply.")
    expected_tail = (available.size, available.size)
    if profiles.ndim != 3 or profiles.shape[0] == 0 or profiles.shape[1:] != expected_tail:
        raise ValueError(
            "trip_counts_by_lag must have shape (n_lags, n_zones, n_zones)."
        )

    for name, values in (
        ("supply", available),
        ("demand", observed),
        ("trip_counts_by_lag", profiles),
    ):
        if not np.isfinite(values).all() or (values < 0.0).any():
            raise ValueError(f"{name} must contain finite non-negative values.")

    served = np.minimum(available, observed)
    idle = available - served
    arrivals = np.zeros((profiles.shape[0], available.size), dtype=np.float64)

    # Sum over arrival lag and destination for each pickup origin.
    origin_totals = profiles.sum(axis=(0, 2))
    for origin in range(available.size):
        if served[origin] == 0.0:
            continue
        if origin_totals[origin] == 0.0:
            arrivals[0, origin] += served[origin]
            continue

        conditional = profiles[:, origin, :] / origin_totals[origin]
        arrivals += served[origin] * conditional

    conserved = float(idle.sum() + arrivals.sum())
    if not np.isclose(conserved, available.sum()):
        raise RuntimeError("Fleet conservation failed during duration-aware dispatch.")

    return DurationTransition(
        idle_next_hour=idle.astype(np.float64),
        arrivals_by_lag=arrivals.astype(np.float64),
    )
