"""Rolling fleet-state transitions driven by realised served trip destinations."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


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

    Args:
        supply: Available vehicles by origin zone after rebalancing.
        demand: Realised pickup demand by origin zone.
        od_counts: Realised within-analysis-zone trip counts, shape ``(n, n)``.

    Returns:
        Next-hour fleet state by zone.

    Raises:
        ValueError: If dimensions or values are invalid.
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
