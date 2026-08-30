"""Fleet-allocation policies for the mobility decision-science project."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import linprog

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class AllocationResult:
    """One fleet-allocation solution."""

    supply: FloatArray
    relocation_matrix: FloatArray
    relocation_cost: float
    objective: float


def _as_nonnegative_vector(values: npt.ArrayLike, *, name: str) -> FloatArray:
    """Validate a finite non-negative one-dimensional vector."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    if not np.isfinite(array).all() or (array < 0.0).any():
        raise ValueError(f"{name} must contain finite non-negative values.")
    return array


def allocate_fleet(
    *,
    initial_supply: npt.ArrayLike,
    target_demand: npt.ArrayLike,
    relocation_cost_matrix: npt.ArrayLike,
    unmet_penalty: npt.ArrayLike,
    idle_penalty: npt.ArrayLike,
) -> AllocationResult:
    """Solve a linear fleet-allocation problem for one decision epoch.

    Decision variables are zone-to-zone relocation flows, unmet demand, and idle
    capacity. Fleet conservation is enforced exactly. The optimized target demand
    can be a point forecast, a predictive quantile, or realised demand for the
    retrospective oracle.
    """
    initial = _as_nonnegative_vector(initial_supply, name="initial_supply")
    target = _as_nonnegative_vector(target_demand, name="target_demand")
    unmet = _as_nonnegative_vector(unmet_penalty, name="unmet_penalty")
    idle = _as_nonnegative_vector(idle_penalty, name="idle_penalty")
    if not (initial.shape == target.shape == unmet.shape == idle.shape):
        raise ValueError("All zone-level vectors must have identical shapes.")

    costs = np.asarray(relocation_cost_matrix, dtype=np.float64)
    n_zones = initial.size
    if costs.shape != (n_zones, n_zones):
        raise ValueError("relocation_cost_matrix must be square with one row per zone.")
    if not np.isfinite(costs).all() or (costs < 0.0).any():
        raise ValueError("relocation costs must be finite and non-negative.")
    if not np.allclose(np.diag(costs), 0.0):
        raise ValueError("Diagonal relocation costs must be zero.")

    n_flow = n_zones * n_zones
    n_vars = n_flow + 2 * n_zones
    objective = np.concatenate([costs.ravel(), unmet, idle])

    # Each origin zone may send no more vehicles than it starts with.
    a_ub = np.zeros((n_zones, n_vars), dtype=np.float64)
    b_ub = initial.copy()
    for i in range(n_zones):
        row_start = i * n_zones
        a_ub[i, row_start : row_start + n_zones] = 1.0

    # Supply after relocations equals initial - outbound + inbound. The equality
    # target - supply = unmet - idle is expressed linearly below.
    a_eq = np.zeros((n_zones, n_vars), dtype=np.float64)
    b_eq = target - initial
    for j in range(n_zones):
        for i in range(n_zones):
            a_eq[j, i * n_zones + j] -= 1.0  # inbound lowers target-minus-supply
            a_eq[j, j * n_zones + i] += 1.0  # outbound raises target-minus-supply
        a_eq[j, n_flow + j] -= 1.0  # unmet
        a_eq[j, n_flow + n_zones + j] += 1.0  # idle

    result = linprog(
        c=objective,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=(0.0, None),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Fleet allocation failed: {result.message}")

    flows = result.x[:n_flow].reshape((n_zones, n_zones))
    outbound = flows.sum(axis=1)
    inbound = flows.sum(axis=0)
    final_supply = initial - outbound + inbound
    relocation = float(np.sum(flows * costs))

    return AllocationResult(
        supply=final_supply.astype(np.float64),
        relocation_matrix=flows.astype(np.float64),
        relocation_cost=relocation,
        objective=float(result.fun),
    )
