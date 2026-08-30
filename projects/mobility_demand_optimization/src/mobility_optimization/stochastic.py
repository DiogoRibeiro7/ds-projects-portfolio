"""Scenario-based stochastic fleet allocation for mobility v2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.optimize import linprog
from scipy.stats import nbinom

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class StochasticAllocationResult:
    """One sample-average stochastic fleet-allocation solution."""

    supply: FloatArray
    relocation_matrix: FloatArray
    relocation_cost: float
    objective: float


def _vector(values: npt.ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    if not np.isfinite(array).all() or (array < 0.0).any():
        raise ValueError(f"{name} must contain finite non-negative values.")
    return array


def sample_nb2_scenarios(
    mean: npt.ArrayLike,
    *,
    alpha: float,
    scenario_count: int,
    seed: int,
) -> FloatArray:
    """Draw conditionally independent NB2 demand scenarios by zone."""
    mu = _vector(mean, name="mean")
    if alpha <= 0.0 or not np.isfinite(alpha):
        raise ValueError("alpha must be finite and strictly positive for NB2 sampling.")
    if scenario_count <= 0:
        raise ValueError("scenario_count must be positive.")
    if seed < 0:
        raise ValueError("seed must be non-negative.")

    shape = 1.0 / alpha
    probability = shape / (shape + mu)
    rng = np.random.default_rng(seed)
    draws = nbinom.rvs(
        n=shape,
        p=probability,
        size=(scenario_count, mu.size),
        random_state=rng,
    )
    return np.asarray(draws, dtype=np.float64)


def timestamp_seed(timestamp_ns: int, *, base_seed: int) -> int:
    """Derive a stable 32-bit scenario seed from an hourly timestamp."""
    if timestamp_ns < 0 or base_seed < 0:
        raise ValueError("timestamp_ns and base_seed must be non-negative.")
    hour_index = timestamp_ns // 3_600_000_000_000
    return int((base_seed + hour_index) % (2**32 - 1))


def allocate_fleet_stochastic_saa(
    *,
    initial_supply: npt.ArrayLike,
    demand_scenarios: npt.ArrayLike,
    relocation_cost_matrix: npt.ArrayLike,
    unmet_penalty: npt.ArrayLike,
    idle_penalty: npt.ArrayLike,
) -> StochasticAllocationResult:
    """Minimise relocation cost plus sample-average realised mismatch loss."""
    initial = _vector(initial_supply, name="initial_supply")
    unmet = _vector(unmet_penalty, name="unmet_penalty")
    idle = _vector(idle_penalty, name="idle_penalty")
    if not (initial.shape == unmet.shape == idle.shape):
        raise ValueError("Zone-level vectors must have identical shapes.")

    scenarios = np.asarray(demand_scenarios, dtype=np.float64)
    if scenarios.ndim != 2 or scenarios.shape[0] == 0 or scenarios.shape[1] != initial.size:
        raise ValueError("demand_scenarios must have shape (n_scenarios, n_zones).")
    if not np.isfinite(scenarios).all() or (scenarios < 0.0).any():
        raise ValueError("demand_scenarios must contain finite non-negative values.")

    costs = np.asarray(relocation_cost_matrix, dtype=np.float64)
    n_zones = initial.size
    if costs.shape != (n_zones, n_zones):
        raise ValueError("relocation_cost_matrix must be square with one row per zone.")
    if not np.isfinite(costs).all() or (costs < 0.0).any():
        raise ValueError("relocation costs must be finite and non-negative.")
    if not np.allclose(np.diag(costs), 0.0):
        raise ValueError("Diagonal relocation costs must be zero.")

    k = scenarios.shape[0]
    n_flow = n_zones * n_zones
    supply_start = n_flow
    unmet_start = supply_start + n_zones
    idle_start = unmet_start + k * n_zones
    n_vars = idle_start + k * n_zones

    objective = np.zeros(n_vars, dtype=np.float64)
    objective[:n_flow] = costs.ravel()
    objective[unmet_start:idle_start] = np.tile(unmet / k, k)
    objective[idle_start:] = np.tile(idle / k, k)

    # Origin relocation limits.
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    b_ub: list[float] = []
    row = 0
    for origin in range(n_zones):
        for destination in range(n_zones):
            rows.append(row)
            cols.append(origin * n_zones + destination)
            vals.append(1.0)
        b_ub.append(float(initial[origin]))
        row += 1

    # Scenario mismatch inequalities: d - s <= u and s - d <= v.
    for scenario_index in range(k):
        for zone in range(n_zones):
            u_index = unmet_start + scenario_index * n_zones + zone
            v_index = idle_start + scenario_index * n_zones + zone

            rows.extend((row, row))
            cols.extend((supply_start + zone, u_index))
            vals.extend((-1.0, -1.0))
            b_ub.append(-float(scenarios[scenario_index, zone]))
            row += 1

            rows.extend((row, row))
            cols.extend((supply_start + zone, v_index))
            vals.extend((1.0, -1.0))
            b_ub.append(float(scenarios[scenario_index, zone]))
            row += 1

    a_ub = sparse.coo_matrix((vals, (rows, cols)), shape=(row, n_vars)).tocsr()

    # Final supply equals initial - outbound + inbound.
    eq_rows: list[int] = []
    eq_cols: list[int] = []
    eq_vals: list[float] = []
    for zone in range(n_zones):
        for destination in range(n_zones):
            eq_rows.append(zone)
            eq_cols.append(zone * n_zones + destination)
            eq_vals.append(1.0)
        for origin in range(n_zones):
            eq_rows.append(zone)
            eq_cols.append(origin * n_zones + zone)
            eq_vals.append(-1.0)
        eq_rows.append(zone)
        eq_cols.append(supply_start + zone)
        eq_vals.append(1.0)
    a_eq = sparse.coo_matrix(
        (eq_vals, (eq_rows, eq_cols)),
        shape=(n_zones, n_vars),
    ).tocsr()

    result = linprog(
        c=objective,
        A_ub=a_ub,
        b_ub=np.asarray(b_ub, dtype=np.float64),
        A_eq=a_eq,
        b_eq=initial,
        bounds=(0.0, None),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Stochastic fleet allocation failed: {result.message}")

    flows = result.x[:n_flow].reshape((n_zones, n_zones))
    supply = result.x[supply_start:unmet_start]
    relocation = float(np.sum(flows * costs))
    if not np.isclose(supply.sum(), initial.sum()):
        raise RuntimeError("Fleet conservation failed after stochastic optimisation.")

    return StochasticAllocationResult(
        supply=supply.astype(np.float64),
        relocation_matrix=flows.astype(np.float64),
        relocation_cost=relocation,
        objective=float(result.fun),
    )
