"""Operational loss functions for mobility allocation policies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class OperationalCost:
    """Decomposition of realised fleet-allocation cost."""

    relocation: float
    unmet_demand: float
    idle_capacity: float

    @property
    def total(self) -> float:
        """Return total realised operational cost."""
        return self.relocation + self.unmet_demand + self.idle_capacity


def realised_operational_cost(
    *,
    demand: npt.ArrayLike,
    supply: npt.ArrayLike,
    unmet_penalty: npt.ArrayLike,
    idle_penalty: npt.ArrayLike,
    relocation_cost: float = 0.0,
) -> OperationalCost:
    """Evaluate one realised allocation decision.

    Args:
        demand: Realised demand by zone.
        supply: Vehicle supply made available by zone.
        unmet_penalty: Cost per unit of unmet demand by zone.
        idle_penalty: Cost per unit of excess supply by zone.
        relocation_cost: Already-computed cost of repositioning vehicles.

    Returns:
        A cost decomposition for the realised decision.

    Raises:
        ValueError: If inputs are malformed, negative, or dimensionally inconsistent.
    """
    arrays: dict[str, FloatArray] = {
        "demand": np.asarray(demand, dtype=np.float64),
        "supply": np.asarray(supply, dtype=np.float64),
        "unmet_penalty": np.asarray(unmet_penalty, dtype=np.float64),
        "idle_penalty": np.asarray(idle_penalty, dtype=np.float64),
    }

    reference_shape: tuple[int, ...] | None = None
    for name, array in arrays.items():
        if array.ndim != 1 or array.size == 0:
            raise ValueError(f"{name} must be a non-empty one-dimensional array.")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values.")
        if np.any(array < 0.0):
            raise ValueError(f"{name} must be non-negative.")
        if reference_shape is None:
            reference_shape = array.shape
        elif array.shape != reference_shape:
            raise ValueError("All zone-level arrays must have identical shapes.")

    if not np.isfinite(relocation_cost) or relocation_cost < 0.0:
        raise ValueError("relocation_cost must be finite and non-negative.")

    unmet = np.maximum(arrays["demand"] - arrays["supply"], 0.0)
    idle = np.maximum(arrays["supply"] - arrays["demand"], 0.0)

    return OperationalCost(
        relocation=float(relocation_cost),
        unmet_demand=float(np.dot(unmet, arrays["unmet_penalty"])),
        idle_capacity=float(np.dot(idle, arrays["idle_penalty"])),
    )


def decision_regret(*, policy_cost: float, oracle_cost: float) -> float:
    """Return realised policy regret relative to the retrospective oracle."""
    if not np.isfinite(policy_cost) or not np.isfinite(oracle_cost):
        raise ValueError("Costs must be finite.")
    if policy_cost < 0.0 or oracle_cost < 0.0:
        raise ValueError("Costs must be non-negative.")
    return float(policy_cost - oracle_cost)
