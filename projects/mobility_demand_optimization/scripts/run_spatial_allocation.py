"""Evaluate fleet allocation under geometry-derived TLC relocation costs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mobility_optimization.allocation import AllocationResult, allocate_fleet
from mobility_optimization.backtest import FROZEN_BACKTEST
from mobility_optimization.decision import realised_operational_cost
from mobility_optimization.probabilistic import count_quantile
from mobility_optimization.spatial import (
    download_taxi_zones,
    load_zone_centroids,
    normalized_distance_cost_matrix,
)

PANEL_PATH = Path("data/processed/mobility_demand_hourly.parquet")
BASELINE_PATH = Path("data/results/baselines/test_poisson_hour_of_week.parquet")
PROBABILISTIC_SUMMARY = Path("data/results/probabilistic_counts/summary.json")
ZONE_ARCHIVE = Path("data/raw/tlc/taxi_zones.zip")
OUTPUT_DIR = Path("data/results/spatial_allocation")
UNMET_PENALTY = 5.0
IDLE_PENALTY = 1.0
REFERENCE_MEDIAN_RELOCATION_COST = 0.25
SERVICE_QUANTILE = UNMET_PENALTY / (UNMET_PENALTY + IDLE_PENALTY)


def _fixed_fleet_size(panel: pd.DataFrame) -> float:
    """Freeze one fleet budget from training aggregate hourly demand only."""
    training = panel.loc[
        panel["timestamp"].ge(FROZEN_BACKTEST.train_start)
        & panel["timestamp"].lt(FROZEN_BACKTEST.train_end)
    ]
    hourly = training.groupby("timestamp", sort=True)["demand"].sum()
    if hourly.empty:
        raise ValueError("Training panel is empty; cannot freeze fleet size.")
    return float(np.rint(hourly.median()))


def _initial_supply(previous_demand: np.ndarray, *, fleet_size: float) -> np.ndarray:
    """Allocate the fixed fleet according to previous-hour observed demand shares."""
    demand = np.asarray(previous_demand, dtype=np.float64)
    if demand.ndim != 1 or demand.size == 0 or (demand < 0.0).any():
        raise ValueError("previous_demand must be a non-empty non-negative vector.")
    total = float(demand.sum())
    if total == 0.0:
        return np.full(demand.size, fleet_size / demand.size, dtype=np.float64)
    return fleet_size * demand / total


def _evaluate_solution(
    *,
    observed: np.ndarray,
    allocation: AllocationResult | None,
    initial_supply: np.ndarray,
) -> dict[str, float]:
    """Evaluate one allocation against realised demand."""
    supply = initial_supply if allocation is None else allocation.supply
    relocation_cost = 0.0 if allocation is None else allocation.relocation_cost
    costs = realised_operational_cost(
        demand=observed,
        supply=supply,
        unmet_penalty=np.full(observed.size, UNMET_PENALTY),
        idle_penalty=np.full(observed.size, IDLE_PENALTY),
        relocation_cost=relocation_cost,
    )
    served = np.minimum(observed, supply).sum()
    demand_total = observed.sum()
    service_level = 1.0 if demand_total == 0.0 else float(served / demand_total)
    return {
        "total_cost": costs.total,
        "relocation_cost": costs.relocation,
        "unmet_cost": costs.unmet_demand,
        "idle_cost": costs.idle_capacity,
        "service_level": service_level,
    }


def main() -> None:
    """Run the spatial relocation-cost sensitivity on the frozen test forecasts."""
    if not PANEL_PATH.exists() or not BASELINE_PATH.exists() or not PROBABILISTIC_SUMMARY.exists():
        raise FileNotFoundError("Panel, point forecasts, and probabilistic calibration are required.")

    panel = pd.read_parquet(PANEL_PATH)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    forecasts = pd.read_parquet(BASELINE_PATH)
    forecasts["timestamp"] = pd.to_datetime(forecasts["timestamp"])
    probabilistic = json.loads(PROBABILISTIC_SUMMARY.read_text(encoding="utf-8"))
    alpha = float(probabilistic["selected_alpha"])
    fleet_size = _fixed_fleet_size(panel)

    zones = tuple(sorted(int(zone) for zone in forecasts["zone_id"].unique()))
    zone_position = {zone: index for index, zone in enumerate(zones)}
    download_taxi_zones(destination=ZONE_ARCHIVE)
    centroids = load_zone_centroids(ZONE_ARCHIVE, zone_ids=zones)
    relocation_matrix = normalized_distance_cost_matrix(
        centroids,
        median_off_diagonal_cost=REFERENCE_MEDIAN_RELOCATION_COST,
    )
    n_zones = len(zones)
    unmet = np.full(n_zones, UNMET_PENALTY, dtype=np.float64)
    idle = np.full(n_zones, IDLE_PENALTY, dtype=np.float64)

    panel_lookup = panel.set_index(["timestamp", "zone_id"])["demand"]
    records: list[dict[str, float | str]] = []

    for timestamp, frame in forecasts.groupby("timestamp", sort=True):
        ordered = frame.assign(_position=frame["zone_id"].map(zone_position)).sort_values("_position")
        observed = ordered["y_true"].to_numpy(dtype=np.float64)
        mean = ordered["y_pred"].to_numpy(dtype=np.float64)
        previous_timestamp = pd.Timestamp(timestamp) - pd.Timedelta(hours=1)
        previous_index = pd.MultiIndex.from_product([[previous_timestamp], zones])
        previous = panel_lookup.reindex(previous_index).to_numpy(dtype=np.float64)
        if np.isnan(previous).any():
            raise ValueError(f"Previous-hour demand is unavailable for {timestamp}.")

        initial = _initial_supply(previous, fleet_size=fleet_size)
        uncertainty_target = count_quantile(mean, alpha=alpha, quantile=SERVICE_QUANTILE)
        policies: dict[str, AllocationResult | None] = {
            "no_rebalancing": None,
            "poisson_mean": allocate_fleet(
                initial_supply=initial,
                target_demand=mean,
                relocation_cost_matrix=relocation_matrix,
                unmet_penalty=unmet,
                idle_penalty=idle,
            ),
            "negative_binomial_service_quantile": allocate_fleet(
                initial_supply=initial,
                target_demand=uncertainty_target,
                relocation_cost_matrix=relocation_matrix,
                unmet_penalty=unmet,
                idle_penalty=idle,
            ),
            "oracle": allocate_fleet(
                initial_supply=initial,
                target_demand=observed,
                relocation_cost_matrix=relocation_matrix,
                unmet_penalty=unmet,
                idle_penalty=idle,
            ),
        }

        evaluated = {
            name: _evaluate_solution(observed=observed, allocation=result, initial_supply=initial)
            for name, result in policies.items()
        }
        oracle_cost = evaluated["oracle"]["total_cost"]
        for name, metrics in evaluated.items():
            records.append(
                {
                    "timestamp": str(timestamp),
                    "policy": name,
                    **metrics,
                    "regret": metrics["total_cost"] - oracle_cost,
                }
            )

    results = pd.DataFrame(records)
    summary = (
        results.groupby("policy", as_index=False)
        .agg(
            mean_total_cost=("total_cost", "mean"),
            mean_regret=("regret", "mean"),
            mean_relocation_cost=("relocation_cost", "mean"),
            mean_unmet_cost=("unmet_cost", "mean"),
            mean_idle_cost=("idle_cost", "mean"),
            mean_service_level=("service_level", "mean"),
        )
        .sort_values("mean_total_cost", ignore_index=True)
    )

    mask = ~np.eye(n_zones, dtype=bool)
    spatial_costs = relocation_matrix[mask]
    metadata = {
        "spatial_source": "official NYC TLC taxi-zone shapefile",
        "projected_crs": "EPSG:2263",
        "distance_definition": "Euclidean distance between projected polygon centroids",
        "normalization": "median off-diagonal relocation cost fixed to 0.25",
        "median_relocation_cost": float(np.median(spatial_costs)),
        "minimum_relocation_cost": float(np.min(spatial_costs)),
        "maximum_relocation_cost": float(np.max(spatial_costs)),
        "fleet_size": fleet_size,
        "unmet_penalty": UNMET_PENALTY,
        "idle_penalty": IDLE_PENALTY,
        "service_quantile": SERVICE_QUANTILE,
        "negative_binomial_alpha": alpha,
        "summary": summary.to_dict(orient="records"),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_DIR / "hourly_policy_results.csv", index=False)
    pd.DataFrame(relocation_matrix, index=zones, columns=zones).to_csv(
        OUTPUT_DIR / "relocation_cost_matrix.csv"
    )
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
