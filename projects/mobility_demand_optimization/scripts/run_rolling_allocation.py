"""Evaluate mobility policies with an endogenous rolling fleet state."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mobility_optimization.allocation import allocate_fleet
from mobility_optimization.decision import realised_operational_cost
from mobility_optimization.fleet_state import transition_fleet_state
from mobility_optimization.probabilistic import count_quantile

RAW_DIR = Path("data/raw/tlc")
PANEL_PATH = Path("data/processed/mobility_demand_hourly.parquet")
FORECAST_PATH = Path("data/results/baselines/test_poisson_hour_of_week.parquet")
PROBABILISTIC_SUMMARY = Path("data/results/probabilistic_counts/summary.json")
SPATIAL_MATRIX_PATH = Path("data/results/spatial_allocation/relocation_cost_matrix.csv")
SPATIAL_SUMMARY_PATH = Path("data/results/spatial_allocation/summary.json")
OUTPUT_DIR = Path("data/results/rolling_allocation")
UNMET_PENALTY = 5.0
IDLE_PENALTY = 1.0
SERVICE_QUANTILE = UNMET_PENALTY / (UNMET_PENALTY + IDLE_PENALTY)


def _initial_supply(previous_demand: np.ndarray, *, fleet_size: float) -> np.ndarray:
    """Seed only the first test hour from previous-hour observed pickup shares."""
    total = float(previous_demand.sum())
    if total == 0.0:
        return np.full(previous_demand.size, fleet_size / previous_demand.size)
    return fleet_size * previous_demand / total


def _load_test_od_counts(zones: tuple[int, ...]) -> pd.DataFrame:
    """Aggregate realised pickup-hour OD counts for March-May 2026 within the region."""
    frames: list[pd.DataFrame] = []
    columns = ["tpep_pickup_datetime", "PULocationID", "DOLocationID"]
    for month in (3, 4, 5):
        path = RAW_DIR / f"yellow_tripdata_2026-{month:02d}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Required TLC source file is missing: {path}")
        frame = pd.read_parquet(path, columns=columns)
        pickup_time = pd.to_datetime(frame["tpep_pickup_datetime"], errors="coerce")
        pickup_zone = pd.to_numeric(frame["PULocationID"], errors="coerce")
        dropoff_zone = pd.to_numeric(frame["DOLocationID"], errors="coerce")
        keep = pickup_time.notna() & pickup_zone.isin(zones) & dropoff_zone.isin(zones)
        valid = pd.DataFrame(
            {
                "timestamp": pickup_time.loc[keep].dt.floor("h"),
                "origin": pickup_zone.loc[keep].astype(int),
                "destination": dropoff_zone.loc[keep].astype(int),
            }
        )
        frames.append(valid)

    combined = pd.concat(frames, ignore_index=True)
    return (
        combined.groupby(["timestamp", "origin", "destination"], as_index=False)
        .size()
        .rename(columns={"size": "trips"})
    )


def _od_matrix(frame: pd.DataFrame, *, zones: tuple[int, ...]) -> np.ndarray:
    """Convert one hour's OD counts to the frozen zone order."""
    position = {zone: index for index, zone in enumerate(zones)}
    matrix = np.zeros((len(zones), len(zones)), dtype=np.float64)
    for row in frame.itertuples(index=False):
        matrix[position[int(row.origin)], position[int(row.destination)]] += float(row.trips)
    return matrix


def _evaluate(observed: np.ndarray, supply: np.ndarray, relocation_cost: float) -> dict[str, float]:
    """Return realised cost and service metrics for one policy-hour."""
    costs = realised_operational_cost(
        demand=observed,
        supply=supply,
        unmet_penalty=np.full(observed.size, UNMET_PENALTY),
        idle_penalty=np.full(observed.size, IDLE_PENALTY),
        relocation_cost=relocation_cost,
    )
    served = np.minimum(observed, supply).sum()
    return {
        "total_cost": costs.total,
        "relocation_cost": costs.relocation,
        "unmet_cost": costs.unmet_demand,
        "idle_cost": costs.idle_capacity,
        "service_level": 1.0 if observed.sum() == 0.0 else float(served / observed.sum()),
    }


def main() -> None:
    """Run a policy-dependent rolling state backtest over the frozen test window."""
    required = [PANEL_PATH, FORECAST_PATH, PROBABILISTIC_SUMMARY, SPATIAL_MATRIX_PATH, SPATIAL_SUMMARY_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required empirical artifacts are missing: {missing}")

    panel = pd.read_parquet(PANEL_PATH)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    forecasts = pd.read_parquet(FORECAST_PATH)
    forecasts["timestamp"] = pd.to_datetime(forecasts["timestamp"])
    zones = tuple(sorted(int(value) for value in forecasts["zone_id"].unique()))
    zone_position = {zone: index for index, zone in enumerate(zones)}

    probabilistic = json.loads(PROBABILISTIC_SUMMARY.read_text(encoding="utf-8"))
    alpha = float(probabilistic["selected_alpha"])
    spatial_metadata = json.loads(SPATIAL_SUMMARY_PATH.read_text(encoding="utf-8"))
    fleet_size = float(spatial_metadata["fleet_size"])
    costs_frame = pd.read_csv(SPATIAL_MATRIX_PATH, index_col=0)
    costs_frame.index = costs_frame.index.astype(int)
    costs_frame.columns = costs_frame.columns.astype(int)
    relocation_costs = costs_frame.loc[list(zones), list(zones)].to_numpy(dtype=np.float64)

    od_counts = _load_test_od_counts(zones)
    panel_lookup = panel.set_index(["timestamp", "zone_id"])["demand"]
    first_timestamp = pd.Timestamp(forecasts["timestamp"].min())
    previous_index = pd.MultiIndex.from_product([[first_timestamp - pd.Timedelta(hours=1)], zones])
    previous = panel_lookup.reindex(previous_index).to_numpy(dtype=np.float64)
    if np.isnan(previous).any():
        raise ValueError("Previous-hour demand is unavailable for rolling-state initialization.")
    seed = _initial_supply(previous, fleet_size=fleet_size)

    states = {
        "no_rebalancing": seed.copy(),
        "poisson_mean": seed.copy(),
        "negative_binomial_service_quantile": seed.copy(),
    }
    records: list[dict[str, float | str]] = []

    for timestamp, frame in forecasts.groupby("timestamp", sort=True):
        ordered = frame.assign(_position=frame["zone_id"].map(zone_position)).sort_values("_position")
        observed = ordered["y_true"].to_numpy(dtype=np.float64)
        mean = ordered["y_pred"].to_numpy(dtype=np.float64)
        nb_target = count_quantile(mean, alpha=alpha, quantile=SERVICE_QUANTILE)
        hourly_od = _od_matrix(od_counts.loc[od_counts["timestamp"].eq(timestamp)], zones=zones)

        policy_outputs: dict[str, tuple[np.ndarray, float]] = {
            "no_rebalancing": (states["no_rebalancing"], 0.0),
        }
        for name, target in (
            ("poisson_mean", mean),
            ("negative_binomial_service_quantile", nb_target),
        ):
            allocation = allocate_fleet(
                initial_supply=states[name],
                target_demand=target,
                relocation_cost_matrix=relocation_costs,
                unmet_penalty=np.full(len(zones), UNMET_PENALTY),
                idle_penalty=np.full(len(zones), IDLE_PENALTY),
            )
            policy_outputs[name] = (allocation.supply, allocation.relocation_cost)

        for name, (post_allocation, relocation_cost) in policy_outputs.items():
            metrics = _evaluate(observed, post_allocation, relocation_cost)
            records.append({"timestamp": str(timestamp), "policy": name, **metrics})
            states[name] = transition_fleet_state(
                supply=post_allocation,
                demand=observed,
                od_counts=hourly_od,
            )

    results = pd.DataFrame(records)
    summary = (
        results.groupby("policy", as_index=False)
        .agg(
            mean_total_cost=("total_cost", "mean"),
            mean_relocation_cost=("relocation_cost", "mean"),
            mean_unmet_cost=("unmet_cost", "mean"),
            mean_idle_cost=("idle_cost", "mean"),
            mean_service_level=("service_level", "mean"),
        )
        .sort_values("mean_total_cost", ignore_index=True)
    )
    metadata = {
        "state_model": "policy-dependent rolling fleet state",
        "initialization": "previous-hour pickup shares used only at first test hour",
        "transition": "served vehicles follow realised within-region OD destination shares; idle vehicles stay",
        "region_boundary": "OD shares condition on trips with pickup and dropoff both inside the frozen top-30 zones",
        "fleet_size": fleet_size,
        "negative_binomial_alpha": alpha,
        "spatial_cost_source": spatial_metadata["spatial_source"],
        "summary": summary.to_dict(orient="records"),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_DIR / "hourly_policy_results.csv", index=False)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
