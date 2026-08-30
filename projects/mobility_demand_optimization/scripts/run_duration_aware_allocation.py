"""Evaluate rolling fleet policies with observed trip-duration availability lags."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from mobility_optimization.allocation import allocate_fleet
from mobility_optimization.decision import realised_operational_cost
from mobility_optimization.fleet_state import dispatch_with_trip_durations
from mobility_optimization.probabilistic import count_quantile

RAW_DIR = Path("data/raw/tlc")
PANEL_PATH = Path("data/processed/mobility_demand_hourly.parquet")
FORECAST_PATH = Path("data/results/baselines/test_poisson_hour_of_week.parquet")
PROBABILISTIC_SUMMARY = Path("data/results/probabilistic_counts/summary.json")
SPATIAL_MATRIX_PATH = Path("data/results/spatial_allocation/relocation_cost_matrix.csv")
SPATIAL_SUMMARY_PATH = Path("data/results/spatial_allocation/summary.json")
OUTPUT_DIR = Path("data/results/duration_aware_allocation")
UNMET_PENALTY = 5.0
IDLE_PENALTY = 1.0
SERVICE_QUANTILE = UNMET_PENALTY / (UNMET_PENALTY + IDLE_PENALTY)
MAX_ARRIVAL_LAG_HOURS = 6


def _initial_supply(previous_demand: np.ndarray, *, fleet_size: float) -> np.ndarray:
    """Seed the first test hour from previous-hour pickup shares only."""
    total = float(previous_demand.sum())
    if total == 0.0:
        return np.full(previous_demand.size, fleet_size / previous_demand.size)
    return fleet_size * previous_demand / total


def _load_trip_profiles(zones: tuple[int, ...]) -> pd.DataFrame:
    """Load realised within-region trips with pickup hour and observed arrival lag."""
    frames: list[pd.DataFrame] = []
    columns = [
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "PULocationID",
        "DOLocationID",
    ]
    for month in (3, 4, 5):
        path = RAW_DIR / f"yellow_tripdata_2026-{month:02d}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Required TLC source file is missing: {path}")
        frame = pd.read_parquet(path, columns=columns)
        pickup = pd.to_datetime(frame["tpep_pickup_datetime"], errors="coerce")
        dropoff = pd.to_datetime(frame["tpep_dropoff_datetime"], errors="coerce")
        origin = pd.to_numeric(frame["PULocationID"], errors="coerce")
        destination = pd.to_numeric(frame["DOLocationID"], errors="coerce")
        keep = (
            pickup.notna()
            & dropoff.notna()
            & dropoff.ge(pickup)
            & origin.isin(zones)
            & destination.isin(zones)
        )
        valid = pd.DataFrame(
            {
                "timestamp": pickup.loc[keep].dt.floor("h"),
                "dropoff_timestamp": dropoff.loc[keep],
                "origin": origin.loc[keep].astype(int),
                "destination": destination.loc[keep].astype(int),
            }
        )
        next_decision = valid["timestamp"] + pd.Timedelta(hours=1)
        lag = np.ceil(
            np.maximum(
                (valid["dropoff_timestamp"] - next_decision).dt.total_seconds().to_numpy() / 3600.0,
                0.0,
            )
        ).astype(int)
        valid["arrival_lag"] = np.minimum(lag, MAX_ARRIVAL_LAG_HOURS - 1)
        frames.append(valid.drop(columns=["dropoff_timestamp"]))

    combined = pd.concat(frames, ignore_index=True)
    return (
        combined.groupby(["timestamp", "arrival_lag", "origin", "destination"], as_index=False)
        .size()
        .rename(columns={"size": "trips"})
    )


def _profile_tensor(frame: pd.DataFrame, *, zones: tuple[int, ...]) -> np.ndarray:
    """Convert one pickup hour's duration/OD counts to a lag-origin-destination tensor."""
    position = {zone: index for index, zone in enumerate(zones)}
    tensor = np.zeros((MAX_ARRIVAL_LAG_HOURS, len(zones), len(zones)), dtype=np.float64)
    for row in frame.itertuples(index=False):
        tensor[int(row.arrival_lag), position[int(row.origin)], position[int(row.destination)]] += float(
            row.trips
        )
    return tensor


def _index_profile_tensors(
    profiles: pd.DataFrame,
    *,
    zones: tuple[int, ...],
) -> dict[pd.Timestamp, np.ndarray]:
    """Build each hourly duration/OD tensor once before the rolling simulation."""
    return {
        pd.Timestamp(timestamp): _profile_tensor(frame, zones=zones)
        for timestamp, frame in profiles.groupby("timestamp", sort=False)
    }


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
    """Run the observed-trip-duration state sensitivity over the frozen test window."""
    required = [
        PANEL_PATH,
        FORECAST_PATH,
        PROBABILISTIC_SUMMARY,
        SPATIAL_MATRIX_PATH,
        SPATIAL_SUMMARY_PATH,
    ]
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

    profiles = _load_trip_profiles(zones)
    profile_tensors = _index_profile_tensors(profiles, zones=zones)
    empty_profile = np.zeros(
        (MAX_ARRIVAL_LAG_HOURS, len(zones), len(zones)),
        dtype=np.float64,
    )

    panel_lookup = panel.set_index(["timestamp", "zone_id"])["demand"]
    first_timestamp = pd.Timestamp(forecasts["timestamp"].min())
    previous_index = pd.MultiIndex.from_product([[first_timestamp - pd.Timedelta(hours=1)], zones])
    previous = panel_lookup.reindex(previous_index).to_numpy(dtype=np.float64)
    if np.isnan(previous).any():
        raise ValueError("Previous-hour demand is unavailable for duration-aware initialization.")
    seed = _initial_supply(previous, fleet_size=fleet_size)

    policy_names = (
        "no_rebalancing",
        "poisson_mean",
        "negative_binomial_service_quantile",
    )
    available = {name: seed.copy() for name in policy_names}
    future_arrivals: dict[str, dict[pd.Timestamp, np.ndarray]] = {
        name: defaultdict(lambda: np.zeros(len(zones), dtype=np.float64)) for name in policy_names
    }
    records: list[dict[str, float | str]] = []

    for timestamp, frame in forecasts.groupby("timestamp", sort=True):
        timestamp = pd.Timestamp(timestamp)
        ordered = frame.assign(_position=frame["zone_id"].map(zone_position)).sort_values("_position")
        observed = ordered["y_true"].to_numpy(dtype=np.float64)
        mean = ordered["y_pred"].to_numpy(dtype=np.float64)
        nb_target = count_quantile(mean, alpha=alpha, quantile=SERVICE_QUANTILE)
        hourly_profiles = profile_tensors.get(timestamp, empty_profile)

        for name in policy_names:
            available[name] = available[name] + future_arrivals[name].pop(
                timestamp,
                np.zeros(len(zones), dtype=np.float64),
            )

        policy_outputs: dict[str, tuple[np.ndarray, float]] = {
            "no_rebalancing": (available["no_rebalancing"], 0.0),
        }
        for name, target in (
            ("poisson_mean", mean),
            ("negative_binomial_service_quantile", nb_target),
        ):
            allocation = allocate_fleet(
                initial_supply=available[name],
                target_demand=target,
                relocation_cost_matrix=relocation_costs,
                unmet_penalty=np.full(len(zones), UNMET_PENALTY),
                idle_penalty=np.full(len(zones), IDLE_PENALTY),
            )
            policy_outputs[name] = (allocation.supply, allocation.relocation_cost)

        for name, (post_allocation, relocation_cost) in policy_outputs.items():
            metrics = _evaluate(observed, post_allocation, relocation_cost)
            records.append({"timestamp": str(timestamp), "policy": name, **metrics})
            transition = dispatch_with_trip_durations(
                supply=post_allocation,
                demand=observed,
                trip_counts_by_lag=hourly_profiles,
            )
            available[name] = transition.idle_next_hour
            for lag_index, arrivals in enumerate(transition.arrivals_by_lag):
                arrival_time = timestamp + pd.Timedelta(hours=lag_index + 1)
                future_arrivals[name][arrival_time] += arrivals

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
        "state_model": "policy-dependent rolling fleet with observed trip-duration arrival lags",
        "initialization": "previous-hour pickup shares used only at first test hour",
        "transition": "served vehicles become available by realised dropoff-time bucket; idle vehicles remain",
        "region_boundary": "trip profiles condition on pickup and dropoff both inside the frozen top-30 zones",
        "maximum_explicit_arrival_lag_hours": MAX_ARRIVAL_LAG_HOURS,
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
