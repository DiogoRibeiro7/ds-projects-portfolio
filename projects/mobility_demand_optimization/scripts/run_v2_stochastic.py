"""Run the preregistered June v2 stochastic fleet-allocation comparison."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from mobility_optimization.allocation import allocate_fleet
from mobility_optimization.decision import realised_operational_cost
from mobility_optimization.fleet_state import dispatch_with_trip_durations
from mobility_optimization.probabilistic import count_quantile
from mobility_optimization.stochastic import (
    allocate_fleet_stochastic_saa,
    sample_nb2_scenarios,
    timestamp_seed,
)

RAW_DIR = Path("data/raw/tlc")
PANEL_PATH = Path("data/v2/processed/mobility_demand_hourly.parquet")
FORECAST_PATH = Path("data/v2/results/forecasts/june_poisson_hour_of_week.parquet")
MATRIX_PATH = Path("evidence/v2_relocation_cost_matrix.csv")
DESIGN_LOCK_PATH = Path("evidence/v2_design_lock.json")
OUTPUT_DIR = Path("data/v2/results/stochastic_primary")
MATRIX_SHA256 = "bf3ebdf7eaa8391c4a5c4554fbb39d0a098f5d4fc31af429cd39f7b4b17bb8b4"
MAX_ARRIVAL_LAG_HOURS = 6
SERVICE_QUANTILE = 5.0 / 6.0


def _load_design() -> dict[str, object]:
    design = json.loads(DESIGN_LOCK_PATH.read_text(encoding="utf-8"))
    required = {
        "design_version": "v2.0-preregistered",
        "scenario_count": 128,
        "alpha": 0.05,
        "fleet_size": 4532.0,
        "unmet_penalty": 5.0,
        "idle_penalty": 1.0,
        "scenario_seed": 20260830,
        "bootstrap_seed": 20260831,
    }
    observed = {
        "design_version": design["design_version"],
        "scenario_count": design["scenario_generation"]["primary_scenarios"],
        "alpha": design["forecast"]["alpha"],
        "fleet_size": design["decision"]["fleet_size"],
        "unmet_penalty": design["decision"]["unmet_penalty"],
        "idle_penalty": design["decision"]["idle_penalty"],
        "scenario_seed": design["scenario_generation"]["base_seed"],
        "bootstrap_seed": design["paired_uncertainty"]["base_seed"],
    }
    if observed != required:
        raise ValueError(f"v2 design lock differs from implementation contract: {observed}")
    return design


def _load_matrix() -> tuple[tuple[int, ...], np.ndarray]:
    digest = hashlib.sha256(MATRIX_PATH.read_bytes()).hexdigest()
    if digest != MATRIX_SHA256:
        raise ValueError(f"Frozen relocation matrix checksum mismatch: {digest}")
    frame = pd.read_csv(MATRIX_PATH, index_col=0)
    frame.index = frame.index.astype(int)
    frame.columns = frame.columns.astype(int)
    zones = tuple(int(value) for value in frame.columns)
    if tuple(int(value) for value in frame.index) != zones or len(zones) != 30:
        raise ValueError("Frozen relocation matrix must be 30x30 with identical zone axes.")
    return zones, frame.to_numpy(dtype=np.float64)


def _load_trip_profiles(zones: tuple[int, ...]) -> dict[pd.Timestamp, np.ndarray]:
    """Load June realised within-region destination/duration profiles."""
    path = RAW_DIR / "yellow_tripdata_2026-06.parquet"
    columns = [
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "PULocationID",
        "DOLocationID",
    ]
    frame = pd.read_parquet(path, columns=columns)
    pickup = pd.to_datetime(frame["tpep_pickup_datetime"], errors="coerce")
    dropoff = pd.to_datetime(frame["tpep_dropoff_datetime"], errors="coerce")
    origin = pd.to_numeric(frame["PULocationID"], errors="coerce")
    destination = pd.to_numeric(frame["DOLocationID"], errors="coerce")
    start = pd.Timestamp("2026-06-01")
    end = pd.Timestamp("2026-07-01")
    keep = (
        pickup.ge(start)
        & pickup.lt(end)
        & dropoff.notna()
        & dropoff.ge(pickup)
        & origin.isin(zones)
        & destination.isin(zones)
    )
    valid = pd.DataFrame(
        {
            "timestamp": pickup.loc[keep].dt.floor("h"),
            "dropoff": dropoff.loc[keep],
            "origin": origin.loc[keep].astype(int),
            "destination": destination.loc[keep].astype(int),
        }
    )
    next_decision = valid["timestamp"] + pd.Timedelta(hours=1)
    lag = np.ceil(
        np.maximum((valid["dropoff"] - next_decision).dt.total_seconds().to_numpy() / 3600.0, 0.0)
    ).astype(int)
    valid["arrival_lag"] = np.minimum(lag, MAX_ARRIVAL_LAG_HOURS - 1)
    grouped = (
        valid.groupby(["timestamp", "arrival_lag", "origin", "destination"], as_index=False)
        .size()
        .rename(columns={"size": "trips"})
    )
    position = {zone: index for index, zone in enumerate(zones)}
    result: dict[pd.Timestamp, np.ndarray] = {}
    for timestamp, hour in grouped.groupby("timestamp", sort=False):
        tensor = np.zeros((MAX_ARRIVAL_LAG_HOURS, len(zones), len(zones)), dtype=np.float64)
        for row in hour.itertuples(index=False):
            tensor[
                int(row.arrival_lag),
                position[int(row.origin)],
                position[int(row.destination)],
            ] += float(row.trips)
        result[pd.Timestamp(timestamp)] = tensor
    return result


def _initial_supply(previous: np.ndarray, *, fleet_size: float) -> np.ndarray:
    total = float(previous.sum())
    if total == 0.0:
        return np.full(previous.size, fleet_size / previous.size, dtype=np.float64)
    return fleet_size * previous / total


def _evaluate(
    observed: np.ndarray,
    supply: np.ndarray,
    relocation_cost: float,
    *,
    unmet_penalty: float,
    idle_penalty: float,
) -> dict[str, float]:
    costs = realised_operational_cost(
        demand=observed,
        supply=supply,
        unmet_penalty=np.full(observed.size, unmet_penalty),
        idle_penalty=np.full(observed.size, idle_penalty),
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


def _moving_block_bootstrap(values: np.ndarray, *, block: int, replications: int, seed: int) -> dict[str, float]:
    if values.ndim != 1 or values.size < block:
        raise ValueError("Paired loss series must be one-dimensional and at least one block long.")
    rng = np.random.default_rng(seed)
    n = values.size
    starts = np.arange(n - block + 1)
    blocks_needed = int(np.ceil(n / block))
    boot = np.empty(replications, dtype=np.float64)
    for replication in range(replications):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate([values[start : start + block] for start in chosen])[:n]
        boot[replication] = sample.mean()
    return {
        "bootstrap_replications": int(replications),
        "block_length_hours": int(block),
        "bootstrap_mean": float(boot.mean()),
        "bootstrap_standard_error": float(boot.std(ddof=1)),
        "bootstrap_ci95_lower": float(np.quantile(boot, 0.025)),
        "bootstrap_ci95_upper": float(np.quantile(boot, 0.975)),
    }


def main() -> None:
    """Execute the primary preregistered stochastic decision comparison once."""
    design = _load_design()
    zones, relocation_costs = _load_matrix()
    forecasts = pd.read_parquet(FORECAST_PATH)
    forecasts["timestamp"] = pd.to_datetime(forecasts["timestamp"])
    panel = pd.read_parquet(PANEL_PATH)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    if tuple(sorted(int(value) for value in forecasts["zone_id"].unique())) != tuple(sorted(zones)):
        raise ValueError("June forecast zones differ from the frozen v1.1 zone set.")

    alpha = float(design["forecast"]["alpha"])
    scenario_count = int(design["scenario_generation"]["primary_scenarios"])
    base_seed = int(design["scenario_generation"]["base_seed"])
    fleet_size = float(design["decision"]["fleet_size"])
    unmet_penalty = float(design["decision"]["unmet_penalty"])
    idle_penalty = float(design["decision"]["idle_penalty"])
    profiles = _load_trip_profiles(zones)
    empty_profile = np.zeros((MAX_ARRIVAL_LAG_HOURS, len(zones), len(zones)), dtype=np.float64)

    zone_position = {zone: index for index, zone in enumerate(zones)}
    panel_lookup = panel.set_index(["timestamp", "zone_id"])["demand"]
    first_timestamp = pd.Timestamp("2026-06-01")
    previous_index = pd.MultiIndex.from_product([[first_timestamp - pd.Timedelta(hours=1)], zones])
    previous = panel_lookup.reindex(previous_index).to_numpy(dtype=np.float64)
    if np.isnan(previous).any():
        raise ValueError("Previous-hour demand is unavailable for v2 initialization.")
    initial = _initial_supply(previous, fleet_size=fleet_size)

    policies = (
        "poisson_mean_deterministic",
        "negative_binomial_stochastic_saa",
        "negative_binomial_service_quantile_v1",
        "no_rebalancing",
    )
    available = {name: initial.copy() for name in policies}
    future_arrivals: dict[str, dict[pd.Timestamp, np.ndarray]] = {
        name: defaultdict(lambda: np.zeros(len(zones), dtype=np.float64)) for name in policies
    }
    records: list[dict[str, float | int | str]] = []

    for timestamp, frame in forecasts.groupby("timestamp", sort=True):
        timestamp = pd.Timestamp(timestamp)
        ordered = frame.assign(_position=frame["zone_id"].map(zone_position)).sort_values("_position")
        observed = ordered["y_true"].to_numpy(dtype=np.float64)
        mean = ordered["y_pred"].to_numpy(dtype=np.float64)
        seed = timestamp_seed(timestamp.value, base_seed=base_seed)
        scenarios = sample_nb2_scenarios(
            mean,
            alpha=alpha,
            scenario_count=scenario_count,
            seed=seed,
        )
        nb_quantile = count_quantile(mean, alpha=alpha, quantile=SERVICE_QUANTILE)

        for name in policies:
            available[name] = available[name] + future_arrivals[name].pop(
                timestamp,
                np.zeros(len(zones), dtype=np.float64),
            )

        poisson = allocate_fleet(
            initial_supply=available["poisson_mean_deterministic"],
            target_demand=mean,
            relocation_cost_matrix=relocation_costs,
            unmet_penalty=np.full(len(zones), unmet_penalty),
            idle_penalty=np.full(len(zones), idle_penalty),
        )
        stochastic = allocate_fleet_stochastic_saa(
            initial_supply=available["negative_binomial_stochastic_saa"],
            demand_scenarios=scenarios,
            relocation_cost_matrix=relocation_costs,
            unmet_penalty=np.full(len(zones), unmet_penalty),
            idle_penalty=np.full(len(zones), idle_penalty),
        )
        quantile = allocate_fleet(
            initial_supply=available["negative_binomial_service_quantile_v1"],
            target_demand=nb_quantile,
            relocation_cost_matrix=relocation_costs,
            unmet_penalty=np.full(len(zones), unmet_penalty),
            idle_penalty=np.full(len(zones), idle_penalty),
        )
        outputs = {
            "poisson_mean_deterministic": (poisson.supply, poisson.relocation_cost),
            "negative_binomial_stochastic_saa": (stochastic.supply, stochastic.relocation_cost),
            "negative_binomial_service_quantile_v1": (quantile.supply, quantile.relocation_cost),
            "no_rebalancing": (available["no_rebalancing"], 0.0),
        }

        profile = profiles.get(timestamp, empty_profile)
        for name, (supply, relocation_cost) in outputs.items():
            metrics = _evaluate(
                observed,
                supply,
                relocation_cost,
                unmet_penalty=unmet_penalty,
                idle_penalty=idle_penalty,
            )
            records.append({"timestamp": str(timestamp), "policy": name, "scenario_seed": seed, **metrics})
            transition = dispatch_with_trip_durations(
                supply=supply,
                demand=observed,
                trip_counts_by_lag=profile,
            )
            available[name] = transition.idle_next_hour
            for lag_index, arrivals in enumerate(transition.arrivals_by_lag):
                future_arrivals[name][timestamp + pd.Timedelta(hours=lag_index + 1)] += arrivals

    results = pd.DataFrame(records)
    if len(results) != 720 * len(policies):
        raise RuntimeError(f"Unexpected v2 policy-hour row count: {len(results)}")
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

    pivot = results.pivot(index="timestamp", columns="policy", values="total_cost")
    paired = (
        pivot["negative_binomial_stochastic_saa"] - pivot["poisson_mean_deterministic"]
    ).to_numpy(dtype=np.float64)
    primary = {
        "estimand": "mean hourly realised cost difference: NB stochastic SAA minus Poisson mean deterministic",
        "mean_paired_cost_difference": float(paired.mean()),
        "median_paired_cost_difference": float(np.median(paired)),
        "paired_hourly_win_rate_nb": float(np.mean(paired < 0.0)),
        "paired_difference_q05": float(np.quantile(paired, 0.05)),
        "paired_difference_q50": float(np.quantile(paired, 0.50)),
        "paired_difference_q95": float(np.quantile(paired, 0.95)),
        **_moving_block_bootstrap(
            paired,
            block=int(design["paired_uncertainty"]["block_length_hours"]),
            replications=int(design["paired_uncertainty"]["replications"]),
            seed=int(design["paired_uncertainty"]["base_seed"]),
        ),
    }
    metadata = {
        "design": "v2.0-preregistered",
        "test_start": "2026-06-01T00:00:00",
        "test_end_exclusive": "2026-07-01T00:00:00",
        "headline_hours": 720,
        "zones": list(zones),
        "fleet_size": fleet_size,
        "negative_binomial_alpha": alpha,
        "scenario_count": scenario_count,
        "scenario_base_seed": base_seed,
        "scenario_dependence": "conditionally independent across zones",
        "relocation_matrix_sha256": MATRIX_SHA256,
        "unmet_penalty": unmet_penalty,
        "idle_penalty": idle_penalty,
        "maximum_explicit_arrival_lag_hours": MAX_ARRIVAL_LAG_HOURS,
        "primary": primary,
        "policy_summary": summary.to_dict(orient="records"),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_DIR / "hourly_policy_results.csv", index=False)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
