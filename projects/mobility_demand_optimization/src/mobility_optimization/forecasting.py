"""Transparent rolling-origin forecasting baselines for mobility demand."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

_REQUIRED_PANEL_COLUMNS = {"timestamp", "zone_id", "demand"}


def _validated_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Return a validated, sorted copy of an hourly zone-demand panel.

    Args:
        panel: Dense hourly panel containing timestamp, zone_id, and demand.

    Returns:
        A sorted copy with integer zone identifiers and non-negative demand.

    Raises:
        ValueError: If the panel schema or values are invalid.
    """
    missing = _REQUIRED_PANEL_COLUMNS.difference(panel.columns)
    if missing:
        raise ValueError(f"Panel is missing columns: {sorted(missing)}")
    if panel.empty:
        raise ValueError("Panel must not be empty.")

    result = panel.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")
    if result["timestamp"].isna().any():
        raise ValueError("Panel contains invalid timestamps.")

    result["zone_id"] = pd.to_numeric(result["zone_id"], errors="coerce")
    result["demand"] = pd.to_numeric(result["demand"], errors="coerce")
    if result[["zone_id", "demand"]].isna().any().any():
        raise ValueError("Panel contains non-numeric zone identifiers or demand.")
    if (result["demand"] < 0).any():
        raise ValueError("Demand must be non-negative.")
    if (result["zone_id"] % 1 != 0).any():
        raise ValueError("zone_id must contain integer-valued identifiers.")
    if result.duplicated(["timestamp", "zone_id"]).any():
        raise ValueError("Panel contains duplicate timestamp-zone rows.")

    result["zone_id"] = result["zone_id"].astype(np.int64)
    result["demand"] = result["demand"].astype(np.float64)
    return result.sort_values(["timestamp", "zone_id"], ignore_index=True)


def _origins(values: Iterable[pd.Timestamp]) -> tuple[pd.Timestamp, ...]:
    """Validate and normalize forecast origins."""
    origins = tuple(pd.Timestamp(value) for value in values)
    if not origins:
        raise ValueError("At least one forecast origin is required.")
    if tuple(sorted(origins)) != origins or len(set(origins)) != len(origins):
        raise ValueError("Forecast origins must be unique and sorted.")
    return origins


def seasonal_naive_forecast(
    panel: pd.DataFrame,
    *,
    origins: Iterable[pd.Timestamp],
    horizon_hours: int = 24,
    seasonal_lag_hours: int = 168,
) -> pd.DataFrame:
    """Forecast each target hour using the same zone and hour from the prior week.

    The lag must be at least as long as the forecast horizon. This guarantees that
    every source observation used for a multi-hour forecast predates the origin.
    """
    if horizon_hours <= 0 or seasonal_lag_hours <= 0:
        raise ValueError("horizon_hours and seasonal_lag_hours must be positive.")
    if seasonal_lag_hours < horizon_hours:
        raise ValueError("seasonal_lag_hours must be at least the forecast horizon.")

    data = _validated_panel(panel)
    forecast_origins = _origins(origins)
    lookup = data.set_index(["timestamp", "zone_id"])["demand"]
    pieces: list[pd.DataFrame] = []

    for origin in forecast_origins:
        end = origin + pd.Timedelta(hours=horizon_hours)
        targets = data.loc[data["timestamp"].ge(origin) & data["timestamp"].lt(end)].copy()
        if targets.empty:
            raise ValueError(f"No target observations are available for origin {origin}.")

        source_index = pd.MultiIndex.from_arrays(
            [
                targets["timestamp"] - pd.Timedelta(hours=seasonal_lag_hours),
                targets["zone_id"],
            ],
            names=["timestamp", "zone_id"],
        )
        predicted = lookup.reindex(source_index)
        if predicted.isna().any():
            raise ValueError(f"Seasonal history is incomplete for origin {origin}.")

        result = targets.rename(columns={"demand": "y_true"})
        result["origin"] = origin
        result["y_pred"] = predicted.to_numpy(dtype=np.float64)
        result["model"] = "seasonal_naive_168h"
        pieces.append(result)

    return pd.concat(pieces, ignore_index=True)


def poisson_hour_of_week_forecast(
    panel: pd.DataFrame,
    *,
    origins: Iterable[pd.Timestamp],
    horizon_hours: int = 24,
) -> pd.DataFrame:
    """Forecast demand with an expanding Poisson hour-of-week rate model.

    For each forecast origin, demand is modelled as

        D[i,t] ~ Poisson(lambda[i,h(t)])

    where ``h(t)`` is the hour of week. The Poisson MLE for each zone/hour-of-week
    cell is its arithmetic mean over observations strictly before the forecast
    origin. This keeps the baseline interpretable and leakage-free.
    """
    if horizon_hours <= 0:
        raise ValueError("horizon_hours must be positive.")

    data = _validated_panel(panel)
    forecast_origins = _origins(origins)
    data["hour_of_week"] = data["timestamp"].dt.dayofweek * 24 + data["timestamp"].dt.hour
    pieces: list[pd.DataFrame] = []

    for origin in forecast_origins:
        history = data.loc[data["timestamp"].lt(origin)]
        if history.empty:
            raise ValueError(f"No training history is available for origin {origin}.")

        rates = (
            history.groupby(["zone_id", "hour_of_week"], as_index=False, sort=True)["demand"]
            .mean()
            .rename(columns={"demand": "y_pred"})
        )
        end = origin + pd.Timedelta(hours=horizon_hours)
        targets = data.loc[data["timestamp"].ge(origin) & data["timestamp"].lt(end)].copy()
        if targets.empty:
            raise ValueError(f"No target observations are available for origin {origin}.")

        result = targets.merge(
            rates,
            on=["zone_id", "hour_of_week"],
            how="left",
            validate="many_to_one",
        )
        if result["y_pred"].isna().any():
            raise ValueError(f"Poisson rate history is incomplete for origin {origin}.")

        result = result.rename(columns={"demand": "y_true"})
        result["origin"] = origin
        result["model"] = "poisson_hour_of_week"
        result = result.drop(columns=["hour_of_week"])
        pieces.append(result)

    return pd.concat(pieces, ignore_index=True)
