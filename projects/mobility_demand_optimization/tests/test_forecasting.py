"""Tests for leakage-free mobility forecasting baselines."""

from __future__ import annotations

import pandas as pd
import pytest

from mobility_optimization.forecasting import (
    poisson_hour_of_week_forecast,
    seasonal_naive_forecast,
)


def _panel(*, periods: int = 24 * 15) -> pd.DataFrame:
    """Build a deterministic dense hourly panel for two zones."""
    timestamps = pd.date_range("2025-01-01", periods=periods, freq="h")
    rows: list[dict[str, object]] = []
    for timestamp in timestamps:
        hour_of_week = timestamp.dayofweek * 24 + timestamp.hour
        rows.append({"timestamp": timestamp, "zone_id": 1, "demand": 10 + hour_of_week})
        rows.append({"timestamp": timestamp, "zone_id": 2, "demand": 20 + hour_of_week})
    return pd.DataFrame(rows)


def test_seasonal_naive_uses_exact_prior_week() -> None:
    """The seasonal-naive baseline should reproduce demand from 168 hours earlier."""
    panel = _panel()
    origin = pd.Timestamp("2025-01-10T00:00:00")

    forecast = seasonal_naive_forecast(panel, origins=[origin], horizon_hours=24)

    assert len(forecast) == 48
    assert (forecast["y_true"] == forecast["y_pred"]).all()


def test_seasonal_naive_rejects_lag_shorter_than_horizon() -> None:
    """A source lag inside the forecast horizon would permit future leakage."""
    panel = _panel()
    with pytest.raises(ValueError, match="at least the forecast horizon"):
        seasonal_naive_forecast(
            panel,
            origins=[pd.Timestamp("2025-01-10")],
            horizon_hours=24,
            seasonal_lag_hours=12,
        )


def test_poisson_hour_of_week_uses_only_pre_origin_history() -> None:
    """Future demand changes must not alter an expanding Poisson forecast."""
    panel = _panel()
    origin = pd.Timestamp("2025-01-10T00:00:00")
    baseline = poisson_hour_of_week_forecast(panel, origins=[origin], horizon_hours=24)

    changed = panel.copy()
    changed.loc[changed["timestamp"].ge(origin), "demand"] = 1_000_000
    rerun = poisson_hour_of_week_forecast(changed, origins=[origin], horizon_hours=24)

    assert baseline["y_pred"].tolist() == pytest.approx(rerun["y_pred"].tolist())
    assert (baseline["y_pred"] >= 0.0).all()


def test_poisson_rate_matches_historical_cell_mean() -> None:
    """The Poisson MLE should equal the expanding mean for each hour-of-week cell."""
    panel = _panel()
    origin = pd.Timestamp("2025-01-10T00:00:00")
    forecast = poisson_hour_of_week_forecast(panel, origins=[origin], horizon_hours=1)

    first = forecast.sort_values("zone_id").reset_index(drop=True)
    expected_hour = origin.dayofweek * 24 + origin.hour
    assert first.loc[0, "y_pred"] == pytest.approx(10 + expected_hour)
    assert first.loc[1, "y_pred"] == pytest.approx(20 + expected_hour)
