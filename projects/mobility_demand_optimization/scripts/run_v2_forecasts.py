"""Generate the preregistered June 2026 conditional-mean forecasts for mobility v2."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mobility_optimization.forecasting import poisson_hour_of_week_forecast

PANEL_PATH = Path("data/v2/processed/mobility_demand_hourly.parquet")
OUTPUT_DIR = Path("data/v2/results/forecasts")
TEST_START = pd.Timestamp("2026-06-01")
TEST_END = pd.Timestamp("2026-07-01")
HORIZON_HOURS = 24


def main() -> None:
    """Run daily-origin 24-hour Poisson forecasts over the untouched June holdout."""
    panel = pd.read_parquet(PANEL_PATH)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    origins = pd.date_range(
        start=TEST_START,
        end=TEST_END - pd.Timedelta(hours=HORIZON_HOURS),
        freq="24h",
    )
    forecasts = poisson_hour_of_week_forecast(
        panel,
        origins=origins,
        horizon_hours=HORIZON_HOURS,
    )
    if len(forecasts) != 30 * 24 * 30:
        raise RuntimeError(f"Unexpected June forecast row count: {len(forecasts)}")
    if forecasts["timestamp"].min() != TEST_START or forecasts["timestamp"].max() != TEST_END - pd.Timedelta(hours=1):
        raise RuntimeError("June forecast timestamps do not match the preregistered holdout.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    forecasts.to_parquet(OUTPUT_DIR / "june_poisson_hour_of_week.parquet", index=False)
    summary = {
        "design": "v2.0-preregistered",
        "test_start": str(TEST_START),
        "test_end_exclusive": str(TEST_END),
        "origins": int(len(origins)),
        "horizon_hours": HORIZON_HOURS,
        "rows": int(len(forecasts)),
        "conditional_mean_model": "expanding Poisson hour-of-week",
        "negative_binomial_alpha_for_scenarios": 0.05,
        "alpha_reselected": false,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
