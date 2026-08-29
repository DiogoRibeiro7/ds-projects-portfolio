"""Evaluate the frozen mobility forecasting baselines on validation and test origins."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mobility_optimization.backtest import FROZEN_BACKTEST
from mobility_optimization.forecasting import (
    poisson_hour_of_week_forecast,
    seasonal_naive_forecast,
)
from mobility_optimization.metrics import weighted_absolute_percentage_error


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel",
        type=Path,
        default=Path("data/processed/mobility_demand_hourly.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/results/baselines"),
    )
    return parser.parse_args()


def _daily_origins(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Return daily origins whose complete frozen horizon fits before ``end``."""
    last_origin = end - pd.Timedelta(hours=FROZEN_BACKTEST.horizon_hours)
    return pd.date_range(
        start=start,
        end=last_origin,
        freq=pd.Timedelta(hours=FROZEN_BACKTEST.origin_frequency_hours),
    )


def _headline_mask(frame: pd.DataFrame) -> pd.Series:
    """Return rows eligible for the headline comparison."""
    if "is_dst_transition_day" not in frame.columns:
        return pd.Series(True, index=frame.index, dtype=bool)
    return ~frame["is_dst_transition_day"].astype(bool)


def _summary(frame: pd.DataFrame) -> dict[str, float | int]:
    """Summarise one model/split forecast table."""
    eligible = frame.loc[_headline_mask(frame)]
    if eligible.empty:
        raise ValueError("No rows remain after the prospective headline exclusions.")
    absolute_error = (eligible["y_true"] - eligible["y_pred"]).abs()
    return {
        "rows": int(len(frame)),
        "headline_rows": int(len(eligible)),
        "mae": float(absolute_error.mean()),
        "wape": float(
            weighted_absolute_percentage_error(
                eligible["y_true"].to_numpy(),
                eligible["y_pred"].to_numpy(),
            )
        ),
    }


def _run_split(
    panel: pd.DataFrame,
    *,
    split: str,
    origins: pd.DatetimeIndex,
    output_dir: Path,
) -> dict[str, dict[str, float | int]]:
    """Run both frozen baselines for one evaluation split."""
    seasonal = seasonal_naive_forecast(
        panel,
        origins=origins,
        horizon_hours=FROZEN_BACKTEST.horizon_hours,
        seasonal_lag_hours=FROZEN_BACKTEST.seasonal_lag_hours,
    )
    poisson = poisson_hour_of_week_forecast(
        panel,
        origins=origins,
        horizon_hours=FROZEN_BACKTEST.horizon_hours,
    )

    results: dict[str, dict[str, float | int]] = {}
    for name, frame in (
        ("seasonal_naive_168h", seasonal),
        ("poisson_hour_of_week", poisson),
    ):
        path = output_dir / f"{split}_{name}.parquet"
        frame.to_parquet(path, index=False)
        results[name] = _summary(frame)
    return results


def main() -> None:
    """Run validation and untouched-test baseline evaluation."""
    args = parse_args()
    panel = pd.read_parquet(args.panel)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    validation_origins = _daily_origins(
        FROZEN_BACKTEST.validation_start,
        FROZEN_BACKTEST.validation_end,
    )
    test_origins = FROZEN_BACKTEST.test_origins()

    summary = {
        "design": {
            "validation_origins": int(len(validation_origins)),
            "test_origins": int(len(test_origins)),
            "horizon_hours": FROZEN_BACKTEST.horizon_hours,
            "seasonal_lag_hours": FROZEN_BACKTEST.seasonal_lag_hours,
        },
        "validation": _run_split(
            panel,
            split="validation",
            origins=validation_origins,
            output_dir=args.output_dir,
        ),
        "test": _run_split(
            panel,
            split="test",
            origins=test_origins,
            output_dir=args.output_dir,
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
