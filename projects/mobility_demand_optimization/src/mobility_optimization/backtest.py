"""Frozen rolling-origin design for the mobility demand study."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class BacktestDesign:
    """Prospectively fixed temporal split and rolling forecast geometry."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_end: pd.Timestamp
    test_end: pd.Timestamp
    horizon_hours: int = 24
    origin_frequency_hours: int = 24
    top_k_zones: int = 30
    seasonal_lag_hours: int = 168

    def __post_init__(self) -> None:
        """Validate split ordering and positive design parameters."""
        if not self.train_start < self.train_end < self.validation_end < self.test_end:
            raise ValueError("Temporal boundaries must satisfy train < validation < test.")
        for name in (
            "horizon_hours",
            "origin_frequency_hours",
            "top_k_zones",
            "seasonal_lag_hours",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")

    @property
    def validation_start(self) -> pd.Timestamp:
        """Return the first validation timestamp."""
        return self.train_end

    @property
    def test_start(self) -> pd.Timestamp:
        """Return the first untouched test timestamp."""
        return self.validation_end

    def test_origins(self) -> pd.DatetimeIndex:
        """Return all test forecast origins whose full horizon is observable."""
        last_origin = self.test_end - pd.Timedelta(hours=self.horizon_hours)
        return pd.date_range(
            start=self.test_start,
            end=last_origin,
            freq=pd.Timedelta(hours=self.origin_frequency_hours),
        )

    def split_label(self, timestamp: pd.Timestamp) -> str:
        """Return the prospective split label for one timestamp."""
        if timestamp < self.train_start or timestamp >= self.test_end:
            return "outside"
        if timestamp < self.train_end:
            return "train"
        if timestamp < self.validation_end:
            return "validation"
        return "test"


FROZEN_BACKTEST = BacktestDesign(
    train_start=pd.Timestamp("2025-01-01 00:00:00"),
    train_end=pd.Timestamp("2026-01-01 00:00:00"),
    validation_end=pd.Timestamp("2026-03-01 00:00:00"),
    test_end=pd.Timestamp("2026-06-01 00:00:00"),
    horizon_hours=24,
    origin_frequency_hours=24,
    top_k_zones=30,
    seasonal_lag_hours=168,
)
