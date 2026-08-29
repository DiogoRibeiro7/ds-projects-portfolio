"""Tests for TLC ingestion and demand-panel construction."""

from __future__ import annotations

import pandas as pd

from mobility_optimization.data import (
    aggregate_pickups_frame,
    build_dense_demand_panel,
    select_top_zones,
)


def test_aggregate_pickups_reports_rejections() -> None:
    """Invalid zones, missing values and out-of-window rows must be explicit."""
    frame = pd.DataFrame(
        {
            "tpep_pickup_datetime": [
                "2025-01-01 00:10:00",
                "2025-01-01 00:40:00",
                None,
                "2025-01-01 01:00:00",
                "2024-12-31 23:59:00",
            ],
            "PULocationID": [1, 1, 1, 999, 1],
        }
    )
    counts, report = aggregate_pickups_frame(
        frame,
        valid_zone_ids=(1, 2),
        start=pd.Timestamp("2025-01-01"),
        end=pd.Timestamp("2025-02-01"),
    )

    assert counts["demand"].tolist() == [2]
    assert report.total_rows == 5
    assert report.valid_rows == 2
    assert report.missing_pickup_time == 1
    assert report.invalid_zone == 1
    assert report.outside_study_window == 1
    assert report.rejected_rows == 3


def test_top_zone_selection_uses_training_only() -> None:
    """A zone that becomes large only in test must not enter via leakage."""
    counts = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2025-06-01", "2025-06-01", "2026-04-01", "2026-04-01"]
            ),
            "zone_id": [1, 2, 2, 3],
            "demand": [100, 50, 1, 1000],
        }
    )
    selected = select_top_zones(
        counts,
        training_start=pd.Timestamp("2025-01-01"),
        training_end=pd.Timestamp("2026-01-01"),
        top_k=2,
    )
    assert selected == (1, 2)


def test_dense_panel_zero_fills_missing_zone_hours() -> None:
    """The modelling panel must contain every selected zone at every clock hour."""
    counts = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2025-01-01 00:00:00")],
            "zone_id": [1],
            "demand": [3],
        }
    )
    panel = build_dense_demand_panel(
        counts,
        zone_ids=(1, 2),
        start=pd.Timestamp("2025-01-01 00:00:00"),
        end=pd.Timestamp("2025-01-01 02:00:00"),
    )

    assert len(panel) == 4
    assert panel["demand"].sum() == 3
    assert panel["demand"].min() == 0
