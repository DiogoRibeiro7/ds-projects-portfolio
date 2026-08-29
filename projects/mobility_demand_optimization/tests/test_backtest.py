"""Tests for the frozen mobility rolling-origin design."""

from __future__ import annotations

import pandas as pd

from mobility_optimization.backtest import FROZEN_BACKTEST


def test_frozen_backtest_has_92_daily_test_origins() -> None:
    """March-May 2026 provides 92 complete daily 24-hour test origins."""
    origins = FROZEN_BACKTEST.test_origins()
    assert len(origins) == 92
    assert origins[0] == pd.Timestamp("2026-03-01 00:00:00")
    assert origins[-1] == pd.Timestamp("2026-05-31 00:00:00")


def test_split_boundaries_are_disjoint() -> None:
    """Boundary timestamps must map to exactly one prospective split."""
    assert FROZEN_BACKTEST.split_label(pd.Timestamp("2025-12-31 23:00")) == "train"
    assert FROZEN_BACKTEST.split_label(pd.Timestamp("2026-01-01 00:00")) == "validation"
    assert FROZEN_BACKTEST.split_label(pd.Timestamp("2026-03-01 00:00")) == "test"
    assert FROZEN_BACKTEST.split_label(pd.Timestamp("2026-06-01 00:00")) == "outside"
