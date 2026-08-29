"""Tests for mobility forecast evaluation metrics."""

from __future__ import annotations

import pytest

from mobility_optimization.metrics import (
    empirical_interval_coverage,
    pinball_loss,
    weighted_absolute_percentage_error,
)


def test_wape_matches_definition() -> None:
    """WAPE should aggregate absolute error over observed demand."""
    result = weighted_absolute_percentage_error([10.0, 20.0], [12.0, 16.0])
    assert result == pytest.approx(6.0 / 30.0)


def test_pinball_loss_is_zero_for_exact_quantile_forecast() -> None:
    """Exact forecasts have zero pinball loss."""
    assert pinball_loss([1.0, 2.0], [1.0, 2.0], quantile=0.9) == pytest.approx(0.0)


def test_empirical_coverage_counts_boundary_values() -> None:
    """Interval endpoints should count as covered."""
    result = empirical_interval_coverage(
        [1.0, 2.0, 5.0],
        [1.0, 1.5, 3.0],
        [1.0, 3.0, 4.0],
    )
    assert result == pytest.approx(2.0 / 3.0)


def test_invalid_quantile_is_rejected() -> None:
    """Quantile levels must lie strictly inside the unit interval."""
    with pytest.raises(ValueError, match="strictly between"):
        pinball_loss([1.0], [1.0], quantile=1.0)
