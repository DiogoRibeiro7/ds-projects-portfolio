"""Tests for probabilistic mobility count calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mobility_optimization.probabilistic import (
    count_quantile,
    probabilistic_summary,
    select_dispersion_alpha,
)


def test_poisson_limit_matches_scipy_shape_contract() -> None:
    """The alpha-zero limit should return finite monotone count quantiles."""
    mean = np.array([1.0, 10.0, 100.0])
    lower = count_quantile(mean, alpha=0.0, quantile=0.1)
    upper = count_quantile(mean, alpha=0.0, quantile=0.9)

    assert np.all(np.isfinite(lower))
    assert np.all(upper >= lower)


def test_negative_binomial_interval_is_wider_than_poisson() -> None:
    """Positive overdispersion should widen predictive intervals at fixed mean."""
    mean = np.array([50.0, 100.0, 200.0])
    poisson_width = count_quantile(mean, alpha=0.0, quantile=0.9) - count_quantile(
        mean, alpha=0.0, quantile=0.1
    )
    nb_width = count_quantile(mean, alpha=0.2, quantile=0.9) - count_quantile(
        mean, alpha=0.2, quantile=0.1
    )

    assert np.all(nb_width > poisson_width)


def test_dispersion_selection_uses_validation_loss() -> None:
    """A highly variable validation sample should prefer an overdispersed candidate."""
    validation = pd.DataFrame(
        {
            "y_true": [0.0, 40.0, 0.0, 40.0, 0.0, 40.0] * 20,
            "y_pred": [20.0] * 120,
        }
    )

    selected, table = select_dispersion_alpha(validation, alpha_grid=(0.0, 0.2, 0.8))

    assert selected > 0.0
    assert set(table["alpha"]) == {0.0, 0.2, 0.8}


def test_summary_reports_interval_calibration() -> None:
    """Probabilistic summaries should expose coverage and sharpness."""
    frame = pd.DataFrame({"y_true": [9.0, 10.0, 11.0], "y_pred": [10.0, 10.0, 10.0]})
    result = probabilistic_summary(frame, alpha=0.0)

    assert 0.0 <= result["interval_coverage"] <= 1.0
    assert result["mean_interval_width"] >= 0.0
    assert result["mean_pinball"] >= 0.0


def test_negative_alpha_is_rejected() -> None:
    """Dispersion cannot be negative."""
    with pytest.raises(ValueError, match="non-negative"):
        count_quantile(np.array([1.0]), alpha=-0.1, quantile=0.5)
