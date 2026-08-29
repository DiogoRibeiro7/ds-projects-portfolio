"""Forecast and operational evaluation metrics for the mobility project."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def _as_float_array(values: Sequence[float] | npt.ArrayLike, *, name: str) -> FloatArray:
    """Convert input values to a finite one-dimensional float array.

    Args:
        values: Numeric values to validate.
        name: Variable name used in validation errors.

    Returns:
        A one-dimensional NumPy array of finite floats.

    Raises:
        ValueError: If the input is empty, non-finite, or not one-dimensional.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def weighted_absolute_percentage_error(
    y_true: Sequence[float] | npt.ArrayLike,
    y_pred: Sequence[float] | npt.ArrayLike,
) -> float:
    """Compute WAPE as total absolute error divided by total observed demand.

    WAPE is undefined when total observed demand is zero; this function raises
    rather than silently returning an arbitrary value.
    """
    observed = _as_float_array(y_true, name="y_true")
    predicted = _as_float_array(y_pred, name="y_pred")
    if observed.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have identical shapes.")
    if np.any(observed < 0.0) or np.any(predicted < 0.0):
        raise ValueError("Demand values must be non-negative.")

    denominator = float(np.sum(observed))
    if denominator == 0.0:
        raise ValueError("WAPE is undefined when total observed demand is zero.")

    return float(np.sum(np.abs(observed - predicted)) / denominator)


def pinball_loss(
    y_true: Sequence[float] | npt.ArrayLike,
    y_quantile: Sequence[float] | npt.ArrayLike,
    *,
    quantile: float,
) -> float:
    """Compute mean pinball loss for a predictive quantile.

    Args:
        y_true: Realised demand values.
        y_quantile: Forecast values for the requested quantile.
        quantile: Quantile level strictly between zero and one.
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must lie strictly between 0 and 1.")

    observed = _as_float_array(y_true, name="y_true")
    forecast = _as_float_array(y_quantile, name="y_quantile")
    if observed.shape != forecast.shape:
        raise ValueError("y_true and y_quantile must have identical shapes.")

    residual = observed - forecast
    losses = np.maximum(quantile * residual, (quantile - 1.0) * residual)
    return float(np.mean(losses))


def empirical_interval_coverage(
    y_true: Sequence[float] | npt.ArrayLike,
    lower: Sequence[float] | npt.ArrayLike,
    upper: Sequence[float] | npt.ArrayLike,
) -> float:
    """Return the empirical fraction of observations inside forecast intervals."""
    observed = _as_float_array(y_true, name="y_true")
    lower_bound = _as_float_array(lower, name="lower")
    upper_bound = _as_float_array(upper, name="upper")
    if observed.shape != lower_bound.shape or observed.shape != upper_bound.shape:
        raise ValueError("y_true, lower, and upper must have identical shapes.")
    if np.any(lower_bound > upper_bound):
        raise ValueError("lower must not exceed upper.")

    covered = (observed >= lower_bound) & (observed <= upper_bound)
    return float(np.mean(covered))
