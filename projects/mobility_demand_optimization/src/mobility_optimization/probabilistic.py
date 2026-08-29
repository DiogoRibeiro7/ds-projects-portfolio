"""Probabilistic count calibration for mobility demand forecasts."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

DEFAULT_ALPHA_GRID: tuple[float, ...] = (0.0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8)
DEFAULT_QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)


def _validate_forecasts(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate a point-forecast table for probabilistic calibration."""
    required = {"y_true", "y_pred"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Forecast frame is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Forecast frame must not be empty.")

    result = frame.copy()
    for column in ("y_true", "y_pred"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
        if result[column].isna().any() or not np.isfinite(result[column]).all():
            raise ValueError(f"{column} must contain only finite numeric values.")
        if (result[column] < 0).any():
            raise ValueError(f"{column} must be non-negative.")
    return result


def _validate_quantiles(quantiles: Sequence[float]) -> tuple[float, ...]:
    """Validate a strictly increasing sequence of quantile levels."""
    values = tuple(float(q) for q in quantiles)
    if not values:
        raise ValueError("At least one quantile is required.")
    if any(not 0.0 < q < 1.0 for q in values):
        raise ValueError("Quantiles must lie strictly between zero and one.")
    if tuple(sorted(set(values))) != values:
        raise ValueError("Quantiles must be unique and strictly increasing.")
    return values


def count_quantile(mean: np.ndarray, *, alpha: float, quantile: float) -> np.ndarray:
    """Return Poisson or Negative-Binomial predictive quantiles.

    The Negative-Binomial parameterization is

    ``Var(Y) = mu + alpha * mu**2``.

    ``alpha=0`` is the Poisson limit.
    """
    if alpha < 0 or not np.isfinite(alpha):
        raise ValueError("alpha must be finite and non-negative.")
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must lie strictly between zero and one.")

    mu = np.asarray(mean, dtype=np.float64)
    if mu.ndim != 1 or mu.size == 0:
        raise ValueError("mean must be a non-empty one-dimensional array.")
    if not np.isfinite(mu).all() or (mu < 0).any():
        raise ValueError("mean must contain finite non-negative values.")

    if alpha == 0.0:
        return poisson.ppf(quantile, mu=mu).astype(np.float64)

    shape = 1.0 / alpha
    probability = shape / (shape + mu)
    return nbinom.ppf(quantile, n=shape, p=probability).astype(np.float64)


def mean_pinball_score(
    frame: pd.DataFrame,
    *,
    alpha: float,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> float:
    """Return the mean pinball loss across requested predictive quantiles."""
    data = _validate_forecasts(frame)
    levels = _validate_quantiles(quantiles)
    observed = data["y_true"].to_numpy(dtype=np.float64)
    mean = data["y_pred"].to_numpy(dtype=np.float64)

    losses: list[float] = []
    for level in levels:
        prediction = count_quantile(mean, alpha=alpha, quantile=level)
        residual = observed - prediction
        loss = np.maximum(level * residual, (level - 1.0) * residual)
        losses.append(float(np.mean(loss)))
    return float(np.mean(losses))


def select_dispersion_alpha(
    validation: pd.DataFrame,
    *,
    alpha_grid: Iterable[float] = DEFAULT_ALPHA_GRID,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> tuple[float, pd.DataFrame]:
    """Select one dispersion value on validation data using mean pinball loss."""
    candidates = tuple(float(alpha) for alpha in alpha_grid)
    if not candidates:
        raise ValueError("alpha_grid must not be empty.")
    if any(alpha < 0 or not np.isfinite(alpha) for alpha in candidates):
        raise ValueError("alpha_grid must contain finite non-negative values.")
    if len(set(candidates)) != len(candidates):
        raise ValueError("alpha_grid must not contain duplicate values.")

    records = [
        {
            "alpha": alpha,
            "mean_pinball": mean_pinball_score(validation, alpha=alpha, quantiles=quantiles),
        }
        for alpha in candidates
    ]
    table = pd.DataFrame(records).sort_values(["mean_pinball", "alpha"], ignore_index=True)
    selected = float(table.iloc[0]["alpha"])
    return selected, table


def probabilistic_summary(
    frame: pd.DataFrame,
    *,
    alpha: float,
    lower_quantile: float = 0.1,
    upper_quantile: float = 0.9,
) -> dict[str, float]:
    """Summarize calibration, sharpness, and probabilistic loss for one count model."""
    if not lower_quantile < 0.5 < upper_quantile:
        raise ValueError("Quantile interval must satisfy lower < 0.5 < upper.")

    data = _validate_forecasts(frame)
    observed = data["y_true"].to_numpy(dtype=np.float64)
    mean = data["y_pred"].to_numpy(dtype=np.float64)
    lower = count_quantile(mean, alpha=alpha, quantile=lower_quantile)
    median = count_quantile(mean, alpha=alpha, quantile=0.5)
    upper = count_quantile(mean, alpha=alpha, quantile=upper_quantile)

    coverage = float(np.mean((observed >= lower) & (observed <= upper)))
    width = float(np.mean(upper - lower))
    quantile_score = mean_pinball_score(
        data,
        alpha=alpha,
        quantiles=(lower_quantile, 0.5, upper_quantile),
    )
    median_mae = float(np.mean(np.abs(observed - median)))

    return {
        "alpha": float(alpha),
        "interval_coverage": coverage,
        "mean_interval_width": width,
        "mean_pinball": quantile_score,
        "median_mae": median_mae,
    }
