"""Variance reduction utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass(frozen=True)
class CupedResult:
    """CUPED-adjusted values and diagnostics."""

    adjusted: NDArray[np.float64]
    theta: float
    variance_reduction: float


def apply_cuped(
    df: pd.DataFrame,
    metric_col: str,
    covariate_col: str,
    *,
    adjusted_col: str | None = None,
) -> pd.DataFrame:
    """Return a DataFrame with a CUPED-adjusted metric column."""

    missing = [col for col in (metric_col, covariate_col) if col not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    clean = df.dropna(subset=[metric_col, covariate_col]).copy()
    if clean.empty:
        raise ValueError("no valid rows remain after dropping missing metric/covariate values")

    result = cuped_values(clean[metric_col].to_numpy(), clean[covariate_col].to_numpy())
    output_col = adjusted_col or f"{metric_col}_cuped"
    clean[output_col] = result.adjusted
    clean["cuped_theta"] = result.theta
    clean["cuped_variance_reduction"] = result.variance_reduction
    return clean


def cuped_values(
    metric: list[float] | NDArray[np.float64],
    covariate: list[float] | NDArray[np.float64],
) -> CupedResult:
    """Return CUPED-adjusted metric values for a single covariate."""

    y = np.asarray(metric, dtype=float)
    x = np.asarray(covariate, dtype=float)
    if y.shape != x.shape:
        raise ValueError("metric and covariate must have the same shape")
    if y.size == 0:
        raise ValueError("metric and covariate cannot be empty")
    if np.isnan(y).any() or np.isnan(x).any():
        raise ValueError("metric and covariate cannot contain NaN values")

    x_variance = float(np.var(x))
    if x_variance == 0:
        return CupedResult(adjusted=y.copy(), theta=0.0, variance_reduction=0.0)

    theta = float(np.cov(y, x)[0, 1] / x_variance)
    adjusted = y - theta * (x - float(np.mean(x)))
    original_variance = float(np.var(y))
    adjusted_variance = float(np.var(adjusted))
    variance_reduction = 0.0
    if original_variance > 0:
        variance_reduction = (original_variance - adjusted_variance) / original_variance
    return CupedResult(
        adjusted=adjusted,
        theta=theta,
        variance_reduction=float(variance_reduction),
    )
