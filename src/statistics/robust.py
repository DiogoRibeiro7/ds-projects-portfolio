"""Robust estimation utilities."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def huber_smooth(values: Iterable[float], delta: float = 1.0) -> np.ndarray:
    values = np.asarray(list(values), dtype=float)
    if delta <= 0:
        raise ValueError("delta must be positive")
    mask = np.abs(values) <= delta
    out = np.empty_like(values)
    out[mask] = values[mask]
    out[~mask] = delta * np.sign(values[~mask])
    return out


def winsorize_by_quantile(
    values: Iterable[float], lower: float = 0.05, upper: float = 0.95
) -> np.ndarray:
    values = np.asarray(list(values), dtype=float)
    if not 0 <= lower < upper <= 1:
        raise ValueError("invalid quantiles")
    lo = np.quantile(values, lower)
    hi = np.quantile(values, upper)
    return np.clip(values, lo, hi)
