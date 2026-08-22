from __future__ import annotations

import numpy as np
import pandas as pd


def representative_years(years: list[int] | np.ndarray) -> list[int]:
    """Pick early, middle, and recent years from an ordered year sequence."""
    ordered = sorted(int(year) for year in years)
    if not ordered:
        return []
    return [ordered[0], ordered[len(ordered) // 2], ordered[-1]]


def clipped_density(pdf_values: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    """Clip unstable density spikes to make comparison plots readable."""
    density = np.asarray(pdf_values, dtype=float)
    density = np.where(np.isfinite(density), density, np.nan)
    finite = density[np.isfinite(density)]
    cap = np.nanpercentile(finite, percentile) if finite.size else 1.0
    return np.clip(density, 0.0, cap)


def grouped_histogram_frame(year_bins: pd.DataFrame, x_cap: float = 5000.0) -> pd.DataFrame:
    """Prepare a grouped-data histogram frame from salary-bracket counts."""
    out = year_bins.copy()
    out["plot_upper"] = np.where(np.isfinite(out["upper"]), out["upper"], x_cap)
    out["width"] = out["plot_upper"] - out["lower"]
    out["share"] = out["count"] / out["count"].sum()
    out["density_height"] = out["share"] / out["width"]
    return out
