"""Descriptive decomposition helpers for Lisbon housing-affordability analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EpisodeDecomposition:
    """Observed change decomposition for one start/end episode."""

    start_year: int
    end_year: int
    rent_change_pct: float
    income_change_pct: float
    rent_income_ratio_change_pct: float
    rent_log_change_pct: float
    income_log_change_pct: float
    affordability_log_gap_pct: float
    tourism_intensity_change_pct: float

    def as_record(self) -> dict[str, int | float]:
        """Return the decomposition as a serialisable record."""
        return asdict(self)


def index_to_base_year(
    frame: pd.DataFrame,
    *,
    value_col: str,
    base_year: int,
    year_col: str = "year",
) -> pd.Series:
    """Index a positive observed series to 100 in one base year."""
    required = {year_col, value_col}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    if not isinstance(base_year, int) or isinstance(base_year, bool):
        raise TypeError("base_year must be an integer.")

    values = pd.to_numeric(frame[value_col], errors="raise").astype(float)
    observed = values.notna()
    if (values.loc[observed] <= 0).any():
        raise ValueError(f"{value_col} must be strictly positive when observed.")

    base_mask = frame[year_col].eq(base_year) & observed
    if int(base_mask.sum()) != 1:
        raise ValueError(f"Expected exactly one observed {value_col} value in {base_year}.")
    base_value = float(values.loc[base_mask].iloc[0])

    result = 100.0 * values / base_value
    result.name = f"{value_col}_index_{base_year}"
    return result


def _percentage_change(start: float, end: float, *, name: str) -> float:
    if not np.isfinite(start) or not np.isfinite(end):
        raise ValueError(f"{name} endpoints must be finite.")
    if start <= 0 or end <= 0:
        raise ValueError(f"{name} endpoints must be strictly positive.")
    return 100.0 * (end / start - 1.0)


def _log_change_pct(start: float, end: float, *, name: str) -> float:
    if not np.isfinite(start) or not np.isfinite(end):
        raise ValueError(f"{name} endpoints must be finite.")
    if start <= 0 or end <= 0:
        raise ValueError(f"{name} endpoints must be strictly positive.")
    return 100.0 * float(np.log(end / start))


def decompose_episode(
    frame: pd.DataFrame,
    *,
    start_year: int,
    end_year: int,
    year_col: str = "year",
    rent_col: str = "rent_eur_m2",
    income_col: str = "income_eur",
    tourism_col: str = "tourism_intensity",
) -> EpisodeDecomposition:
    """Decompose observed affordability change into rent and income log changes."""
    required = {year_col, rent_col, income_col, tourism_col}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    if end_year <= start_year:
        raise ValueError("end_year must be greater than start_year.")

    indexed = frame.set_index(year_col)
    if start_year not in indexed.index or end_year not in indexed.index:
        raise ValueError("Both episode endpoints must be present in the frame.")
    if not indexed.index.is_unique:
        raise ValueError(f"{year_col} must uniquely identify rows.")

    start = indexed.loc[start_year]
    end = indexed.loc[end_year]
    start_rent = float(start[rent_col])
    end_rent = float(end[rent_col])
    start_income = float(start[income_col])
    end_income = float(end[income_col])
    start_tourism = float(start[tourism_col])
    end_tourism = float(end[tourism_col])

    rent_log = _log_change_pct(start_rent, end_rent, name=rent_col)
    income_log = _log_change_pct(start_income, end_income, name=income_col)
    ratio_start = start_rent / start_income
    ratio_end = end_rent / end_income

    return EpisodeDecomposition(
        start_year=start_year,
        end_year=end_year,
        rent_change_pct=_percentage_change(start_rent, end_rent, name=rent_col),
        income_change_pct=_percentage_change(start_income, end_income, name=income_col),
        rent_income_ratio_change_pct=_percentage_change(
            ratio_start, ratio_end, name="rent_income_ratio"
        ),
        rent_log_change_pct=rent_log,
        income_log_change_pct=income_log,
        affordability_log_gap_pct=rent_log - income_log,
        tourism_intensity_change_pct=_percentage_change(
            start_tourism, end_tourism, name=tourism_col
        ),
    )
