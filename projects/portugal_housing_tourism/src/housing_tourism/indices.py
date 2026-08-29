"""Index definitions used in the housing-tourism analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


def _validate_numeric_series(series: pd.Series, *, name: str) -> None:
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} must be a pandas Series, got {type(series)!r}.")
    if not is_numeric_dtype(series):
        raise TypeError(f"{name} must be numeric, got dtype {series.dtype!r}.")


def tourist_housing_conversion_rate(
    al_units: pd.Series,
    housing_stock: pd.Series,
    *,
    per: float = 1_000.0,
) -> pd.Series:
    """Compute short-term accommodation units per conventional dwellings."""
    _validate_numeric_series(al_units, name="al_units")
    _validate_numeric_series(housing_stock, name="housing_stock")
    if not isinstance(per, (int, float)) or isinstance(per, bool):
        raise TypeError("per must be a real number.")
    if not np.isfinite(float(per)) or float(per) <= 0:
        raise ValueError("per must be finite and strictly positive.")
    observed = al_units.notna() & housing_stock.notna()
    if (al_units[observed] < 0).any():
        raise ValueError("al_units cannot be negative.")
    if (housing_stock[observed] <= 0).any():
        raise ValueError("housing_stock must be strictly positive when observed.")
    result = float(per) * al_units.astype(float) / housing_stock.astype(float)
    result.name = "thcr"
    return result


def tourism_intensity(
    overnight_stays: pd.Series,
    resident_population: pd.Series,
) -> pd.Series:
    """Compute annual tourist overnight stays per resident."""
    _validate_numeric_series(overnight_stays, name="overnight_stays")
    _validate_numeric_series(resident_population, name="resident_population")
    observed = overnight_stays.notna() & resident_population.notna()
    if (overnight_stays[observed] < 0).any():
        raise ValueError("overnight_stays cannot be negative.")
    if (resident_population[observed] <= 0).any():
        raise ValueError("resident_population must be strictly positive when observed.")
    result = overnight_stays.astype(float) / resident_population.astype(float)
    result.name = "tourism_intensity"
    return result


def local_housing_decoupling_index(
    frame: pd.DataFrame,
    *,
    group_col: str = "geo_code",
    year_col: str = "year",
    rent_col: str = "rent_eur_m2",
    income_col: str = "income_eur",
    base_year: int | None = 2017,
) -> pd.Series:
    """Compute a municipality-specific rent-to-income decoupling index."""
    required = {group_col, year_col, rent_col, income_col}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    _validate_numeric_series(frame[rent_col], name=rent_col)
    _validate_numeric_series(frame[income_col], name=income_col)
    observed = frame[rent_col].notna() & frame[income_col].notna()
    if (frame.loc[observed, rent_col] <= 0).any():
        raise ValueError(f"{rent_col} must be strictly positive when observed.")
    if (frame.loc[observed, income_col] <= 0).any():
        raise ValueError(f"{income_col} must be strictly positive when observed.")
    ratio = frame[rent_col].astype(float) / frame[income_col].astype(float)
    if base_year is None:
        ordered = frame.assign(_ratio=ratio).sort_values([group_col, year_col])
        base_by_group = ordered.groupby(group_col, sort=False)["_ratio"].first()
    else:
        if not isinstance(base_year, int) or isinstance(base_year, bool):
            raise TypeError("base_year must be an int or None.")
        base_rows = frame.loc[frame[year_col] == base_year, [group_col]].copy()
        base_rows["_ratio"] = ratio.loc[base_rows.index]
        base_by_group = base_rows.drop_duplicates(group_col).set_index(group_col)["_ratio"]
    base_ratio = frame[group_col].map(base_by_group)
    result = 100.0 * ratio / base_ratio
    result.name = "lhdi"
    return result


def external_housing_pressure_index(
    predicted_observed: pd.Series,
    predicted_counterfactual: pd.Series,
) -> pd.Series:
    """Compute the model-based observed/counterfactual rent ratio index."""
    _validate_numeric_series(predicted_observed, name="predicted_observed")
    _validate_numeric_series(predicted_counterfactual, name="predicted_counterfactual")
    observed = predicted_observed.notna() & predicted_counterfactual.notna()
    if (predicted_observed[observed] < 0).any():
        raise ValueError("predicted_observed cannot be negative.")
    if (predicted_counterfactual[observed] <= 0).any():
        raise ValueError("predicted_counterfactual must be strictly positive.")
    result = 100.0 * predicted_observed.astype(float) / predicted_counterfactual.astype(float)
    result.name = "ehpi"
    return result
