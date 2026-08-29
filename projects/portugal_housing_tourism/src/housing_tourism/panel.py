"""Construction and validation of the canonical municipality-year panel."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .indices import (
    local_housing_decoupling_index,
    tourism_intensity,
    tourist_housing_conversion_rate,
)

KEYS = ["geo_code", "year"]


def validate_canonical_series(
    frame: pd.DataFrame,
    *,
    value_col: str,
    require_geo_name: bool = True,
) -> None:
    """Validate one canonical annual geography series before a panel merge."""
    required = {*KEYS, value_col}
    if require_geo_name:
        required.add("geo_name")
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    if frame.duplicated(KEYS).any():
        raise ValueError("Duplicate geography-year rows found.")
    if not pd.api.types.is_numeric_dtype(frame[value_col]):
        raise TypeError(f"{value_col} must be numeric.")


def merge_canonical_series(series: Iterable[tuple[pd.DataFrame, str]]) -> pd.DataFrame:
    """Outer-merge canonical annual series on geography and year."""
    pairs = list(series)
    if not pairs:
        raise ValueError("At least one series is required.")
    merged: pd.DataFrame | None = None
    for frame, value_col in pairs:
        validate_canonical_series(frame, value_col=value_col)
        subset = frame[["geo_code", "geo_name", "year", value_col]].copy()
        if merged is None:
            merged = subset
            continue
        merged = merged.merge(
            subset,
            on=["geo_code", "year"],
            how="outer",
            suffixes=("", "_new"),
            validate="one_to_one",
        )
        if "geo_name_new" in merged.columns:
            conflicting = (
                merged["geo_name"].notna()
                & merged["geo_name_new"].notna()
                & (merged["geo_name"] != merged["geo_name_new"])
            )
            if conflicting.any():
                raise ValueError("Conflicting geography names found.")
            merged["geo_name"] = merged["geo_name"].fillna(merged["geo_name_new"])
            merged = merged.drop(columns="geo_name_new")
    assert merged is not None
    return merged.sort_values(KEYS).reset_index(drop=True)


def add_core_indices(
    panel: pd.DataFrame,
    *,
    base_year: int | None = 2017,
    al_col: str = "al_units",
) -> pd.DataFrame:
    """Add THCR, LHDI and tourism intensity to a canonical panel."""
    required = {
        "geo_code",
        "year",
        "rent_eur_m2",
        "income_eur",
        "housing_stock",
        "resident_population",
        "overnight_stays",
        al_col,
    }
    missing = required.difference(panel.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    result = panel.copy()
    result["thcr"] = tourist_housing_conversion_rate(
        result[al_col],
        result["housing_stock"],
    )
    result["lhdi"] = local_housing_decoupling_index(result, base_year=base_year)
    result["tourism_intensity"] = tourism_intensity(
        result["overnight_stays"],
        result["resident_population"],
    )
    result["log_rent"] = np.where(
        result["rent_eur_m2"] > 0,
        np.log(result["rent_eur_m2"]),
        np.nan,
    )
    result["log_income"] = np.where(
        result["income_eur"] > 0,
        np.log(result["income_eur"]),
        np.nan,
    )
    return result
