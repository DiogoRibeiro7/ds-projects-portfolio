"""Schema validation for UHI exposure input tables."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype, is_string_dtype


@dataclass(frozen=True)
class ExposureColumns:
    """Column names expected by the exposure analysis.

    The defaults match the CSV templates used in this repository. A dataclass is
    used so the same computation functions can be reused with renamed columns.
    """

    city: str = "city"
    cell_id: str = "cell_id"
    population_total: str = "population_total"
    age_0_14: str = "age_0_14"
    age_65_plus: str = "age_65_plus"
    employed: str = "employed"
    born_outside_eu: str = "born_outside_eu"
    uhi_intensity_celsius: str = "uhi_intensity_celsius"
    is_uhi_exposed: str = "is_uhi_exposed"

    @property
    def required(self) -> tuple[str, ...]:
        """Return required columns for a prepared exposure table."""
        return (
            self.city,
            self.cell_id,
            self.population_total,
            self.age_0_14,
            self.age_65_plus,
            self.employed,
            self.born_outside_eu,
            self.uhi_intensity_celsius,
        )

    @property
    def count_columns(self) -> tuple[str, ...]:
        """Return columns that must contain non-negative population counts."""
        return (
            self.population_total,
            self.age_0_14,
            self.age_65_plus,
            self.employed,
            self.born_outside_eu,
        )


def validate_exposure_cells(df: pd.DataFrame, columns: ExposureColumns | None = None) -> None:
    """Validate a cell-level UHI exposure table.

    Parameters
    ----------
    df:
        Input table with one row per grid cell or small area.
    columns:
        Optional column mapping. Defaults to :class:`ExposureColumns`.

    Raises
    ------
    ValueError
        If required columns are missing, have invalid types, or contain invalid values.
    """
    columns = columns or ExposureColumns()

    missing_columns = [column for column in columns.required if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if not is_string_dtype(df[columns.city]):
        raise ValueError(f"Column {columns.city!r} must contain city names as strings.")

    for column in columns.count_columns:
        if not is_numeric_dtype(df[column]):
            raise ValueError(f"Column {column!r} must be numeric.")
        if (df[column] < 0).any():
            raise ValueError(f"Column {column!r} contains negative values.")

    if not is_numeric_dtype(df[columns.uhi_intensity_celsius]):
        raise ValueError(f"Column {columns.uhi_intensity_celsius!r} must be numeric.")

    if columns.is_uhi_exposed in df.columns and not is_bool_dtype(df[columns.is_uhi_exposed]):
        valid_values = {0, 1, True, False}
        observed_values = set(df[columns.is_uhi_exposed].dropna().unique().tolist())
        if not observed_values.issubset(valid_values):
            raise ValueError(
                f"Column {columns.is_uhi_exposed!r} must be boolean or contain 0/1 values."
            )

    total = df[columns.population_total]
    for column in (columns.age_0_14, columns.age_65_plus, columns.employed, columns.born_outside_eu):
        if (df[column] > total).any():
            raise ValueError(
                f"Column {column!r} has values larger than {columns.population_total!r}."
            )
