"""Metrics for urban heat island exposure and representation analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from uhi_exposure.schema import ExposureColumns, validate_exposure_cells

GROUP_COLUMNS: dict[str, str] = {
    "children_0_14": "age_0_14",
    "older_65_plus": "age_65_plus",
    "not_employed": "not_employed",
    "born_outside_eu": "born_outside_eu",
}


def add_exposure_flag(
    df: pd.DataFrame,
    threshold: float = 2.0,
    columns: ExposureColumns | None = None,
) -> pd.DataFrame:
    """Return a copy of ``df`` with a boolean urban heat exposure flag.

    Parameters
    ----------
    df:
        Cell-level input table.
    threshold:
        UHI intensity threshold in degrees Celsius. Cells with intensity greater
        than or equal to this threshold are marked as exposed.
    columns:
        Optional column mapping.

    Returns
    -------
    pandas.DataFrame
        A validated copy of the input with ``is_uhi_exposed`` as a boolean column.
    """
    columns = columns or ExposureColumns()
    validate_exposure_cells(df, columns)

    result = df.copy()
    if columns.is_uhi_exposed in result.columns:
        result[columns.is_uhi_exposed] = result[columns.is_uhi_exposed].astype(bool)
    else:
        result[columns.is_uhi_exposed] = result[columns.uhi_intensity_celsius] >= threshold

    result["not_employed"] = result[columns.population_total] - result[columns.employed]
    return result


def _safe_divide(numerator: float, denominator: float) -> float:
    """Divide two numbers and return NaN for zero denominators."""
    if denominator == 0:
        return float("nan")
    return numerator / denominator


def compute_city_exposure_summary(
    df: pd.DataFrame,
    threshold: float = 2.0,
    columns: ExposureColumns | None = None,
) -> pd.DataFrame:
    """Compute city-level population exposure summary.

    Parameters
    ----------
    df:
        Cell-level input table.
    threshold:
        UHI intensity threshold used when ``is_uhi_exposed`` is absent.
    columns:
        Optional column mapping.

    Returns
    -------
    pandas.DataFrame
        One row per city with total population, exposed population, and exposure share.
    """
    columns = columns or ExposureColumns()
    prepared = add_exposure_flag(df, threshold=threshold, columns=columns)

    records: list[dict[str, float | str]] = []
    for city, city_df in prepared.groupby(columns.city, sort=True):
        total_population = float(city_df[columns.population_total].sum())
        exposed_population = float(
            city_df.loc[city_df[columns.is_uhi_exposed], columns.population_total].sum()
        )
        records.append(
            {
                "city": city,
                "total_population": total_population,
                "exposed_population": exposed_population,
                "exposed_population_share": _safe_divide(exposed_population, total_population),
                "mean_uhi_intensity_celsius": float(
                    np.average(
                        city_df[columns.uhi_intensity_celsius],
                        weights=city_df[columns.population_total],
                    )
                ),
                "mean_uhi_intensity_exposed_celsius": float(
                    np.average(
                        city_df.loc[
                            city_df[columns.is_uhi_exposed], columns.uhi_intensity_celsius
                        ],
                        weights=city_df.loc[
                            city_df[columns.is_uhi_exposed], columns.population_total
                        ],
                    )
                )
                if exposed_population > 0
                else float("nan"),
            }
        )

    return pd.DataFrame.from_records(records)


def compute_group_representation(
    df: pd.DataFrame,
    threshold: float = 2.0,
    columns: ExposureColumns | None = None,
) -> pd.DataFrame:
    """Compute group representation inside exposed UHI areas.

    For each city and population group, this function compares the group's share
    inside UHI-exposed cells with the group's share in the city as a whole.

    Parameters
    ----------
    df:
        Cell-level input table.
    threshold:
        UHI intensity threshold used when ``is_uhi_exposed`` is absent.
    columns:
        Optional column mapping.

    Returns
    -------
    pandas.DataFrame
        Long table with one row per city and group.
    """
    columns = columns or ExposureColumns()
    prepared = add_exposure_flag(df, threshold=threshold, columns=columns)

    records: list[dict[str, float | str]] = []
    for city, city_df in prepared.groupby(columns.city, sort=True):
        total_population = float(city_df[columns.population_total].sum())
        exposed_df = city_df.loc[city_df[columns.is_uhi_exposed]].copy()
        exposed_population = float(exposed_df[columns.population_total].sum())

        for group_name, group_column in GROUP_COLUMNS.items():
            full_group_count = float(city_df[group_column].sum())
            exposed_group_count = float(exposed_df[group_column].sum())
            city_share = _safe_divide(full_group_count, total_population)
            exposed_share = _safe_divide(exposed_group_count, exposed_population)
            difference_pp = (exposed_share - city_share) * 100
            ratio = _safe_divide(exposed_share, city_share)

            records.append(
                {
                    "city": city,
                    "group": group_name,
                    "city_group_count": full_group_count,
                    "exposed_group_count": exposed_group_count,
                    "city_group_share": city_share,
                    "exposed_group_share": exposed_share,
                    "difference_percentage_points": difference_pp,
                    "representation_ratio": ratio,
                    "is_overrepresented": bool(ratio > 1.0) if not np.isnan(ratio) else False,
                }
            )

    return pd.DataFrame.from_records(records)
