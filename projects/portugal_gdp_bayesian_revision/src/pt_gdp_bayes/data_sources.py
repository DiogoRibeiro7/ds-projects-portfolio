"""Data loading utilities for the Portugal GDP Bayesian revision project.

The project is designed to be reproducible when run on a machine with internet
access. The execution environment used to create this repository may not have
external network access, so these functions fail with clear error messages when
an API cannot be reached.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd
import requests


@dataclass(frozen=True)
class WorldBankIndicator:
    """World Bank indicator metadata.

    Attributes
    ----------
    column_name:
        Name to use in the returned dataframe.
    code:
        World Bank indicator code.
    """

    column_name: str
    code: str


class DataDownloadError(RuntimeError):
    """Raised when an external data source cannot be downloaded."""


def load_known_observations(path: str | Path) -> pd.DataFrame:
    """Load manually anchored observations used by the notebook.

    Parameters
    ----------
    path:
        Path to ``known_observations.csv``.

    Returns
    -------
    pandas.DataFrame
        Normalized table with variable, country, year, value and source fields.
    """

    df = pd.read_csv(path)
    required = {"variable", "country", "year", "value", "unit", "source"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in known observations: {missing}")
    df["year"] = df["year"].astype(int)
    df["value"] = pd.to_numeric(df["value"], errors="raise")
    return df


def get_known_value(observations: pd.DataFrame, variable: str, year: int) -> float:
    """Extract a single numeric value from the known-observations table."""

    rows = observations[(observations["variable"] == variable) & (observations["year"] == year)]
    if rows.empty:
        raise KeyError(f"No known observation found for variable={variable!r}, year={year}")
    if len(rows) > 1:
        raise ValueError(f"Multiple observations found for variable={variable!r}, year={year}")
    return float(rows.iloc[0]["value"])


def fetch_world_bank_indicator(
    country: str,
    indicator: WorldBankIndicator,
    *,
    start_year: int | None = None,
    end_year: int | None = None,
    timeout_seconds: float = 30.0,
) -> pd.DataFrame:
    """Fetch one World Bank indicator as a tidy dataframe.

    Parameters
    ----------
    country:
        ISO-3 or World Bank country code, e.g. ``"PRT"``.
    indicator:
        Indicator metadata.
    start_year, end_year:
        Optional inclusive year bounds.
    timeout_seconds:
        HTTP timeout.

    Returns
    -------
    pandas.DataFrame
        Dataframe with columns ``year`` and ``indicator.column_name``.
    """

    url = (
        f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator.code}"
        "?format=json&per_page=20000"
    )
    try:
        response = requests.get(url, timeout=timeout_seconds)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # pragma: no cover - depends on network
        raise DataDownloadError(f"Could not download World Bank indicator {indicator.code}: {exc}") from exc

    if not isinstance(payload, list) or len(payload) < 2:
        raise DataDownloadError(f"Unexpected World Bank response for {indicator.code}")

    records = []
    for item in payload[1]:
        value = item.get("value")
        if value is None:
            continue
        year = int(item["date"])
        if start_year is not None and year < start_year:
            continue
        if end_year is not None and year > end_year:
            continue
        records.append({"year": year, indicator.column_name: float(value)})

    return pd.DataFrame.from_records(records).sort_values("year").reset_index(drop=True)


def fetch_world_bank_panel(
    country: str,
    indicators: Mapping[str, str],
    *,
    start_year: int | None = None,
    end_year: int | None = None,
) -> pd.DataFrame:
    """Fetch and merge several World Bank indicators.

    Parameters
    ----------
    country:
        Country code, e.g. ``"PRT"``.
    indicators:
        Mapping from output column name to World Bank indicator code.
    start_year, end_year:
        Optional inclusive year bounds.

    Returns
    -------
    pandas.DataFrame
        One row per year and one column per requested indicator.
    """

    frames: list[pd.DataFrame] = []
    for column_name, code in indicators.items():
        frames.append(
            fetch_world_bank_indicator(
                country,
                WorldBankIndicator(column_name=column_name, code=code),
                start_year=start_year,
                end_year=end_year,
            )
        )

    if not frames:
        raise ValueError("At least one indicator is required")

    panel = frames[0]
    for frame in frames[1:]:
        panel = panel.merge(frame, on="year", how="outer")

    return panel.sort_values("year").reset_index(drop=True)


def patch_manual_observation(
    df: pd.DataFrame,
    *,
    year: int,
    column: str,
    value: float,
    source_column: str = "manual_patch_notes",
    note: str | None = None,
) -> pd.DataFrame:
    """Patch or append a manually anchored observation.

    This is used for the revised 2025 INE population estimate, because many
    international APIs can lag official national revisions.
    """

    out = df.copy()
    if "year" not in out.columns:
        raise ValueError("Input dataframe must contain a 'year' column")
    if column not in out.columns:
        out[column] = pd.NA
    if source_column not in out.columns:
        out[source_column] = ""

    mask = out["year"] == year
    if mask.any():
        out.loc[mask, column] = float(value)
        out.loc[mask, source_column] = note or "manual observation patch"
    else:
        new_row = {col: pd.NA for col in out.columns}
        new_row["year"] = year
        new_row[column] = float(value)
        new_row[source_column] = note or "manual observation patch"
        out = pd.concat([out, pd.DataFrame([new_row])], ignore_index=True)

    return out.sort_values("year").reset_index(drop=True)


def keep_longest_population_window(*frames: pd.DataFrame, population_col: str = "population") -> pd.DataFrame:
    """Return the population dataframe with the largest non-null year span.

    Parameters
    ----------
    *frames:
        Candidate dataframes. Each must contain ``year`` and ``population_col``.
    population_col:
        Population column name.

    Returns
    -------
    pandas.DataFrame
        Candidate with the largest observed span.
    """

    if not frames:
        raise ValueError("At least one dataframe is required")

    best_frame: pd.DataFrame | None = None
    best_span = -1
    for frame in frames:
        if "year" not in frame.columns or population_col not in frame.columns:
            continue
        valid = frame.dropna(subset=[population_col])
        if valid.empty:
            continue
        span = int(valid["year"].max() - valid["year"].min())
        if span > best_span:
            best_span = span
            best_frame = frame.copy()

    if best_frame is None:
        raise ValueError("No candidate contained a valid population series")
    return best_frame.sort_values("year").reset_index(drop=True)
