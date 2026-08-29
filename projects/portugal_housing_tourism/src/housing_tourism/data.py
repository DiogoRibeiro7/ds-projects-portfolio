"""Data acquisition and normalisation helpers."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

INE_DATA_URL = "https://www.ine.pt/ine/json_indicador/pindica.jsp"
INE_META_URL = "https://www.ine.pt/ine/json_indicador/pindicaMeta.jsp"


@dataclass(frozen=True)
class INEIndicator:
    """Definition of one INE indicator used by the project."""

    code: str
    value_name: str
    description: str

    def __post_init__(self) -> None:
        if not re.fullmatch(r"\d{7}", self.code):
            raise ValueError(f"INE indicator code must have 7 digits: {self.code!r}")
        if not self.value_name.strip():
            raise ValueError("value_name cannot be empty.")
        if not self.description.strip():
            raise ValueError("description cannot be empty.")


class INEClient:
    """Minimal client for the public INE JSON indicator endpoint."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        timeout_seconds: float = 60.0,
        user_agent: str = "portugal-housing-tourism/0.2",
    ) -> None:
        if not isinstance(cache_dir, Path):
            raise TypeError("cache_dir must be a pathlib.Path.")
        if not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool):
            raise TypeError("timeout_seconds must be a real number.")
        if float(timeout_seconds) <= 0:
            raise ValueError("timeout_seconds must be positive.")
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = float(timeout_seconds)
        self.headers = {"User-Agent": user_agent}

    def fetch_indicator(
        self,
        indicator_code: str,
        *,
        dimensions: Mapping[str, str] | None = None,
        language: str = "PT",
        refresh: bool = False,
    ) -> list[dict[str, Any]]:
        """Fetch and cache an INE indicator JSON payload."""
        if not re.fullmatch(r"\d{7}", indicator_code):
            raise ValueError("indicator_code must contain exactly seven digits.")
        if dimensions is not None and not isinstance(dimensions, Mapping):
            raise TypeError("dimensions must be a mapping or None.")
        suffix = "_all" if not dimensions else "_filtered"
        cache_path = self.cache_dir / f"{indicator_code}{suffix}.json"
        if cache_path.exists() and not refresh:
            return _validate_payload(
                json.loads(cache_path.read_text(encoding="utf-8")),
            )
        params: dict[str, str] = {
            "op": "2",
            "varcd": indicator_code,
            "lang": language,
        }
        if dimensions:
            for key, value in dimensions.items():
                if not re.fullmatch(r"Dim\d+", str(key)):
                    raise ValueError(f"Invalid INE dimension name: {key!r}")
                params[str(key)] = str(value)
        response = requests.get(
            INE_DATA_URL,
            params=params,
            headers=self.headers,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = _validate_payload(response.json())
        cache_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return payload

    def fetch_metadata(
        self,
        indicator_code: str,
        *,
        language: str = "PT",
        refresh: bool = False,
    ) -> Any:
        """Fetch and cache INE metadata for one indicator."""
        if not re.fullmatch(r"\d{7}", indicator_code):
            raise ValueError("indicator_code must contain exactly seven digits.")
        cache_path = self.cache_dir / f"{indicator_code}_meta.json"
        if cache_path.exists() and not refresh:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        response = requests.get(
            INE_META_URL,
            params={"varcd": indicator_code, "lang": language},
            headers=self.headers,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        cache_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return payload


def _validate_payload(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, list) or not payload:
        raise ValueError("Unexpected INE payload: expected a non-empty list.")
    if not all(isinstance(item, dict) for item in payload):
        raise ValueError("Unexpected INE payload: all top-level items must be objects.")
    return payload


def parse_ine_numeric(value: Any) -> float:
    """Convert INE numeric strings to float while preserving missing values."""
    if value is None:
        return float("nan")
    if isinstance(value, bool):
        raise TypeError("Boolean values are not valid numeric observations.")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    if text in {"", ":", "x", "X", "...", "..", "-"}:
        return float("nan")
    return float(text.replace("\u00a0", "").replace(" ", "").replace(",", "."))


def flatten_ine_payload(payload: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten a standard INE indicator response without guessing dimensions."""
    validated = _validate_payload(payload)
    records: list[dict[str, Any]] = []
    for top in validated:
        data = top.get("Dados")
        if not isinstance(data, dict):
            raise ValueError("Unexpected INE payload: missing object-valued 'Dados'.")
        metadata = {
            "indicator_code": top.get("IndicadorCod"),
            "indicator_name": top.get("IndicadorDsg"),
            "last_update": top.get("DataUltimoAtualizacao"),
        }
        for period, observations in data.items():
            if not isinstance(observations, list):
                raise ValueError("Unexpected INE payload: each period must contain a list.")
            for observation in observations:
                if not isinstance(observation, dict):
                    raise ValueError("Unexpected INE payload: observation must be an object.")
                row = {**metadata, "period": str(period), **observation}
                if "valor" in row:
                    row["value"] = parse_ine_numeric(row["valor"])
                records.append(row)
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise ValueError("INE payload contained no observations.")
    return frame


def infer_year(period: pd.Series) -> pd.Series:
    """Extract the first four-digit year from INE period labels."""
    if not isinstance(period, pd.Series):
        raise TypeError("period must be a pandas Series.")
    year = period.astype(str).str.extract(r"((?:19|20)\d{2})", expand=False)
    return pd.to_numeric(year, errors="coerce").astype("Int64")


def save_flat_ine_indicator(
    client: INEClient,
    indicator_code: str,
    output_path: Path,
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch, flatten and save one INE indicator to Parquet."""
    if not isinstance(output_path, Path):
        raise TypeError("output_path must be a pathlib.Path.")
    payload = client.fetch_indicator(indicator_code, refresh=refresh)
    frame = flatten_ine_payload(payload)
    frame["year"] = infer_year(frame["period"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    return frame


def dimension_columns(frame: pd.DataFrame) -> list[str]:
    """Return human-readable INE dimension-label columns found in a flat frame."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")
    return sorted(column for column in frame.columns if re.fullmatch(r"dim_\d+_t", str(column)))


def describe_dimensions(frame: pd.DataFrame, *, max_values: int = 20) -> dict[str, list[str]]:
    """Summarise dimension labels before any scientific filtering."""
    if not isinstance(max_values, int) or isinstance(max_values, bool) or max_values <= 0:
        raise ValueError("max_values must be a positive integer.")
    return {
        column: frame[column].dropna().astype(str).drop_duplicates().tolist()[:max_values]
        for column in dimension_columns(frame)
    }


def canonicalise_ine_measure(
    frame: pd.DataFrame,
    *,
    value_name: str,
    filters: Mapping[str, str] | None = None,
    geo_code_col: str = "geocod",
    geo_name_col: str = "geodsg",
    year_col: str = "year",
    minimum_year: int | None = 2017,
) -> pd.DataFrame:
    """Convert a filtered flat INE indicator into the canonical annual schema."""
    required = {geo_code_col, geo_name_col, year_col, "value"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing INE columns: {sorted(missing)}")
    data = frame.copy()
    if filters:
        for column, expected in filters.items():
            if column not in data.columns:
                raise KeyError(f"Filter column not found: {column!r}")
            data = data.loc[data[column].astype(str).eq(str(expected))].copy()
    if minimum_year is not None:
        if not isinstance(minimum_year, int) or isinstance(minimum_year, bool):
            raise TypeError("minimum_year must be an int or None.")
        data = data.loc[pd.to_numeric(data[year_col], errors="coerce") >= minimum_year].copy()
    output = data[[geo_code_col, geo_name_col, year_col, "value"]].rename(
        columns={
            geo_code_col: "geo_code",
            geo_name_col: "geo_name",
            year_col: "year",
            "value": value_name,
        }
    )
    output["geo_code"] = output["geo_code"].astype(str)
    output["geo_name"] = output["geo_name"].astype(str)
    output["year"] = pd.to_numeric(output["year"], errors="raise").astype(int)
    output[value_name] = pd.to_numeric(output[value_name], errors="coerce")
    output = output.dropna(subset=[value_name])
    if output.empty:
        raise ValueError("No observations remain after applying INE filters.")
    if output.duplicated(["geo_code", "year"]).any():
        raise ValueError(
            "Filters do not identify a unique geography-year measure. Inspect remaining dimensions."
        )
    return output.sort_values(["geo_code", "year"]).reset_index(drop=True)


def infer_total_filters(
    frame: pd.DataFrame,
    *,
    candidates: tuple[str, ...] = ("Total", "TOTAL", "Total geral", "Total Geral"),
) -> dict[str, str]:
    """Infer explicit total filters only when the choice is unambiguous."""
    candidate_keys = {candidate.casefold() for candidate in candidates}
    filters: dict[str, str] = {}
    for column in dimension_columns(frame):
        values = frame[column].dropna().astype(str).drop_duplicates().tolist()
        if len(values) <= 1:
            continue
        matches = [value for value in values if value.strip().casefold() in candidate_keys]
        if len(matches) != 1:
            raise ValueError(
                f"Cannot infer a unique total category for {column!r}. "
                f"Observed labels include {values[:12]!r}. Set this filter manually."
            )
        filters[column] = matches[0]
    return filters
