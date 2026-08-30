"""Audit repeated municipality support for the current-vintage 2022-2024 panel."""

from __future__ import annotations

import json
import sys
import time
import tomllib
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_lisbon_longitudinal import (  # noqa: E402
    _categories,
    _dimension_numbers,
    _metadata,
    _mirror_call,
    _period_codes,
    _total_codes,
)

from housing_tourism.data import (  # noqa: E402
    canonicalise_ine_measure,
    flatten_ine_payload,
    infer_year,
)
from housing_tourism.panel_support import summarise_panel_support  # noqa: E402

YEARS = (2022, 2023, 2024)
MEASURES = {
    "rent": "rent_eur_m2",
    "income": "income_eur",
    "population": "resident_population",
    "overnight_stays": "overnight_stays",
}
PANEL_PATH = ROOT / "results" / "processed" / "municipality_panel_support_2022_2024.csv"
SUMMARY_PATH = ROOT / "results" / "processed" / "municipality_panel_support_2022_2024.json"


def _load_current_sources() -> dict[str, str]:
    with (ROOT / "config" / "sources.toml").open("rb") as handle:
        config = tomllib.load(handle)
    current = config.get("ine_current")
    if not isinstance(current, dict):
        raise TypeError("config/sources.toml must define [ine_current.*] tables.")
    indicators: dict[str, str] = {}
    for measure in MEASURES:
        source = current.get(measure)
        if not isinstance(source, dict) or "indicator" not in source:
            raise KeyError(f"Missing current INE indicator for {measure!r}.")
        indicators[measure] = str(source["indicator"])
    return indicators


def _municipality_codes(categories: list[dict[str, Any]], geography_dim: str) -> dict[str, str]:
    codes = {
        str(record["categ_cod"]): str(record.get("categ_dsg", "")).strip()
        for record in categories
        if str(record.get("dim_num")) == geography_dim
        and str(record.get("categ_nivel")) == "5"
        and record.get("categ_cod") is not None
    }
    if len(codes) < 300:
        raise ValueError(
            f"Expected national municipality metadata coverage, found only {len(codes)} codes."
        )
    return codes


def _period_code(categories: list[dict[str, Any]], time_dim: str, year: int) -> str:
    matches = [
        code for observed_year, code in _period_codes(categories, time_dim) if observed_year == year
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one period code for {year}, found {len(matches)}.")
    return matches[0]


def _fetch_measure(measure: str, indicator: str) -> pd.DataFrame:
    metadata = _metadata(indicator)
    categories = _categories(metadata)
    time_dim, geography_dim, other_dims = _dimension_numbers(metadata, categories)
    municipality_codes = _municipality_codes(categories, geography_dim)
    totals = _total_codes(metadata, categories, other_dims)
    value_name = MEASURES[measure]

    observations: list[pd.DataFrame] = []
    for year in YEARS:
        dimensions = {
            f"Dim{time_dim}": _period_code(categories, time_dim, year),
            **{f"Dim{dim}": code for dim, code in totals.items()},
        }
        payload: object = []
        for attempt in range(3):
            payload = _mirror_call(
                "get_indicator",
                {"varcd": indicator, "dims": dimensions, "lang": "PT"},
            )
            if isinstance(payload, list) and payload:
                break
            if attempt < 2:
                time.sleep(1.0 + attempt)
        if not isinstance(payload, list) or not payload:
            raise ValueError(f"No bulk {measure} payload returned for {year}.")

        flat = flatten_ine_payload(payload)
        flat["year"] = infer_year(flat["period"])
        flat = flat.loc[
            flat["year"].eq(year) & flat["geocod"].astype(str).isin(municipality_codes)
        ].copy()
        canonical = canonicalise_ine_measure(flat, value_name=value_name, minimum_year=year)
        canonical = canonical.loc[canonical["year"].eq(year)].copy()
        canonical["geo_name"] = canonical["geo_code"].map(municipality_codes)
        observations.append(canonical[["geo_code", "geo_name", "year", value_name]])

    result = pd.concat(observations, ignore_index=True)
    if result.duplicated(["geo_code", "year"]).any():
        raise ValueError(f"Duplicate municipality-year observations returned for {measure}.")
    print(
        f"{measure}: indicator={indicator}, municipalities={result['geo_code'].nunique()}, "
        f"rows={len(result)}"
    )
    return result


def _build_panel() -> pd.DataFrame:
    frames = [
        _fetch_measure(measure, indicator) for measure, indicator in _load_current_sources().items()
    ]
    panel = frames[0]
    for frame in frames[1:]:
        panel = panel.merge(
            frame,
            on=["geo_code", "geo_name", "year"],
            how="outer",
            validate="one_to_one",
        )
    return panel.sort_values(["geo_code", "year"]).reset_index(drop=True)


def main() -> None:
    """Build the current-vintage support panel and export repeated-observation diagnostics."""
    panel = _build_panel()
    summary = summarise_panel_support(
        panel,
        years=YEARS,
        value_columns=tuple(MEASURES.values()),
    )
    summary["window_basis"] = (
        "Common current NUTS-2024 availability: rent, income, population and comparable "
        "overnight-stay indicators all cover 2022-2024."
    )
    summary["model_status"] = "support-audit-only"

    PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(PANEL_PATH, index=False, float_format="%.6f")
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("\nMunicipality panel support audit:")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
