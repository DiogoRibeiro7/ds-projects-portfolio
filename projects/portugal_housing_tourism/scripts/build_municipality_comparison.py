"""Build an observed-municipality comparison for the 2022-2023 affordability change."""

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
from housing_tourism.municipality import (  # noqa: E402
    municipality_change_panel,
    summarise_reference_municipality,
)

YEARS = (2022, 2023)
MIN_AFFORDABILITY_COVERAGE = 0.50
MEASURES = {
    "rent": "rent_eur_m2",
    "income": "income_eur",
    "population": "resident_population",
    "overnight_stays": "overnight_stays",
}
ENDPOINTS_PATH = ROOT / "results" / "processed" / "municipality_endpoints_2022_2023.csv"
COMPARISON_PATH = ROOT / "results" / "processed" / "municipality_comparison_2022_2023.csv"
SUMMARY_PATH = ROOT / "results" / "processed" / "municipality_comparison_summary.json"


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


def _fetch_bulk_measure(measure: str, indicator: str) -> pd.DataFrame:
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


def _build_endpoints() -> pd.DataFrame:
    indicators = _load_current_sources()
    frames = [_fetch_bulk_measure(measure, indicator) for measure, indicator in indicators.items()]
    result = frames[0]
    for frame in frames[1:]:
        result = result.merge(
            frame, on=["geo_code", "geo_name", "year"], how="outer", validate="one_to_one"
        )
    return result.sort_values(["geo_code", "year"]).reset_index(drop=True)


def _measure_coverage(endpoints: pd.DataFrame, column: str) -> dict[str, int]:
    coverage: dict[str, int] = {}
    for year in YEARS:
        observed = endpoints.loc[endpoints["year"].eq(year) & endpoints[column].notna(), "geo_code"]
        coverage[str(year)] = int(observed.nunique())
    complete = endpoints.loc[endpoints[column].notna()].groupby("geo_code")["year"].nunique()
    coverage["both_years"] = int(complete.ge(2).sum())
    return coverage


def main() -> None:
    """Fetch endpoint data and rank the observed 2022-2023 affordability sample."""
    endpoints = _build_endpoints()
    comparison = municipality_change_panel(endpoints, start_year=2022, end_year=2023)
    municipality_universe = int(endpoints["geo_code"].nunique())
    affordability_coverage = len(comparison) / municipality_universe
    if affordability_coverage < MIN_AFFORDABILITY_COVERAGE:
        raise ValueError(
            f"Affordability coverage {affordability_coverage:.1%} is below the "
            f"{MIN_AFFORDABILITY_COVERAGE:.0%} minimum guard."
        )

    lisbon = summarise_reference_municipality(comparison, geo_name="Lisboa")
    porto = summarise_reference_municipality(comparison, geo_name="Porto")
    tourism_observed = int(comparison["tourism_intensity_change_pct"].notna().sum())
    summary = {
        "start_year": 2022,
        "end_year": 2023,
        "municipality_universe": municipality_universe,
        "complete_affordability_municipalities": len(comparison),
        "affordability_coverage_pct": 100.0 * affordability_coverage,
        "tourism_change_observed_municipalities": tourism_observed,
        "measure_coverage": {
            value_name: _measure_coverage(endpoints, value_name) for value_name in MEASURES.values()
        },
        "lisboa": lisbon,
        "porto": porto,
        "median_rent_income_ratio_change_pct": float(
            comparison["rent_income_ratio_change_pct"].median()
        ),
        "mean_rent_income_ratio_change_pct": float(
            comparison["rent_income_ratio_change_pct"].mean()
        ),
    }

    ENDPOINTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    endpoints.to_csv(ENDPOINTS_PATH, index=False, float_format="%.6f")
    comparison.to_csv(COMPARISON_PATH, index=False, float_format="%.6f")
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("\nObserved 2022-2023 municipality comparison:")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("\nTop 15 affordability deteriorations:")
    print(
        comparison[
            [
                "geo_name",
                "rent_change_pct",
                "income_change_pct",
                "rent_income_ratio_change_pct",
                "tourism_intensity_change_pct",
                "affordability_rank_desc",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
