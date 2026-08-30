"""Build the Lisbon longitudinal series across the validated NUTS-vintage bridge."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_lisbon_longitudinal import (  # noqa: E402
    VALUE_NAMES,
    _categories,
    _dimension_numbers,
    _lisboa_code,
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

LEGACY_PATH = ROOT / "results" / "processed" / "lisbon_longitudinal_legacy.csv"
AUDIT_PATH = ROOT / "results" / "processed" / "lisbon_nuts2024_bridge_summary.csv"


def _validated_audit() -> pd.DataFrame:
    """Load the frozen bridge audit and require exact overlap equality for every measure."""
    audit = pd.read_csv(AUDIT_PATH, dtype={"old_indicator": str, "current_indicator": str})
    required = {
        "measure",
        "old_indicator",
        "current_indicator",
        "old_last_year",
        "current_last_year",
        "overlap_years",
        "max_abs_relative_difference",
    }
    missing = required.difference(audit.columns)
    if missing:
        raise KeyError(f"Bridge audit is missing columns: {sorted(missing)}")
    if set(audit["measure"]) != set(VALUE_NAMES):
        raise ValueError("Bridge audit does not cover exactly the configured Lisbon measures.")
    if (audit["overlap_years"] <= 0).any():
        raise ValueError("Every bridged measure must have at least one overlap observation.")
    if not audit["max_abs_relative_difference"].eq(0.0).all():
        raise ValueError("Bridge audit no longer establishes exact overlap equality.")
    return audit


def _fetch_tail_years(
    measure: str,
    indicator: str,
    *,
    after_year: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Fetch only current-vintage years later than the frozen legacy endpoint."""
    metadata = _metadata(indicator)
    categories = _categories(metadata)
    time_dim, geography_dim, other_dims = _dimension_numbers(metadata, categories)
    lisboa = _lisboa_code(categories, geography_dim)
    totals = _total_codes(metadata, categories, other_dims)
    periods = [(year, code) for year, code in _period_codes(categories, time_dim) if year > after_year]

    value_name = VALUE_NAMES[measure]
    observations: list[pd.DataFrame] = []
    extraction_dates: list[str] = []
    for year, period_code in periods:
        dimensions = {
            f"Dim{time_dim}": period_code,
            f"Dim{geography_dim}": lisboa,
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
            raise ValueError(f"No current-vintage {measure} observation returned for {year}.")

        flat = flatten_ine_payload(payload)
        flat["year"] = infer_year(flat["period"])
        canonical = canonicalise_ine_measure(
            flat,
            value_name=value_name,
            minimum_year=year,
        )
        canonical = canonical.loc[canonical["year"] == year, ["year", value_name]]
        if len(canonical) != 1:
            raise ValueError(f"Expected one current-vintage {measure} observation for {year}.")
        observations.append(canonical)
        if isinstance(payload[0], dict):
            extraction_dates.append(str(payload[0].get("DataExtracao", "")))

    if observations:
        tail = pd.concat(observations, ignore_index=True).sort_values("year").reset_index(drop=True)
    else:
        tail = pd.DataFrame(columns=["year", value_name])

    provenance: dict[str, object] = {
        "measure": measure,
        "indicator_code": indicator,
        "geo_code": lisboa,
        "geo_name": "Lisboa",
        "bridge_after_year": after_year,
        "tail_first_year": int(tail["year"].min()) if not tail.empty else None,
        "tail_last_year": int(tail["year"].max()) if not tail.empty else None,
        "tail_rows": len(tail),
        "overlap_equality": "exact_frozen_audit",
        "statistical_source": "Instituto Nacional de Estatística (INE), Portugal",
        "transport": "Pipeworx INE proxy",
        "extraction_date": max(extraction_dates) if extraction_dates else None,
    }
    return tail, provenance


def _apply_current_tails(legacy: pd.DataFrame, audit: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill only post-legacy years using current NUTS-2024 observations."""
    result = legacy.copy()
    provenance: list[dict[str, object]] = []
    for record in audit.to_dict(orient="records"):
        measure = str(record["measure"])
        value_name = VALUE_NAMES[measure]
        legacy_last_year = int(record["old_last_year"])
        current_last_year = int(record["current_last_year"])
        indicator = str(record["current_indicator"])
        if current_last_year <= legacy_last_year:
            provenance.append(
                {
                    "measure": measure,
                    "indicator_code": indicator,
                    "bridge_after_year": legacy_last_year,
                    "tail_rows": 0,
                    "overlap_equality": "exact_frozen_audit",
                }
            )
            continue

        tail, source_provenance = _fetch_tail_years(
            measure,
            indicator,
            after_year=legacy_last_year,
        )
        result = result.merge(tail, on="year", how="outer", suffixes=("", "_current"), validate="one_to_one")
        current_column = f"{value_name}_current"
        if current_column in result.columns:
            result[value_name] = result[value_name].combine_first(result[current_column])
            result = result.drop(columns=[current_column])
        provenance.append(source_provenance)

    return result.sort_values("year").reset_index(drop=True), pd.DataFrame(provenance)


def _recompute_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Recompute derived metrics after appending current-vintage tails."""
    result = frame.copy()
    result["rent_income_ratio"] = result["rent_eur_m2"] / result["income_eur"]
    baseline = result.loc[result["year"] == 2017, "rent_income_ratio"]
    if len(baseline) != 1 or pd.isna(baseline.iloc[0]) or float(baseline.iloc[0]) <= 0:
        raise ValueError("A valid 2017 rent/income baseline is required for bridged LHDI.")
    result["lhdi"] = 100.0 * result["rent_income_ratio"] / float(baseline.iloc[0])
    result["tourism_intensity"] = result["overnight_stays"] / result["resident_population"]
    result["listed_units_per_1000_residents"] = (
        1000.0 * result["listed_units"] / result["resident_population"]
    )
    result["entire_home_per_1000_residents"] = (
        1000.0 * result["entire_home_units"] / result["resident_population"]
    )
    result["entire_home_per_1000_dwellings"] = (
        1000.0 * result["entire_home_units"] / result["housing_stock"]
    )
    return result


def main() -> None:
    """Execute the frozen-audit bridge and export the extended Lisbon series."""
    legacy = pd.read_csv(LEGACY_PATH, parse_dates=["snapshot_date"])
    audit = _validated_audit()
    bridged, provenance = _apply_current_tails(legacy, audit)
    result = _recompute_metrics(bridged)

    output_dir = ROOT / "results" / "processed"
    result.to_csv(output_dir / "lisbon_longitudinal_bridged.csv", index=False)
    provenance.to_csv(output_dir / "lisbon_ine_bridge_provenance.csv", index=False)

    columns = [
        "year",
        "rent_eur_m2",
        "income_eur",
        "lhdi",
        "tourism_intensity",
        "listed_units",
        "entire_home_units",
        "listed_units_per_1000_residents",
        "entire_home_per_1000_residents",
        "entire_home_per_1000_dwellings",
    ]
    print("\nFrozen-audit bridged Lisbon longitudinal series:")
    print(result[columns].to_string(index=False))


if __name__ == "__main__":
    main()
