"""Build the Lisbon longitudinal series across the validated NUTS-vintage bridge."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_lisbon_longitudinal import (  # noqa: E402
    VALUE_NAMES,
    _add_indices,
    _fetch_lisbon_indicator,
    _join_airbnb,
    _load_sources,
)


def bridge_measure_series(
    measure: str,
    legacy_indicator: str,
    current_indicator: str,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Append the current tail only after exact equality on all overlap years."""
    legacy, legacy_provenance = _fetch_lisbon_indicator(measure, legacy_indicator)
    current, current_provenance = _fetch_lisbon_indicator(measure, current_indicator)
    value_name = VALUE_NAMES[measure]

    overlap = legacy.merge(
        current,
        on="year",
        how="inner",
        suffixes=("_legacy", "_current"),
        validate="one_to_one",
    )
    if overlap.empty:
        raise ValueError(f"Cannot bridge {measure}: no overlapping observations.")

    legacy_values = overlap[f"{value_name}_legacy"].to_numpy(dtype=float)
    current_values = overlap[f"{value_name}_current"].to_numpy(dtype=float)
    if not np.array_equal(legacy_values, current_values, equal_nan=True):
        raise ValueError(f"Cannot bridge {measure}: overlap is not exactly equal.")

    legacy_last_year = int(legacy["year"].max())
    current_tail = current.loc[current["year"] > legacy_last_year].copy()
    bridged = (
        pd.concat([legacy, current_tail], ignore_index=True)
        .sort_values("year")
        .reset_index(drop=True)
    )
    if bridged.duplicated("year").any():
        raise ValueError(f"Duplicate years after bridging {measure}.")

    for record, role in (
        (legacy_provenance, "legacy"),
        (current_provenance, "current_bridge"),
    ):
        record["series_role"] = role
        record["bridge_after_year"] = legacy_last_year
        record["overlap_equality"] = "exact"

    print(
        f"{measure}: {legacy_indicator} -> {current_indicator}; "
        f"legacy through {legacy_last_year}, bridged through {bridged['year'].max()}"
    )
    return bridged, [legacy_provenance, current_provenance]


def build_bridged_ine_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build all five Lisbon INE measures using the validated direct bridge."""
    sources = _load_sources()
    legacy = sources.get("ine")
    current = sources.get("ine_current")
    if not isinstance(legacy, dict) or not isinstance(current, dict):
        raise TypeError("Both [ine] and [ine_current] source groups are required.")

    series: list[pd.DataFrame] = []
    provenance: list[dict[str, object]] = []
    for measure in VALUE_NAMES:
        legacy_source = legacy.get(measure)
        current_source = current.get(measure)
        if not isinstance(legacy_source, dict) or not isinstance(current_source, dict):
            raise KeyError(f"Missing legacy/current source definition for {measure}.")
        frame, records = bridge_measure_series(
            measure,
            str(legacy_source["indicator"]),
            str(current_source["indicator"]),
        )
        series.append(frame)
        provenance.extend(records)

    panel = series[0]
    for frame in series[1:]:
        panel = panel.merge(frame, on="year", how="outer", validate="one_to_one")
    return panel.sort_values("year").reset_index(drop=True), pd.DataFrame(provenance)


def add_platform_population_rates(frame: pd.DataFrame) -> pd.DataFrame:
    """Add platform exposure per resident without substituting for housing-stock exposure."""
    result = frame.copy()
    result["listed_units_per_1000_residents"] = (
        1000.0 * result["listed_units"] / result["resident_population"]
    )
    result["entire_home_per_1000_residents"] = (
        1000.0 * result["entire_home_units"] / result["resident_population"]
    )
    return result


def main() -> None:
    """Execute the guarded bridge and export the extended Lisbon series."""
    ine, provenance = build_bridged_ine_panel()
    result = add_platform_population_rates(_join_airbnb(_add_indices(ine)))

    output_dir = ROOT / "results" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
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
    print("\nGuarded bridged Lisbon longitudinal series:")
    print(result[columns].to_string(index=False))


if __name__ == "__main__":
    main()
