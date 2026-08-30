"""Audit overlap between the legacy NUTS-2013 and current NUTS-2024 INE series."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_lisbon_longitudinal import VALUE_NAMES, _fetch_lisbon_indicator  # noqa: E402


def _load_sources() -> dict[str, object]:
    with (ROOT / "config" / "sources.toml").open("rb") as handle:
        return tomllib.load(handle)


def _relative_difference(old: pd.Series, current: pd.Series) -> pd.Series:
    denominator = old.abs().replace(0.0, np.nan)
    return (current - old) / denominator


def _audit_measure(
    measure: str,
    old_indicator: str,
    current_indicator: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    old, old_provenance = _fetch_lisbon_indicator(measure, old_indicator)
    current, current_provenance = _fetch_lisbon_indicator(measure, current_indicator)
    value_name = VALUE_NAMES[measure]

    overlap = old.merge(
        current,
        on="year",
        how="inner",
        suffixes=("_old", "_current"),
        validate="one_to_one",
    )
    if overlap.empty:
        raise ValueError(f"No overlapping years for {measure}.")

    overlap["measure"] = measure
    overlap["old_indicator"] = old_indicator
    overlap["current_indicator"] = current_indicator
    overlap["absolute_difference"] = (
        overlap[f"{value_name}_current"] - overlap[f"{value_name}_old"]
    )
    overlap["relative_difference"] = _relative_difference(
        overlap[f"{value_name}_old"],
        overlap[f"{value_name}_current"],
    )

    summary: dict[str, object] = {
        "measure": measure,
        "old_indicator": old_indicator,
        "current_indicator": current_indicator,
        "old_last_year": int(old["year"].max()),
        "current_last_year": int(current["year"].max()),
        "overlap_first_year": int(overlap["year"].min()),
        "overlap_last_year": int(overlap["year"].max()),
        "overlap_years": len(overlap),
        "max_abs_relative_difference": float(overlap["relative_difference"].abs().max()),
        "mean_abs_relative_difference": float(overlap["relative_difference"].abs().mean()),
        "old_geo_code": old_provenance["geo_code"],
        "current_geo_code": current_provenance["geo_code"],
    }
    return overlap, summary


def main() -> None:
    """Run the bridge audit and export overlap diagnostics without splicing series."""
    sources = _load_sources()
    legacy = sources.get("ine")
    current = sources.get("ine_current")
    if not isinstance(legacy, dict) or not isinstance(current, dict):
        raise TypeError("Both [ine] and [ine_current] source groups are required.")

    overlaps: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for measure in VALUE_NAMES:
        legacy_source = legacy.get(measure)
        current_source = current.get(measure)
        if not isinstance(legacy_source, dict) or not isinstance(current_source, dict):
            raise KeyError(f"Missing legacy/current source definition for {measure}.")
        overlap, summary = _audit_measure(
            measure,
            str(legacy_source["indicator"]),
            str(current_source["indicator"]),
        )
        overlaps.append(overlap)
        summaries.append(summary)

    overlap_table = pd.concat(overlaps, ignore_index=True)
    summary_table = pd.DataFrame(summaries)
    output_dir = ROOT / "results" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    overlap_table.to_csv(output_dir / "lisbon_nuts2024_overlap.csv", index=False)
    summary_table.to_csv(output_dir / "lisbon_nuts2024_bridge_summary.csv", index=False)

    print("NUTS-2013 vs NUTS-2024 overlap audit:")
    print(summary_table.to_string(index=False))
    print("\nNo series have been spliced by this audit.")


if __name__ == "__main__":
    main()
