"""Support diagnostics for municipality-year housing panels."""

from __future__ import annotations

import pandas as pd


def summarise_panel_support(
    frame: pd.DataFrame,
    *,
    years: tuple[int, ...],
    value_columns: tuple[str, ...],
) -> dict[str, object]:
    """Summarise per-year and repeated-observation support for a municipality panel."""
    required = {"geo_code", "geo_name", "year", *value_columns}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    if not years:
        raise ValueError("years cannot be empty.")
    if len(set(years)) != len(years):
        raise ValueError("years must be unique.")
    if frame.duplicated(["geo_code", "year"]).any():
        raise ValueError("geo_code and year must uniquely identify panel rows.")

    subset = frame.loc[frame["year"].isin(years)].copy()
    universe = int(subset["geo_code"].nunique())
    if universe == 0:
        raise ValueError("No municipalities are available in the requested years.")

    measures: dict[str, object] = {}
    for column in value_columns:
        observed = subset.loc[subset[column].notna(), ["geo_code", "year"]]
        counts = observed.groupby("geo_code")["year"].nunique()
        measures[column] = {
            "by_year": {
                str(year): int(
                    subset.loc[subset["year"].eq(year) & subset[column].notna(), "geo_code"].nunique()
                )
                for year in years
            },
            "at_least_two_years": int(counts.ge(2).sum()),
            "all_years": int(counts.ge(len(years)).sum()),
        }

    affordability_mask = subset["rent_eur_m2"].notna() & subset["income_eur"].notna()
    affordability_counts = subset.loc[affordability_mask].groupby("geo_code")["year"].nunique()
    tourism_mask = subset["overnight_stays"].notna() & subset["resident_population"].notna()
    tourism_counts = subset.loc[tourism_mask].groupby("geo_code")["year"].nunique()

    return {
        "years": list(years),
        "municipality_universe": universe,
        "measures": measures,
        "affordability_at_least_two_years": int(affordability_counts.ge(2).sum()),
        "affordability_all_years": int(affordability_counts.ge(len(years)).sum()),
        "tourism_at_least_two_years": int(tourism_counts.ge(2).sum()),
        "tourism_all_years": int(tourism_counts.ge(len(years)).sum()),
    }
