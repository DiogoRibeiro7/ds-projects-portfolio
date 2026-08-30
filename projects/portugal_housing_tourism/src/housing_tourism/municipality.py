"""Municipality-level comparison helpers for the housing-tourism case study."""

from __future__ import annotations

import numpy as np
import pandas as pd


def municipality_change_panel(
    frame: pd.DataFrame,
    *,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Compute municipality-level affordability and tourism changes between two years."""
    required = {
        "geo_code",
        "geo_name",
        "year",
        "rent_eur_m2",
        "income_eur",
        "resident_population",
        "overnight_stays",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    if end_year <= start_year:
        raise ValueError("end_year must be greater than start_year.")

    subset = frame.loc[frame["year"].isin([start_year, end_year])].copy()
    if subset.duplicated(["geo_code", "year"]).any():
        raise ValueError("geo_code and year must uniquely identify observations.")

    value_columns = [
        "rent_eur_m2",
        "income_eur",
        "resident_population",
        "overnight_stays",
    ]
    for column in value_columns:
        subset[column] = pd.to_numeric(subset[column], errors="coerce")

    subset["rent_income_ratio"] = subset["rent_eur_m2"] / subset["income_eur"]
    subset["tourism_intensity"] = subset["overnight_stays"] / subset["resident_population"]

    names = subset[["geo_code", "geo_name"]].drop_duplicates("geo_code")
    wide = subset.pivot(index="geo_code", columns="year", values=[
        "rent_eur_m2",
        "income_eur",
        "rent_income_ratio",
        "tourism_intensity",
    ])
    required_pairs = [(column, year) for column in wide.columns.levels[0] for year in (start_year, end_year)]
    if not all(pair in wide.columns for pair in required_pairs):
        raise ValueError("The panel does not contain both endpoint years for every requested measure.")

    result = names.set_index("geo_code").join(wide).reset_index()
    complete = result[
        [
            ("rent_eur_m2", start_year),
            ("rent_eur_m2", end_year),
            ("income_eur", start_year),
            ("income_eur", end_year),
            ("rent_income_ratio", start_year),
            ("rent_income_ratio", end_year),
            ("tourism_intensity", start_year),
            ("tourism_intensity", end_year),
        ]
    ].notna().all(axis=1)
    result = result.loc[complete].copy()

    def pct_change(column: str) -> pd.Series:
        start = result[(column, start_year)].astype(float)
        end = result[(column, end_year)].astype(float)
        if (start <= 0).any() or (end <= 0).any():
            raise ValueError(f"{column} endpoints must be strictly positive.")
        return 100.0 * (end / start - 1.0)

    output = pd.DataFrame(
        {
            "geo_code": result["geo_code"].astype(str),
            "geo_name": result["geo_name"].astype(str),
            "rent_change_pct": pct_change("rent_eur_m2"),
            "income_change_pct": pct_change("income_eur"),
            "rent_income_ratio_change_pct": pct_change("rent_income_ratio"),
            "tourism_intensity_change_pct": pct_change("tourism_intensity"),
        }
    )
    ratio_change = output["rent_income_ratio_change_pct"]
    output["affordability_percentile"] = ratio_change.rank(method="average", pct=True) * 100.0
    output["affordability_rank_desc"] = ratio_change.rank(method="min", ascending=False).astype(int)
    output["municipality_count"] = len(output)
    return output.sort_values(["affordability_rank_desc", "geo_name"]).reset_index(drop=True)


def summarise_reference_municipality(panel: pd.DataFrame, *, geo_name: str) -> dict[str, float | int]:
    """Extract one municipality's rank and change metrics from a completed comparison panel."""
    matches = panel.loc[panel["geo_name"].astype(str).str.casefold() == geo_name.casefold()]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one municipality named {geo_name!r}, found {len(matches)}.")
    row = matches.iloc[0]
    return {
        "rent_change_pct": float(row["rent_change_pct"]),
        "income_change_pct": float(row["income_change_pct"]),
        "rent_income_ratio_change_pct": float(row["rent_income_ratio_change_pct"]),
        "tourism_intensity_change_pct": float(row["tourism_intensity_change_pct"]),
        "affordability_percentile": float(row["affordability_percentile"]),
        "affordability_rank_desc": int(row["affordability_rank_desc"]),
        "municipality_count": int(row["municipality_count"]),
    }
