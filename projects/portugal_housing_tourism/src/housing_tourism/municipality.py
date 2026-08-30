"""Municipality-level comparison helpers for the housing-tourism case study."""

from __future__ import annotations

import pandas as pd


def municipality_change_panel(
    frame: pd.DataFrame,
    *,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Compute municipality-level affordability changes and supplementary tourism changes."""
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

    endpoint_columns = [
        "geo_code",
        "geo_name",
        "rent_eur_m2",
        "income_eur",
        "rent_income_ratio",
        "tourism_intensity",
    ]
    start = subset.loc[subset["year"].eq(start_year), endpoint_columns].copy()
    end = subset.loc[subset["year"].eq(end_year), endpoint_columns].copy()
    start = start.rename(
        columns={column: f"{column}_start" for column in endpoint_columns if column != "geo_code"}
    )
    end = end.rename(
        columns={column: f"{column}_end" for column in endpoint_columns if column != "geo_code"}
    )
    result = start.merge(end, on="geo_code", how="inner", validate="one_to_one")

    affordability_columns = [
        "rent_eur_m2_start",
        "rent_eur_m2_end",
        "income_eur_start",
        "income_eur_end",
        "rent_income_ratio_start",
        "rent_income_ratio_end",
    ]
    result = result.loc[result[affordability_columns].notna().all(axis=1)].copy()
    if not result["geo_name_start"].astype(str).eq(result["geo_name_end"].astype(str)).all():
        raise ValueError("Municipality names changed between endpoint observations.")

    def pct_change(column: str) -> pd.Series:
        start_values = result[f"{column}_start"].astype(float)
        end_values = result[f"{column}_end"].astype(float)
        if (start_values <= 0).any() or (end_values <= 0).any():
            raise ValueError(f"{column} endpoints must be strictly positive.")
        return 100.0 * (end_values / start_values - 1.0)

    tourism_start = result["tourism_intensity_start"].astype(float)
    tourism_end = result["tourism_intensity_end"].astype(float)
    tourism_observed = (
        tourism_start.notna() & tourism_end.notna() & (tourism_start > 0) & (tourism_end > 0)
    )
    tourism_change = pd.Series(float("nan"), index=result.index, dtype=float)
    tourism_change.loc[tourism_observed] = 100.0 * (
        tourism_end.loc[tourism_observed] / tourism_start.loc[tourism_observed] - 1.0
    )

    output = pd.DataFrame(
        {
            "geo_code": result["geo_code"].astype(str),
            "geo_name": result["geo_name_start"].astype(str),
            "rent_change_pct": pct_change("rent_eur_m2"),
            "income_change_pct": pct_change("income_eur"),
            "rent_income_ratio_change_pct": pct_change("rent_income_ratio"),
            "tourism_intensity_change_pct": tourism_change,
        }
    )
    ratio_change = output["rent_income_ratio_change_pct"]
    output["affordability_percentile"] = ratio_change.rank(method="average", pct=True) * 100.0
    output["affordability_rank_desc"] = ratio_change.rank(method="min", ascending=False).astype(int)
    output["municipality_count"] = len(output)
    return output.sort_values(["affordability_rank_desc", "geo_name"]).reset_index(drop=True)


def summarise_reference_municipality(
    panel: pd.DataFrame, *, geo_name: str
) -> dict[str, float | int | None]:
    """Extract one municipality's rank and change metrics from a completed comparison panel."""
    matches = panel.loc[panel["geo_name"].astype(str).str.casefold() == geo_name.casefold()]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one municipality named {geo_name!r}, found {len(matches)}."
        )
    row = matches.iloc[0]
    tourism_change = row["tourism_intensity_change_pct"]
    return {
        "rent_change_pct": float(row["rent_change_pct"]),
        "income_change_pct": float(row["income_change_pct"]),
        "rent_income_ratio_change_pct": float(row["rent_income_ratio_change_pct"]),
        "tourism_intensity_change_pct": (
            None if pd.isna(tourism_change) else float(tourism_change)
        ),
        "affordability_percentile": float(row["affordability_percentile"]),
        "affordability_rank_desc": int(row["affordability_rank_desc"]),
        "municipality_count": int(row["municipality_count"]),
    }
