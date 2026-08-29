"""Build the first observed Lisbon housing-tourism longitudinal dataset."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from housing_tourism.data import (  # noqa: E402
    INEClient,
    canonicalise_ine_measure,
    flatten_ine_payload,
    infer_total_filters,
    infer_year,
)

VALUE_NAMES = {
    "rent": "rent_eur_m2",
    "income": "income_eur",
    "housing_stock": "housing_stock",
    "population": "resident_population",
    "overnight_stays": "overnight_stays",
}


def _load_sources() -> dict[str, object]:
    with (ROOT / "config" / "sources.toml").open("rb") as handle:
        return tomllib.load(handle)


def _lisbon_series(frame: pd.DataFrame, *, value_name: str) -> pd.DataFrame:
    """Select the unique annual geography series named Lisboa."""
    matches = frame.loc[frame["geo_name"].str.strip().str.casefold().eq("lisboa")].copy()
    candidates = matches[["geo_code", "geo_name"]].drop_duplicates().sort_values("geo_code")
    if matches.empty:
        raise ValueError("INE series contains no geography named 'Lisboa'.")
    if candidates["geo_code"].nunique() != 1:
        raise ValueError(
            "INE geography 'Lisboa' is ambiguous after filtering. Candidates: "
            f"{candidates.to_dict(orient='records')}"
        )
    return matches[["year", value_name]].sort_values("year").reset_index(drop=True)


def _build_ine_series() -> pd.DataFrame:
    sources = _load_sources()
    ine_sources = sources["ine"]
    if not isinstance(ine_sources, dict):
        raise TypeError("config/sources.toml must define an [ine] table.")

    client = INEClient(ROOT / "data" / "raw" / "ine")
    series: list[pd.DataFrame] = []
    for measure, source in ine_sources.items():
        if measure not in VALUE_NAMES:
            continue
        if not isinstance(source, dict):
            raise TypeError(f"INE source {measure!r} must be a table.")
        indicator = str(source["indicator"])
        payload = client.fetch_indicator(indicator, refresh=True)
        flat = flatten_ine_payload(payload)
        flat["year"] = infer_year(flat["period"])
        filters = infer_total_filters(flat)
        canonical = canonicalise_ine_measure(
            flat,
            value_name=VALUE_NAMES[measure],
            filters=filters,
            minimum_year=2017,
        )
        lisbon = _lisbon_series(canonical, value_name=VALUE_NAMES[measure])
        print(
            f"{measure}: indicator={indicator}, filters={filters}, "
            f"years={lisbon['year'].min()}-{lisbon['year'].max()}, rows={len(lisbon)}"
        )
        series.append(lisbon)

    if not series:
        raise ValueError("No configured INE series were loaded.")
    result = series[0]
    for next_series in series[1:]:
        result = result.merge(next_series, on="year", how="outer", validate="one_to_one")
    return result.sort_values("year").reset_index(drop=True)


def _add_indices(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["rent_income_ratio"] = result["rent_eur_m2"] / result["income_eur"]
    base = result.loc[result["year"] == 2017, "rent_income_ratio"]
    if base.empty or pd.isna(base.iloc[0]) or float(base.iloc[0]) <= 0:
        raise ValueError("A valid 2017 rent/income baseline is required for LHDI.")
    result["lhdi"] = 100.0 * result["rent_income_ratio"] / float(base.iloc[0])
    result["tourism_intensity"] = result["overnight_stays"] / result["resident_population"]
    return result


def _join_airbnb(frame: pd.DataFrame) -> pd.DataFrame:
    path = ROOT / "results" / "processed" / "inside_airbnb_lisbon_annual.csv"
    airbnb = pd.read_csv(path, parse_dates=["snapshot_date"])
    required = {"year", "listed_units", "entire_home_units"}
    missing = required.difference(airbnb.columns)
    if missing:
        raise KeyError(f"Inside Airbnb result is missing columns: {sorted(missing)}")
    airbnb["entire_home_share"] = airbnb["entire_home_units"] / airbnb["listed_units"]
    output = frame.merge(airbnb, on="year", how="outer", validate="one_to_one")
    output["entire_home_per_1000_dwellings"] = (
        1000.0 * output["entire_home_units"] / output["housing_stock"]
    )
    return output.sort_values("year").reset_index(drop=True)


def _save_figures(frame: pd.DataFrame) -> None:
    figure_dir = ROOT / "results" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    lhd = frame.dropna(subset=["lhdi"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lhd["year"], lhd["lhdi"], marker="o")
    ax.axhline(100.0, linewidth=1)
    ax.set_title("Lisbon local housing decoupling index")
    ax.set_xlabel("Year")
    ax.set_ylabel("LHDI (2017 = 100)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "lisbon_lhdi.png", dpi=180)
    plt.close(fig)

    tourism = frame.dropna(subset=["tourism_intensity"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        tourism["year"],
        tourism["tourism_intensity"],
        marker="o",
        label="Tourist nights / resident",
    )
    observed = frame.dropna(subset=["entire_home_per_1000_dwellings"])
    if not observed.empty:
        ax.scatter(
            observed["year"],
            observed["entire_home_per_1000_dwellings"],
            marker="s",
            label="Entire-home listings / 1,000 dwellings",
        )
    ax.set_title("Lisbon tourism intensity and observed platform exposure")
    ax.set_xlabel("Year")
    ax.set_ylabel("Observed intensity")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "lisbon_tourism_platform_exposure.png", dpi=180)
    plt.close(fig)


def main() -> None:
    """Build, export, and print the observed Lisbon longitudinal dataset."""
    ine = _add_indices(_build_ine_series())
    result = _join_airbnb(ine)
    output_dir = ROOT / "results" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "lisbon_longitudinal.csv", index=False)
    _save_figures(result)

    columns = [
        "year",
        "rent_eur_m2",
        "income_eur",
        "lhdi",
        "tourism_intensity",
        "listed_units",
        "entire_home_units",
        "entire_home_per_1000_dwellings",
    ]
    print("\nLisbon longitudinal series:")
    print(result[columns].to_string(index=False))


if __name__ == "__main__":
    main()
