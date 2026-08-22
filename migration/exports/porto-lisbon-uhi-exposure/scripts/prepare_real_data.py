"""Prepare a reproducible Porto/Lisbon UHI exposure table from official sources."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from uhi_exposure.io import write_dataframe
from uhi_exposure.real_data import (
    CITY_BOUNDARIES_URL,
    GRID_ARCHIVE_URL,
    UHI_IDENTIFY_URL,
    ensure_population_grid_files,
    finalize_prepared_cells,
    intersect_grid_with_cities,
    load_city_boundaries,
    load_population_grid,
    sample_uhi_values,
    summarize_fill_stats,
    validate_prepared_output,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="data/manual/porto_lisbon_cells.csv",
        help="Output CSV or Parquet path for the prepared Porto/Lisbon table.",
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Directory used to cache downloaded source files.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=2.0,
        help="UHI threshold, in °C, used to compute is_uhi_exposed.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum concurrent requests used to sample the UHI raster service.",
    )
    return parser.parse_args()


def main() -> None:
    """Prepare the real-data exposure table and print quality checks."""
    args = parse_args()
    output_path = Path(args.output)
    raw_dir = Path(args.raw_dir)

    grid_gpkg_path, grid_readme_path = ensure_population_grid_files(raw_dir)
    cities = load_city_boundaries()
    grid = load_population_grid(grid_gpkg_path, tuple(cities.total_bounds))
    city_cells = intersect_grid_with_cities(grid, cities)
    sampled_cells = sample_uhi_values(city_cells, max_workers=args.max_workers)
    prepared = finalize_prepared_cells(sampled_cells, threshold=args.threshold)
    validate_prepared_output(prepared)
    write_dataframe(prepared, output_path)

    fill_summary = summarize_fill_stats(sampled_cells)
    city_totals = (
        prepared.groupby("city", as_index=False)
        .agg(
            cells=("cell_id", "count"),
            population_total=("population_total", "sum"),
            age_0_14=("age_0_14", "sum"),
            age_65_plus=("age_65_plus", "sum"),
            employed=("employed", "sum"),
            born_outside_eu=("born_outside_eu", "sum"),
        )
    )
    exposed_population = (
        prepared.loc[prepared["is_uhi_exposed"]]
        .groupby("city", as_index=False)["population_total"]
        .sum()
        .rename(columns={"population_total": "exposed_population"})
    )
    city_totals = city_totals.merge(exposed_population, on="city", how="left")
    city_totals["exposed_population"] = city_totals["exposed_population"].fillna(0.0)
    city_totals["exposed_population_share"] = (
        city_totals["exposed_population"] / city_totals["population_total"]
    )

    print(f"Saved prepared table: {output_path}")
    print("\nSource files")
    print(f"- Eurostat Census 2021 grid archive: {GRID_ARCHIVE_URL}")
    print(f"- Extracted grid GeoPackage: {grid_gpkg_path}")
    print(f"- Extracted readme: {grid_readme_path}")
    print(f"- Eurostat Urban Audit city boundaries: {CITY_BOUNDARIES_URL}")
    print(f"- EEA/UrbClim UHI raster service: {UHI_IDENTIFY_URL}")
    print("\nUHI fill summary")
    print(fill_summary.to_string(index=False))
    print("\nCity totals")
    with pd.option_context("display.max_columns", None, "display.width", 140):
        print(city_totals)


if __name__ == "__main__":
    main()
