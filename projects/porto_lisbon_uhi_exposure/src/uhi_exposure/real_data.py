"""Real-data preparation helpers for the Porto/Lisbon UHI workflow."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Final

import geopandas as gpd
import pandas as pd
import requests

from uhi_exposure.schema import validate_exposure_cells

GRID_ARCHIVE_URL: Final[str] = (
    "https://gisco-services.ec.europa.eu/census/2021/Eurostat_Census-GRID_2021_V3.zip"
)
GRID_ARCHIVE_NAME: Final[str] = "Eurostat_Census-GRID_2021_V3.zip"
GRID_DIRECTORY_NAME: Final[str] = "Eurostat_Census-GRID_2021_V3"
GRID_GPKG_NAME: Final[str] = "ESTAT_Census_2021_V3.gpkg"
GRID_README_NAME: Final[str] = "read.me"
CITY_BOUNDARIES_URL: Final[str] = (
    "https://gisco-services.ec.europa.eu/distribution/v2/urau/gpkg/URAU_RG_100K_2024_3035_CITIES.gpkg"
)
UHI_IDENTIFY_URL: Final[str] = (
    "https://climate.discomap.eea.europa.eu/arcgis/rest/services/"
    "UAMV/Urban_Heat_Island_modelling/MapServer/identify"
)
TARGET_CRS: Final[int] = 3035
UHI_QUERY_CRS: Final[int] = 3857
CITY_CODE_TO_NAME: Final[dict[str, str]] = {
    "PT001C": "Lisbon",
    "PT002C": "Porto",
}
GRID_COUNT_COLUMNS: Final[tuple[str, ...]] = ("T", "Y_LT15", "Y_GE65", "EMP", "OTH")
GRID_MISSING_CODES: Final[tuple[int, int]] = (-8888, -9999)
PREPARED_COUNT_COLUMNS: Final[tuple[str, ...]] = (
    "population_total",
    "age_0_14",
    "age_65_plus",
    "employed",
    "born_outside_eu",
)


def download_file(url: str, destination: Path) -> Path:
    """Download ``url`` to ``destination`` if it is not already present."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination

    response = requests.get(url, timeout=120)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def ensure_population_grid_files(raw_dir: Path) -> tuple[Path, Path]:
    """Download and extract the Eurostat Census 2021 grid GeoPackage."""
    import zipfile

    archive_path = download_file(GRID_ARCHIVE_URL, raw_dir / GRID_ARCHIVE_NAME)
    extract_dir = raw_dir / GRID_DIRECTORY_NAME
    gpkg_path = extract_dir / GRID_GPKG_NAME
    readme_path = extract_dir / GRID_README_NAME

    if gpkg_path.exists() and readme_path.exists():
        return gpkg_path, readme_path

    with zipfile.ZipFile(archive_path) as archive:
        for member in (
            f"{GRID_DIRECTORY_NAME}/{GRID_GPKG_NAME}",
            f"{GRID_DIRECTORY_NAME}/{GRID_README_NAME}",
        ):
            archive.extract(member, path=raw_dir)

    return gpkg_path, readme_path


def load_city_boundaries(boundary_source: str = CITY_BOUNDARIES_URL) -> gpd.GeoDataFrame:
    """Load the Eurostat Urban Audit city polygons for Lisbon and Porto."""
    cities = gpd.read_file(boundary_source)
    cities = cities.loc[cities["URAU_CODE"].isin(CITY_CODE_TO_NAME)].copy()
    if cities.empty:
        raise ValueError("Could not find Porto and Lisbon in the Urban Audit boundary layer.")

    cities["city"] = cities["URAU_CODE"].map(CITY_CODE_TO_NAME)
    return cities.to_crs(TARGET_CRS)


def load_population_grid(grid_gpkg_path: Path, city_bounds: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    """Load only the Eurostat grid cells that intersect the Porto/Lisbon bounding box."""
    grid = gpd.read_file(grid_gpkg_path, bbox=city_bounds)
    if grid.empty:
        raise ValueError("No Eurostat grid cells found for the Porto/Lisbon bounding box.")
    return grid.to_crs(TARGET_CRS)


def _check_for_missing_grid_values(grid: gpd.GeoDataFrame) -> None:
    cleaned = grid[list(GRID_COUNT_COLUMNS)].replace(list(GRID_MISSING_CODES), pd.NA)
    missing_counts = cleaned.isna().sum()
    if int(missing_counts.sum()) == 0:
        return

    missing_summary = {
        column: int(count) for column, count in missing_counts.items() if int(count) > 0
    }
    raise ValueError(
        "The selected Eurostat grid subset contains confidentiality or missing-value codes. "
        f"Resolve these before preparing the final table: {missing_summary}"
    )


def intersect_grid_with_cities(
    grid: gpd.GeoDataFrame,
    cities: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Intersect 1 km grid cells with city boundaries and apply area weights."""
    _check_for_missing_grid_values(grid)

    source_grid = grid.copy()
    source_grid["cell_area_m2"] = source_grid.geometry.area
    parts: list[gpd.GeoDataFrame] = []

    for _, city_row in cities.iterrows():
        city_gdf = gpd.GeoDataFrame([city_row], crs=cities.crs)
        intersection = gpd.overlay(source_grid, city_gdf, how="intersection")
        if intersection.empty:
            continue

        intersection["share"] = intersection.geometry.area / intersection["cell_area_m2"]
        intersection["city"] = city_row["city"]
        intersection["cell_id"] = intersection["GRD_ID"]
        intersection["population_total"] = intersection["T"] * intersection["share"]
        intersection["age_0_14"] = intersection["Y_LT15"] * intersection["share"]
        intersection["age_65_plus"] = intersection["Y_GE65"] * intersection["share"]
        intersection["employed"] = intersection["EMP"] * intersection["share"]
        intersection["born_outside_eu"] = intersection["OTH"] * intersection["share"]

        parts.append(
            intersection[
                [
                    "city",
                    "cell_id",
                    "population_total",
                    "age_0_14",
                    "age_65_plus",
                    "employed",
                    "born_outside_eu",
                    "geometry",
                ]
            ]
        )

    if not parts:
        raise ValueError("No intersected Porto/Lisbon grid cells were produced.")

    result = pd.concat(parts, ignore_index=True)
    result = gpd.GeoDataFrame(result, geometry="geometry", crs=TARGET_CRS)
    return enforce_group_count_bounds(result)


def _query_uhi_value(x: float, y: float, request_timeout: int) -> float | None:
    params = {
        "f": "pjson",
        "geometry": f"{x},{y}",
        "geometryType": "esriGeometryPoint",
        "sr": str(UHI_QUERY_CRS),
        "layers": "all:0",
        "tolerance": "1",
        "mapExtent": f"{x - 500},{y - 500},{x + 500},{y + 500}",
        "imageDisplay": "400,400,96",
        "returnGeometry": "false",
    }
    response = requests.get(UHI_IDENTIFY_URL, params=params, timeout=request_timeout)
    response.raise_for_status()
    payload = response.json()
    results = payload.get("results", [])
    if not results:
        return None

    pixel_value = results[0]["attributes"].get("Stretch.Pixel Value")
    if pixel_value is None:
        return None
    if isinstance(pixel_value, str) and pixel_value.strip().lower() == "nodata":
        return None
    return float(pixel_value)


def sample_uhi_values(
    cells: gpd.GeoDataFrame,
    max_workers: int = 8,
    request_timeout: int = 30,
) -> gpd.GeoDataFrame:
    """Sample the EEA UHI raster service at each fragment's representative point."""
    sample_points = gpd.GeoSeries(cells.geometry.representative_point(), crs=cells.crs).to_crs(
        UHI_QUERY_CRS
    )
    coordinates = list(zip(sample_points.x, sample_points.y))
    sampled_values: list[float | None] = [None] * len(coordinates)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_query_uhi_value, x, y, request_timeout): index
            for index, (x, y) in enumerate(coordinates)
        }
        for future in as_completed(futures):
            index = futures[future]
            sampled_values[index] = future.result()

    result = cells.copy()
    result["uhi_intensity_celsius"] = sampled_values
    return fill_missing_uhi_values(result)


def fill_missing_uhi_values(cells: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Fill no-data UHI samples with the nearest non-missing cell in the same city."""
    filled_parts: list[gpd.GeoDataFrame] = []

    for city, city_cells in cells.groupby("city", sort=True):
        city_cells = city_cells.copy()
        missing_mask = city_cells["uhi_intensity_celsius"].isna()
        if not missing_mask.any():
            filled_parts.append(city_cells)
            continue

        valid = city_cells.loc[~missing_mask, ["uhi_intensity_celsius", "geometry"]].copy()
        if valid.empty:
            raise ValueError(f"UHI sampling returned no valid values for {city}.")

        valid.geometry = valid.geometry.representative_point()
        missing = city_cells.loc[missing_mask, ["geometry"]].copy()
        missing.geometry = missing.geometry.representative_point()

        nearest = gpd.sjoin_nearest(
            missing,
            valid,
            how="left",
            distance_col="_uhi_fill_distance_m",
            lsuffix="missing",
            rsuffix="valid",
        )
        city_cells.loc[missing_mask, "uhi_intensity_celsius"] = nearest[
            "uhi_intensity_celsius"
        ].to_numpy()
        city_cells.loc[missing_mask, "uhi_fill_distance_m"] = nearest[
            "_uhi_fill_distance_m"
        ].to_numpy()
        filled_parts.append(city_cells)

    return gpd.GeoDataFrame(pd.concat(filled_parts, ignore_index=True), geometry="geometry", crs=cells.crs)


def enforce_group_count_bounds(cells: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Clip subgroup counts so they cannot exceed total population in a fragment."""
    result = cells.copy()
    for column in PREPARED_COUNT_COLUMNS[1:]:
        result[column] = result[[column, "population_total"]].min(axis=1)
    return result


def finalize_prepared_cells(
    cells: gpd.GeoDataFrame,
    threshold: float = 2.0,
) -> pd.DataFrame:
    """Return the final CSV-ready table."""
    result = cells.copy()
    result["is_uhi_exposed"] = result["uhi_intensity_celsius"] >= threshold
    return pd.DataFrame(
        result[
            [
                "city",
                "cell_id",
                "population_total",
                "age_0_14",
                "age_65_plus",
                "employed",
                "born_outside_eu",
                "uhi_intensity_celsius",
                "is_uhi_exposed",
            ]
        ]
    )


def summarize_fill_stats(cells: pd.DataFrame) -> pd.DataFrame:
    """Summarize how many cells required nearest-neighbour UHI filling."""
    fill_mask = cells["uhi_fill_distance_m"].notna() if "uhi_fill_distance_m" in cells.columns else False
    if isinstance(fill_mask, bool):
        fill_mask = pd.Series([False] * len(cells), index=cells.index)

    summary = (
        cells.assign(_filled=fill_mask)
        .groupby(["city", "_filled"], as_index=False)
        .agg(
            cells=("cell_id", "count"),
            population_total=("population_total", "sum"),
        )
        .rename(columns={"_filled": "used_nearest_uhi_fill"})
    )
    return summary


def validate_prepared_output(df: pd.DataFrame) -> None:
    """Run repository schema checks plus Porto/Lisbon-specific assertions."""
    validate_exposure_cells(df)

    observed_cities = set(df["city"].unique().tolist())
    expected_cities = {"Lisbon", "Porto"}
    if observed_cities != expected_cities:
        raise ValueError(
            f"Expected cities {sorted(expected_cities)}, found {sorted(observed_cities)}."
        )
