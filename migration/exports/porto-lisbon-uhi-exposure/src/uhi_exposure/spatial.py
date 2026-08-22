"""Spatial helpers for prepared UHI exposure tables."""

from __future__ import annotations

import re
from typing import Final

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

CELL_ID_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"CRS3035RES(?P<resolution>\d+)mN(?P<northing>\d+)E(?P<easting>\d+)"
)
TARGET_CRS: Final[int] = 3035


def parse_grid_cell_id(cell_id: str) -> tuple[float, float, float]:
    """Parse a Eurostat grid cell id into easting, northing, and resolution."""
    match = CELL_ID_PATTERN.fullmatch(cell_id)
    if match is None:
        raise ValueError(f"Unsupported grid cell id format: {cell_id!r}")

    resolution = float(match.group("resolution"))
    northing = float(match.group("northing"))
    easting = float(match.group("easting"))
    return easting, northing, resolution


def cell_id_to_geometry(cell_id: str):
    """Return the square geometry encoded by a Eurostat 1 km cell id."""
    easting, northing, resolution = parse_grid_cell_id(cell_id)
    return box(easting, northing, easting + resolution, northing + resolution)


def prepared_cells_to_geodataframe(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Reconstruct grid cell geometries from the prepared CSV cell ids."""
    geometry = df["cell_id"].map(cell_id_to_geometry)
    return gpd.GeoDataFrame(df.copy(), geometry=geometry, crs=TARGET_CRS)
