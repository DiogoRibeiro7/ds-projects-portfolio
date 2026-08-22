"""Green-space cover per grid cell from OpenStreetMap, for the green-vs-heat analysis.

We query OpenStreetMap (Overpass API) for parks and other vegetated land, build the
union of those polygons per city, and compute the fraction of each 1 km grid cell that
is green. That fraction is then related to the cell's modelled urban-heat-island
intensity to test whether greener cells are cooler.

Results are cached to ``data/raw`` so the notebook only hits the network on first run.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Polygon
from shapely.ops import unary_union

from uhi_exposure.spatial import TARGET_CRS, prepared_cells_to_geodataframe

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
USER_AGENT = "porto-lisbon-uhi-green/1.0 (portfolio research)"

# OSM tags treated as green / park land for the cooling analysis.
GREEN_TAG_FILTERS: tuple[str, ...] = (
    '["leisure"~"park|garden|nature_reserve|recreation_ground|dog_park|common"]',
    '["landuse"~"forest|grass|meadow|recreation_ground|village_green|greenfield"]',
    '["natural"~"wood|grassland|scrub|heath"]',
)


def _overpass_query(south: float, west: float, north: float, east: float) -> str:
    parts = "".join(
        f"way{t}({south},{west},{north},{east});relation{t}({south},{west},{north},{east});"
        for t in GREEN_TAG_FILTERS
    )
    return f"[out:json][timeout:180];({parts});out geom;"


def _polygons_from_overpass(payload: dict) -> list[Polygon]:
    """Build valid polygons from an Overpass ``out geom`` response (ways + relation members)."""
    polygons: list[Polygon] = []

    def _add(geometry: list[dict]) -> None:
        points = [(node["lon"], node["lat"]) for node in geometry]
        if len(points) >= 4:
            try:
                polygon = Polygon(points).buffer(0)
            except Exception:
                return
            if polygon.is_valid and not polygon.is_empty:
                polygons.append(polygon)

    for element in payload.get("elements", []):
        if element.get("type") == "way" and element.get("geometry"):
            _add(element["geometry"])
        elif element.get("type") == "relation":
            for member in element.get("members", []):
                if member.get("type") == "way" and member.get("geometry"):
                    _add(member["geometry"])
    return polygons


def fetch_city_green_union(
    cells_geo: gpd.GeoDataFrame,
    city: str,
    *,
    timeout_seconds: int = 200,
):
    """Fetch and union all OSM green polygons covering one city's grid extent (EPSG:3035)."""
    city_cells = cells_geo.loc[cells_geo["city"] == city]
    west, south, east, north = city_cells.to_crs(4326).total_bounds
    response = requests.post(
        OVERPASS_URL,
        data={"data": _overpass_query(south, west, north, east)},
        timeout=timeout_seconds,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    polygons = _polygons_from_overpass(response.json())
    if not polygons:
        raise RuntimeError(f"Overpass returned no green polygons for {city}.")
    green = gpd.GeoSeries(polygons, crs=4326).to_crs(TARGET_CRS)
    return unary_union(green.values), len(polygons)


def attach_green_fraction(
    cells: pd.DataFrame,
    cache_path: str | Path,
    *,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return ``cells`` with a ``green_fraction`` column (share of the cell that is green).

    The per-cell fractions are cached as CSV at ``cache_path``; pass
    ``force_refresh=True`` to re-query OpenStreetMap.
    """
    cache_path = Path(cache_path)
    if cache_path.exists() and not force_refresh:
        cached = pd.read_csv(cache_path)
        return cells.merge(cached[["city", "cell_id", "green_fraction"]], on=["city", "cell_id"], how="left")

    cells_geo = prepared_cells_to_geodataframe(cells)
    # One geometry per cell (fragments of a cell share its 1 km box).
    unique_cells = cells_geo.drop_duplicates(subset=["city", "cell_id"]).copy()

    fractions: list[pd.DataFrame] = []
    for city in sorted(unique_cells["city"].unique()):
        green_union, n_polygons = fetch_city_green_union(cells_geo, city)
        city_cells = unique_cells.loc[unique_cells["city"] == city].copy()
        cell_area = city_cells.geometry.area
        green_area = city_cells.geometry.intersection(green_union).area
        city_cells["green_fraction"] = (green_area / cell_area).clip(0.0, 1.0)
        print(f"{city}: {len(city_cells)} cells, {n_polygons:,} OSM green polygons, "
              f"mean green cover {city_cells['green_fraction'].mean():.1%}")
        fractions.append(city_cells[["city", "cell_id", "green_fraction"]])

    green_table = pd.concat(fractions, ignore_index=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    green_table.to_csv(cache_path, index=False)
    return cells.merge(green_table, on=["city", "cell_id"], how="left")
