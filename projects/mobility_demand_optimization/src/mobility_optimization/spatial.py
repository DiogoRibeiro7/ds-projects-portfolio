"""Spatial relocation-cost construction from official TLC taxi-zone geometry."""

from __future__ import annotations

from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryDirectory
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile

import geopandas as gpd
import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
TLC_TAXI_ZONES_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
TLC_PROJECTED_CRS = "EPSG:2263"


def download_taxi_zones(*, destination: Path, overwrite: bool = False) -> Path:
    """Download the official TLC taxi-zone shapefile archive atomically.

    Args:
        destination: Local ZIP destination.
        overwrite: Whether to replace an existing archive.

    Returns:
        The downloaded or already-existing archive path.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        return destination

    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urlopen(TLC_TAXI_ZONES_URL, timeout=120) as response, temporary.open("wb") as output:  # noqa: S310
            copyfileobj(response, output, length=1024 * 1024)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _extract_shapefile(archive: Path, *, destination: Path) -> Path:
    """Validate and extract one ESRI shapefile from the TLC ZIP archive.

    Args:
        archive: Downloaded TLC ZIP archive.
        destination: Empty directory used for extraction.

    Returns:
        Path to the extracted ``.shp`` file.

    Raises:
        ValueError: If the archive is invalid or does not contain exactly one shapefile.
    """
    if not archive.is_file():
        raise ValueError(f"Taxi-zone archive does not exist: {archive}")

    try:
        with ZipFile(archive) as bundle:
            names = bundle.namelist()
            shapefiles = [name for name in names if name.lower().endswith(".shp")]
            if len(shapefiles) != 1:
                raise ValueError(
                    "Taxi-zone archive must contain exactly one .shp file; "
                    f"found {len(shapefiles)}."
                )
            bundle.extractall(destination)
    except BadZipFile as exc:
        raise ValueError("Taxi-zone download is not a valid ZIP archive.") from exc

    shapefile = destination / shapefiles[0]
    if not shapefile.is_file():
        raise ValueError("Extracted taxi-zone shapefile is missing.")
    return shapefile


def load_zone_centroids(
    archive: Path,
    *,
    zone_ids: tuple[int, ...],
    projected_crs: str = TLC_PROJECTED_CRS,
) -> FloatArray:
    """Return projected centroid coordinates ordered by requested TLC zone IDs.

    Args:
        archive: Official taxi-zone ZIP archive.
        zone_ids: Unique zone identifiers in desired matrix order.
        projected_crs: Projected CRS used before centroid calculation.

    Returns:
        ``(n_zones, 2)`` array of centroid x/y coordinates.

    Raises:
        ValueError: If zone IDs are invalid, missing, duplicated, or geometry is unusable.
    """
    if not zone_ids or len(set(zone_ids)) != len(zone_ids):
        raise ValueError("zone_ids must be a non-empty sequence of unique identifiers.")

    with TemporaryDirectory(prefix="tlc-taxi-zones-") as temporary:
        shapefile = _extract_shapefile(archive, destination=Path(temporary))
        zones = gpd.read_file(shapefile)

    if "LocationID" not in zones.columns:
        raise ValueError("Taxi-zone geometry is missing LocationID.")
    if zones.crs is None:
        raise ValueError("Taxi-zone geometry must declare a coordinate reference system.")

    zones = zones[["LocationID", "geometry"]].copy()
    zones["LocationID"] = zones["LocationID"].astype(int)
    if zones["LocationID"].duplicated().any():
        raise ValueError("Taxi-zone geometry contains duplicate LocationID values.")

    selected = zones.loc[zones["LocationID"].isin(zone_ids)].copy()
    missing = sorted(set(zone_ids).difference(selected["LocationID"]))
    if missing:
        raise ValueError(f"Taxi-zone geometry is missing requested zones: {missing}")
    if selected.geometry.isna().any() or selected.geometry.is_empty.any():
        raise ValueError("Taxi-zone geometry contains missing or empty shapes.")

    projected = selected.to_crs(projected_crs)
    centroids = projected.geometry.centroid
    coordinates = {
        int(zone_id): (float(point.x), float(point.y))
        for zone_id, point in zip(projected["LocationID"], centroids, strict=True)
    }
    return np.asarray([coordinates[zone_id] for zone_id in zone_ids], dtype=np.float64)


def normalized_distance_cost_matrix(
    coordinates: npt.ArrayLike,
    *,
    median_off_diagonal_cost: float = 0.25,
) -> FloatArray:
    """Construct Euclidean relocation costs with a frozen median price level.

    Distances preserve spatial heterogeneity while scaling the median non-zero
    relocation cost to ``median_off_diagonal_cost``. This makes the spatial
    sensitivity comparable to the earlier uniform-cost experiment.
    """
    points = np.asarray(coordinates, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 2:
        raise ValueError("coordinates must have shape (n_zones, 2) with at least two zones.")
    if not np.isfinite(points).all():
        raise ValueError("coordinates must contain only finite values.")
    if not np.isfinite(median_off_diagonal_cost) or median_off_diagonal_cost <= 0.0:
        raise ValueError("median_off_diagonal_cost must be finite and positive.")

    delta = points[:, None, :] - points[None, :, :]
    distance = np.sqrt(np.sum(delta**2, axis=2))
    mask = ~np.eye(points.shape[0], dtype=bool)
    off_diagonal = distance[mask]
    median_distance = float(np.median(off_diagonal))
    if median_distance <= 0.0:
        raise ValueError("Distinct zones must not all share identical centroids.")

    costs = distance * (median_off_diagonal_cost / median_distance)
    np.fill_diagonal(costs, 0.0)
    return costs.astype(np.float64)
