from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

from uhi_exposure.real_data import (
    enforce_group_count_bounds,
    finalize_prepared_cells,
    intersect_grid_with_cities,
    validate_prepared_output,
)
from uhi_exposure.spatial import cell_id_to_geometry, parse_grid_cell_id


def test_intersect_grid_with_cities_applies_area_weights() -> None:
    grid = gpd.GeoDataFrame(
        {
            "GRD_ID": ["cell_1"],
            "T": [100],
            "Y_LT15": [20],
            "Y_GE65": [10],
            "EMP": [60],
            "OTH": [5],
        },
        geometry=[Polygon([(0, 0), (1000, 0), (1000, 1000), (0, 1000)])],
        crs=3035,
    )
    cities = gpd.GeoDataFrame(
        {
            "URAU_CODE": ["PT001C"],
            "URAU_NAME": ["Lisboa"],
            "city": ["Lisbon"],
        },
        geometry=[Polygon([(0, 0), (500, 0), (500, 1000), (0, 1000)])],
        crs=3035,
    )

    result = intersect_grid_with_cities(grid, cities)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["city"] == "Lisbon"
    assert row["cell_id"] == "cell_1"
    assert row["population_total"] == pytest.approx(50.0)
    assert row["age_0_14"] == pytest.approx(10.0)
    assert row["age_65_plus"] == pytest.approx(5.0)
    assert row["employed"] == pytest.approx(30.0)
    assert row["born_outside_eu"] == pytest.approx(2.5)


def test_enforce_group_count_bounds_clips_group_values() -> None:
    cells = gpd.GeoDataFrame(
        {
            "city": ["Porto"],
            "cell_id": ["cell_1"],
            "population_total": [10.0],
            "age_0_14": [12.0],
            "age_65_plus": [3.0],
            "employed": [11.0],
            "born_outside_eu": [8.0],
        },
        geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        crs=3035,
    )

    result = enforce_group_count_bounds(cells)

    row = result.iloc[0]
    assert row["age_0_14"] == pytest.approx(10.0)
    assert row["employed"] == pytest.approx(10.0)
    assert row["age_65_plus"] == pytest.approx(3.0)
    assert row["born_outside_eu"] == pytest.approx(8.0)


def test_finalize_and_validate_prepared_output() -> None:
    sampled = gpd.GeoDataFrame(
        {
            "city": ["Lisbon", "Porto"],
            "cell_id": ["L1", "P1"],
            "population_total": [100.0, 50.0],
            "age_0_14": [20.0, 8.0],
            "age_65_plus": [15.0, 12.0],
            "employed": [60.0, 20.0],
            "born_outside_eu": [10.0, 5.0],
            "uhi_intensity_celsius": [2.5, 1.5],
        },
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
        ],
        crs=3035,
    )

    prepared = finalize_prepared_cells(sampled, threshold=2.0)

    assert prepared["is_uhi_exposed"].tolist() == [True, False]
    validate_prepared_output(prepared)


def test_validate_prepared_output_requires_both_cities() -> None:
    prepared = pd.DataFrame(
        {
            "city": ["Lisbon"],
            "cell_id": ["L1"],
            "population_total": [10.0],
            "age_0_14": [2.0],
            "age_65_plus": [1.0],
            "employed": [6.0],
            "born_outside_eu": [1.0],
            "uhi_intensity_celsius": [2.1],
            "is_uhi_exposed": [True],
        }
    )

    with pytest.raises(ValueError, match="Expected cities"):
        validate_prepared_output(prepared)


def test_parse_grid_cell_id_and_geometry() -> None:
    easting, northing, resolution = parse_grid_cell_id("CRS3035RES1000mN1929000E2657000")
    geometry = cell_id_to_geometry("CRS3035RES1000mN1929000E2657000")

    assert easting == pytest.approx(2657000.0)
    assert northing == pytest.approx(1929000.0)
    assert resolution == pytest.approx(1000.0)
    assert geometry.bounds == pytest.approx((2657000.0, 1929000.0, 2658000.0, 1930000.0))
