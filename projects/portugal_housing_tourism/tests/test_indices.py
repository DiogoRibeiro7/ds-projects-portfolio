from __future__ import annotations

import pandas as pd
import pytest

from housing_tourism.indices import (
    external_housing_pressure_index,
    local_housing_decoupling_index,
    tourism_intensity,
    tourist_housing_conversion_rate,
)


def test_thcr_is_units_per_thousand() -> None:
    al = pd.Series([50.0, 100.0])
    stock = pd.Series([10_000.0, 20_000.0])
    result = tourist_housing_conversion_rate(al, stock)
    assert result.tolist() == pytest.approx([5.0, 5.0])


def test_tourism_intensity() -> None:
    result = tourism_intensity(pd.Series([1_000.0]), pd.Series([100.0]))
    assert result.iloc[0] == pytest.approx(10.0)


def test_lhdi_uses_rent_income_ratio() -> None:
    frame = pd.DataFrame(
        {
            "geo_code": ["A", "A"],
            "year": [2017, 2018],
            "rent_eur_m2": [10.0, 15.0],
            "income_eur": [100.0, 120.0],
        }
    )
    result = local_housing_decoupling_index(frame, base_year=2017)
    assert result.tolist() == pytest.approx([100.0, 125.0])


def test_ehpi_is_prediction_ratio() -> None:
    result = external_housing_pressure_index(
        pd.Series([12.5, 10.0]),
        pd.Series([10.0, 10.0]),
    )
    assert result.tolist() == pytest.approx([125.0, 100.0])
