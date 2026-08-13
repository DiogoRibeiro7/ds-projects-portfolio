from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


@pytest.fixture
def synthetic_panel() -> pd.DataFrame:
    """A well-behaved macro panel: steady population and nominal GDP growth.

    Deterministic so posterior assertions are stable, but with enough variation
    that the growth regressions are identified.
    """

    years = np.arange(1970, 2025)
    n = len(years)
    drift = 0.004 + 0.002 * np.sin(np.arange(n) / 6.0)
    population = 9_000_000 * np.exp(np.cumsum(drift))

    deflator_pct = 2.5 + 0.8 * np.sin(np.arange(n) / 5.0)
    real_growth = 0.02 + 0.005 * np.cos(np.arange(n) / 4.0)
    nominal_growth = np.log1p(deflator_pct / 100.0) + np.log1p(real_growth)
    gdp = 50e9 * np.exp(np.cumsum(nominal_growth))

    return pd.DataFrame(
        {
            "year": years,
            "population": population,
            "gdp_current_eur": gdp,
            "gdp_deflator_pct": deflator_pct,
            "real_gdp_growth_pct": real_growth * 100.0,
        }
    )


@pytest.fixture
def pps_inputs() -> pd.DataFrame:
    """Published PPS inputs for Portugal, as Eurostat reports them.

    Real values rather than round numbers, because the point under test is a
    property of the actual data: ``gdp_meur / gdp_pc_eur`` recovers
    ``na_population`` (10.80M) and not ``population_avg`` (11.41M).
    """

    return pd.DataFrame(
        [
            {
                "year": 2024,
                "gdp_meur": 289_784.3,
                "gdp_pc_eur": 27_100.0,
                "gdp_pc_pps": 32_937.1,
                "eu_gdp_pc_pps": 39_985.4,
                "published_index": 82.0,
                "na_population": 10_694_680.0,
                "population_avg": 11_295_785.0,
            },
            {
                "year": 2025,
                "gdp_meur": 306_749.6,
                "gdp_pc_eur": 28_390.0,
                "gdp_pc_pps": 33_825.0,
                "eu_gdp_pc_pps": 41_565.7,
                "published_index": 81.0,
                "na_population": 10_804_160.0,
                "population_avg": 11_405_627.0,
            },
        ]
    )
