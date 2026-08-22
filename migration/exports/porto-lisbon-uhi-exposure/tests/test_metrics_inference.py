"""Tests for the statistical-inference metrics (bootstrap, dose-response, double burden)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from uhi_exposure.metrics import (
    bootstrap_group_representation,
    compute_double_burden,
    compute_dose_response,
    compute_group_representation,
)


def _make_cells(born_outside_eu_hot: int, born_outside_eu_cool: int, n_each: int = 60) -> pd.DataFrame:
    """Synthetic city: ``n_each`` hot cells (UHI 3 C) and ``n_each`` cool cells (UHI 0 C)."""
    rows = []
    for i in range(2 * n_each):
        hot = i < n_each
        rows.append(
            {
                "city": "TestCity",
                "cell_id": f"CRS3035RES1000mN{1000 + i}E{2000 + i}",
                "population_total": 100.0,
                "age_0_14": 10.0,
                "age_65_plus": (30.0 if hot else 10.0),  # elderly concentrated in hot
                "employed": 60.0,
                "born_outside_eu": float(born_outside_eu_hot if hot else born_outside_eu_cool),
                "uhi_intensity_celsius": 3.0 if hot else 0.0,
                "is_uhi_exposed": bool(hot),
            }
        )
    return pd.DataFrame(rows)


def test_bootstrap_ci_brackets_observed_ratio() -> None:
    cells = _make_cells(born_outside_eu_hot=40, born_outside_eu_cool=10)
    observed = compute_group_representation(cells, threshold=2.0)
    boot = bootstrap_group_representation(cells, threshold=2.0, n_boot=500, random_state=1)
    merged = observed.merge(boot, on=["city", "group"], suffixes=("_obs", "_boot"))
    for _, row in merged.iterrows():
        assert row["ci_low"] <= row["representation_ratio_obs"] + 1e-9
        assert row["representation_ratio_obs"] <= row["ci_high"] + 1e-9


def test_bootstrap_flags_concentrated_group_as_significant() -> None:
    cells = _make_cells(born_outside_eu_hot=40, born_outside_eu_cool=10)
    boot = bootstrap_group_representation(cells, threshold=2.0, n_boot=1000, random_state=2)
    migrants = boot.loc[boot["group"] == "born_outside_eu"].iloc[0]
    assert migrants["representation_ratio"] > 1.0
    assert migrants["ci_low"] > 1.0
    assert migrants["significant"]


def test_bootstrap_uniform_group_not_significant() -> None:
    # Same share in hot and cool cells -> ratio ~ 1, interval should straddle 1.
    cells = _make_cells(born_outside_eu_hot=25, born_outside_eu_cool=25)
    boot = bootstrap_group_representation(cells, threshold=2.0, n_boot=1000, random_state=3)
    migrants = boot.loc[boot["group"] == "born_outside_eu"].iloc[0]
    assert migrants["ci_low"] <= 1.0 <= migrants["ci_high"]
    assert not migrants["significant"]


def test_dose_response_sign_matches_concentration() -> None:
    cells = _make_cells(born_outside_eu_hot=40, born_outside_eu_cool=10)
    dose = compute_dose_response(cells).set_index("group")
    # Migrants and elderly are concentrated in hot cells -> positive association.
    assert dose.loc["born_outside_eu", "weighted_corr_uhi_share"] > 0
    assert dose.loc["older_65_plus", "weighted_corr_uhi_share"] > 0


def test_double_burden_conserves_population() -> None:
    cells = _make_cells(born_outside_eu_hot=40, born_outside_eu_cool=10)
    db = compute_double_burden(cells)
    assert db["population"].sum() == pytest.approx(cells["population_total"].sum())
    assert db["population_share"].sum() == pytest.approx(1.0)
    assert set(db["quadrant"]).issuperset({"high heat + high vulnerability (double burden)"})


def _make_green_cells() -> pd.DataFrame:
    """Synthetic city where higher green cover goes with lower UHI (greener = cooler)."""
    rng = np.random.default_rng(0)
    green = np.linspace(0.0, 0.6, 80)
    uhi = 2.0 - 2.5 * green + rng.normal(0, 0.05, green.size)  # strong negative relation
    return pd.DataFrame(
        {
            "city": "TestCity",
            "cell_id": [f"CRS3035RES1000mN{1000 + i}E{2000 + i}" for i in range(green.size)],
            "population_total": 100.0,
            "age_0_14": 10.0,
            "age_65_plus": 10.0,
            "employed": 60.0,
            "born_outside_eu": 10.0,
            "uhi_intensity_celsius": uhi,
            "is_uhi_exposed": uhi >= 2.0,
            "green_fraction": green,
        }
    )


def test_green_relationship_is_negative() -> None:
    from uhi_exposure.metrics import compute_green_uhi_relationship

    rel = compute_green_uhi_relationship(_make_green_cells()).iloc[0]
    assert rel["weighted_corr_green_uhi"] < -0.5
    assert rel["celsius_per_10pp_green"] < 0


def test_uhi_by_green_band_decreases_with_green() -> None:
    from uhi_exposure.metrics import compute_uhi_by_green_band

    band = compute_uhi_by_green_band(_make_green_cells())
    band = band.set_index("green_band")["mean_uhi_celsius"]
    # the most-green band must be cooler than the least-green band present
    assert band.loc["<5%"] > band.iloc[-1]
