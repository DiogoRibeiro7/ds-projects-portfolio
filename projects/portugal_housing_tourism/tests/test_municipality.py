"""Tests for municipality-level descriptive comparisons."""

from __future__ import annotations

import pandas as pd
import pytest

from housing_tourism.municipality import municipality_change_panel, summarise_reference_municipality


def _panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "geo_code": ["A", "A", "B", "B", "C", "C"],
            "geo_name": ["Alpha", "Alpha", "Beta", "Beta", "Gamma", "Gamma"],
            "year": [2022, 2023, 2022, 2023, 2022, 2023],
            "rent_eur_m2": [10.0, 12.0, 10.0, 11.0, 10.0, 10.5],
            "income_eur": [100.0, 102.0, 100.0, 105.0, 100.0, 110.0],
            "resident_population": [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0],
            "overnight_stays": [1000.0, 1200.0, 1000.0, 900.0, 1000.0, 1100.0],
        }
    )


def test_municipality_change_panel_ranks_largest_deterioration_first() -> None:
    result = municipality_change_panel(_panel(), start_year=2022, end_year=2023)

    assert result.iloc[0]["geo_name"] == "Alpha"
    assert result.iloc[0]["affordability_rank_desc"] == 1
    assert result.iloc[0]["municipality_count"] == 3
    assert result.iloc[0]["rent_change_pct"] == pytest.approx(20.0)
    assert result.iloc[0]["income_change_pct"] == pytest.approx(2.0)
    assert result.iloc[0]["rent_income_ratio_change_pct"] == pytest.approx(17.6470588)


def test_reference_summary_extracts_rank_and_changes() -> None:
    result = municipality_change_panel(_panel(), start_year=2022, end_year=2023)
    summary = summarise_reference_municipality(result, geo_name="beta")

    assert summary["municipality_count"] == 3
    assert summary["affordability_rank_desc"] == 2
    assert summary["rent_change_pct"] == pytest.approx(10.0)
    assert summary["income_change_pct"] == pytest.approx(5.0)


def test_duplicate_geography_year_is_rejected() -> None:
    frame = pd.concat([_panel(), _panel().iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="uniquely identify"):
        municipality_change_panel(frame, start_year=2022, end_year=2023)


def test_invalid_episode_order_is_rejected() -> None:
    with pytest.raises(ValueError, match="greater"):
        municipality_change_panel(_panel(), start_year=2023, end_year=2022)
