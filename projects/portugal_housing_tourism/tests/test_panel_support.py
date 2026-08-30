"""Tests for municipality panel support diagnostics."""

from __future__ import annotations

import pandas as pd
import pytest

from housing_tourism.panel_support import summarise_panel_support


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "geo_code": ["A", "A", "A", "B", "B", "B"],
            "geo_name": ["Alpha", "Alpha", "Alpha", "Beta", "Beta", "Beta"],
            "year": [2022, 2023, 2024, 2022, 2023, 2024],
            "rent_eur_m2": [10.0, 11.0, 12.0, 8.0, None, 9.0],
            "income_eur": [100.0, 102.0, 104.0, 90.0, 92.0, 94.0],
            "resident_population": [1000.0, 1000.0, 1000.0, 800.0, 800.0, 800.0],
            "overnight_stays": [1000.0, 1100.0, 1200.0, 400.0, None, 500.0],
        }
    )


def test_support_summary_preserves_unbalanced_affordability_panel() -> None:
    summary = summarise_panel_support(
        _frame(),
        years=(2022, 2023, 2024),
        value_columns=(
            "rent_eur_m2",
            "income_eur",
            "resident_population",
            "overnight_stays",
        ),
    )

    assert summary["municipality_universe"] == 2
    assert summary["affordability_at_least_two_years"] == 2
    assert summary["affordability_all_years"] == 1
    assert summary["tourism_at_least_two_years"] == 2
    assert summary["tourism_all_years"] == 1


def test_duplicate_municipality_year_is_rejected() -> None:
    frame = pd.concat([_frame(), _frame().iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="uniquely identify"):
        summarise_panel_support(
            frame,
            years=(2022, 2023, 2024),
            value_columns=("rent_eur_m2", "income_eur"),
        )


def test_empty_years_are_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        summarise_panel_support(
            _frame(),
            years=(),
            value_columns=("rent_eur_m2", "income_eur"),
        )
