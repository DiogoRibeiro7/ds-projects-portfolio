"""
Regression tests for data cleaning edge cases.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_processing.cleaning import clean_ab_data

pytestmark = pytest.mark.regression


def test_clean_ab_data_handles_missing_metric_columns():
    """
    Bug repro: historically passing a metric column that did not exist would
    raise a KeyError during outlier detection. Ensure the helper skips cleanly.
    """
    df = pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4],
            "group": ["A", "A", "B", "B"],
            "converted": [0, 1, 0, 1],
        }
    )

    cleaned, report = clean_ab_data(
        df,
        user_col="user_id",
        group_col="group",
        metric_cols=["not_a_metric"],
        generate_report=True,
    )

    assert len(cleaned) == len(df)
    step_names = {step["step"] for step in report["cleaning_steps"]}
    assert "duplicate_removal" not in step_names  # no duplicates removed


def test_clean_ab_data_retains_group_balance_after_missing_values():
    """
    Regression guard: removing rows with missing critical columns must preserve
    remaining groups and not introduce NaNs in the cleaned frame.
    """
    df = pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4, None],
            "group": ["A", "B", "A", None, "B"],
            "converted": [0, 1, 0, 1, 0],
        }
    )

    cleaned = clean_ab_data(df, user_col="user_id", group_col="group")

    assert cleaned["user_id"].isnull().sum() == 0
    assert cleaned["group"].isnull().sum() == 0
    assert set(cleaned["group"].unique()) == {"A", "B"}
    assert len(cleaned) == 3
