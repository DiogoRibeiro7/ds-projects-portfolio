"""Integration tests covering the experiment data workflow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_processing.cleaning import clean_ab_data, validate_experiment_data
from src.statistics.core import ExperimentAnalyzer

pytestmark = pytest.mark.integration


def _make_raw_experiment_frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "user_id": list(range(40)) + [0, 1, 2],  # duplicates
            "group": ["control"] * 20 + ["treatment"] * 23,
            "converted": np.concatenate(
                [rng.integers(0, 2, size=20), rng.integers(0, 2, size=23)]
            ),
            "metric": rng.normal(100, 10, size=43),
        }
    )
    df.loc[5, "converted"] = np.nan
    df.loc[10, "metric"] = 9999  # extreme outlier
    return df


def test_clean_validate_and_analyze_end_to_end():
    raw = _make_raw_experiment_frame()

    cleaned, report = clean_ab_data(
        raw,
        user_col="user_id",
        group_col="group",
        metric_cols=["converted", "metric"],
        outlier_method="iqr",
        generate_report=True,
    )

    # Duplicates and outliers removed
    assert len(cleaned) < len(raw)
    step_names = {step["step"] for step in report["cleaning_steps"]}
    assert {"duplicate_removal", "outlier_removal"}.issubset(step_names)

    # Validate cleaned data shape/invariants before analysis
    validation = validate_experiment_data(
        cleaned, required_groups=["control", "treatment"]
    )
    assert validation["is_valid"]
    assert validation["warnings"] or validation["data_quality_score"] <= 100

    analyzer = ExperimentAnalyzer()
    results = analyzer.analyze_conversion(cleaned, conversion_col="converted")

    # Ensure report surfaces the usual experiment stats
    assert set(results["conversion_rates"].keys()) == {"control", "treatment"}
    assert "absolute_lift" in results
    assert "z_statistic" in results["statistics"]
