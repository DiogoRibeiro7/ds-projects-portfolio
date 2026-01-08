"""
Unit tests for pure helpers in src.data_processing.cleaning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_processing.cleaning import _detect_outliers, validate_experiment_data

pytestmark = pytest.mark.unit


class TestDetectOutliers:
    """Exercises deterministic behavior of the private outlier helper."""

    def test_iqr_outlier_detection_marks_extremes(self):
        series = pd.Series([10, 12, 11, 9, 10, 500])
        mask = _detect_outliers(series, method="iqr", threshold=1.5)
        assert mask.sum() == 1
        assert mask.iloc[-1]

    def test_zscore_constant_series_has_no_outliers(self):
        series = pd.Series([5.0] * 10)
        mask = _detect_outliers(series, method="zscore", threshold=3.0)
        assert not mask.any()
        # Deterministic — same mask every call
        pd.testing.assert_series_equal(mask, _detect_outliers(series, method="zscore"))

    def test_invalid_method_raises_value_error(self):
        series = pd.Series([1, 2, 3])
        with pytest.raises(ValueError, match="Unknown outlier detection method"):
            _detect_outliers(series, method="does-not-exist")


class TestValidateExperimentData:
    """Covers validation invariants and error paths."""

    def test_valid_dataset_returns_true(self):
        df = pd.DataFrame(
            {
                "user_id": range(6),
                "group": ["A", "A", "A", "B", "B", "B"],
                "metric": np.linspace(0.1, 0.6, 6),
            }
        )
        results = validate_experiment_data(df, required_groups=["A", "B"])
        assert results["is_valid"]
        assert results["errors"] == []
        assert results["data_quality_score"] == 100

    def test_missing_required_column_marks_invalid(self):
        df = pd.DataFrame({"user_id": [1, 2, 3]})
        results = validate_experiment_data(df)
        assert not results["is_valid"]
        assert "Missing required columns" in results["errors"][0]

    def test_missing_values_trigger_errors_and_score_reduction(self):
        df = pd.DataFrame(
            {
                "user_id": [1, 2, None, 4],
                "group": ["A", "B", "A", None],
            }
        )
        results = validate_experiment_data(df)
        assert not results["is_valid"]
        assert any("missing values" in err for err in results["errors"])
        assert results["data_quality_score"] < 100

    def test_sample_ratio_warning_when_groups_unbalanced(self):
        df = pd.DataFrame(
            {
                "user_id": range(11),
                "group": ["A"] * 10 + ["B"],
            }
        )
        results = validate_experiment_data(df, min_sample_ratio=0.8)
        assert results["is_valid"]
        assert any("Sample ratio mismatch" in warn for warn in results["warnings"])
        assert results["data_quality_score"] < 100
