"""Regression tests for data-quality helpers used in the experimentation stack."""

import numpy as np
import pandas as pd
import pytest

from src.data_processing.cleaning import (
    DataQualityChecker,
    clean_ab_data,
    validate_experiment_data,
)

pytestmark = pytest.mark.integration


class TestDataCleaning:
    """Test suite for data cleaning functions."""

    @pytest.fixture
    def messy_data(self):
        """Create messy test data."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "user_id": list(range(100)) + [0, 1, 2],  # Duplicates
                "group": np.random.choice(["A", "B"], 103),
                "metric": np.random.normal(100, 20, 103),
            }
        )
        # Add outliers
        df.loc[50, "metric"] = 1000
        df.loc[51, "metric"] = -1000
        # Add missing values
        df.loc[60:65, "metric"] = np.nan
        return df

    def test_duplicate_removal(self, messy_data):
        """Test duplicate user removal."""
        cleaned = clean_ab_data(messy_data, user_col="user_id")
        assert len(cleaned) == 100  # Should remove 3 duplicates
        assert cleaned["user_id"].nunique() == len(cleaned)

    def test_outlier_removal(self, messy_data):
        """Test outlier detection and removal."""
        cleaned = clean_ab_data(
            messy_data,
            user_col="user_id",
            metric_cols=["metric"],
            outlier_method="zscore",
            outlier_threshold=3,
        )
        assert cleaned["metric"].max() < 1000
        assert cleaned["metric"].min() > -1000

    def test_missing_value_handling(self, messy_data):
        """Test missing value handling."""
        cleaned = clean_ab_data(messy_data, user_col="user_id")
        assert cleaned["metric"].isna().sum() == 0


class TestDataValidation:
    """Test suite for data validation."""

    def test_valid_experiment_data(self):
        """Test validation with valid data."""
        df = pd.DataFrame(
            {
                "user_id": range(2000),
                "group": ["control"] * 1000 + ["treatment"] * 1000,
                "converted": np.random.binomial(1, 0.1, 2000),
            }
        )

        validation = validate_experiment_data(df)
        assert validation["is_valid"]
        assert len(validation["errors"]) == 0

    def test_missing_required_columns(self):
        """Test validation with missing columns."""
        df = pd.DataFrame({"some_col": range(100)})

        validation = validate_experiment_data(df, user_col="user_id")
        assert not validation["is_valid"]
        assert "Missing required columns" in str(validation["errors"])

    def test_sample_ratio_mismatch(self):
        """Test detection of sample ratio mismatch."""
        df = pd.DataFrame(
            {"user_id": range(1300), "group": ["control"] * 1000 + ["treatment"] * 300}
        )

        validation = validate_experiment_data(df, min_sample_ratio=0.8)
        assert "Sample ratio mismatch" in str(validation["warnings"])


class TestDataQualityChecker:
    """Test suite for DataQualityChecker."""

    def test_comprehensive_check(self):
        """Test comprehensive data quality check."""
        df = pd.DataFrame(
            {
                "user_id": range(1000),
                "group": np.random.choice(["A", "B"], 1000),
                "metric": np.random.normal(100, 20, 1000),
                "date": pd.date_range("2024-01-01", periods=1000, freq="h"),
            }
        )

        config = {
            "user_col": "user_id",
            "group_col": "group",
            "date_col": "date",
            "metric_cols": ["metric"],
        }

        checker = DataQualityChecker()
        report = checker.run_full_check(df, config)

        assert "overall_score" in report
        assert report["overall_score"] > 80  # Should be good quality
        assert "recommendations" in report
