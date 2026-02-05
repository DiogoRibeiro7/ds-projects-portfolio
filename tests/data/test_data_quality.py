"""
Data quality testing with Great Expectations and other validation tools.
"""

import pytest
import pandas as pd
import numpy as np
import great_expectations as gx
from great_expectations.core.batch import RuntimeBatchRequest
from great_expectations.core import ExpectationSuite, ExpectationConfiguration
from great_expectations.checkpoint import SimpleCheckpoint
import pandera as pa
from pandera import Column, DataFrameSchema, Check
from typing import Dict, Any, List, Optional
import json
from pathlib import Path
from faker import Faker
from hypothesis import given, strategies as st, assume
from hypothesis.extra.pandas import data_frames, column, range_indexes
import hashlib
from datetime import datetime, timedelta


class TestGreatExpectations:
    """Test data quality with Great Expectations."""

    @pytest.mark.data
    def test_customer_data_expectations(self, sample_dataframe, data_context):
        """Test customer data meets expectations."""
        # Create expectations suite
        suite = ExpectationSuite(expectation_suite_name="customer_data_suite")

        # Add expectations
        expectations = [
            # Column existence
            {
                "expectation_type": "expect_column_to_exist",
                "kwargs": {"column": "customer_id"},
            },
            {"expectation_type": "expect_column_to_exist", "kwargs": {"column": "age"}},
            {
                "expectation_type": "expect_column_to_exist",
                "kwargs": {"column": "churn"},
            },
            # Data types
            {
                "expectation_type": "expect_column_values_to_be_of_type",
                "kwargs": {"column": "age", "type_": "int"},
            },
            {
                "expectation_type": "expect_column_values_to_be_of_type",
                "kwargs": {"column": "balance", "type_": "float"},
            },
            # Value constraints
            {
                "expectation_type": "expect_column_values_to_be_between",
                "kwargs": {"column": "age", "min_value": 18, "max_value": 100},
            },
            {
                "expectation_type": "expect_column_values_to_be_between",
                "kwargs": {
                    "column": "credit_score",
                    "min_value": 300,
                    "max_value": 850,
                },
            },
            {
                "expectation_type": "expect_column_values_to_be_in_set",
                "kwargs": {"column": "churn", "value_set": [0, 1]},
            },
            # Uniqueness
            {
                "expectation_type": "expect_column_values_to_be_unique",
                "kwargs": {"column": "customer_id"},
            },
            # Null checks
            {
                "expectation_type": "expect_column_values_to_not_be_null",
                "kwargs": {"column": "customer_id"},
            },
            {
                "expectation_type": "expect_column_values_to_not_be_null",
                "kwargs": {"column": "age", "mostly": 0.95},
            },
            # Statistical expectations
            {
                "expectation_type": "expect_column_mean_to_be_between",
                "kwargs": {"column": "age", "min_value": 30, "max_value": 60},
            },
            {
                "expectation_type": "expect_column_stdev_to_be_between",
                "kwargs": {"column": "balance", "min_value": 100, "max_value": 100000},
            },
            # Distribution expectations
            {
                "expectation_type": "expect_column_kl_divergence_to_be_less_than",
                "kwargs": {
                    "column": "credit_score",
                    "partition_object": {"bins": 10},
                    "threshold": 0.5,
                },
            },
            # Categorical expectations
            {
                "expectation_type": "expect_column_values_to_be_in_set",
                "kwargs": {
                    "column": "geography",
                    "value_set": ["USA", "UK", "Germany"],
                },
            },
            {
                "expectation_type": "expect_column_values_to_be_in_set",
                "kwargs": {"column": "gender", "value_set": ["Male", "Female"]},
            },
        ]

        for exp in expectations:
            suite.add_expectation(
                ExpectationConfiguration(
                    expectation_type=exp["expectation_type"], kwargs=exp["kwargs"]
                )
            )

        # Add suite to context
        data_context.add_expectation_suite(expectation_suite=suite)

        # Create batch request
        batch_request = RuntimeBatchRequest(
            datasource_name="pandas_datasource",
            data_connector_name="default_runtime_data_connector_name",
            data_asset_name="customer_data",
            batch_identifiers={"default_identifier_name": "default_identifier"},
            runtime_parameters={"batch_data": sample_dataframe},
        )

        # Validate
        validator = data_context.get_validator(
            batch_request=batch_request, expectation_suite_name="customer_data_suite"
        )

        validation_result = validator.validate()
        assert validation_result.success

    @pytest.mark.data
    def test_data_profiling(self, sample_dataframe, data_context):
        """Test automatic data profiling."""
        # Create profiler
        from great_expectations.profile.user_configurable_profiler import (
            UserConfigurableProfiler,
        )

        profiler = UserConfigurableProfiler(
            profile_dataset=sample_dataframe,
            excluded_expectations=[
                "expect_column_values_to_be_unique",
                "expect_compound_columns_to_be_unique",
            ],
            value_set_threshold="MANY",
        )

        # Build expectations suite
        suite = profiler.build_suite()

        assert len(suite.expectations) > 0
        assert suite.expectation_suite_name is not None

        # Validate against the same data (should pass)
        batch_request = RuntimeBatchRequest(
            datasource_name="pandas_datasource",
            data_connector_name="default_runtime_data_connector_name",
            data_asset_name="profiled_data",
            batch_identifiers={"default_identifier_name": "default_identifier"},
            runtime_parameters={"batch_data": sample_dataframe},
        )

        validator = data_context.get_validator(
            batch_request=batch_request, expectation_suite=suite
        )

        validation_result = validator.validate()
        assert validation_result.statistics["success_percent"] > 90

    @pytest.mark.data
    def test_data_drift_detection(self, sample_dataframe):
        """Test data drift detection between datasets."""
        # Create reference dataset
        reference_data = sample_dataframe.copy()

        # Create drifted dataset
        drifted_data = sample_dataframe.copy()
        drifted_data["age"] = drifted_data["age"] + 10  # Age drift
        drifted_data["balance"] = drifted_data["balance"] * 1.5  # Balance drift

        # Create expectations from reference
        context = gx.get_context()
        suite = ExpectationSuite(expectation_suite_name="drift_detection")

        # Add statistical expectations based on reference
        ref_age_mean = reference_data["age"].mean()
        ref_age_std = reference_data["age"].std()
        ref_balance_mean = reference_data["balance"].mean()

        suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_mean_to_be_between",
                kwargs={
                    "column": "age",
                    "min_value": ref_age_mean - ref_age_std,
                    "max_value": ref_age_mean + ref_age_std,
                },
            )
        )

        suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_mean_to_be_between",
                kwargs={
                    "column": "balance",
                    "min_value": ref_balance_mean * 0.9,
                    "max_value": ref_balance_mean * 1.1,
                },
            )
        )

        # Validate drifted data (should fail)
        batch_request = RuntimeBatchRequest(
            datasource_name="pandas_datasource",
            data_connector_name="default_runtime_data_connector_name",
            data_asset_name="drifted_data",
            batch_identifiers={"default_identifier_name": "default_identifier"},
            runtime_parameters={"batch_data": drifted_data},
        )

        validator = context.get_validator(
            batch_request=batch_request, expectation_suite=suite
        )

        validation_result = validator.validate()
        assert not validation_result.success  # Should detect drift

    @pytest.mark.data
    def test_custom_expectations(self, sample_dataframe):
        """Test custom business logic expectations."""
        from great_expectations.expectations.expectation import ColumnMapExpectation
        from great_expectations.expectations.metrics import ColumnMapMetricProvider

        # Custom expectation: Credit score should correlate with balance
        class ExpectCreditScoreBalanceCorrelation(ColumnMapExpectation):
            """Expect credit score to correlate with account balance."""

            map_metric = "column_values.credit_balance_correlation"

            @classmethod
            def _prescriptive_renderer(
                cls, configuration=None, result=None, runtime_configuration=None
            ):
                return "Credit score should positively correlate with balance"

        # Register custom expectation
        context = gx.get_context()
        suite = ExpectationSuite(expectation_suite_name="custom_expectations")

        # Add custom business rules
        # Rule 1: High balance customers should have low churn
        high_balance = sample_dataframe[
            sample_dataframe["balance"] > sample_dataframe["balance"].quantile(0.75)
        ]
        high_balance_churn_rate = high_balance["churn"].mean()
        assert high_balance_churn_rate < 0.3, (
            "High balance customers have unexpectedly high churn"
        )

        # Rule 2: Active members should have more products
        active_members = sample_dataframe[sample_dataframe["is_active"] == 1]
        inactive_members = sample_dataframe[sample_dataframe["is_active"] == 0]
        assert (
            active_members["num_products"].mean()
            > inactive_members["num_products"].mean()
        )

        # Rule 3: Credit score distribution should be approximately normal
        from scipy import stats

        _, p_value = stats.normaltest(sample_dataframe["credit_score"])
        assert p_value > 0.01, "Credit score distribution is not normal"


class TestPanderaValidation:
    """Test data validation with Pandera."""

    @pytest.mark.data
    def test_schema_validation(self, sample_dataframe):
        """Test dataframe schema validation."""
        # Define schema
        schema = DataFrameSchema(
            {
                "customer_id": Column(
                    int, Check.greater_than(0), unique=True, nullable=False
                ),
                "age": Column(
                    int,
                    [
                        Check.greater_than_or_equal_to(18),
                        Check.less_than_or_equal_to(100),
                    ],
                ),
                "tenure": Column(float, Check.greater_than_or_equal_to(0)),
                "balance": Column(float, Check.greater_than_or_equal_to(0)),
                "num_products": Column(int, Check.isin([1, 2, 3, 4])),
                "credit_score": Column(
                    float,
                    [
                        Check.greater_than_or_equal_to(300),
                        Check.less_than_or_equal_to(850),
                    ],
                ),
                "is_active": Column(int, Check.isin([0, 1])),
                "salary": Column(float, Check.greater_than(0)),
                "churn": Column(int, Check.isin([0, 1])),
                "geography": Column(str, Check.isin(["USA", "UK", "Germany"])),
                "gender": Column(str, Check.isin(["Male", "Female"])),
            }
        )

        # Validate
        validated_df = schema.validate(sample_dataframe)
        assert validated_df is not None

    @pytest.mark.data
    def test_statistical_checks(self, sample_dataframe):
        """Test statistical validation checks."""
        schema = DataFrameSchema(
            {
                "age": Column(
                    int,
                    [
                        Check(lambda s: s.mean() > 30, element_wise=False),
                        Check(lambda s: s.std() < 30, element_wise=False),
                    ],
                ),
                "churn": Column(
                    int,
                    [
                        Check(
                            lambda s: s.mean() < 0.5,
                            element_wise=False,
                            error="Churn rate too high",
                        )
                    ],
                ),
            }
        )

        validated_df = schema.validate(sample_dataframe[["age", "churn"]])
        assert validated_df is not None

    @pytest.mark.data
    def test_custom_checks(self, sample_dataframe):
        """Test custom validation checks."""

        def check_balance_income_ratio(df):
            """Check if balance to income ratio is reasonable."""
            ratio = df["balance"] / df["salary"].clip(lower=1)
            return (ratio < 10).all()

        def check_age_tenure_consistency(df):
            """Check if tenure is consistent with age."""
            return (df["tenure"] / 12 < df["age"] - 18).all()

        schema = DataFrameSchema(
            checks=[
                Check(
                    check_balance_income_ratio,
                    element_wise=False,
                    error="Balance to income ratio too high",
                ),
                Check(
                    check_age_tenure_consistency,
                    element_wise=False,
                    error="Tenure inconsistent with age",
                ),
            ]
        )

        try:
            schema.validate(sample_dataframe)
        except pa.errors.SchemaError as e:
            # Handle validation errors
            print(f"Validation error: {e}")


class TestSyntheticDataGeneration:
    """Test synthetic data generation for testing."""

    @pytest.mark.data
    def test_faker_data_generation(self):
        """Test synthetic data generation with Faker."""
        fake = Faker()
        Faker.seed(42)

        # Generate synthetic customer data
        n_samples = 1000
        synthetic_data = pd.DataFrame(
            {
                "customer_id": [fake.uuid4() for _ in range(n_samples)],
                "name": [fake.name() for _ in range(n_samples)],
                "email": [fake.email() for _ in range(n_samples)],
                "phone": [fake.phone_number() for _ in range(n_samples)],
                "address": [fake.address() for _ in range(n_samples)],
                "birth_date": [
                    fake.date_of_birth(minimum_age=18, maximum_age=80)
                    for _ in range(n_samples)
                ],
                "account_created": [
                    fake.date_between(start_date="-5y", end_date="today")
                    for _ in range(n_samples)
                ],
                "credit_card": [fake.credit_card_number() for _ in range(n_samples)],
                "company": [fake.company() for _ in range(n_samples)],
                "job": [fake.job() for _ in range(n_samples)],
            }
        )

        # Validate generated data
        assert len(synthetic_data) == n_samples
        assert synthetic_data["email"].str.contains("@").all()
        assert synthetic_data["customer_id"].nunique() == n_samples

    @pytest.mark.data
    def test_statistical_synthetic_data(self, sample_dataframe):
        """Test synthetic data that preserves statistical properties."""
        from scipy import stats

        # Calculate statistics from original data
        original_stats = {
            "age_mean": sample_dataframe["age"].mean(),
            "age_std": sample_dataframe["age"].std(),
            "balance_mean": sample_dataframe["balance"].mean(),
            "balance_std": sample_dataframe["balance"].std(),
            "churn_rate": sample_dataframe["churn"].mean(),
        }

        # Generate synthetic data with similar properties
        n_samples = 1000
        synthetic_data = pd.DataFrame(
            {
                "age": np.random.normal(
                    original_stats["age_mean"], original_stats["age_std"], n_samples
                )
                .astype(int)
                .clip(18, 80),
                "balance": np.random.lognormal(
                    np.log(original_stats["balance_mean"]),
                    np.log(original_stats["balance_std"])
                    if original_stats["balance_std"] > 1
                    else 0.5,
                    n_samples,
                ).clip(0, None),
                "churn": np.random.choice(
                    [0, 1],
                    n_samples,
                    p=[1 - original_stats["churn_rate"], original_stats["churn_rate"]],
                ),
            }
        )

        # Verify statistical properties are preserved
        assert abs(synthetic_data["age"].mean() - original_stats["age_mean"]) < 5
        assert abs(synthetic_data["churn"].mean() - original_stats["churn_rate"]) < 0.05

        # Test distribution similarity
        _, p_value = stats.ks_2samp(sample_dataframe["age"], synthetic_data["age"])
        assert p_value > 0.01  # Distributions are similar

    @pytest.mark.data
    @given(
        n_rows=st.integers(min_value=10, max_value=100),
        n_numeric_cols=st.integers(min_value=1, max_value=10),
        n_categorical_cols=st.integers(min_value=1, max_value=5),
    )
    def test_property_based_synthetic_data(
        self, n_rows, n_numeric_cols, n_categorical_cols
    ):
        """Property-based test for synthetic data generation."""
        # Generate synthetic dataframe
        numeric_data = {
            f"numeric_{i}": np.random.randn(n_rows) for i in range(n_numeric_cols)
        }

        categorical_data = {
            f"categorical_{i}": np.random.choice(["A", "B", "C"], n_rows)
            for i in range(n_categorical_cols)
        }

        synthetic_df = pd.DataFrame({**numeric_data, **categorical_data})

        # Properties that should hold
        assert len(synthetic_df) == n_rows
        assert len(synthetic_df.columns) == n_numeric_cols + n_categorical_cols
        assert (
            synthetic_df.select_dtypes(include=[np.number]).shape[1] == n_numeric_cols
        )
        assert (
            synthetic_df.select_dtypes(include=["object"]).shape[1]
            == n_categorical_cols
        )
        assert not synthetic_df.isnull().all().any()


class TestDataPrivacy:
    """Test data privacy and compliance."""

    @pytest.mark.data
    def test_pii_detection(self, sample_dataframe):
        """Test PII detection in data."""
        # Add PII columns for testing
        fake = Faker()
        sample_dataframe["ssn"] = [fake.ssn() for _ in range(len(sample_dataframe))]
        sample_dataframe["email"] = [fake.email() for _ in range(len(sample_dataframe))]
        sample_dataframe["phone"] = [
            fake.phone_number() for _ in range(len(sample_dataframe))
        ]

        # PII detection patterns
        pii_patterns = {
            "ssn": r"^\d{3}-\d{2}-\d{4}$",
            "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
            "phone": r"^\+?1?\d{9,15}$",
            "credit_card": r"^\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}$",
        }

        detected_pii = []
        for column in sample_dataframe.columns:
            # Check column name
            if any(
                pii_type in column.lower()
                for pii_type in ["ssn", "email", "phone", "credit"]
            ):
                detected_pii.append(column)
                continue

            # Check data patterns
            if sample_dataframe[column].dtype == "object":
                sample = sample_dataframe[column].dropna().astype(str).head(10)
                for pii_type, pattern in pii_patterns.items():
                    if sample.str.match(pattern).any():
                        detected_pii.append(column)
                        break

        assert len(detected_pii) >= 3  # Should detect ssn, email, phone

    @pytest.mark.data
    def test_data_anonymization(self, sample_dataframe):
        """Test data anonymization techniques."""
        # Add PII
        fake = Faker()
        sample_dataframe["name"] = [fake.name() for _ in range(len(sample_dataframe))]
        sample_dataframe["email"] = [fake.email() for _ in range(len(sample_dataframe))]

        # Anonymization function
        def anonymize_data(df):
            """Anonymize PII in dataframe."""
            anonymized = df.copy()

            # Hash PII columns
            pii_columns = ["name", "email"]
            for col in pii_columns:
                if col in anonymized.columns:
                    anonymized[col] = anonymized[col].apply(
                        lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:8]
                    )

            # Generalize age to ranges
            if "age" in anonymized.columns:
                anonymized["age_range"] = pd.cut(
                    anonymized["age"],
                    bins=[0, 25, 35, 45, 55, 65, 100],
                    labels=["<25", "25-34", "35-44", "45-54", "55-64", "65+"],
                )
                anonymized = anonymized.drop("age", axis=1)

            return anonymized

        anonymized_data = anonymize_data(sample_dataframe)

        # Verify anonymization
        assert (
            "name" not in anonymized_data.columns
            or not anonymized_data["name"].str.contains("@").any()
        )
        assert "age_range" in anonymized_data.columns
        assert "age" not in anonymized_data.columns

    @pytest.mark.data
    def test_differential_privacy(self, sample_dataframe):
        """Test differential privacy implementation."""
        epsilon = 1.0  # Privacy parameter

        def add_laplace_noise(value, sensitivity, epsilon):
            """Add Laplace noise for differential privacy."""
            scale = sensitivity / epsilon
            noise = np.random.laplace(0, scale)
            return value + noise

        # Calculate private statistics
        true_mean = sample_dataframe["balance"].mean()
        sensitivity = (
            sample_dataframe["balance"].max() - sample_dataframe["balance"].min()
        )

        private_mean = add_laplace_noise(true_mean, sensitivity, epsilon)

        # Private mean should be close but not exact
        assert abs(private_mean - true_mean) > 0
        assert abs(private_mean - true_mean) < sensitivity


class TestDataCorruption:
    """Test handling of corrupted and malformed data."""

    @pytest.mark.data
    def test_missing_data_patterns(self):
        """Test different missing data patterns."""
        n_samples = 1000

        # MCAR - Missing Completely At Random
        mcar_data = pd.DataFrame({"value": np.random.randn(n_samples)})
        mcar_mask = np.random.random(n_samples) < 0.1
        mcar_data.loc[mcar_mask, "value"] = np.nan

        # MAR - Missing At Random (depends on other variables)
        mar_data = pd.DataFrame(
            {
                "age": np.random.randint(18, 80, n_samples),
                "income": np.random.lognormal(10, 1, n_samples),
            }
        )
        # Income missing more often for younger people
        mar_mask = (mar_data["age"] < 30) & (np.random.random(n_samples) < 0.3)
        mar_data.loc[mar_mask, "income"] = np.nan

        # MNAR - Missing Not At Random (depends on missing value itself)
        mnar_data = pd.DataFrame({"salary": np.random.lognormal(11, 0.5, n_samples)})
        # High salaries more likely to be missing
        mnar_mask = (mnar_data["salary"] > mnar_data["salary"].quantile(0.8)) & (
            np.random.random(n_samples) < 0.5
        )
        mnar_data.loc[mnar_mask, "salary"] = np.nan

        # Verify patterns
        assert mcar_data["value"].isna().sum() > 0
        assert mar_data["income"].isna().sum() > 0
        assert mnar_data["salary"].isna().sum() > 0

    @pytest.mark.data
    def test_data_type_inconsistencies(self):
        """Test handling of data type inconsistencies."""
        # Mixed type column
        mixed_data = pd.DataFrame(
            {"mixed": [1, "2", 3.0, "4.5", None, "text", True, False]}
        )

        # Attempt to convert to numeric
        numeric = pd.to_numeric(mixed_data["mixed"], errors="coerce")
        assert numeric.notna().sum() == 5  # 1, 2, 3.0, 4.5, and boolean True

        # Detect mixed types
        def detect_mixed_types(series):
            types = series.dropna().apply(type).unique()
            return len(types) > 1

        assert detect_mixed_types(mixed_data["mixed"])

    @pytest.mark.data
    def test_outlier_detection(self, sample_dataframe):
        """Test outlier detection methods."""
        # Add artificial outliers
        data = sample_dataframe.copy()
        data.loc[0, "balance"] = 1e8  # Extreme outlier

        # Z-score method
        z_scores = np.abs(
            (data["balance"] - data["balance"].mean()) / data["balance"].std()
        )
        outliers_z = data[z_scores > 3]
        assert len(outliers_z) > 0

        # IQR method
        Q1 = data["balance"].quantile(0.25)
        Q3 = data["balance"].quantile(0.75)
        IQR = Q3 - Q1
        outliers_iqr = data[
            (data["balance"] < Q1 - 1.5 * IQR) | (data["balance"] > Q3 + 1.5 * IQR)
        ]
        assert len(outliers_iqr) > 0

        # Isolation Forest
        from sklearn.ensemble import IsolationForest

        iso = IsolationForest(contamination=0.1, random_state=42)
        outliers_iso = iso.fit_predict(data[["balance", "age"]].fillna(0))
        assert (outliers_iso == -1).sum() > 0

    @pytest.mark.data
    def test_duplicate_detection(self):
        """Test duplicate record detection."""
        # Create data with duplicates
        data = pd.DataFrame(
            {
                "id": [1, 2, 2, 3, 4, 5, 5],
                "name": ["Alice", "Bob", "Bob", "Charlie", "David", "Eve", "Eva"],
                "email": [
                    "a@test.com",
                    "b@test.com",
                    "b@test.com",
                    "c@test.com",
                    "d@test.com",
                    "e@test.com",
                    "e@test.com",
                ],
            }
        )

        # Exact duplicates
        exact_duplicates = data.duplicated()
        assert exact_duplicates.sum() == 2

        # Duplicates by key
        id_duplicates = data.duplicated(subset=["id"])
        assert id_duplicates.sum() == 2

        # Fuzzy duplicates (similar but not exact)
        from difflib import SequenceMatcher

        def similar(a, b):
            return SequenceMatcher(None, a, b).ratio()

        # Find similar names
        similar_names = []
        for i in range(len(data)):
            for j in range(i + 1, len(data)):
                if similar(data.iloc[i]["name"], data.iloc[j]["name"]) > 0.8:
                    similar_names.append((i, j))

        assert len(similar_names) > 0  # Eve and Eva are similar


class TestDataMonitoring:
    """Test data monitoring and alerting."""

    @pytest.mark.data
    def test_data_drift_monitoring(self, sample_dataframe):
        """Test continuous data drift monitoring."""
        from scipy import stats

        # Simulate data over time
        baseline = sample_dataframe.copy()
        drift_detected = []

        for week in range(10):
            # Simulate gradual drift
            current_data = sample_dataframe.copy()
            current_data["age"] = current_data["age"] + week * 0.5
            current_data["balance"] = current_data["balance"] * (1 + week * 0.02)

            # KS test for distribution drift
            ks_age = stats.ks_2samp(baseline["age"], current_data["age"])
            ks_balance = stats.ks_2samp(baseline["balance"], current_data["balance"])

            if ks_age.pvalue < 0.05 or ks_balance.pvalue < 0.05:
                drift_detected.append(week)

        assert len(drift_detected) > 0  # Should detect drift in later weeks

    @pytest.mark.data
    def test_data_quality_metrics(self, corrupted_dataframe):
        """Test data quality metric calculation."""

        def calculate_quality_score(df):
            """Calculate overall data quality score."""
            metrics = {
                "completeness": 1
                - df.isnull().sum().sum() / (len(df) * len(df.columns)),
                "uniqueness": df.nunique().mean() / len(df),
                "validity": 0.9,  # Placeholder for validation rules
                "consistency": 0.95,  # Placeholder for consistency checks
                "timeliness": 1.0,  # Placeholder for freshness checks
            }

            # Weighted average
            weights = {
                "completeness": 0.3,
                "uniqueness": 0.2,
                "validity": 0.2,
                "consistency": 0.2,
                "timeliness": 0.1,
            }

            quality_score = sum(metrics[k] * weights[k] for k in metrics)
            return quality_score, metrics

        score, metrics = calculate_quality_score(corrupted_dataframe)

        assert 0 <= score <= 1
        assert metrics["completeness"] < 1  # Has missing values
        assert all(0 <= v <= 1 for v in metrics.values())
