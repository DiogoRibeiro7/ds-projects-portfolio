"""
Performance benchmark tests for critical components.

This module contains performance benchmarks for data processing,
statistical computations, and ML operations.
"""

import numpy as np
import pandas as pd
import pytest
from typing import List, Dict, Any


# Data Processing Benchmarks
class TestDataProcessingBenchmarks:
    """Benchmark tests for data processing operations."""

    def setup_method(self):
        """Set up test data for benchmarks."""
        np.random.seed(42)
        self.small_df = pd.DataFrame(
            np.random.randn(1000, 10), columns=[f"col_{i}" for i in range(10)]
        )
        self.medium_df = pd.DataFrame(
            np.random.randn(10000, 50), columns=[f"col_{i}" for i in range(50)]
        )
        self.large_df = pd.DataFrame(
            np.random.randn(100000, 100), columns=[f"col_{i}" for i in range(100)]
        )

    @pytest.mark.benchmark(group="data-cleaning")
    def test_remove_duplicates_small(self, benchmark):
        """Benchmark duplicate removal on small dataset."""
        from src.data_processing.cleaning import DataCleaner

        cleaner = DataCleaner()

        # Add some duplicates
        df_with_dups = pd.concat([self.small_df, self.small_df.iloc[:100]])

        result = benchmark(cleaner.remove_duplicates, df_with_dups)
        assert len(result) == len(self.small_df)

    @pytest.mark.benchmark(group="data-cleaning")
    def test_remove_duplicates_large(self, benchmark):
        """Benchmark duplicate removal on large dataset."""
        from src.data_processing.cleaning import DataCleaner

        cleaner = DataCleaner()

        # Add some duplicates
        df_with_dups = pd.concat([self.large_df, self.large_df.iloc[:1000]])

        result = benchmark(cleaner.remove_duplicates, df_with_dups)
        assert len(result) == len(self.large_df)

    @pytest.mark.benchmark(group="data-transformation")
    def test_feature_scaling(self, benchmark):
        """Benchmark feature scaling operations."""
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.fit(self.medium_df)

        result = benchmark(scaler.transform, self.medium_df)
        assert result.shape == self.medium_df.shape

    @pytest.mark.benchmark(group="data-aggregation")
    def test_groupby_aggregation(self, benchmark):
        """Benchmark groupby aggregation operations."""
        # Add a categorical column for grouping
        self.large_df["group"] = np.random.choice(
            ["A", "B", "C", "D"], len(self.large_df)
        )

        def aggregate_data(df):
            return df.groupby("group").agg(
                {
                    "col_0": ["mean", "std", "min", "max"],
                    "col_1": ["sum", "count"],
                    "col_2": ["median", "var"],
                }
            )

        result = benchmark(aggregate_data, self.large_df)
        assert len(result) == 4  # 4 groups


# Statistical Method Benchmarks
class TestStatisticalBenchmarks:
    """Benchmark tests for statistical computations."""

    def setup_method(self):
        """Set up test data for statistical benchmarks."""
        np.random.seed(42)
        self.small_sample = np.random.randn(100)
        self.medium_sample = np.random.randn(10000)
        self.large_sample = np.random.randn(1000000)

    @pytest.mark.benchmark(group="hypothesis-testing")
    def test_two_sample_ttest(self, benchmark):
        """Benchmark two-sample t-test."""
        from scipy import stats

        sample1 = np.random.randn(10000)
        sample2 = np.random.randn(10000) + 0.1

        result = benchmark(stats.ttest_ind, sample1, sample2)
        assert result.pvalue < 1.0

    @pytest.mark.benchmark(group="hypothesis-testing")
    def test_bootstrap_confidence_interval(self, benchmark):
        """Benchmark bootstrap confidence interval calculation."""
        from src.statistics.core import bootstrap_ci_diff

        control = np.random.randn(5000)
        treatment = np.random.randn(5000) + 0.2

        result = benchmark(
            bootstrap_ci_diff,
            control,
            treatment,
            n_bootstrap=1000,
            confidence_level=0.95,
        )
        assert len(result) == 2

    @pytest.mark.benchmark(group="correlation")
    def test_correlation_matrix(self, benchmark):
        """Benchmark correlation matrix computation."""
        data = np.random.randn(10000, 50)
        df = pd.DataFrame(data)

        result = benchmark(df.corr)
        assert result.shape == (50, 50)

    @pytest.mark.benchmark(group="power-analysis")
    def test_sample_size_calculation(self, benchmark):
        """Benchmark sample size calculation."""
        from src.statistics.core import calculate_sample_size

        result = benchmark(
            calculate_sample_size, alpha=0.05, power=0.8, effect_size=0.2
        )
        assert result > 0


# Machine Learning Benchmarks
class TestMLBenchmarks:
    """Benchmark tests for ML operations."""

    def setup_method(self):
        """Set up test data for ML benchmarks."""
        np.random.seed(42)
        self.X_small = np.random.randn(1000, 20)
        self.y_small = np.random.randint(0, 2, 1000)

        self.X_medium = np.random.randn(10000, 50)
        self.y_medium = np.random.randint(0, 2, 10000)

        self.X_large = np.random.randn(50000, 100)
        self.y_large = np.random.randint(0, 2, 50000)

    @pytest.mark.benchmark(group="model-training")
    def test_random_forest_training(self, benchmark):
        """Benchmark Random Forest training."""
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=100, random_state=42)

        benchmark(model.fit, self.X_medium, self.y_medium)
        assert model.score(self.X_medium, self.y_medium) > 0.5

    @pytest.mark.benchmark(group="model-training")
    def test_xgboost_training(self, benchmark):
        """Benchmark XGBoost training."""
        pytest.importorskip("xgboost")
        import xgboost as xgb

        model = xgb.XGBClassifier(n_estimators=100, random_state=42)

        benchmark(model.fit, self.X_medium, self.y_medium)
        assert model.score(self.X_medium, self.y_medium) > 0.5

    @pytest.mark.benchmark(group="model-inference")
    def test_batch_prediction(self, benchmark):
        """Benchmark batch prediction."""
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(self.X_small, self.y_small)

        result = benchmark(model.predict, self.X_large)
        assert len(result) == len(self.X_large)

    @pytest.mark.benchmark(group="feature-importance")
    def test_shap_values(self, benchmark):
        """Benchmark SHAP value calculation."""
        pytest.importorskip("shap")
        import shap
        from sklearn.ensemble import RandomForestClassifier

        # Train a small model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(self.X_small[:500], self.y_small[:500])

        # Create explainer
        explainer = shap.TreeExplainer(model)

        # Benchmark SHAP value calculation
        sample_data = self.X_small[:100]
        result = benchmark(explainer.shap_values, sample_data)
        assert result is not None


# Dashboard and API Benchmarks
class TestAPIBenchmarks:
    """Benchmark tests for API and dashboard operations."""

    @pytest.mark.benchmark(group="api-serialization")
    def test_json_serialization(self, benchmark):
        """Benchmark JSON serialization of large responses."""
        import json

        # Create a large response object
        data = {
            "results": [
                {
                    "id": i,
                    "features": np.random.randn(100).tolist(),
                    "metadata": {
                        "timestamp": "2024-01-01",
                        "version": "1.0.0",
                        "model": "test_model",
                    },
                }
                for i in range(1000)
            ]
        }

        result = benchmark(json.dumps, data)
        assert len(result) > 0

    @pytest.mark.benchmark(group="api-validation")
    def test_request_validation(self, benchmark):
        """Benchmark request validation with pydantic."""
        from pydantic import BaseModel, Field
        from typing import List

        class PredictionRequest(BaseModel):
            features: List[float] = Field(..., min_items=1, max_items=1000)
            model_id: str = Field(..., min_length=1, max_length=100)
            options: Dict[str, Any] = Field(default_factory=dict)

        # Create test data
        test_requests = [
            {
                "features": np.random.randn(100).tolist(),
                "model_id": f"model_{i}",
                "options": {"threshold": 0.5},
            }
            for i in range(100)
        ]

        def validate_batch(requests):
            return [PredictionRequest(**req) for req in requests]

        result = benchmark(validate_batch, test_requests)
        assert len(result) == 100


# Memory and Resource Benchmarks
class TestResourceBenchmarks:
    """Benchmark tests for memory and resource usage."""

    @pytest.mark.benchmark(group="memory-usage")
    def test_dataframe_memory_optimization(self, benchmark):
        """Benchmark DataFrame memory optimization."""
        # Create a large DataFrame with different dtypes
        df = pd.DataFrame(
            {
                "int_col": np.random.randint(0, 100, 100000),
                "float_col": np.random.randn(100000),
                "string_col": ["category_" + str(i % 100) for i in range(100000)],
                "bool_col": np.random.choice([True, False], 100000),
            }
        )

        def optimize_memory(df):
            """Optimize DataFrame memory usage."""
            optimized = df.copy()

            # Convert strings to categories if beneficial
            for col in optimized.select_dtypes(include=["object"]):
                if optimized[col].nunique() / len(optimized) < 0.5:
                    optimized[col] = optimized[col].astype("category")

            # Downcast numeric types
            for col in optimized.select_dtypes(include=["int"]):
                optimized[col] = pd.to_numeric(optimized[col], downcast="integer")

            for col in optimized.select_dtypes(include=["float"]):
                optimized[col] = pd.to_numeric(optimized[col], downcast="float")

            return optimized

        result = benchmark(optimize_memory, df)
        assert result.memory_usage(deep=True).sum() < df.memory_usage(deep=True).sum()

    @pytest.mark.benchmark(group="caching")
    def test_cache_performance(self, benchmark):
        """Benchmark caching performance."""
        from functools import lru_cache
        import time

        # Simulate expensive computation
        def expensive_function(n):
            time.sleep(0.001)  # Simulate work
            return sum(range(n))

        @lru_cache(maxsize=128)
        def cached_function(n):
            return expensive_function(n)

        # Warm up cache
        for i in range(50):
            cached_function(i)

        # Benchmark cache hits
        def test_cache_hits():
            results = []
            for _ in range(10):
                for i in range(50):
                    results.append(cached_function(i))
            return results

        result = benchmark(test_cache_hits)
        assert len(result) == 500


# Parallel Processing Benchmarks
class TestParallelBenchmarks:
    """Benchmark tests for parallel processing."""

    @pytest.mark.benchmark(group="parallel-processing")
    def test_multiprocessing_performance(self, benchmark):
        """Benchmark multiprocessing vs sequential processing."""
        from multiprocessing import Pool
        import math

        def compute_intensive_task(n):
            """Simulate CPU-intensive task."""
            result = 0
            for i in range(n):
                result += math.sqrt(i) * math.log(i + 1)
            return result

        # Test data
        tasks = [10000 for _ in range(100)]

        def parallel_processing(tasks):
            with Pool(processes=4) as pool:
                return pool.map(compute_intensive_task, tasks)

        result = benchmark(parallel_processing, tasks)
        assert len(result) == 100

    @pytest.mark.benchmark(group="async-io")
    def test_async_io_performance(self, benchmark):
        """Benchmark async I/O operations."""
        import asyncio

        async def async_task(n):
            """Simulate async I/O task."""
            await asyncio.sleep(0.001)
            return n * 2

        async def process_batch(items):
            """Process items in parallel using async."""
            tasks = [async_task(item) for item in items]
            return await asyncio.gather(*tasks)

        # Test data
        items = list(range(100))

        def run_async_benchmark(items):
            return asyncio.run(process_batch(items))

        result = benchmark(run_async_benchmark, items)
        assert len(result) == 100


if __name__ == "__main__":
    pytest.main([__file__, "--benchmark-only", "-v"])
