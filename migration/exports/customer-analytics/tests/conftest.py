from __future__ import annotations

import os
import tempfile
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import settings


settings.register_profile("dev", max_examples=10, deadline=None)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))


@pytest.fixture(scope="function")
def temp_dir() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="function")
def sample_dataframe() -> pd.DataFrame:
    np.random.seed(42)
    n_samples = 1000
    return pd.DataFrame(
        {
            "customer_id": range(1, n_samples + 1),
            "age": np.random.randint(18, 80, n_samples),
            "tenure": np.random.exponential(24, n_samples),
            "balance": np.random.lognormal(10, 1.5, n_samples),
            "num_products": np.random.choice([1, 2, 3, 4], n_samples, p=[0.4, 0.35, 0.2, 0.05]),
            "credit_score": np.random.normal(650, 100, n_samples),
            "is_active": np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
            "salary": np.random.lognormal(11, 0.5, n_samples),
            "churn": np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            "geography": np.random.choice(["USA", "UK", "Germany"], n_samples, p=[0.5, 0.3, 0.2]),
            "gender": np.random.choice(["Male", "Female"], n_samples, p=[0.55, 0.45]),
        }
    )


@pytest.fixture(scope="function")
def small_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50],
            "c": ["x", "y", "x", "y", "x"],
            "target": [0, 1, 0, 1, 1],
        }
    )


@pytest.fixture(scope="function")
def corrupted_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 2, 4, 5],
            "value": [10, np.nan, 30, 40, 50],
            "text": ["abc", "123", None, "def", ""],
            "date": ["2024-01-01", "2024-13-01", "2024-01-32", None, ""],
            "amount": [100, -50, 1e10, 0, 0.001],
            "category": ["A", "B", "C", "D", "Z"],
        }
    )


@pytest.fixture(scope="function")
def trained_model(sample_dataframe):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    feature_cols = ["age", "tenure", "balance", "num_products", "credit_score"]
    X = sample_dataframe[feature_cols]
    y = sample_dataframe["churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test


@pytest.fixture(scope="function")
def pipeline_config():
    from modern_bank_churn.ml_pipeline_orchestrator import PipelineConfig

    return PipelineConfig(
        feature_selection_method="mutual_info",
        feature_selection_k=5,
        model_type="random_forest",
        hyperparameter_tuning=False,
        cross_validation_folds=3,
        random_state=42,
    )
