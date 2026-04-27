"""Global pytest fixtures and configuration for all tests.
"""

import os
import sys
import tempfile
import types
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import great_expectations as gx
import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pytest
from faker import Faker
from fakeredis import FakeRedis
from hypothesis import settings
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "projects"))


def _install_test_stub(name: str, module: types.ModuleType) -> None:
    if name not in sys.modules:
        sys.modules[name] = module


if "flask_httpauth" not in sys.modules:
    flask_httpauth_stub = types.ModuleType("flask_httpauth")

    class _AuthBase:
        def __init__(self, *args, **kwargs):
            pass

        def verify_password(self, func):
            return func

        def verify_token(self, func):
            return func

        def login_required(self, func):
            return func

    flask_httpauth_stub.HTTPBasicAuth = _AuthBase
    flask_httpauth_stub.HTTPTokenAuth = _AuthBase
    _install_test_stub("flask_httpauth", flask_httpauth_stub)

if "flask_restful" not in sys.modules:
    flask_restful_stub = types.ModuleType("flask_restful")

    class _Api:
        def __init__(self, *args, **kwargs):
            pass

        def add_resource(self, *args, **kwargs):
            return None

    class _Resource:
        pass

    flask_restful_stub.Api = _Api
    flask_restful_stub.Resource = _Resource
    _install_test_stub("flask_restful", flask_restful_stub)

if "flask_swagger_ui" not in sys.modules:
    flask_swagger_ui_stub = types.ModuleType("flask_swagger_ui")
    flask_swagger_ui_stub.get_swaggerui_blueprint = lambda *args, **kwargs: None
    _install_test_stub("flask_swagger_ui", flask_swagger_ui_stub)

# Configure Hypothesis
settings.register_profile("dev", max_examples=10, deadline=None)
settings.register_profile("ci", max_examples=100, deadline=5000)
settings.register_profile("production", max_examples=500, deadline=10000)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))

# Initialize Faker
fake = Faker()
Faker.seed(42)

# ============================================================================
# Session-level fixtures
# ============================================================================


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Return path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ============================================================================
# Database fixtures
# ============================================================================


@pytest.fixture(scope="function")
def db_engine():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    yield engine
    engine.dispose()


@pytest.fixture(scope="function")
def db_session(db_engine):
    """Create a database session for testing."""
    Session = sessionmaker(bind=db_engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture(scope="function")
def redis_client():
    """Create a fake Redis client for testing."""
    client = FakeRedis()
    yield client
    client.flushall()


# ============================================================================
# Data fixtures
# ============================================================================


@pytest.fixture(scope="function")
def sample_dataframe() -> pd.DataFrame:
    """Generate a sample dataframe for testing."""
    np.random.seed(42)
    n_samples = 1000

    return pd.DataFrame(
        {
            "customer_id": range(1, n_samples + 1),
            "age": np.random.randint(18, 80, n_samples),
            "tenure": np.random.exponential(24, n_samples),
            "balance": np.random.lognormal(10, 1.5, n_samples),
            "num_products": np.random.choice(
                [1, 2, 3, 4], n_samples, p=[0.4, 0.35, 0.2, 0.05]
            ),
            "credit_score": np.random.normal(650, 100, n_samples),
            "is_active": np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
            "salary": np.random.lognormal(11, 0.5, n_samples),
            "churn": np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            "geography": np.random.choice(
                ["USA", "UK", "Germany"], n_samples, p=[0.5, 0.3, 0.2]
            ),
            "gender": np.random.choice(["Male", "Female"], n_samples, p=[0.55, 0.45]),
        }
    )


@pytest.fixture(scope="function")
def small_dataframe() -> pd.DataFrame:
    """Generate a small dataframe for quick tests."""
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
    """Generate a dataframe with various data quality issues."""
    return pd.DataFrame(
        {
            "id": [1, 2, 2, 4, 5],  # Duplicate
            "value": [10, np.nan, 30, 40, 50],  # Missing value
            "text": ["abc", "123", None, "def", ""],  # Mixed types and nulls
            "date": [
                "2024-01-01",
                "2024-13-01",
                "2024-01-32",
                None,
                "",
            ],  # Invalid dates
            "amount": [100, -50, 1e10, 0, 0.001],  # Outliers and edge cases
            "category": ["A", "B", "C", "D", "Z"],  # Unknown category
        }
    )


@pytest.fixture(scope="function")
def time_series_data() -> pd.DataFrame:
    """Generate time series data for testing."""
    dates = pd.date_range("2022-01-01", periods=365, freq="D")
    values = np.cumsum(np.random.randn(365)) + 100

    return pd.DataFrame(
        {
            "date": dates,
            "value": values,
            "seasonality": np.sin(np.arange(365) * 2 * np.pi / 365) * 10,
            "trend": np.arange(365) * 0.1,
        }
    )


# ============================================================================
# Model fixtures
# ============================================================================


@pytest.fixture(scope="function")
def mock_model():
    """Create a mock machine learning model."""
    model = MagicMock()
    model.predict = MagicMock(return_value=np.array([0, 1, 0, 1]))
    model.predict_proba = MagicMock(
        return_value=np.array([[0.3, 0.7], [0.1, 0.9], [0.8, 0.2], [0.4, 0.6]])
    )
    model.feature_importances_ = np.array([0.3, 0.5, 0.1, 0.1])
    return model


@pytest.fixture(scope="function")
def trained_model(sample_dataframe):
    """Train a simple model for testing."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    feature_cols = ["age", "tenure", "balance", "num_products", "credit_score"]
    X = sample_dataframe[feature_cols]
    y = sample_dataframe["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    return model, X_test, y_test


# ============================================================================
# Configuration fixtures
# ============================================================================


@pytest.fixture(scope="function")
def pipeline_config():
    """Create a pipeline configuration for testing."""
    from modern_bank_churn.ml_pipeline_orchestrator import PipelineConfig

    return PipelineConfig(
        feature_selection_method="mutual_info",
        feature_selection_k=5,
        model_type="random_forest",
        hyperparameter_tuning=False,
        cross_validation_folds=3,
        random_state=42,
    )


@pytest.fixture(scope="function")
def dashboard_config():
    """Create a dashboard configuration for testing."""
    from dashboard_enhanced.dashboard_framework import DashboardConfig

    return DashboardConfig(
        app_name="Test Dashboard",
        port=8050,
        debug=True,
        enable_realtime=False,
        enable_export=True,
        cache_timeout=60,
    )


# ============================================================================
# Great Expectations fixtures
# ============================================================================


@pytest.fixture(scope="function")
def data_context(temp_dir):
    """Create a Great Expectations data context for testing."""
    context = gx.get_context()
    return context


@pytest.fixture(scope="function")
def expectation_suite():
    """Create a basic expectation suite for testing."""
    suite = gx.core.ExpectationSuite(expectation_suite_name="test_suite")

    # Add expectations
    suite.add_expectation(
        gx.core.ExpectationConfiguration(
            expectation_type="expect_column_to_exist", kwargs={"column": "customer_id"}
        )
    )

    suite.add_expectation(
        gx.core.ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "customer_id"},
        )
    )

    return suite


# ============================================================================
# Hypothesis strategies
# ============================================================================


@st.composite
def dataframe_strategy(draw, min_rows=1, max_rows=100):
    """Generate random dataframes using Hypothesis."""
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    n_cols = draw(st.integers(min_value=1, max_value=10))

    columns = [f"col_{i}" for i in range(n_cols)]
    data = {}

    for col in columns:
        dtype = draw(st.sampled_from(["int", "float", "str", "bool"]))

        if dtype == "int":
            data[col] = draw(st.lists(st.integers(), min_size=n_rows, max_size=n_rows))
        elif dtype == "float":
            data[col] = draw(
                st.lists(
                    st.floats(allow_nan=False, allow_infinity=False),
                    min_size=n_rows,
                    max_size=n_rows,
                )
            )
        elif dtype == "str":
            data[col] = draw(
                st.lists(
                    st.text(min_size=1, max_size=10), min_size=n_rows, max_size=n_rows
                )
            )
        else:  # bool
            data[col] = draw(st.lists(st.booleans(), min_size=n_rows, max_size=n_rows))

    return pd.DataFrame(data)


@st.composite
def model_params_strategy(draw):
    """Generate random model parameters for testing."""
    return {
        "n_estimators": draw(st.integers(min_value=10, max_value=200)),
        "max_depth": draw(st.integers(min_value=1, max_value=20)),
        "min_samples_split": draw(st.integers(min_value=2, max_value=10)),
        "min_samples_leaf": draw(st.integers(min_value=1, max_value=5)),
        "random_state": 42,
    }


# ============================================================================
# Performance monitoring fixtures
# ============================================================================


@pytest.fixture
def benchmark_data():
    """Generate data for benchmarking tests."""
    sizes = [100, 1000, 10000, 100000]
    datasets = {}

    for size in sizes:
        datasets[size] = pd.DataFrame(
            {
                "x": np.random.randn(size),
                "y": np.random.randn(size),
                "target": np.random.choice([0, 1], size),
            }
        )

    return datasets


# ============================================================================
# Notebook testing fixtures
# ============================================================================


@pytest.fixture(scope="function")
def notebook_runner():
    """Create a notebook runner for testing."""
    import papermill as pm

    def run_notebook(notebook_path, output_path, parameters=None):
        """Run a notebook with parameters."""
        pm.execute_notebook(
            notebook_path,
            output_path,
            parameters=parameters or {},
            kernel_name="python3",
        )
        return output_path

    return run_notebook


# ============================================================================
# Mock API responses
# ============================================================================


@pytest.fixture
def mock_api_response():
    """Mock API response for testing."""
    return {
        "status": "success",
        "data": {
            "prediction": 0.75,
            "confidence": 0.92,
            "features": {"age": 35, "balance": 50000, "tenure": 24},
        },
        "metadata": {"model_version": "1.2.0", "timestamp": "2024-01-01T12:00:00Z"},
    }


# ============================================================================
# Cleanup fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def cleanup(request):
    """Cleanup after each test."""
    yield
    # Clean up any temporary files
    import glob

    for pattern in ["*.tmp", "*.log", "*.pkl"]:
        for file in glob.glob(pattern):
            try:
                os.remove(file)
            except:
                pass


# ============================================================================
# Markers and configuration
# ============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "data: mark test as data validation test")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "data" in str(item.fspath):
            item.add_marker(pytest.mark.data)
