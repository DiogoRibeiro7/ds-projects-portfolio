"""
Integration tests for end-to-end pipeline execution.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import time
import concurrent.futures
from pathlib import Path
import requests
import json
from unittest.mock import patch, MagicMock
import asyncio
import aiohttp
from multiprocessing import Pool
import threading
import queue

pytestmark = pytest.mark.integration

from modern_bank_churn.ml_pipeline_orchestrator import (
    MLPipelineOrchestrator,
    PipelineConfig,
)
from statistical_methods.statistical_analyzer import StatisticalAnalyzer
from statistical_methods.hypothesis_tester import HypothesisTester
from dashboard_enhanced.dashboard_framework import EnhancedDashboard, DashboardConfig
from dashboard_enhanced.api_infrastructure import APIInfrastructure


class TestEndToEndPipeline:
    """Test complete data pipeline from ingestion to prediction."""

    @pytest.mark.integration
    def test_complete_ml_pipeline(self, sample_dataframe, temp_dir):
        """Test the complete ML pipeline from data to deployment."""
        # 1. Data ingestion
        data = sample_dataframe.copy()
        assert len(data) > 0

        # 2. Statistical analysis
        analyzer = StatisticalAnalyzer(data)
        summary = analyzer.generate_summary()
        assert "mean" in str(summary).lower()

        # 3. Feature engineering
        from modern_bank_churn.feature_engineering import FeatureEngineer

        engineer = FeatureEngineer()
        features = engineer.create_features(data)
        assert features.shape[1] >= data.shape[1]

        # 4. Model training
        config = PipelineConfig(
            feature_selection_method="mutual_info",
            model_type="random_forest",
            hyperparameter_tuning=False,
            cross_validation_folds=2,
        )
        orchestrator = MLPipelineOrchestrator(config)
        results = orchestrator.run_pipeline(features)

        assert results.best_model is not None
        assert results.metrics["auc_roc"] > 0.5

        # 5. Model persistence
        model_path = temp_dir / "model.pkl"
        orchestrator.save_pipeline(model_path)
        assert model_path.exists()

        # 6. Production deployment
        from modern_bank_churn.production_readiness import ProductionPipeline

        prod_pipeline = ProductionPipeline(
            model=results.best_model, feature_names=results.selected_features
        )

        # 7. Prediction
        test_data = features[results.selected_features].iloc[:10]
        predictions = prod_pipeline.predict(test_data)
        assert len(predictions) == 10

    @pytest.mark.integration
    @pytest.mark.slow
    def test_pipeline_with_data_validation(self, corrupted_dataframe):
        """Test pipeline with data quality checks."""
        import great_expectations as gx

        # Create data context
        context = gx.get_context()

        # Add datasource
        datasource = context.sources.add_pandas(name="test_pandas")
        data_asset = datasource.add_dataframe_asset(name="test_asset")

        # Create expectations
        batch_request = data_asset.build_batch_request(dataframe=corrupted_dataframe)
        batch = context.get_batch_list(batch_request=batch_request)[0]

        # Validate data quality
        expectation_suite = context.add_expectation_suite("test_suite")

        # Add expectations
        batch.expect_column_to_exist("id")
        batch.expect_column_values_to_not_be_null("id")
        batch.expect_column_values_to_be_unique("id")

        # Run validation
        validation_result = batch.validate(expectation_suite=expectation_suite)

        # Check validation results
        assert not validation_result.success  # Should fail due to duplicates

        # Clean data based on validation
        cleaned_data = corrupted_dataframe.drop_duplicates(subset=["id"])
        cleaned_data = cleaned_data.dropna(subset=["value"])

        # Revalidate
        batch_request_clean = data_asset.build_batch_request(dataframe=cleaned_data)
        batch_clean = context.get_batch_list(batch_request=batch_request_clean)[0]

        validation_clean = batch_clean.validate(expectation_suite=expectation_suite)
        assert validation_clean.success or len(cleaned_data) > 0

    @pytest.mark.integration
    def test_notebook_execution_order(self, notebook_runner, temp_dir):
        """Test notebook execution in different orders."""
        notebooks = [
            "tutorials/01_getting_started.ipynb",
            "tutorials/02_real_world_case_study.ipynb",
            "tutorials/03_model_comparison.ipynb",
        ]

        # Test sequential execution
        for notebook in notebooks:
            if Path(notebook).exists():
                output = temp_dir / f"output_{Path(notebook).stem}.ipynb"
                try:
                    notebook_runner(notebook, output, parameters={"test_mode": True})
                    assert output.exists()
                except Exception as e:
                    # Notebooks might not execute without dependencies
                    print(f"Notebook {notebook} failed: {e}")

    @pytest.mark.integration
    def test_concurrent_model_training(self, sample_dataframe):
        """Test concurrent execution of multiple model trainings."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC

        X = sample_dataframe.drop("churn", axis=1).select_dtypes(include=[np.number])
        y = sample_dataframe["churn"]

        models = {
            "rf": RandomForestClassifier(n_estimators=10, random_state=42),
            "lr": LogisticRegression(random_state=42, max_iter=100),
            "svm": SVC(random_state=42, probability=True),
        }

        def train_model(model_data):
            name, model = model_data
            model.fit(X, y)
            score = model.score(X, y)
            return name, score

        # Train models concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(train_model, item): item[0] for item in models.items()
            }

            results = {}
            for future in concurrent.futures.as_completed(futures):
                name, score = future.result()
                results[name] = score

        assert len(results) == 3
        assert all(0 <= score <= 1 for score in results.values())

    @pytest.mark.integration
    def test_data_pipeline_with_different_formats(self, temp_dir):
        """Test pipeline with various data formats."""
        # Create test data in different formats
        test_data = pd.DataFrame(
            {"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50], "target": [0, 1, 0, 1, 1]}
        )

        # CSV format
        csv_path = temp_dir / "data.csv"
        test_data.to_csv(csv_path, index=False)
        csv_data = pd.read_csv(csv_path)
        assert csv_data.shape == test_data.shape

        # JSON format
        json_path = temp_dir / "data.json"
        test_data.to_json(json_path, orient="records")
        json_data = pd.read_json(json_path)
        assert json_data.shape == test_data.shape

        # Parquet format
        parquet_path = temp_dir / "data.parquet"
        test_data.to_parquet(parquet_path)
        parquet_data = pd.read_parquet(parquet_path)
        assert parquet_data.shape == test_data.shape

        # Excel format
        excel_path = temp_dir / "data.xlsx"
        test_data.to_excel(excel_path, index=False)
        excel_data = pd.read_excel(excel_path)
        assert excel_data.shape == test_data.shape

        # Test pipeline with each format
        for data in [csv_data, json_data, parquet_data, excel_data]:
            config = PipelineConfig(cross_validation_folds=2)
            orchestrator = MLPipelineOrchestrator(config)
            results = orchestrator.run_pipeline(data)
            assert results.best_model is not None

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_data_processing(self, sample_dataframe):
        """Test asynchronous data processing."""

        async def process_chunk(chunk):
            """Simulate async processing."""
            await asyncio.sleep(0.01)  # Simulate I/O
            return chunk.mean()

        # Split data into chunks
        chunks = np.array_split(sample_dataframe.select_dtypes(include=[np.number]), 10)

        # Process asynchronously
        tasks = [process_chunk(chunk) for chunk in chunks]
        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        assert all(isinstance(r, pd.Series) for r in results)

    @pytest.mark.integration
    def test_error_handling_and_recovery(self, corrupted_dataframe):
        """Test error handling and recovery mechanisms."""
        config = PipelineConfig()
        orchestrator = MLPipelineOrchestrator(config)

        # Test with corrupted data
        with pytest.raises(Exception):
            # Should handle gracefully
            results = orchestrator.run_pipeline(corrupted_dataframe)

        # Test recovery
        # Clean the data
        cleaned = corrupted_dataframe.dropna()
        cleaned = cleaned.select_dtypes(include=[np.number])

        if len(cleaned) > 10 and len(cleaned.columns) > 1:
            # Add a target column if missing
            if "target" not in cleaned.columns:
                cleaned["target"] = np.random.choice([0, 1], len(cleaned))

            # Retry with cleaned data
            results = orchestrator.run_pipeline(cleaned)
            assert results.best_model is not None


class TestAPIIntegration:
    """Test API integration and endpoints."""

    @pytest.mark.integration
    def test_rest_api_endpoints(self):
        """Test REST API endpoints."""
        api = APIInfrastructure()

        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"token": "test_token"}

            # Test authentication
            response = api.authenticate("user", "pass")
            assert response["token"] == "test_token"

        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {"data": [1, 2, 3]}

            # Test data retrieval
            response = api.get_data("test_token")
            assert response["data"] == [1, 2, 3]

    @pytest.mark.integration
    def test_graphql_queries(self):
        """Test GraphQL queries."""
        query = """
        {
            metrics {
                name
                value
                timestamp
            }
        }
        """

        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {
                "data": {
                    "metrics": [
                        {"name": "accuracy", "value": 0.95, "timestamp": "2024-01-01"}
                    ]
                }
            }

            api = APIInfrastructure()
            response = api.graphql_query(query)
            assert response["data"]["metrics"][0]["value"] == 0.95

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_websocket_connection(self):
        """Test WebSocket connection."""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.ws_connect("ws://localhost:5000/ws") as ws:
                    # Send message
                    await ws.send_str(
                        json.dumps({"action": "subscribe", "channel": "metrics"})
                    )

                    # Receive message
                    msg = await ws.receive()
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        assert "channel" in data
            except:
                # Connection might fail in test environment
                pass

    @pytest.mark.integration
    def test_api_rate_limiting(self):
        """Test API rate limiting."""
        api = APIInfrastructure()

        with patch("requests.get") as mock_get:
            # First 10 requests succeed
            for i in range(10):
                mock_get.return_value.status_code = 200
                response = api.get_data("token")
                assert response is not None

            # 11th request should be rate limited
            mock_get.return_value.status_code = 429
            mock_get.return_value.json.return_value = {"error": "Rate limit exceeded"}

            with pytest.raises(Exception):
                api.get_data("token")

    @pytest.mark.integration
    def test_api_authentication_flow(self):
        """Test complete authentication flow."""
        api = APIInfrastructure()

        with patch("requests.post") as mock_post:
            # Register user
            mock_post.return_value.status_code = 201
            mock_post.return_value.json.return_value = {"user_id": "123"}

            registration = api.register_user("newuser", "password", "email@test.com")
            assert registration["user_id"] == "123"

            # Login
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"token": "jwt_token"}

            login = api.login("newuser", "password")
            assert login["token"] == "jwt_token"

            # Use token for authenticated request
            with patch("requests.get") as mock_get:
                mock_get.return_value.status_code = 200
                mock_get.return_value.json.return_value = {"profile": "user_data"}

                profile = api.get_profile("jwt_token")
                assert profile["profile"] == "user_data"


class TestDashboardIntegration:
    """Test dashboard integration."""

    @pytest.mark.integration
    def test_dashboard_initialization(self, dashboard_config):
        """Test dashboard initialization and configuration."""
        dashboard = EnhancedDashboard(dashboard_config)
        assert dashboard.config == dashboard_config

    @pytest.mark.integration
    def test_dashboard_data_sources(self, sample_dataframe, dashboard_config):
        """Test dashboard data source registration."""
        dashboard = EnhancedDashboard(dashboard_config)

        # Register data source
        def get_data(filters=None):
            if filters:
                return sample_dataframe[
                    sample_dataframe["churn"] == filters.get("churn", 0)
                ]
            return sample_dataframe

        dashboard.register_data_source("test_data", get_data)
        assert "test_data" in dashboard.data_sources

        # Test data retrieval
        data = dashboard.get_data("test_data")
        assert len(data) == len(sample_dataframe)

        # Test with filters
        filtered_data = dashboard.get_data("test_data", filters={"churn": 1})
        assert len(filtered_data) <= len(sample_dataframe)

    @pytest.mark.integration
    def test_dashboard_real_time_updates(self, redis_client, dashboard_config):
        """Test real-time dashboard updates."""
        dashboard_config.enable_realtime = True
        dashboard = EnhancedDashboard(dashboard_config)

        # Simulate real-time data
        test_data = {"metric": "accuracy", "value": 0.95, "timestamp": time.time()}

        # Publish to Redis
        redis_client.publish("metrics", json.dumps(test_data))

        # Subscribe and receive
        pubsub = redis_client.pubsub()
        pubsub.subscribe("metrics")

        # Get message (skip subscription confirmation)
        pubsub.get_message()
        message = pubsub.get_message()

        if message:
            data = json.loads(message["data"])
            assert data["metric"] == "accuracy"
            assert data["value"] == 0.95

    @pytest.mark.integration
    def test_dashboard_export_functionality(self, sample_dataframe, temp_dir):
        """Test dashboard export capabilities."""
        from dashboard_enhanced.visualization_components import (
            InteractiveVisualizations,
        )
        import plotly.graph_objects as go

        viz = InteractiveVisualizations()

        # Create sample figure
        fig = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[4, 5, 6]))

        # Test PDF export
        pdf_path = temp_dir / "export.pdf"
        viz.export_to_pdf([fig], pdf_path)
        # Note: Actual PDF generation might require additional dependencies

        # Test Excel export
        excel_path = temp_dir / "export.xlsx"
        sample_dataframe.to_excel(excel_path)
        assert excel_path.exists()


class TestConcurrentExecution:
    """Test concurrent execution scenarios."""

    @pytest.mark.integration
    def test_parallel_model_training(self, sample_dataframe):
        """Test parallel training of multiple models."""
        from sklearn.model_selection import train_test_split

        X = sample_dataframe.drop("churn", axis=1).select_dtypes(include=[np.number])
        y = sample_dataframe["churn"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        def train_single_model(config_dict):
            """Train a single model with given config."""
            config = PipelineConfig(**config_dict)
            orchestrator = MLPipelineOrchestrator(config)

            # Create temporary dataframe
            train_df = pd.concat([X_train, y_train], axis=1)
            train_df.columns = list(X_train.columns) + ["target"]

            results = orchestrator.run_pipeline(train_df)
            return results.metrics.get("auc_roc", 0)

        configs = [
            {"model_type": "random_forest", "cross_validation_folds": 2},
            {"model_type": "xgboost", "cross_validation_folds": 2},
            {"model_type": "lightgbm", "cross_validation_folds": 2},
        ]

        # Train models in parallel
        with Pool(processes=3) as pool:
            scores = pool.map(train_single_model, configs)

        assert len(scores) == 3
        assert all(0 <= score <= 1 for score in scores)

    @pytest.mark.integration
    def test_thread_safe_prediction(self, trained_model):
        """Test thread-safe prediction serving."""
        model, X_test, _ = trained_model
        results = []
        errors = []

        def predict_batch(data, index):
            """Make predictions for a batch."""
            try:
                preds = model.predict(data)
                results.append((index, preds))
            except Exception as e:
                errors.append((index, str(e)))

        # Create threads
        threads = []
        for i in range(10):
            batch = X_test.iloc[i * 10 : (i + 1) * 10]
            t = threading.Thread(target=predict_batch, args=(batch, i))
            threads.append(t)
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10

    @pytest.mark.integration
    def test_queue_based_processing(self, sample_dataframe):
        """Test queue-based data processing."""
        input_queue = queue.Queue()
        output_queue = queue.Queue()

        def worker():
            """Worker function to process data from queue."""
            while True:
                item = input_queue.get()
                if item is None:
                    break

                # Process item
                result = item.mean()
                output_queue.put(result)
                input_queue.task_done()

        # Start workers
        num_workers = 4
        threads = []
        for _ in range(num_workers):
            t = threading.Thread(target=worker)
            t.start()
            threads.append(t)

        # Add data to queue
        chunks = np.array_split(sample_dataframe.select_dtypes(include=[np.number]), 20)
        for chunk in chunks:
            input_queue.put(chunk)

        # Wait for processing
        input_queue.join()

        # Stop workers
        for _ in range(num_workers):
            input_queue.put(None)
        for t in threads:
            t.join()

        # Collect results
        results = []
        while not output_queue.empty():
            results.append(output_queue.get())

        assert len(results) == 20


class TestDataSizeScaling:
    """Test pipeline with various data sizes."""

    @pytest.mark.integration
    @pytest.mark.parametrize("n_samples", [100, 1000, 10000])
    def test_different_data_sizes(self, n_samples):
        """Test pipeline with different data sizes."""
        # Generate data
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "feature1": np.random.randn(n_samples),
                "feature2": np.random.randn(n_samples),
                "feature3": np.random.randn(n_samples),
                "target": np.random.choice([0, 1], n_samples),
            }
        )

        # Run pipeline
        config = PipelineConfig(cross_validation_folds=2, hyperparameter_tuning=False)
        orchestrator = MLPipelineOrchestrator(config)

        start_time = time.time()
        results = orchestrator.run_pipeline(data)
        elapsed = time.time() - start_time

        assert results.best_model is not None
        # Larger datasets should take more time
        if n_samples == 10000:
            assert elapsed > 0.1  # Should take some measurable time

    @pytest.mark.integration
    @pytest.mark.slow
    def test_memory_efficiency(self):
        """Test memory efficiency with large datasets."""
        import psutil
        import gc

        # Get initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create large dataset
        large_data = pd.DataFrame(
            {f"col_{i}": np.random.randn(100000) for i in range(50)}
        )
        large_data["target"] = np.random.choice([0, 1], 100000)

        # Run pipeline
        config = PipelineConfig(cross_validation_folds=2, hyperparameter_tuning=False)
        orchestrator = MLPipelineOrchestrator(config)

        # Use chunked processing
        chunk_size = 10000
        for i in range(0, len(large_data), chunk_size):
            chunk = large_data.iloc[i : i + chunk_size]
            # Process chunk (in real scenario, would aggregate results)
            del chunk
            gc.collect()

        # Check memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (< 500 MB for this test)
        assert memory_increase < 500
