"""
Test suite for the ML API module.
"""

import pytest
import json
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from fastapi.testclient import TestClient
from src.api.ml_api import (
    app,
    ModelManager,
    PredictionRequest,
    PredictionResponse,
    ModelMetrics,
    BatchPredictionRequest
)

class TestModelManager:
    """Test suite for ModelManager class."""

    @pytest.fixture
    def model_manager(self):
        """Create a model manager instance."""
        return ModelManager()

    @pytest.fixture
    def sample_model(self):
        """Create a sample model mock."""
        model = Mock()
        model.predict = Mock(return_value=np.array([0.8, 0.2]))
        model.predict_proba = Mock(return_value=np.array([[0.2, 0.8], [0.8, 0.2]]))
        return model

    def test_register_model(self, model_manager, sample_model):
        """Test model registration."""
        model_manager.register_model(
            'test_model',
            sample_model,
            version='1.0.0',
            metadata={'framework': 'sklearn'}
        )

        assert 'test_model' in model_manager.models
        assert model_manager.models['test_model']['version'] == '1.0.0'
        assert model_manager.models['test_model']['metadata']['framework'] == 'sklearn'

    def test_get_model(self, model_manager, sample_model):
        """Test getting a registered model."""
        model_manager.register_model('test_model', sample_model)
        retrieved_model = model_manager.get_model('test_model')

        assert retrieved_model is sample_model

    def test_get_nonexistent_model(self, model_manager):
        """Test getting a non-existent model raises error."""
        with pytest.raises(KeyError):
            model_manager.get_model('nonexistent')

    def test_list_models(self, model_manager, sample_model):
        """Test listing all registered models."""
        model_manager.register_model('model1', sample_model, version='1.0')
        model_manager.register_model('model2', sample_model, version='2.0')

        models = model_manager.list_models()
        assert len(models) == 2
        assert 'model1' in models
        assert 'model2' in models

    def test_update_model_metrics(self, model_manager, sample_model):
        """Test updating model metrics."""
        model_manager.register_model('test_model', sample_model)

        metrics = {
            'accuracy': 0.95,
            'precision': 0.92,
            'recall': 0.93,
            'f1_score': 0.925
        }

        model_manager.update_metrics('test_model', metrics)
        stored_metrics = model_manager.get_metrics('test_model')

        assert stored_metrics['accuracy'] == 0.95
        assert stored_metrics['precision'] == 0.92

    def test_model_versioning(self, model_manager):
        """Test model versioning support."""
        model_v1 = Mock()
        model_v2 = Mock()

        model_manager.register_model('model', model_v1, version='1.0.0')
        model_manager.register_model('model', model_v2, version='2.0.0')

        # Should keep the latest version by default
        assert model_manager.models['model']['version'] == '2.0.0'

class TestPredictionAPI:
    """Test suite for prediction API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def mock_model_manager(self):
        """Create mock model manager."""
        with patch('src.api.ml_api.model_manager') as mock:
            model = Mock()
            model.predict = Mock(return_value=np.array([1, 0]))
            model.predict_proba = Mock(return_value=np.array([[0.3, 0.7], [0.9, 0.1]]))

            mock.get_model = Mock(return_value=model)
            mock.models = {'test_model': {'model': model, 'version': '1.0'}}
            yield mock

    def test_single_prediction_endpoint(self, client, mock_model_manager):
        """Test single prediction endpoint."""
        request_data = {
            'model_name': 'test_model',
            'features': {'feature1': 1.0, 'feature2': 2.0},
            'return_probabilities': False
        }

        response = client.post('/predict', json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert 'prediction' in data
        assert 'model_version' in data
        assert 'timestamp' in data

    def test_batch_prediction_endpoint(self, client, mock_model_manager):
        """Test batch prediction endpoint."""
        request_data = {
            'model_name': 'test_model',
            'features': [
                {'feature1': 1.0, 'feature2': 2.0},
                {'feature1': 3.0, 'feature2': 4.0}
            ]
        }

        response = client.post('/predict/batch', json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert 'predictions' in data
        assert len(data['predictions']) == 2

    def test_prediction_with_probabilities(self, client, mock_model_manager):
        """Test prediction with probability scores."""
        request_data = {
            'model_name': 'test_model',
            'features': {'feature1': 1.0, 'feature2': 2.0},
            'return_probabilities': True
        }

        response = client.post('/predict', json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert 'probabilities' in data
        assert isinstance(data['probabilities'], list)

    def test_model_not_found(self, client):
        """Test prediction with non-existent model."""
        request_data = {
            'model_name': 'nonexistent_model',
            'features': {'feature1': 1.0}
        }

        response = client.post('/predict', json=request_data)

        assert response.status_code == 404
        assert 'error' in response.json()

    def test_invalid_features(self, client, mock_model_manager):
        """Test prediction with invalid features."""
        mock_model_manager.get_model.side_effect = ValueError("Invalid features")

        request_data = {
            'model_name': 'test_model',
            'features': {'invalid': 'data'}
        }

        response = client.post('/predict', json=request_data)

        assert response.status_code == 400

class TestModelMetrics:
    """Test suite for model metrics endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_get_model_metrics(self, client):
        """Test getting model metrics."""
        with patch('src.api.ml_api.model_manager') as mock_manager:
            mock_manager.get_metrics = Mock(return_value={
                'accuracy': 0.95,
                'precision': 0.92,
                'recall': 0.93
            })

            response = client.get('/models/test_model/metrics')

            assert response.status_code == 200
            data = response.json()
            assert data['accuracy'] == 0.95

    def test_update_model_metrics(self, client):
        """Test updating model metrics."""
        with patch('src.api.ml_api.model_manager') as mock_manager:
            mock_manager.update_metrics = Mock()

            metrics_data = {
                'accuracy': 0.96,
                'precision': 0.94
            }

            response = client.post('/models/test_model/metrics', json=metrics_data)

            assert response.status_code == 200
            mock_manager.update_metrics.assert_called_once()

    def test_list_all_models(self, client):
        """Test listing all available models."""
        with patch('src.api.ml_api.model_manager') as mock_manager:
            mock_manager.list_models = Mock(return_value={
                'model1': {'version': '1.0', 'status': 'active'},
                'model2': {'version': '2.0', 'status': 'active'}
            })

            response = client.get('/models')

            assert response.status_code == 200
            data = response.json()
            assert len(data) == 2
            assert 'model1' in data

class TestModelDeployment:
    """Test suite for model deployment features."""

    @pytest.fixture
    def deployment_manager(self):
        """Create deployment manager instance."""
        from src.api.ml_api import DeploymentManager
        return DeploymentManager()

    def test_deploy_model(self, deployment_manager):
        """Test model deployment."""
        model = Mock()

        deployment_id = deployment_manager.deploy(
            model=model,
            name='test_deployment',
            endpoint='/predict/test'
        )

        assert deployment_id is not None
        assert deployment_manager.get_deployment(deployment_id) is not None

    def test_rollback_deployment(self, deployment_manager):
        """Test deployment rollback."""
        model_v1 = Mock()
        model_v2 = Mock()

        # Deploy v1
        deployment_id = deployment_manager.deploy(model_v1, 'test', '/test')

        # Deploy v2
        deployment_manager.deploy(model_v2, 'test', '/test')

        # Rollback
        deployment_manager.rollback(deployment_id)

        current = deployment_manager.get_current_deployment('test')
        assert current['model'] is model_v1

    def test_canary_deployment(self, deployment_manager):
        """Test canary deployment strategy."""
        model_stable = Mock()
        model_canary = Mock()

        # Deploy stable version
        deployment_manager.deploy(model_stable, 'prod', '/prod')

        # Deploy canary with 10% traffic
        deployment_manager.deploy_canary(
            model_canary,
            'prod',
            traffic_percentage=0.1
        )

        # Verify traffic split
        traffic_config = deployment_manager.get_traffic_config('prod')
        assert traffic_config['canary_percentage'] == 0.1

class TestFeatureEngineering:
    """Test suite for feature engineering utilities."""

    def test_feature_scaling(self):
        """Test feature scaling functionality."""
        from src.api.ml_api import FeatureScaler

        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })

        scaler = FeatureScaler(method='standard')
        scaled_data = scaler.fit_transform(data)

        # Check mean is approximately 0
        assert abs(scaled_data['feature1'].mean()) < 0.01
        # Check std is approximately 1
        assert abs(scaled_data['feature1'].std() - 1) < 0.01

    def test_feature_encoding(self):
        """Test categorical feature encoding."""
        from src.api.ml_api import FeatureEncoder

        data = pd.DataFrame({
            'category': ['A', 'B', 'A', 'C', 'B']
        })

        encoder = FeatureEncoder(method='onehot')
        encoded_data = encoder.fit_transform(data)

        assert 'category_A' in encoded_data.columns
        assert 'category_B' in encoded_data.columns
        assert 'category_C' in encoded_data.columns

    def test_feature_selection(self):
        """Test feature selection methods."""
        from src.api.ml_api import FeatureSelector

        X = pd.DataFrame(np.random.randn(100, 10))
        y = np.random.binomial(1, 0.5, 100)

        selector = FeatureSelector(method='mutual_info', k=5)
        selected_features = selector.fit_transform(X, y)

        assert selected_features.shape[1] == 5

class TestModelMonitoring:
    """Test suite for model monitoring capabilities."""

    @pytest.fixture
    def monitor(self):
        """Create model monitor instance."""
        from src.api.ml_api import ModelMonitor
        return ModelMonitor()

    def test_data_drift_detection(self, monitor):
        """Test data drift detection."""
        reference_data = pd.DataFrame(np.random.randn(1000, 5))
        current_data = pd.DataFrame(np.random.randn(1000, 5) + 0.5)  # Shifted

        drift_report = monitor.detect_drift(reference_data, current_data)

        assert 'has_drift' in drift_report
        assert 'drift_score' in drift_report
        assert drift_report['has_drift'] == True

    def test_performance_monitoring(self, monitor):
        """Test model performance monitoring."""
        predictions = np.array([0, 1, 1, 0, 1])
        actuals = np.array([0, 1, 0, 0, 1])

        monitor.log_predictions(predictions, actuals)

        metrics = monitor.get_performance_metrics()
        assert 'accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1

    def test_alert_triggering(self, monitor):
        """Test alert triggering on threshold breach."""
        monitor.set_alert_threshold('accuracy', 0.9)

        # Log poor performance
        predictions = np.array([0, 1, 1, 0, 1])
        actuals = np.array([1, 0, 0, 1, 0])

        monitor.log_predictions(predictions, actuals)

        alerts = monitor.check_alerts()
        assert len(alerts) > 0
        assert any('accuracy' in alert['metric'] for alert in alerts)

class TestAPIRateLimiting:
    """Test suite for API rate limiting."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_rate_limiting(self, client):
        """Test rate limiting on prediction endpoint."""
        # Make multiple rapid requests
        responses = []
        for _ in range(15):
            response = client.post('/predict', json={
                'model_name': 'test',
                'features': {}
            })
            responses.append(response)

        # Should get rate limited after threshold
        assert any(r.status_code == 429 for r in responses)

    def test_rate_limit_headers(self, client):
        """Test rate limit headers in response."""
        response = client.post('/predict', json={
            'model_name': 'test',
            'features': {}
        })

        assert 'X-RateLimit-Limit' in response.headers
        assert 'X-RateLimit-Remaining' in response.headers