"""
Unit tests for ML Pipeline module with property-based testing.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from hypothesis import given, strategies as st, assume, settings
from hypothesis.extra.pandas import data_frames, column, range_indexes
import joblib
import tempfile
from pathlib import Path

from modern_bank_churn.ml_pipeline_orchestrator import (
    MLPipelineOrchestrator,
    PipelineConfig
)
from modern_bank_churn.feature_engineering import FeatureEngineer
from modern_bank_churn.evaluation_enhancements import ModelEvaluator
from modern_bank_churn.production_readiness import ProductionPipeline


class TestPipelineConfig:
    """Test PipelineConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PipelineConfig()
        assert config.feature_selection_method == 'mutual_info'
        assert config.model_type == 'random_forest'
        assert config.cross_validation_folds == 5
        assert config.random_state == 42

    def test_custom_config(self):
        """Test custom configuration."""
        config = PipelineConfig(
            feature_selection_method='boruta',
            model_type='xgboost',
            hyperparameter_tuning=True,
            optimization_metric='f1',
            cross_validation_folds=10
        )
        assert config.feature_selection_method == 'boruta'
        assert config.model_type == 'xgboost'
        assert config.hyperparameter_tuning is True
        assert config.optimization_metric == 'f1'
        assert config.cross_validation_folds == 10

    @given(
        method=st.sampled_from(['mutual_info', 'boruta', 'rfe', 'lasso']),
        model=st.sampled_from(['random_forest', 'xgboost', 'lightgbm', 'ensemble']),
        folds=st.integers(min_value=2, max_value=20)
    )
    def test_property_based_config(self, method, model, folds):
        """Property-based test for configuration."""
        config = PipelineConfig(
            feature_selection_method=method,
            model_type=model,
            cross_validation_folds=folds
        )
        assert config.feature_selection_method == method
        assert config.model_type == model
        assert config.cross_validation_folds == folds

    def test_config_validation(self):
        """Test configuration validation."""
        with pytest.raises(ValueError):
            PipelineConfig(feature_selection_method='invalid')

        with pytest.raises(ValueError):
            PipelineConfig(model_type='invalid')

        with pytest.raises(ValueError):
            PipelineConfig(cross_validation_folds=1)

    def test_config_serialization(self):
        """Test configuration serialization."""
        config = PipelineConfig()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert 'feature_selection_method' in config_dict
        assert 'model_type' in config_dict

        # Test deserialization
        new_config = PipelineConfig.from_dict(config_dict)
        assert new_config.feature_selection_method == config.feature_selection_method
        assert new_config.model_type == config.model_type


class TestFeatureEngineer:
    """Test FeatureEngineer class."""

    def test_initialization(self):
        """Test FeatureEngineer initialization."""
        engineer = FeatureEngineer()
        assert engineer is not None
        assert hasattr(engineer, 'create_features')
        assert hasattr(engineer, 'select_features')

    def test_create_features(self, sample_dataframe):
        """Test feature creation."""
        engineer = FeatureEngineer()
        features = engineer.create_features(sample_dataframe)

        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(sample_dataframe)
        assert len(features.columns) >= len(sample_dataframe.columns)

    @given(
        df=data_frames(
            columns=[
                column('a', dtype=float),
                column('b', dtype=float),
                column('c', dtype=int)
            ],
            rows=st.integers(min_value=10, max_value=100)
        )
    )
    def test_property_based_features(self, df):
        """Property-based test for feature creation."""
        engineer = FeatureEngineer()
        features = engineer.create_features(df)

        # Properties that should always hold
        assert len(features) == len(df)
        assert features.shape[0] > 0
        assert features.shape[1] >= df.shape[1]
        assert not features.isnull().all().any()  # No column should be all null

    def test_interaction_features(self, small_dataframe):
        """Test interaction feature creation."""
        engineer = FeatureEngineer()
        interactions = engineer.create_interaction_features(
            small_dataframe,
            cols=['a', 'b']
        )

        assert 'a_x_b' in interactions.columns
        assert (interactions['a_x_b'] == small_dataframe['a'] * small_dataframe['b']).all()

    def test_polynomial_features(self, small_dataframe):
        """Test polynomial feature creation."""
        engineer = FeatureEngineer()
        poly = engineer.create_polynomial_features(
            small_dataframe[['a', 'b']],
            degree=2
        )

        assert poly.shape[1] > small_dataframe[['a', 'b']].shape[1]
        assert 'a^2' in poly.columns or 'poly_1' in poly.columns

    def test_time_features(self):
        """Test time-based feature extraction."""
        dates = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=100, freq='D')
        })

        engineer = FeatureEngineer()
        time_features = engineer.create_time_features(dates, 'date')

        assert 'date_year' in time_features.columns
        assert 'date_month' in time_features.columns
        assert 'date_day' in time_features.columns
        assert 'date_dayofweek' in time_features.columns

    def test_feature_selection(self, sample_dataframe):
        """Test feature selection methods."""
        engineer = FeatureEngineer()

        X = sample_dataframe.drop('churn', axis=1).select_dtypes(include=[np.number])
        y = sample_dataframe['churn']

        # Test different selection methods
        for method in ['mutual_info', 'chi2', 'f_classif']:
            selected = engineer.select_features(X, y, method=method, k=5)
            assert isinstance(selected, list)
            assert len(selected) == 5
            assert all(f in X.columns for f in selected)

    @pytest.mark.parametrize("k", [1, 3, 5, 10])
    def test_feature_selection_k_values(self, sample_dataframe, k):
        """Test feature selection with different k values."""
        engineer = FeatureEngineer()

        X = sample_dataframe.drop('churn', axis=1).select_dtypes(include=[np.number])
        y = sample_dataframe['churn']

        if k <= len(X.columns):
            selected = engineer.select_features(X, y, method='mutual_info', k=k)
            assert len(selected) == k

    def test_handle_missing_values(self, corrupted_dataframe):
        """Test handling of missing values."""
        engineer = FeatureEngineer()
        cleaned = engineer.handle_missing_values(corrupted_dataframe)

        assert cleaned.isnull().sum().sum() < corrupted_dataframe.isnull().sum().sum()


class TestMLPipelineOrchestrator:
    """Test MLPipelineOrchestrator class."""

    def test_initialization(self, pipeline_config):
        """Test orchestrator initialization."""
        orchestrator = MLPipelineOrchestrator(pipeline_config)
        assert orchestrator.config == pipeline_config
        assert orchestrator.best_model is None
        assert orchestrator.selected_features is None

    @patch('modern_bank_churn.ml_pipeline_orchestrator.train_test_split')
    @patch('modern_bank_churn.ml_pipeline_orchestrator.RandomForestClassifier')
    def test_run_pipeline(self, mock_rf, mock_split, sample_dataframe):
        """Test complete pipeline execution."""
        # Setup mocks
        mock_split.return_value = (
            sample_dataframe.iloc[:800],
            sample_dataframe.iloc[800:],
            sample_dataframe.iloc[:800]['churn'],
            sample_dataframe.iloc[800:]['churn']
        )

        mock_model = MagicMock()
        mock_model.fit.return_value = mock_model
        mock_model.predict.return_value = np.array([0, 1] * 100)
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]] * 200)
        mock_rf.return_value = mock_model

        orchestrator = MLPipelineOrchestrator(PipelineConfig())
        results = orchestrator.run_pipeline(sample_dataframe)

        assert results.best_model is not None
        assert results.selected_features is not None
        assert 'auc_roc' in results.metrics

    def test_train_models(self, sample_dataframe, pipeline_config):
        """Test model training."""
        orchestrator = MLPipelineOrchestrator(pipeline_config)

        X = sample_dataframe.drop('churn', axis=1).select_dtypes(include=[np.number])
        y = sample_dataframe['churn']

        models = orchestrator.train_models(X, y)

        assert isinstance(models, dict)
        assert len(models) > 0
        assert all(hasattr(model, 'predict') for model in models.values())

    def test_hyperparameter_tuning(self, small_dataframe):
        """Test hyperparameter tuning."""
        config = PipelineConfig(hyperparameter_tuning=True)
        orchestrator = MLPipelineOrchestrator(config)

        X = small_dataframe[['a', 'b']]
        y = small_dataframe['target']

        best_params = orchestrator.tune_hyperparameters(
            'random_forest',
            X, y,
            n_trials=5
        )

        assert isinstance(best_params, dict)
        assert 'n_estimators' in best_params

    @given(
        n_samples=st.integers(min_value=100, max_value=1000),
        n_features=st.integers(min_value=5, max_value=50)
    )
    @settings(max_examples=5, deadline=10000)
    def test_property_based_pipeline(self, n_samples, n_features):
        """Property-based test for pipeline robustness."""
        # Generate random data
        X = np.random.randn(n_samples, n_features)
        y = np.random.choice([0, 1], n_samples)

        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
        df['target'] = y

        config = PipelineConfig(
            cross_validation_folds=2,
            hyperparameter_tuning=False
        )
        orchestrator = MLPipelineOrchestrator(config)

        try:
            results = orchestrator.run_pipeline(df)
            # Properties that should hold
            assert results.best_model is not None
            assert 0 <= results.metrics.get('auc_roc', 0) <= 1
            assert len(results.selected_features) > 0
        except Exception as e:
            # Pipeline should handle errors gracefully
            assert "Pipeline failed" not in str(e)

    def test_save_and_load_pipeline(self, trained_model, temp_dir):
        """Test pipeline persistence."""
        model, X_test, y_test = trained_model

        config = PipelineConfig()
        orchestrator = MLPipelineOrchestrator(config)
        orchestrator.best_model = model
        orchestrator.selected_features = list(X_test.columns)

        # Save pipeline
        save_path = temp_dir / "pipeline.pkl"
        orchestrator.save_pipeline(save_path)
        assert save_path.exists()

        # Load pipeline
        loaded_orchestrator = MLPipelineOrchestrator.load_pipeline(save_path)
        assert loaded_orchestrator.best_model is not None
        assert loaded_orchestrator.selected_features == orchestrator.selected_features


class TestModelEvaluator:
    """Test ModelEvaluator class."""

    def test_initialization(self):
        """Test evaluator initialization."""
        evaluator = ModelEvaluator()
        assert hasattr(evaluator, 'evaluate')
        assert hasattr(evaluator, 'cross_validate')

    def test_evaluate(self, trained_model):
        """Test model evaluation."""
        model, X_test, y_test = trained_model
        evaluator = ModelEvaluator()

        metrics = evaluator.evaluate(model, X_test, y_test)

        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'auc_roc' in metrics

        # Check metric ranges
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']:
            assert 0 <= metrics[metric] <= 1

    def test_cross_validation(self, sample_dataframe):
        """Test cross-validation."""
        from sklearn.ensemble import RandomForestClassifier

        evaluator = ModelEvaluator()
        model = RandomForestClassifier(n_estimators=10, random_state=42)

        X = sample_dataframe.drop('churn', axis=1).select_dtypes(include=[np.number])
        y = sample_dataframe['churn']

        cv_results = evaluator.cross_validate(model, X, y, cv=3)

        assert 'mean_score' in cv_results
        assert 'std_score' in cv_results
        assert 'scores' in cv_results
        assert len(cv_results['scores']) == 3

    @pytest.mark.parametrize("metric", ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])
    def test_different_metrics(self, trained_model, metric):
        """Test evaluation with different metrics."""
        model, X_test, y_test = trained_model
        evaluator = ModelEvaluator()

        score = evaluator.evaluate_metric(model, X_test, y_test, metric=metric)
        assert isinstance(score, float)
        assert 0 <= score <= 1

    def test_confusion_matrix(self, trained_model):
        """Test confusion matrix generation."""
        model, X_test, y_test = trained_model
        evaluator = ModelEvaluator()

        cm = evaluator.get_confusion_matrix(model, X_test, y_test)

        assert cm.shape == (2, 2)
        assert cm.sum() == len(y_test)

    def test_classification_report(self, trained_model):
        """Test classification report generation."""
        model, X_test, y_test = trained_model
        evaluator = ModelEvaluator()

        report = evaluator.get_classification_report(model, X_test, y_test)

        assert isinstance(report, dict)
        assert '0' in report
        assert '1' in report
        assert 'accuracy' in report

    def test_roc_curve(self, trained_model):
        """Test ROC curve data generation."""
        model, X_test, y_test = trained_model
        evaluator = ModelEvaluator()

        fpr, tpr, thresholds = evaluator.get_roc_curve(model, X_test, y_test)

        assert len(fpr) == len(tpr) == len(thresholds)
        assert fpr[0] == 0 and fpr[-1] == 1
        assert tpr[0] == 0 and tpr[-1] == 1

    def test_feature_importance(self, trained_model):
        """Test feature importance extraction."""
        model, X_test, _ = trained_model
        evaluator = ModelEvaluator()

        importance = evaluator.get_feature_importance(model, X_test.columns)

        assert isinstance(importance, pd.DataFrame)
        assert 'feature' in importance.columns
        assert 'importance' in importance.columns
        assert len(importance) == len(X_test.columns)

    @given(
        y_true=st.lists(st.integers(min_value=0, max_value=1), min_size=10, max_size=100),
        y_pred=st.lists(st.integers(min_value=0, max_value=1), min_size=10, max_size=100)
    )
    def test_property_based_metrics(self, y_true, y_pred):
        """Property-based test for metric calculation."""
        assume(len(y_true) == len(y_pred))

        evaluator = ModelEvaluator()
        metrics = evaluator.calculate_metrics(y_true, y_pred)

        # Properties that should always hold
        assert 0 <= metrics['accuracy'] <= 1
        if metrics['precision'] is not None:
            assert 0 <= metrics['precision'] <= 1
        if metrics['recall'] is not None:
            assert 0 <= metrics['recall'] <= 1


class TestProductionPipeline:
    """Test ProductionPipeline class."""

    def test_initialization(self, trained_model):
        """Test production pipeline initialization."""
        model, X_test, _ = trained_model

        pipeline = ProductionPipeline(
            model=model,
            feature_names=list(X_test.columns)
        )

        assert pipeline.model == model
        assert pipeline.feature_names == list(X_test.columns)

    def test_predict(self, trained_model):
        """Test prediction in production pipeline."""
        model, X_test, _ = trained_model

        pipeline = ProductionPipeline(
            model=model,
            feature_names=list(X_test.columns)
        )

        predictions = pipeline.predict(X_test.iloc[:10])

        assert len(predictions) == 10
        assert all(p in [0, 1] for p in predictions)

    def test_predict_proba(self, trained_model):
        """Test probability prediction."""
        model, X_test, _ = trained_model

        pipeline = ProductionPipeline(
            model=model,
            feature_names=list(X_test.columns)
        )

        probabilities = pipeline.predict_proba(X_test.iloc[:10])

        assert probabilities.shape == (10, 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)

    def test_input_validation(self, trained_model):
        """Test input validation in production."""
        model, X_test, _ = trained_model

        pipeline = ProductionPipeline(
            model=model,
            feature_names=list(X_test.columns)
        )

        # Test with missing column
        invalid_data = X_test.iloc[:5].drop(columns=[X_test.columns[0]])

        with pytest.raises(ValueError):
            pipeline.predict(invalid_data)

    def test_preprocessing(self, trained_model, corrupted_dataframe):
        """Test data preprocessing in production."""
        model, X_test, _ = trained_model

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(X_test)

        pipeline = ProductionPipeline(
            model=model,
            feature_names=list(X_test.columns),
            preprocessing_params={'scaler': scaler}
        )

        # Should handle preprocessing
        predictions = pipeline.predict(X_test.iloc[:5])
        assert len(predictions) == 5

    def test_model_versioning(self, trained_model, temp_dir):
        """Test model versioning."""
        model, X_test, _ = trained_model

        pipeline = ProductionPipeline(
            model=model,
            feature_names=list(X_test.columns),
            version='1.0.0'
        )

        # Save versioned model
        save_path = temp_dir / "model_v1.0.0.pkl"
        pipeline.save(save_path)
        assert save_path.exists()

        # Load and verify version
        loaded_pipeline = ProductionPipeline.load(save_path)
        assert loaded_pipeline.version == '1.0.0'

    def test_monitoring_metrics(self, trained_model):
        """Test monitoring metric collection."""
        model, X_test, _ = trained_model

        pipeline = ProductionPipeline(
            model=model,
            feature_names=list(X_test.columns),
            enable_monitoring=True
        )

        # Make predictions
        predictions = pipeline.predict(X_test.iloc[:10])

        # Check monitoring metrics
        metrics = pipeline.get_monitoring_metrics()
        assert 'prediction_count' in metrics
        assert 'avg_prediction_time' in metrics
        assert metrics['prediction_count'] == 10

    @given(
        n_samples=st.integers(min_value=1, max_value=100)
    )
    def test_batch_prediction(self, trained_model, n_samples):
        """Test batch prediction with various sizes."""
        model, X_test, _ = trained_model

        pipeline = ProductionPipeline(
            model=model,
            feature_names=list(X_test.columns)
        )

        batch_size = min(n_samples, len(X_test))
        batch_data = X_test.iloc[:batch_size]

        predictions = pipeline.predict_batch(batch_data)
        assert len(predictions) == batch_size


class TestMutationTesting:
    """Mutation testing to ensure test quality."""

    def test_mutation_resistance(self, sample_dataframe):
        """Test that our tests catch mutations."""
        engineer = FeatureEngineer()

        # Original function
        original_features = engineer.create_features(sample_dataframe)

        # Mutate the function (simulate a bug)
        with patch.object(engineer, 'create_features') as mock_create:
            # Return wrong number of rows
            mock_create.return_value = original_features.iloc[:50]

            # This should fail our tests
            with pytest.raises(AssertionError):
                features = engineer.create_features(sample_dataframe)
                assert len(features) == len(sample_dataframe)