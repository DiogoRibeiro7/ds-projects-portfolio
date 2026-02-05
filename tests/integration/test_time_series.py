"""Test suite for time series analysis module.
"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from time_series import (
    AnomalyDetector,
    ForecastModel,
    SeasonalDecomposer,
    TimeSeriesAnalyzer,
    TimeSeriesPreprocessor,
)

pytestmark = pytest.mark.integration


class TestTimeSeriesPreprocessor:
    """Test suite for time series preprocessing."""

    @pytest.fixture
    def sample_ts_data(self):
        """Create sample time series data."""
        dates = pd.date_range("2020-01-01", periods=365, freq="D")
        values = np.sin(np.arange(365) * 2 * np.pi / 365) * 100 + 1000
        noise = np.random.normal(0, 10, 365)

        return pd.DataFrame({"date": dates, "value": values + noise})

    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance."""
        return TimeSeriesPreprocessor()

    def test_missing_value_imputation(self, preprocessor, sample_ts_data):
        """Test missing value imputation methods."""
        # Add missing values
        data = sample_ts_data.copy()
        data.loc[10:15, "value"] = np.nan

        # Forward fill
        filled_ff = preprocessor.impute_missing(data, method="forward_fill")
        assert not filled_ff["value"].isna().any()

        # Linear interpolation
        filled_linear = preprocessor.impute_missing(data, method="linear")
        assert not filled_linear["value"].isna().any()

        # Seasonal interpolation
        filled_seasonal = preprocessor.impute_missing(data, method="seasonal", period=7)
        assert not filled_seasonal["value"].isna().any()

    def test_outlier_detection(self, preprocessor, sample_ts_data):
        """Test outlier detection methods."""
        # Add outliers
        data = sample_ts_data.copy()
        data.loc[50, "value"] = 2000  # Extreme outlier

        # Z-score method
        outliers_z = preprocessor.detect_outliers(data, method="zscore", threshold=3)
        assert 50 in outliers_z

        # IQR method
        outliers_iqr = preprocessor.detect_outliers(data, method="iqr")
        assert 50 in outliers_iqr

        # Isolation Forest
        outliers_if = preprocessor.detect_outliers(data, method="isolation_forest")
        assert len(outliers_if) > 0

    def test_resampling(self, preprocessor, sample_ts_data):
        """Test time series resampling."""
        # Resample to weekly
        weekly_data = preprocessor.resample(
            sample_ts_data, freq="W", aggregation="mean"
        )

        assert len(weekly_data) < len(sample_ts_data)
        assert weekly_data["value"].notna().all()

        # Resample to hourly (upsampling)
        hourly_data = preprocessor.resample(
            sample_ts_data.head(7), freq="H", interpolation="linear"
        )

        assert len(hourly_data) > len(sample_ts_data.head(7))

    def test_differencing(self, preprocessor, sample_ts_data):
        """Test differencing for stationarity."""
        # First difference
        diff_1 = preprocessor.difference(sample_ts_data["value"], order=1)
        assert len(diff_1) == len(sample_ts_data) - 1

        # Seasonal difference
        diff_seasonal = preprocessor.difference(
            sample_ts_data["value"], order=1, seasonal_period=7
        )
        assert len(diff_seasonal) == len(sample_ts_data) - 7


class TestTimeSeriesAnalyzer:
    """Test suite for time series analysis."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return TimeSeriesAnalyzer()

    @pytest.fixture
    def trend_data(self):
        """Create data with trend."""
        t = np.arange(100)
        trend = 100 + 2 * t
        seasonal = 10 * np.sin(2 * np.pi * t / 12)
        noise = np.random.normal(0, 5, 100)

        return pd.Series(
            trend + seasonal + noise, index=pd.date_range("2020-01-01", periods=100)
        )

    def test_stationarity_test(self, analyzer, trend_data):
        """Test stationarity testing."""
        results = analyzer.test_stationarity(trend_data)

        assert "adf_statistic" in results
        assert "p_value" in results
        assert "is_stationary" in results
        assert not results["is_stationary"]  # Trend data is non-stationary

    def test_autocorrelation_analysis(self, analyzer, trend_data):
        """Test autocorrelation analysis."""
        acf_values = analyzer.calculate_acf(trend_data, nlags=20)
        pacf_values = analyzer.calculate_pacf(trend_data, nlags=20)

        assert len(acf_values) == 21  # Including lag 0
        assert len(pacf_values) == 21
        assert acf_values[0] == pytest.approx(1.0)  # Lag 0 correlation

    def test_seasonality_detection(self, analyzer):
        """Test seasonality detection."""
        # Create data with known seasonality
        t = np.arange(365)
        seasonal_data = pd.Series(
            100 + 50 * np.sin(2 * np.pi * t / 7),  # Weekly seasonality
            index=pd.date_range("2020-01-01", periods=365),
        )

        seasonality = analyzer.detect_seasonality(seasonal_data)

        assert seasonality["has_seasonality"]
        assert seasonality["period"] == 7

    def test_trend_analysis(self, analyzer, trend_data):
        """Test trend extraction and analysis."""
        trend_info = analyzer.analyze_trend(trend_data)

        assert "trend_type" in trend_info
        assert "slope" in trend_info
        assert trend_info["trend_type"] == "linear"
        assert trend_info["slope"] > 0  # Positive trend

    def test_spectral_analysis(self, analyzer):
        """Test frequency domain analysis."""
        # Create data with multiple frequencies
        t = np.linspace(0, 1, 500)
        signal = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 10 * t)
        ts_data = pd.Series(signal)

        spectrum = analyzer.spectral_analysis(ts_data)

        assert "frequencies" in spectrum
        assert "power" in spectrum
        # Should detect peaks at frequencies 5 and 10
        peak_freqs = spectrum["dominant_frequencies"]
        assert len(peak_freqs) >= 2


class TestForecastModel:
    """Test suite for forecasting models."""

    @pytest.fixture
    def forecast_model(self):
        """Create forecast model instance."""
        return ForecastModel()

    @pytest.fixture
    def train_data(self):
        """Create training data."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        values = 100 + np.cumsum(np.random.normal(0, 1, 100))

        return pd.Series(values, index=dates)

    def test_arima_forecast(self, forecast_model, train_data):
        """Test ARIMA forecasting."""
        model = forecast_model.fit_arima(
            train_data, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)
        )

        forecast = forecast_model.predict(model, steps=10)

        assert len(forecast) == 10
        assert forecast.index[0] > train_data.index[-1]

    def test_prophet_forecast(self, forecast_model, train_data):
        """Test Prophet forecasting."""
        with patch("fbprophet.Prophet") as mock_prophet:
            mock_model = Mock()
            mock_prophet.return_value = mock_model
            mock_model.predict.return_value = pd.DataFrame(
                {"yhat": np.random.randn(10)}
            )

            model = forecast_model.fit_prophet(train_data)
            forecast = forecast_model.predict_prophet(model, periods=10)

            assert len(forecast) == 10

    def test_exponential_smoothing(self, forecast_model, train_data):
        """Test Exponential Smoothing."""
        model = forecast_model.fit_exponential_smoothing(
            train_data, trend="add", seasonal=None
        )

        forecast = model.forecast(steps=10)

        assert len(forecast) == 10

    def test_lstm_forecast(self, forecast_model, train_data):
        """Test LSTM neural network forecasting."""
        # Prepare data for LSTM
        X, y = forecast_model.prepare_lstm_data(train_data, lookback=10)

        assert X.shape[0] == len(train_data) - 10
        assert X.shape[1] == 10
        assert len(y) == len(X)

        # Test model creation (mock actual training)
        with patch("tensorflow.keras.models.Sequential") as mock_model:
            lstm_model = forecast_model.create_lstm_model(input_shape=(10, 1), units=50)
            mock_model.assert_called_once()

    def test_ensemble_forecast(self, forecast_model, train_data):
        """Test ensemble forecasting."""
        # Mock individual models
        models = [Mock() for _ in range(3)]
        for i, model in enumerate(models):
            model.forecast.return_value = pd.Series([100 + i] * 10)

        ensemble_forecast = forecast_model.ensemble_predict(
            models, weights=[0.3, 0.3, 0.4], steps=10
        )

        assert len(ensemble_forecast) == 10
        # Weighted average should be around 101
        assert ensemble_forecast[0] == pytest.approx(101, rel=0.01)

    def test_forecast_accuracy_metrics(self, forecast_model):
        """Test forecast accuracy metrics."""
        actual = pd.Series([100, 102, 104, 106, 108])
        predicted = pd.Series([101, 103, 103, 107, 109])

        metrics = forecast_model.calculate_accuracy_metrics(actual, predicted)

        assert "mae" in metrics
        assert "rmse" in metrics
        assert "mape" in metrics
        assert "mase" in metrics

        assert metrics["mae"] == pytest.approx(1.0, rel=0.01)


class TestAnomalyDetector:
    """Test suite for anomaly detection."""

    @pytest.fixture
    def detector(self):
        """Create anomaly detector instance."""
        return AnomalyDetector()

    @pytest.fixture
    def normal_data(self):
        """Create normal time series data."""
        t = np.arange(1000)
        data = 100 + 10 * np.sin(2 * np.pi * t / 50) + np.random.normal(0, 2, 1000)

        # Add anomalies
        data[100] = 150  # Point anomaly
        data[500:510] = 80  # Contextual anomaly

        return pd.Series(data)

    def test_statistical_anomaly_detection(self, detector, normal_data):
        """Test statistical anomaly detection methods."""
        # Z-score method
        anomalies_z = detector.detect_zscore(normal_data, threshold=3)
        assert 100 in anomalies_z

        # Moving average method
        anomalies_ma = detector.detect_moving_average(
            normal_data, window=20, threshold=3
        )
        assert len(anomalies_ma) > 0

    def test_isolation_forest(self, detector, normal_data):
        """Test Isolation Forest anomaly detection."""
        anomalies = detector.detect_isolation_forest(normal_data, contamination=0.01)

        assert len(anomalies) > 0
        assert len(anomalies) < len(normal_data) * 0.05

    def test_lstm_autoencoder(self, detector, normal_data):
        """Test LSTM autoencoder for anomaly detection."""
        with patch("tensorflow.keras.models.Sequential") as mock_model:
            # Mock the model and its methods
            mock_instance = Mock()
            mock_model.return_value = mock_instance
            mock_instance.predict.return_value = normal_data.values.reshape(-1, 1)

            anomalies = detector.detect_lstm_autoencoder(
                normal_data, sequence_length=10, threshold_percentile=95
            )

            assert isinstance(anomalies, list)

    def test_seasonal_hybrid_decomposition(self, detector):
        """Test STL decomposition for anomaly detection."""
        # Create seasonal data with anomalies
        t = np.arange(365)
        seasonal = 100 + 20 * np.sin(2 * np.pi * t / 30)
        seasonal[150:160] = 200  # Anomaly period

        ts_data = pd.Series(seasonal, index=pd.date_range("2020-01-01", periods=365))

        anomalies = detector.detect_stl_anomalies(ts_data, period=30)

        assert len(anomalies) > 0
        assert any(150 <= idx <= 160 for idx in anomalies)


class TestSeasonalDecomposer:
    """Test suite for seasonal decomposition."""

    @pytest.fixture
    def decomposer(self):
        """Create decomposer instance."""
        return SeasonalDecomposer()

    @pytest.fixture
    def seasonal_data(self):
        """Create data with trend and seasonality."""
        t = np.arange(365 * 2)
        trend = 100 + 0.1 * t
        seasonal = 20 * np.sin(2 * np.pi * t / 365) + 10 * np.sin(2 * np.pi * t / 7)
        noise = np.random.normal(0, 5, len(t))

        return pd.Series(
            trend + seasonal + noise, index=pd.date_range("2020-01-01", periods=len(t))
        )

    def test_classical_decomposition(self, decomposer, seasonal_data):
        """Test classical decomposition."""
        components = decomposer.classical_decompose(
            seasonal_data, period=7, model="additive"
        )

        assert "trend" in components
        assert "seasonal" in components
        assert "residual" in components

        # Reconstruction should match original (minus NaN values)
        reconstructed = (
            components["trend"] + components["seasonal"] + components["residual"]
        )
        valid_idx = ~np.isnan(reconstructed)
        assert np.allclose(reconstructed[valid_idx], seasonal_data[valid_idx])

    def test_stl_decomposition(self, decomposer, seasonal_data):
        """Test STL decomposition."""
        components = decomposer.stl_decompose(
            seasonal_data,
            period=7,
            seasonal=13,  # Seasonal window
        )

        assert all(key in components for key in ["trend", "seasonal", "residual"])
        assert len(components["trend"]) == len(seasonal_data)

    def test_x11_decomposition(self, decomposer, seasonal_data):
        """Test X-11 decomposition."""
        with patch("statsmodels.tsa.x13.x13_arima_analysis") as mock_x11:
            mock_x11.return_value = Mock(
                trend=seasonal_data.values,
                seasonal=np.zeros_like(seasonal_data),
                irregular=np.zeros_like(seasonal_data),
            )

            components = decomposer.x11_decompose(seasonal_data)

            assert "trend" in components
            assert "seasonal" in components

    def test_multiple_seasonality(self, decomposer):
        """Test decomposition with multiple seasonal patterns."""
        # Create data with daily and weekly seasonality
        t = np.arange(365)
        daily = 10 * np.sin(2 * np.pi * t)
        weekly = 20 * np.sin(2 * np.pi * t / 7)
        data = pd.Series(100 + daily + weekly)

        components = decomposer.decompose_multiple_seasonality(data, periods=[1, 7])

        assert "seasonal_1" in components
        assert "seasonal_7" in components


class TestTimeSeriesValidation:
    """Test suite for time series cross-validation."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        from time_series import TimeSeriesValidator

        return TimeSeriesValidator()

    def test_time_series_split(self, validator):
        """Test time series splitting."""
        data = pd.Series(range(100))

        splits = validator.time_series_split(data, n_splits=5)

        assert len(splits) == 5
        # Each split should have increasing training size
        for i in range(1, len(splits)):
            assert len(splits[i][0]) > len(splits[i - 1][0])

    def test_sliding_window_validation(self, validator):
        """Test sliding window validation."""
        data = pd.Series(range(100))

        windows = validator.sliding_window_split(
            data, train_size=50, test_size=10, step=10
        )

        for train, test in windows:
            assert len(train) == 50
            assert len(test) == 10
            assert train.index[-1] < test.index[0]

    def test_blocked_cross_validation(self, validator):
        """Test blocked time series CV."""
        data = pd.Series(range(365))

        blocks = validator.blocked_cv(data, block_size=30, n_blocks=5)

        assert len(blocks) == 5
        for train, test in blocks:
            assert len(test) == 30
