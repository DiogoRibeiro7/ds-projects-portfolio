# Time Series Analysis and Forecasting Toolkit

A comprehensive collection of Jupyter notebooks and utilities for time series analysis, forecasting, anomaly detection, and real-time monitoring.

## 📁 Contents

### Notebooks

1. **00_complete_time_series_example.ipynb**
   - End-to-end example demonstrating the complete workflow
   - Energy consumption forecasting case study
   - Integration of all components

2. **01_exploratory_time_series_analysis.ipynb**
   - `TimeSeriesAnalyzer`: Comprehensive EDA toolkit
   - `TimeSeriesVisualizer`: Interactive visualizations
   - Stationarity tests (ADF, KPSS)
   - Seasonality and trend detection
   - Outlier analysis
   - Missing data handling

3. **02_advanced_forecasting_models.ipynb**
   - `AdvancedForecaster`: Multi-model forecasting engine
   - Models implemented:
     - ARIMA/SARIMA
     - Prophet
     - Exponential Smoothing (Holt-Winters)
     - LSTM/BiLSTM neural networks
     - XGBoost/LightGBM
     - Ensemble methods
   - `ModelEvaluator`: Comprehensive evaluation metrics
   - Hyperparameter optimization

4. **03_model_comparison_and_ensemble.ipynb**
   - `TimeSeriesCrossValidator`: Proper time series CV
   - `ModelComparator`: Statistical comparison tests
     - Diebold-Mariano test
     - Friedman test
     - Model Confidence Set
   - `EnsembleForecaster`: Advanced ensemble methods
     - Simple/weighted averaging
     - Stacking
     - Dynamic ensembles

5. **04_real_time_forecasting_dashboard.ipynb**
   - `RealTimeForecastingDashboard`: Interactive Dash application
   - `DataStreamSimulator`: Realistic data streaming
   - `RealTimeForecaster`: Live model predictions
   - `AlertSystem`: Anomaly detection and alerts
   - Performance monitoring

6. **05_anomaly_detection_system.ipynb**
   - `AnomalyDetector`: Multi-method detection system
   - Statistical methods (Z-score, IQR, Grubbs test)
   - ML methods (Isolation Forest, LOF, One-Class SVM)
   - Time series specific (S-H-ESD, Matrix Profile)
   - Deep learning (LSTM Autoencoder)
   - `RealTimeAnomalyDetector`: Streaming anomaly detection
   - `AnomalyVisualizer`: Comprehensive visualization tools
   - Ensemble voting methods

### Utilities

**utils.py**: Comprehensive utility functions
- Data preprocessing and validation
- Feature engineering (lag, rolling, expanding features)
- Statistical tests (stationarity, normality, white noise)
- Model evaluation metrics
- Visualization helpers
- Cross-validation utilities
- Transformation functions

**anomaly_utils.py**: Specialized anomaly detection utilities
- `AnomalyGenerator`: Generate synthetic anomalies for testing
- `StatisticalTests`: Advanced statistical tests (Dixon, Modified Z-score, Peirce)
- `AnomalyFeatures`: Feature engineering for anomaly detection
- `AnomalyMetrics`: Comprehensive evaluation metrics
- `AdaptiveThreshold`: Dynamic threshold adjustment
- `AnomalyExplainer`: Explain detected anomalies

## 🚀 Quick Start

### Installation

```bash
# Install required dependencies
pip install -r requirements.txt
```

Required packages:
```
numpy
pandas
matplotlib
seaborn
plotly
dash
scikit-learn
statsmodels
prophet
tensorflow  # for LSTM models
xgboost
lightgbm
scipy
```

### Basic Usage

```python
# Import utilities
from utils import *

# Load your data
data = pd.read_csv('your_data.csv', parse_dates=['date'], index_col='date')

# Validate and preprocess
data = validate_time_series(data)
data = handle_missing_values(data, method='interpolate')

# Test for stationarity
stationarity = test_stationarity(data['value'])
print(f"Is stationary: {stationarity['conclusion']['is_stationary']}")

# Create features
data = create_time_features(data)
data = create_lag_features(data, 'value', [1, 7, 30])
data = create_rolling_features(data, 'value', [7, 30])

# Evaluate model
metrics = calculate_metrics(actual, predicted)
print(f"MAE: {metrics['mae']:.2f}")
```

## 📊 Examples

### 1. Exploratory Analysis

```python
from TimeSeriesAnalyzer import TimeSeriesAnalyzer

analyzer = TimeSeriesAnalyzer(data)
analyzer.plot_time_series()
analyzer.test_stationarity()
analyzer.detect_seasonality()
analyzer.decompose_time_series()
```

### 2. Forecasting

```python
from AdvancedForecaster import AdvancedForecaster

forecaster = AdvancedForecaster()
forecaster.fit(train_data, model='ensemble')
predictions = forecaster.predict(horizon=30)
forecaster.plot_forecast(test_data)
```

### 3. Model Comparison

```python
from ModelComparator import ModelComparator

comparator = ModelComparator()
results = comparator.compare_models(models, test_data)
comparator.plot_comparison()
```

### 4. Real-time Dashboard

```python
from RealTimeForecastingDashboard import RealTimeForecastingDashboard

dashboard = RealTimeForecastingDashboard()
dashboard.run(port=8050)
# Open browser to http://localhost:8050
```

## 🔧 Features

### Data Preprocessing
- Missing value imputation
- Outlier detection and removal
- Data resampling and aggregation
- Transformation (Box-Cox, differencing)

### Feature Engineering
- Time-based features (hour, day, month, cyclical)
- Lag features
- Rolling window statistics
- Expanding window features
- Holiday and event indicators

### Statistical Analysis
- Stationarity tests (ADF, KPSS)
- Seasonality detection
- Normality tests
- White noise tests
- Autocorrelation analysis

### Forecasting Models
- Classical: ARIMA, Exponential Smoothing
- Machine Learning: Random Forest, XGBoost, LightGBM
- Deep Learning: LSTM, BiLSTM, GRU
- Prophet for business time series
- Ensemble methods

### Model Evaluation
- Multiple metrics (MAE, RMSE, MAPE, SMAPE, MASE)
- Directional accuracy
- Residual diagnostics
- Cross-validation for time series
- Statistical comparison tests

### Visualization
- Interactive time series plots
- Decomposition plots
- ACF/PACF plots
- Forecast vs actual comparisons
- Residual analysis
- Performance dashboards

## 📈 Performance Tips

1. **Data Quality**
   - Ensure consistent frequency in your time series
   - Handle missing values before modeling
   - Check for and remove outliers if appropriate

2. **Feature Selection**
   - Don't use future information (data leakage)
   - Select relevant lag features based on ACF/PACF
   - Consider domain-specific features

3. **Model Selection**
   - Start with simple models (moving average, linear trend)
   - Use cross-validation for model selection
   - Consider ensemble methods for better performance

4. **Real-time Considerations**
   - Implement incremental learning for streaming data
   - Monitor model performance continuously
   - Set up alerts for model degradation

## 🎯 Use Cases

- **Energy Consumption Forecasting**: Predict future energy demand
- **Sales Forecasting**: Forecast product sales with seasonality
- **Stock Price Prediction**: Financial time series analysis
- **Weather Forecasting**: Temperature and precipitation prediction
- **Traffic Flow Prediction**: Urban traffic pattern analysis
- **Anomaly Detection**: Identify unusual patterns in time series

## 📝 Best Practices

1. **Always validate stationarity** before using ARIMA models
2. **Use proper time series cross-validation** (no random splits!)
3. **Check residuals** for patterns after fitting models
4. **Combine multiple models** for robust predictions
5. **Monitor performance** in production environments
6. **Update models regularly** with new data
7. **Document assumptions** and limitations

## 🔍 Troubleshooting

### Common Issues

1. **Non-stationary data**: Apply differencing or transformation
2. **Seasonality not detected**: Check frequency and use domain knowledge
3. **Poor forecast accuracy**: Try ensemble methods or more features
4. **Computational performance**: Use sampling or reduce forecast horizon
5. **Missing values**: Use appropriate imputation based on pattern

## 📚 References

- [Forecasting: Principles and Practice](https://otexts.com/fpp3/)
- [Time Series Analysis and Its Applications](https://www.stat.pitt.edu/stoffer/tsa4/)
- [Prophet Documentation](https://facebook.github.io/prophet/)
- [statsmodels Time Series](https://www.statsmodels.org/stable/user-guide.html#time-series-analysis)

## 📄 License

This project is part of the Data Science Portfolio and follows the project's licensing terms.

## 🤝 Contributing

Contributions are welcome! Please follow the project's contribution guidelines.

## 📧 Contact

For questions or support, please open an issue in the repository.