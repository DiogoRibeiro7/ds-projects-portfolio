# Feature Engineering System

A comprehensive automated feature engineering framework with advanced transformation, selection, and time series capabilities.

## 📁 Contents

### Core Modules

1. **automated_feature_creator.py**
   - Intelligent feature type detection
   - Automated feature generation
   - Statistical, interaction, and polynomial features
   - Text and temporal feature extraction
   - Deep feature synthesis with Featuretools
   - Target encoding strategies
   - Integrated feature selection

2. **feature_transformers.py**
   - Outlier detection and handling
   - Skewness correction
   - Dimensionality reduction (PCA, ICA, NMF, Autoencoder)
   - Clustering-based features
   - Interaction transformations
   - Binning strategies
   - Target transformation
   - Feature augmentation

3. **time_series_features.py**
   - Statistical time series features
   - Frequency domain analysis
   - Entropy-based features
   - Autocorrelation and stationarity
   - Seasonal decomposition
   - Window-based features
   - Change point detection
   - TSFresh integration

4. **feature_selection.py**
   - Variance-based selection
   - Correlation analysis
   - Mutual information
   - Tree-based importance
   - L1 regularization (Lasso)
   - Recursive Feature Elimination
   - Boruta algorithm
   - SHAP-based selection
   - Forward/Backward selection
   - Genetic algorithms

5. **utils.py**
   - Feature validation
   - Feature profiling
   - Drift detection
   - Pipeline management
   - Cross-validation engineering
   - Reporting utilities

### Notebooks

- **feature_engineering_demo.ipynb**: Comprehensive demonstration of all capabilities

## 🚀 Quick Start

### Installation

```bash
# Core dependencies
pip install pandas numpy scikit-learn scipy

# Advanced features (optional)
pip install featuretools category_encoders
pip install tsfresh statsmodels
pip install shap boruta lightgbm
pip install tensorflow  # For autoencoder

# Visualization
pip install matplotlib seaborn plotly
```

### Basic Usage

```python
from src.feature_engineering.automated_feature_creator import AutomatedFeatureEngineer

# Initialize engineer
engineer = AutomatedFeatureEngineer(
    task_type='classification',
    max_features=100,
    verbosity=1
)

# Engineer features
X_engineered = engineer.engineer_features(
    df,
    target=y,
    include_interactions=True,
    include_polynomial=True,
    include_aggregates=True
)

# Transform new data
X_new_engineered = engineer.transform(X_new)
```

## 📊 Feature Engineering Capabilities

### 1. Automated Feature Creation

```python
# Automatic type detection and feature generation
engineer = AutomatedFeatureEngineer(task_type='regression')

# Comprehensive feature engineering
X_engineered = engineer.engineer_features(
    df,
    target=y,
    entity_col='customer_id',  # For aggregations
    time_col='timestamp',       # For temporal features
    include_interactions=True,   # Feature interactions
    include_polynomial=True,     # Polynomial features
    include_aggregates=True,     # Group statistics
    include_deep_features=True   # Featuretools
)
```

**Automatic Features:**
- Missing value indicators and statistics
- Statistical features (mean, std, skew, kurtosis)
- Row-wise aggregations
- Interaction features (multiply, divide, ratios)
- Polynomial combinations
- Temporal features (hour, day, season, lags)
- Text statistics (length, word count, patterns)
- Group-based aggregations
- Target encoding

### 2. Advanced Transformations

```python
from src.feature_engineering.feature_transformers import (
    OutlierTransformer,
    SkewnessTransformer,
    DimensionalityReducer,
    ClusteringTransformer
)

# Handle outliers
outlier_transformer = OutlierTransformer(method='iqr', threshold=1.5)
X_clean = outlier_transformer.fit_transform(X)

# Correct skewness
skew_transformer = SkewnessTransformer(threshold=0.5, method='boxcox')
X_normal = skew_transformer.fit_transform(X)

# Reduce dimensions
reducer = DimensionalityReducer(method='pca', n_components=50)
X_reduced = reducer.fit_transform(X)

# Add cluster features
clusterer = ClusteringTransformer(method='kmeans', n_clusters=5)
X_clustered = clusterer.fit_transform(X)
```

### 3. Time Series Features

```python
from src.feature_engineering.time_series_features import TimeSeriesFeatureExtractor

# Extract comprehensive time series features
ts_extractor = TimeSeriesFeatureExtractor(
    window_sizes=[5, 10, 20],
    include_statistical=True,
    include_frequency=True,
    include_entropy=True,
    include_autocorrelation=True
)

features = ts_extractor.extract_features(
    df,
    time_col='timestamp',
    value_cols=['value1', 'value2'],
    entity_col='entity_id'
)
```

**Time Series Features:**
- Rolling statistics (mean, std, min, max)
- Frequency domain (FFT, spectral features)
- Entropy measures (Shannon, approximate, sample)
- Autocorrelation and partial autocorrelation
- Stationarity tests (ADF, KPSS)
- Lag features and differences
- Seasonal decomposition
- Change point detection

### 4. Feature Selection

```python
from src.feature_engineering.feature_selection import AdvancedFeatureSelector

# Multi-method selection
selector = AdvancedFeatureSelector(
    task_type='classification',
    selection_methods=['mutual_info', 'importance', 'lasso', 'boruta'],
    max_features=50,
    cv_folds=5
)

# Select best features
X_selected, scores = selector.fit_select(X, y, return_scores=True)
```

**Selection Methods:**
- Statistical tests (chi-square, ANOVA)
- Mutual information
- Tree-based importance
- L1/L2 regularization
- Recursive Feature Elimination
- Boruta algorithm
- SHAP values
- Forward/Backward selection
- Genetic algorithms

## 🔧 Advanced Features

### Feature Validation

```python
from src.feature_engineering.utils import FeatureValidator

validator = FeatureValidator()
report = validator.validate(X_engineered, y)

# Check for issues
print(f"Missing values: {report['checks']['missing_values']['status']}")
print(f"Constant features: {report['checks']['constant_features']['n_constant']}")
print(f"Duplicate features: {report['checks']['duplicate_features']['n_duplicates']}")
print(f"High correlations: {report['checks']['high_correlation']['n_high_corr']}")
```

### Feature Profiling

```python
from src.feature_engineering.utils import FeatureProfiler

profiler = FeatureProfiler()
profile = profiler.profile(X_engineered)

# View statistics for each feature
print(profile[['feature', 'dtype', 'mean', 'std', 'skew', 'missing_pct']])
```

### Drift Detection

```python
from src.feature_engineering.utils import FeatureMonitor

monitor = FeatureMonitor()
monitor.set_reference(X_train)

# Check for drift in new data
drift_report = monitor.detect_drift(X_new, method='psi')
print(f"Drifted features: {len(drift_report['drifted_features'])}")
```

### Pipeline Creation

```python
from src.feature_engineering.utils import FeatureEngineringPipeline
from src.feature_engineering.feature_transformers import create_transformation_pipeline

# Create pipeline
pipeline = FeatureEngineringPipeline([
    ('outlier', OutlierTransformer()),
    ('skewness', SkewnessTransformer()),
    ('interactions', InteractionTransformer()),
    ('selection', AdvancedFeatureSelector(max_features=100))
])

# Fit and transform
X_transformed = pipeline.fit_transform(X, y)
```

### Cross-Validation Engineering

```python
from src.feature_engineering.utils import CrossValidationFeatureEngineer

# Engineer features with CV to prevent overfitting
cv_engineer = CrossValidationFeatureEngineer(
    engineer=AutomatedFeatureEngineer(),
    cv_folds=5,
    stratified=True
)

X_cv_engineered = cv_engineer.fit_transform_cv(X, y)
```

## 📈 Performance Optimization

### Memory Optimization
- Use sparse matrices for high-dimensional data
- Process in chunks for large datasets
- Select features early to reduce dimensionality
- Use efficient data types (int8, float32)

### Speed Optimization
- Parallel processing for independent features
- Caching intermediate results
- Vectorized operations with NumPy
- Early stopping in selection algorithms

### Quality Optimization
- Cross-validation for robust features
- Multiple selection methods with voting
- Validation checks for data quality
- Monitoring for feature drift

## 🎯 Use Cases

### Classification Tasks
```python
engineer = AutomatedFeatureEngineer(task_type='classification')
X_engineered = engineer.engineer_features(X, y)
```

### Regression Tasks
```python
engineer = AutomatedFeatureEngineer(task_type='regression')
target_transformer = TargetTransformer(method='auto')
y_transformed = target_transformer.fit_transform(y)
```

### Time Series Forecasting
```python
ts_extractor = TimeSeriesFeatureExtractor()
features = ts_extractor.extract_features(df, 'date', ['sales'])
```

### Text Classification
```python
# Automatic text feature extraction
X_text = engineer.engineer_features(
    df_text,
    include_interactions=False,
    include_polynomial=False
)
```

## 📊 Feature Importance Analysis

```python
from feature_selection import FeatureImportanceAnalyzer

analyzer = FeatureImportanceAnalyzer()
importance_df = analyzer.analyze(X, y, models=['rf', 'gb', 'lr'])

# Get consensus importance
top_features = importance_df.nlargest(20, 'mean')
print(top_features[['mean', 'std', 'rank']])
```

## 🔍 Best Practices

1. **Start Simple**: Begin with basic features before complex transformations
2. **Validate Features**: Check for leakage, duplicates, and constants
3. **Use Domain Knowledge**: Incorporate business understanding
4. **Monitor Drift**: Track feature distributions over time
5. **Document Features**: Keep clear descriptions of engineered features
6. **Test Robustness**: Validate on hold-out data
7. **Consider Interpretability**: Balance complexity with explainability

## 📝 Configuration Management

```python
from src.feature_engineering.utils import save_feature_engineering_config, load_feature_engineering_config

# Save configuration
config = {
    'task_type': 'classification',
    'max_features': 100,
    'methods': ['statistical', 'interactions', 'polynomial'],
    'selection': ['mutual_info', 'importance']
}
save_feature_engineering_config(config, 'feature_config.json')

# Load configuration
config = load_feature_engineering_config('feature_config.json')
```

## 🚨 Common Issues and Solutions

### High Memory Usage
- Reduce polynomial degree
- Limit interaction features
- Use sparse matrices
- Process in batches

### Slow Processing
- Reduce window sizes for time series
- Limit deep feature synthesis depth
- Use sampling for large datasets
- Parallelize where possible

### Poor Feature Quality
- Check for data leakage
- Validate feature distributions
- Use cross-validation
- Combine multiple selection methods

### Overfitting
- Use regularization in selection
- Apply cross-validation engineering
- Limit feature complexity
- Monitor validation performance

## 📚 References

- [Feature Engineering for Machine Learning](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)
- [Featuretools Documentation](https://featuretools.alteryx.com/en/stable/)
- [TSFresh Documentation](https://tsfresh.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Boruta Algorithm](https://github.com/scikit-learn-contrib/boruta_py)

## 📄 License

This project is part of the Data Science Portfolio and follows the project's licensing terms.

## 🤝 Contributing

Contributions are welcome! Please follow the project's contribution guidelines.

## 📧 Contact

For questions or support, please open an issue in the repository.
