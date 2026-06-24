# 🎯 Best Practices Guide

This guide provides recommended practices for using the Data Science Portfolio effectively in production environments.

## Table of Contents
1. [Data Handling](#data-handling)
2. [Model Development](#model-development)
3. [Statistical Analysis](#statistical-analysis)
4. [Dashboard Design](#dashboard-design)
5. [Performance Optimization](#performance-optimization)
6. [Security](#security)
7. [Testing](#testing)
8. [Documentation](#documentation)

## Data Handling

### 1. Data Validation
Always validate data before processing:

```python
from modern_bank_churn.data_validator import DataValidator

validator = DataValidator()
validation_report = validator.validate(data)

if validation_report['has_issues']:
    # Handle data quality issues
    print(validation_report['issues'])
    data = validator.clean_data(data)
```

### 2. Feature Engineering
Follow these principles:

```python
# ✅ Good: Meaningful feature names
data['customer_lifetime_value'] = data['total_purchases'] * data['avg_order_value']

# ❌ Bad: Cryptic names
data['f1'] = data['col1'] * data['col2']
```

### 3. Data Versioning
Track dataset changes in a lightweight, reproducible way:

- Use immutable raw data snapshots and append-only version metadata.
- Store dataset metadata alongside each snapshot: source, schema, timestamp, owner, checksum, and quality status.
- Keep derivation logic separate from raw data; do not overwrite original source files.
- Prefer a simple manifest-based workflow before adopting heavier tools like DVC or Delta Lake.

#### Lightweight data versioning workflow

1. Save each dataset snapshot in a versioned folder or with a timestamped filename.
2. Create a manifest entry for every snapshot that records the dataset hash, row count, schema, and tags.
3. Use a small utility to compare versions and identify schema or row-count drift.
4. Keep version metadata in source control when it is small and high-value.
5. Archive or cleanup old versions according to retention policies.

```python
import hashlib
import json
from datetime import datetime
from pathlib import Path


def compute_dataframe_hash(df):
    hash_value = hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()
    return hash_value


def write_version_manifest(dataset_name, version_id, df, metadata, output_dir="data_versions"):
    manifest = {
        "dataset": dataset_name,
        "version": version_id,
        "hash": compute_dataframe_hash(df),
        "shape": list(df.shape),
        "columns": list(df.columns),
        "created_at": datetime.now().isoformat(),
        "metadata": metadata,
    }
    path = Path(output_dir) / dataset_name
    path.mkdir(parents=True, exist_ok=True)
    manifest_path = path / f"manifest_{version_id}.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path
```

#### Practical guidelines

- `dataset_name`: use a stable, descriptive identifier.
- `version_id`: use semver, dates, or incremental counters like `2026-04-14` or `v1.0`.
- `metadata`: include source file(s), extraction query, upstream dependency version, and validation results.
- For small teams, a JSON manifest plus versioned CSV/Parquet snapshots is usually enough.
- For larger projects, adopt DVC or other tooling once the lightweight workflow is stable.

## Model Development

### 1. Pipeline Structure
Always use pipelines for reproducibility:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest

# ✅ Good: Complete pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(k=20)),
    ('model', XGBClassifier())
])

# ❌ Bad: Manual steps
X_scaled = scaler.fit_transform(X)
X_selected = selector.fit_transform(X_scaled)
model.fit(X_selected, y)
```

### 2. Hyperparameter Tuning
Use systematic optimization:

```python
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0)
    }

    model = XGBClassifier(**params)
    score = cross_val_score(model, X, y, cv=5, scoring='roc_auc').mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

### 3. Model selection criteria and tradeoffs

Use a structured scoring matrix before locking a model:

- Define a primary business metric (e.g. recall for fraud detection, RMSE for forecasting).
- Add secondary constraints (latency budget, interpretability, memory, and maintenance cost).
- Compare at least two model families on the same cross-validation protocol and report the tradeoff table (quality, speed, complexity).
- Prefer simpler models when gains are within an acceptable margin and operational cost is high.

Example lightweight decision rule:

```python
candidates = [
    {"name": "LogisticRegression", "metric": 0.82, "latency_ms": 4, "train_min": 1, "interpretability": "high"},
    {"name": "XGBoost", "metric": 0.86, "latency_ms": 18, "train_min": 12, "interpretability": "medium"},
]

selected = max(
    candidates,
    key=lambda c: (c["metric"], -c["latency_ms"] if c["latency_ms"] < 25 else -999),
)
print("Selected model:", selected["name"])
```

### 4. Model Versioning
Track all models:

```python
import mlflow

mlflow.start_run()
mlflow.log_params(config.to_dict())
mlflow.log_metrics({'auc': auc, 'accuracy': accuracy})
mlflow.sklearn.log_model(model, "model")
mlflow.end_run()
```

## Statistical Analysis

### 1. Assumption Checking
Always verify assumptions:

```python
from statistical_methods.assumption_checker import AssumptionChecker

checker = AssumptionChecker()

# Before t-test
assumptions = checker.check_t_test_assumptions(group1, group2)
if not assumptions['normality']:
    # Use non-parametric test
    result = tester.mann_whitney_u(group1, group2)
else:
    result = tester.t_test(group1, group2)
```

### 2. Multiple Testing Correction
Apply corrections for multiple comparisons:

```python
from statsmodels.stats.multitest import multipletests

p_values = [0.01, 0.04, 0.03, 0.05, 0.02]
rejected, p_adjusted, _, _ = multipletests(p_values, method='fdr_bh')
```

### 3. Effect Size Reporting
Always report effect sizes with p-values:

```python
def comprehensive_test(group1, group2):
    """Perform test with effect size."""
    result = tester.t_test(group1, group2)

    # Add effect size
    from scipy.stats import cohen_d
    result['cohens_d'] = cohen_d(group1, group2)
    result['interpretation'] = interpret_effect_size(result['cohens_d'])

    return result
```

## Dashboard Design

### 1. Responsive Design
Ensure mobile compatibility:

```python
config = DashboardConfig(
    enable_mobile=True,
    responsive_breakpoints={
        'mobile': 480,
        'tablet': 768,
        'desktop': 1024
    }
)
```

### 2. Performance Optimization
Implement caching and lazy loading:

```python
from functools import lru_cache
import redis

# Redis caching for dashboard
cache = redis.Redis(host='localhost', port=6379)

@lru_cache(maxsize=100)
def get_expensive_data(filters):
    """Cache expensive computations."""
    cache_key = f"data:{hash(frozenset(filters.items()))}"

    # Check cache
    cached = cache.get(cache_key)
    if cached:
        return pd.read_json(cached)

    # Compute and cache
    data = expensive_computation(filters)
    cache.setex(cache_key, 300, data.to_json())
    return data
```

### 3. Error Handling
Graceful error handling:

```python
@dashboard.callback(
    Output('chart', 'figure'),
    Input('filter', 'value')
)
def update_chart(filter_value):
    try:
        data = get_data(filter_value)
        return create_chart(data)
    except DataNotFoundError:
        return create_empty_chart("No data available")
    except Exception as e:
        logger.error(f"Chart update failed: {e}")
        return create_error_chart("An error occurred")
```

## Performance Optimization

### 1. Parallel Processing
Utilize multiple cores:

```python
from joblib import Parallel, delayed

def process_chunk(chunk):
    # Process data chunk
    return chunk.apply(complex_function)

# Parallel processing
results = Parallel(n_jobs=-1)(
    delayed(process_chunk)(chunk)
    for chunk in np.array_split(data, 10)
)
final_result = pd.concat(results)
```

### 2. Memory Management
Handle large datasets efficiently:

```python
# Use chunking for large files
def process_large_csv(filepath, chunksize=10000):
    results = []

    for chunk in pd.read_csv(filepath, chunksize=chunksize):
        # Process chunk
        result = process_chunk(chunk)
        results.append(result)

    return pd.concat(results, ignore_index=True)
```

### 3. GPU Acceleration
Leverage GPU when available:

```python
# Check GPU availability
import torch

if torch.cuda.is_available():
    device = 'cuda'
    model = XGBClassifier(tree_method='gpu_hist', gpu_id=0)
else:
    device = 'cpu'
    model = XGBClassifier()
```

## Security

### 1. Input Validation
Always validate user inputs:

```python
from werkzeug.utils import secure_filename
import re

def validate_input(user_input):
    """Validate and sanitize user input."""
    # Check for SQL injection patterns
    sql_pattern = re.compile(r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|CREATE)\b)', re.I)
    if sql_pattern.search(user_input):
        raise ValueError("Invalid input detected")

    # Sanitize
    return secure_filename(user_input)
```

### 2. API Security
Implement proper authentication:

```python
from functools import wraps
import jwt

def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')

        if not token:
            return jsonify({'message': 'No token provided'}), 401

        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
            request.user_id = payload['user_id']
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Invalid token'}), 401

        return f(*args, **kwargs)

    return decorated_function
```

### 3. Data Privacy
Implement data anonymization:

```python
def anonymize_pii(data):
    """Remove or hash PII data."""
    import hashlib

    # Hash sensitive columns
    for col in ['ssn', 'email', 'phone']:
        if col in data.columns:
            data[col] = data[col].apply(
                lambda x: hashlib.sha256(str(x).encode()).hexdigest()
            )

    # Remove unnecessary PII
    drop_cols = ['name', 'address', 'dob']
    data = data.drop(columns=[col for col in drop_cols if col in data.columns])

    return data
```

## Testing

### 1. Unit Testing
Test individual components:

```python
import unittest
from unittest.mock import Mock, patch

class TestMLPipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = MLPipelineOrchestrator(config)

    def test_feature_selection(self):
        X, y = make_classification(n_samples=100, n_features=20)
        selected = self.pipeline.select_features(X, y, k=10)
        self.assertEqual(len(selected), 10)

    @patch('pipeline.load_data')
    def test_data_loading(self, mock_load):
        mock_load.return_value = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        data = self.pipeline.load_data('test.csv')
        self.assertEqual(len(data), 2)
```

### 2. Integration Testing
Test component interactions:

```python
def test_end_to_end_pipeline():
    """Test complete pipeline flow."""
    # Load test data
    data = load_test_data()

    # Run pipeline
    results = orchestrator.run_pipeline(data)

    # Assertions
    assert results.best_model is not None
    assert results.metrics['auc_roc'] > 0.5
    assert len(results.selected_features) > 0
```

### 3. Performance Testing
Monitor performance metrics:

```python
import time
import memory_profiler

@memory_profiler.profile
def test_memory_usage():
    """Profile memory usage."""
    data = generate_large_dataset(1000000)
    model = train_model(data)
    return model

def test_inference_speed():
    """Test prediction speed."""
    model = load_model()
    X_test = generate_test_data(1000)

    start = time.time()
    predictions = model.predict(X_test)
    elapsed = time.time() - start

    assert elapsed < 1.0  # Should predict 1000 samples in < 1 second
```

## Documentation

### 1. Code Documentation
Use clear docstrings:

```python
def calculate_metrics(y_true, y_pred, sample_weight=None):
    """
    Calculate comprehensive evaluation metrics.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels.
    y_pred : array-like of shape (n_samples,)
        Predicted binary labels.
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights for weighted metrics.

    Returns
    -------
    dict
        Dictionary containing:
        - accuracy: float
            Classification accuracy
        - precision: float
            Precision score
        - recall: float
            Recall score
        - f1: float
            F1 score
        - auc_roc: float
            Area under ROC curve

    Examples
    --------
    >>> y_true = [0, 1, 1, 0]
    >>> y_pred = [0, 1, 0, 0]
    >>> metrics = calculate_metrics(y_true, y_pred)
    >>> print(f"Accuracy: {metrics['accuracy']:.2f}")
    Accuracy: 0.75

    Notes
    -----
    For imbalanced datasets, consider using sample_weight or
    stratified metrics for better evaluation.

    References
    ----------
    .. [1] Scikit-learn metrics documentation
           https://scikit-learn.org/stable/modules/model_evaluation.html
    """
    # Implementation
    pass
```

### 2. API Documentation
Document all endpoints:

```python
@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict customer churn probability.

    ---
    tags:
      - Prediction
    parameters:
      - in: body
        name: customer_data
        description: Customer features for prediction
        required: true
        schema:
          type: object
          properties:
            age:
              type: integer
              example: 35
            balance:
              type: number
              example: 50000.00
            tenure:
              type: integer
              example: 24
    responses:
      200:
        description: Successful prediction
        schema:
          type: object
          properties:
            probability:
              type: number
              example: 0.73
            prediction:
              type: integer
              example: 1
            risk_level:
              type: string
              example: "High"
      400:
        description: Invalid input
      500:
        description: Server error
    """
    # Implementation
    pass
```

### 3. Change Documentation
Maintain changelog:

```markdown
# Changelog

## [1.2.0] - 2024-01-15
### Added
- Real-time dashboard updates via WebSocket
- Bayesian hyperparameter optimization
- GPU acceleration support

### Changed
- Improved feature selection algorithm
- Updated to XGBoost 2.0

### Fixed
- Memory leak in data preprocessing
- Dashboard rendering on mobile devices

### Deprecated
- Legacy API endpoints (will be removed in 2.0)
```

## Summary

Following these best practices will help you:

1. ✅ Build robust, production-ready models
2. ✅ Ensure statistical validity
3. ✅ Create performant dashboards
4. ✅ Maintain secure applications
5. ✅ Write maintainable code

Remember:
- **Test early, test often**
- **Document as you code**
- **Profile before optimizing**
- **Validate all assumptions**
- **Security is not optional**

For more specific guidance, see:
- [API Reference](api/index.md)
- [Usage Guide](usage.md)
- [Development Guide](contributor/development.md)
