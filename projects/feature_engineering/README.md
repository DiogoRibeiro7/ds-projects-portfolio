# Feature Engineering Utilities

This folder contains portfolio examples for automated feature creation,
transformations, time-series feature extraction, feature selection, and
feature-quality checks. Treat the code as reference material for review and
local experimentation, not as a packaged feature-engineering product.

## Contents

- `automated_feature_creator.py`: `AutomatedFeatureEngineer` for type-aware
  feature generation, interactions, polynomial features, temporal fields, text
  summaries, aggregations, optional Featuretools integration, and target
  encoding.
- `feature_transformers.py`: scikit-learn-style transformers for outlier
  handling, skew correction, dimensionality reduction, clustering features,
  interactions, binning, target transforms, and simple augmentation.
- `time_series_features.py`: rolling, lag, frequency, entropy, stationarity,
  seasonal, window, and change-point feature examples.
- `feature_selection.py`: feature selection helpers using statistical tests,
  mutual information, model importance, regularization, RFE, optional Boruta,
  optional SHAP, and search-based methods.
- `utils.py`: compatibility shim for shared feature-engineering utilities in
  `src.feature_engineering.utils`.
- `feature_engineering_demo.ipynb`: notebook walkthrough for the project
  modules.

## Quick Start

Install the core scientific stack first:

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn
```

Some paths use optional libraries when they are installed:

```bash
pip install featuretools category_encoders tsfresh statsmodels shap boruta lightgbm
```

Run examples from this project folder, or add `projects/feature_engineering` to
`PYTHONPATH` before importing the modules.

```python
from automated_feature_creator import AutomatedFeatureEngineer

engineer = AutomatedFeatureEngineer(
    task_type="classification",
    max_features=100,
    verbosity=1,
)

X_engineered = engineer.engineer_features(
    df,
    target=y,
    include_interactions=True,
    include_polynomial=True,
    include_aggregates=True,
)
```

## Example Components

### Transformations

```python
from feature_transformers import (
    ClusteringTransformer,
    DimensionalityReducer,
    OutlierTransformer,
    SkewnessTransformer,
)

X_clean = OutlierTransformer(method="iqr", threshold=1.5).fit_transform(X)
X_normal = SkewnessTransformer(threshold=0.5, method="boxcox").fit_transform(X_clean)
X_reduced = DimensionalityReducer(method="pca", n_components=20).fit_transform(X_normal)
X_clustered = ClusteringTransformer(method="kmeans", n_clusters=5).fit_transform(X_reduced)
```

### Time-Series Features

```python
from time_series_features import TimeSeriesFeatureExtractor

extractor = TimeSeriesFeatureExtractor(
    window_sizes=[5, 10, 20],
    include_statistical=True,
    include_frequency=True,
    include_entropy=True,
    include_autocorrelation=True,
)

features = extractor.extract_features(
    df,
    time_col="timestamp",
    value_cols=["value"],
    entity_col="entity_id",
)
```

### Feature Selection

```python
from feature_selection import AdvancedFeatureSelector

selector = AdvancedFeatureSelector(
    task_type="classification",
    selection_methods=["mutual_info", "importance", "lasso"],
    max_features=50,
    cv_folds=5,
)

X_selected, scores = selector.fit_select(X, y, return_scores=True)
```

## Review Notes

- The modules are intentionally broad because they demonstrate feature
  engineering patterns rather than a single production pipeline.
- Optional integrations degrade gracefully when dependencies are missing.
- Validate leakage, target encoding, and time-aware splits before reusing these
  patterns on real data.
- Prefer the notebook for a quick portfolio review and the module files for
  implementation details.
