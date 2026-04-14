# Feature Store Usage

A feature store centralizes feature generation, storage, and reuse for machine learning systems. It helps enforce consistency between offline training and online serving while tracking feature versioning and metadata.

## Why use a feature store?

- **Consistency**: Ensures the same feature logic is applied in training and production.
- **Reusability**: Shares features across models and teams.
- **Versioning**: Stores feature sets with metadata so you can reproduce experiments.
- **Serving**: Supports online and batch access to precomputed feature values.
- **Governance**: Tracks feature lineage, ownership, and quality.

## Core feature store concepts

- **Feature set**: A collection of related features derived from one or more data sources.
- **Feature version**: A timestamped or semantic version of a saved feature set.
- **Metadata**: Information about feature definitions, data sources, schema, and generation logic.
- **Online store**: Low-latency storage for serving features in production.
- **Offline store**: Batch storage used for training and backtesting.

## Lightweight feature store pattern

For projects that are not yet ready for a full managed feature store, use a lightweight, folder-based store that captures both features and metadata.

### Recommended workflow

1. Define feature engineering pipelines that produce a stable feature schema.
2. Save each feature set as a versioned artifact, e.g. `features_20260414_123456/`.
3. Store metadata alongside the feature dataset, including source, creation time, schema, and calculation notes.
4. Load the feature set by ID for training and evaluation.
5. Use a separate manifest or registry file to look up available feature versions.

## Example: local feature store usage

The `modern-bank-churn` project includes a simple `FeatureStore` implementation in `modern-bank-churn/mlops_pipeline.py`.

```python
from modern_bank_churn.mlops_pipeline import FeatureStore

feature_store = FeatureStore(base_path="data/feature_store")

metadata = {
    "dataset": "churn_features",
    "description": "Customer churn prediction features",
    "source": "raw_customers, raw_transactions",
    "created_by": "data_scientist",
}

feature_set_id = feature_store.save_features(df_features, metadata)
print(f"Saved feature set: {feature_set_id}")

# Later, load the latest features for training / scoring
latest_features = feature_store.get_latest_features()
```

### What the example stores

- `features.parquet` for the actual feature values
- `metadata.json` describing the feature set contents
- a global `metadata.json` index that tracks all saved versions

## When to adopt a managed feature store

Use a managed feature store platform once your project needs:

- Online feature serving with strict latency targets
- Multi-team feature sharing and discovery
- Complex feature pipelines with incremental updates
- Embedded monitoring and data freshness checks

Common managed feature store tools:

- Feast
- Tecton
- Hopsworks
- Databricks Feature Store

## Practical guidance for this portfolio

- Start with a simple local versioned store for prototypes.
- Keep raw data immutable and derive feature sets in a reproducible pipeline.
- Document feature definitions clearly in metadata.
- Use consistent naming conventions and avoid feature name collisions.
- When possible, separate feature generation, storage, and model training into distinct steps.

## Related resources

- `modern-bank-churn/mlops_pipeline.py` — local `FeatureStore` implementation
- `recommendation-system/README.md` — production notes on using feature stores for recommender features
- `docs/methodology/evaluation_metrics_standardization.md` — metrics standardization for model evaluation
