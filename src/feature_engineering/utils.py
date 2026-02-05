"""
Feature Engineering Utilities

Helper functions and utilities for feature engineering workflows.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Tuple, Any, Callable
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import make_scorer
import warnings

warnings.filterwarnings("ignore")
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureValidator:
    """Validate features for quality and consistency."""

    def __init__(self, checks: List[str] = None):
        """
        Initialize feature validator.

        Args:
            checks: List of checks to perform
        """
        self.checks = checks or [
            "missing_values",
            "constant_features",
            "duplicate_features",
            "high_correlation",
            "infinite_values",
            "data_types",
        ]
        self.validation_report = {}

    def validate(self, df: pd.DataFrame, target: Optional[pd.Series] = None) -> Dict:
        """
        Validate features.

        Args:
            df: Feature dataframe
            target: Optional target series

        Returns:
            Validation report
        """
        self.validation_report = {
            "n_samples": len(df),
            "n_features": len(df.columns),
            "checks": {},
        }

        for check in self.checks:
            if check == "missing_values":
                self._check_missing_values(df)
            elif check == "constant_features":
                self._check_constant_features(df)
            elif check == "duplicate_features":
                self._check_duplicate_features(df)
            elif check == "high_correlation":
                self._check_high_correlation(df)
            elif check == "infinite_values":
                self._check_infinite_values(df)
            elif check == "data_types":
                self._check_data_types(df)
            elif check == "target_leakage" and target is not None:
                self._check_target_leakage(df, target)

        return self.validation_report

    def _check_missing_values(self, df: pd.DataFrame):
        """Check for missing values."""
        missing_counts = df.isnull().sum()
        missing_pct = (missing_counts / len(df)) * 100

        problematic = missing_pct[missing_pct > 50]

        self.validation_report["checks"]["missing_values"] = {
            "total_missing": int(missing_counts.sum()),
            "features_with_missing": int((missing_counts > 0).sum()),
            "high_missing_features": problematic.to_dict(),
            "status": "warning" if len(problematic) > 0 else "ok",
        }

    def _check_constant_features(self, df: pd.DataFrame):
        """Check for constant features."""
        constant_features = []

        for col in df.columns:
            if df[col].nunique() <= 1:
                constant_features.append(col)

        self.validation_report["checks"]["constant_features"] = {
            "n_constant": len(constant_features),
            "constant_features": constant_features,
            "status": "warning" if len(constant_features) > 0 else "ok",
        }

    def _check_duplicate_features(self, df: pd.DataFrame):
        """Check for duplicate features."""
        duplicates = []

        for i, col1 in enumerate(df.columns):
            for col2 in df.columns[i + 1 :]:
                if df[col1].equals(df[col2]):
                    duplicates.append((col1, col2))

        self.validation_report["checks"]["duplicate_features"] = {
            "n_duplicates": len(duplicates),
            "duplicate_pairs": duplicates,
            "status": "warning" if len(duplicates) > 0 else "ok",
        }

    def _check_high_correlation(self, df: pd.DataFrame, threshold: float = 0.95):
        """Check for highly correlated features."""
        numeric_df = df.select_dtypes(include=[np.number])

        if len(numeric_df.columns) < 2:
            self.validation_report["checks"]["high_correlation"] = {
                "status": "skip",
                "message": "Not enough numeric features",
            }
            return

        corr_matrix = numeric_df.corr().abs()
        upper_tri = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if upper_tri[i, j] and corr_matrix.iloc[i, j] > threshold:
                    high_corr_pairs.append(
                        {
                            "feature1": corr_matrix.columns[i],
                            "feature2": corr_matrix.columns[j],
                            "correlation": corr_matrix.iloc[i, j],
                        }
                    )

        self.validation_report["checks"]["high_correlation"] = {
            "n_high_corr": len(high_corr_pairs),
            "high_corr_pairs": high_corr_pairs[:10],  # Limit output
            "threshold": threshold,
            "status": "warning" if len(high_corr_pairs) > 0 else "ok",
        }

    def _check_infinite_values(self, df: pd.DataFrame):
        """Check for infinite values."""
        numeric_df = df.select_dtypes(include=[np.number])
        inf_counts = np.isinf(numeric_df).sum()
        features_with_inf = inf_counts[inf_counts > 0].to_dict()

        self.validation_report["checks"]["infinite_values"] = {
            "total_infinite": int(inf_counts.sum()),
            "features_with_infinite": features_with_inf,
            "status": "error" if len(features_with_inf) > 0 else "ok",
        }

    def _check_data_types(self, df: pd.DataFrame):
        """Check data types."""
        type_counts = df.dtypes.value_counts().to_dict()
        type_counts = {str(k): v for k, v in type_counts.items()}

        self.validation_report["checks"]["data_types"] = {
            "type_distribution": type_counts,
            "numeric_features": len(df.select_dtypes(include=[np.number]).columns),
            "categorical_features": len(df.select_dtypes(include=["object"]).columns),
            "status": "ok",
        }

    def _check_target_leakage(self, df: pd.DataFrame, target: pd.Series):
        """Check for potential target leakage."""
        suspicious_features = []

        for col in df.select_dtypes(include=[np.number]).columns:
            correlation = df[col].corr(target)
            if abs(correlation) > 0.99:  # Nearly perfect correlation
                suspicious_features.append({"feature": col, "correlation": correlation})

        self.validation_report["checks"]["target_leakage"] = {
            "suspicious_features": suspicious_features,
            "status": "error" if len(suspicious_features) > 0 else "ok",
        }


class FeatureProfiler:
    """Profile features to understand their characteristics."""

    def profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive feature profile.

        Args:
            df: Feature dataframe

        Returns:
            Profile dataframe
        """
        profile_data = []

        for col in df.columns:
            profile = self._profile_column(df[col])
            profile["feature"] = col
            profile_data.append(profile)

        return pd.DataFrame(profile_data)

    def _profile_column(self, series: pd.Series) -> Dict:
        """Profile a single column."""
        profile = {
            "dtype": str(series.dtype),
            "n_unique": series.nunique(),
            "n_missing": series.isnull().sum(),
            "missing_pct": (series.isnull().sum() / len(series)) * 100,
        }

        if pd.api.types.is_numeric_dtype(series):
            profile.update(
                {
                    "mean": series.mean(),
                    "std": series.std(),
                    "min": series.min(),
                    "max": series.max(),
                    "q25": series.quantile(0.25),
                    "median": series.median(),
                    "q75": series.quantile(0.75),
                    "skew": series.skew(),
                    "kurtosis": series.kurtosis(),
                    "n_zeros": (series == 0).sum(),
                    "n_negative": (series < 0).sum()
                    if pd.api.types.is_numeric_dtype(series)
                    else 0,
                }
            )
        else:
            # Categorical features
            top_values = series.value_counts().head(3)
            profile.update(
                {
                    "most_frequent": top_values.index[0]
                    if len(top_values) > 0
                    else None,
                    "most_frequent_count": top_values.iloc[0]
                    if len(top_values) > 0
                    else 0,
                    "category_counts": len(series.value_counts()),
                }
            )

        return profile


class FeatureMonitor:
    """Monitor feature drift and changes over time."""

    def __init__(self):
        self.reference_stats = {}
        self.drift_history = []

    def set_reference(self, df: pd.DataFrame):
        """
        Set reference distribution for monitoring.

        Args:
            df: Reference dataframe
        """
        for col in df.select_dtypes(include=[np.number]).columns:
            self.reference_stats[col] = {
                "mean": df[col].mean(),
                "std": df[col].std(),
                "min": df[col].min(),
                "max": df[col].max(),
                "q25": df[col].quantile(0.25),
                "q75": df[col].quantile(0.75),
            }

    def detect_drift(self, df: pd.DataFrame, method: str = "psi") -> Dict:
        """
        Detect feature drift.

        Args:
            df: Current dataframe
            method: Drift detection method ('psi', 'ks', 'wasserstein')

        Returns:
            Drift report
        """
        drift_report = {
            "timestamp": pd.Timestamp.now(),
            "method": method,
            "drifted_features": [],
        }

        for col in df.select_dtypes(include=[np.number]).columns:
            if col not in self.reference_stats:
                continue

            if method == "psi":
                drift_score = self._calculate_psi(self.reference_stats[col], df[col])
                threshold = 0.2
            elif method == "ks":
                drift_score = self._calculate_ks(self.reference_stats[col], df[col])
                threshold = 0.05
            else:
                drift_score = 0
                threshold = 0

            if drift_score > threshold:
                drift_report["drifted_features"].append(
                    {"feature": col, "score": drift_score, "threshold": threshold}
                )

        self.drift_history.append(drift_report)
        return drift_report

    def _calculate_psi(self, reference_stats: Dict, current: pd.Series) -> float:
        """Calculate Population Stability Index."""
        # Simplified PSI calculation
        ref_mean = reference_stats["mean"]
        ref_std = reference_stats["std"]

        current_mean = current.mean()
        current_std = current.std()

        if ref_std == 0 or current_std == 0:
            return 0

        # Normalize differences
        mean_diff = abs(current_mean - ref_mean) / ref_std
        std_diff = abs(current_std - ref_std) / ref_std

        return mean_diff + std_diff

    def _calculate_ks(self, reference_stats: Dict, current: pd.Series) -> float:
        """Calculate Kolmogorov-Smirnov statistic."""
        from scipy import stats

        # Create synthetic reference distribution
        ref_samples = np.random.normal(
            reference_stats["mean"], reference_stats["std"], len(current)
        )

        ks_stat, p_value = stats.ks_2samp(ref_samples, current.values)
        return p_value


class FeatureEngineringPipeline:
    """End-to-end feature engineering pipeline."""

    def __init__(self, steps: List[Tuple[str, Any]]):
        """
        Initialize pipeline.

        Args:
            steps: List of (name, transformer) tuples
        """
        self.steps = steps
        self.fitted = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit all transformers in pipeline."""
        X_temp = X.copy()

        for name, transformer in self.steps:
            logger.info(f"Fitting {name}...")

            if hasattr(transformer, "fit_transform"):
                X_temp = transformer.fit_transform(X_temp, y)
            elif hasattr(transformer, "fit"):
                transformer.fit(X_temp, y)
                if hasattr(transformer, "transform"):
                    X_temp = transformer.transform(X_temp)
            else:
                raise ValueError(f"Transformer {name} must have fit method")

        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data through pipeline."""
        if not self.fitted:
            raise ValueError("Pipeline must be fitted before transform")

        X_temp = X.copy()

        for name, transformer in self.steps:
            logger.info(f"Transforming with {name}...")

            if hasattr(transformer, "transform"):
                X_temp = transformer.transform(X_temp)
            elif callable(transformer):
                X_temp = transformer(X_temp)
            else:
                raise ValueError(f"Transformer {name} must have transform method")

        return X_temp

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)


class CrossValidationFeatureEngineer:
    """Feature engineering with cross-validation."""

    def __init__(self, engineer, cv_folds: int = 5, stratified: bool = True):
        """
        Initialize CV feature engineer.

        Args:
            engineer: Feature engineering object
            cv_folds: Number of CV folds
            stratified: Use stratified splits
        """
        self.engineer = engineer
        self.cv_folds = cv_folds
        self.stratified = stratified
        self.fold_engineers = []

    def fit_transform_cv(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fit and transform with cross-validation.

        Args:
            X: Feature dataframe
            y: Target series

        Returns:
            Transformed dataframe with CV-based features
        """
        if self.stratified and y.dtype == "object":
            kf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        else:
            kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

        X_transformed_list = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            logger.info(f"Processing fold {fold + 1}/{self.cv_folds}")

            X_train_fold = X.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_train_fold = y.iloc[train_idx]

            # Clone engineer for this fold
            fold_engineer = self._clone_engineer()

            # Fit on training fold
            fold_engineer.fit(X_train_fold, y_train_fold)

            # Transform validation fold
            X_val_transformed = fold_engineer.transform(X_val_fold)
            X_val_transformed.index = val_idx

            X_transformed_list.append(X_val_transformed)
            self.fold_engineers.append(fold_engineer)

        # Combine all folds
        X_transformed = pd.concat(X_transformed_list).sort_index()

        return X_transformed

    def _clone_engineer(self):
        """Clone the feature engineer."""
        from copy import deepcopy

        return deepcopy(self.engineer)


def create_feature_engineering_report(
    df_original: pd.DataFrame,
    df_engineered: pd.DataFrame,
    y: Optional[pd.Series] = None,
) -> str:
    """
    Create comprehensive feature engineering report.

    Args:
        df_original: Original dataframe
        df_engineered: Engineered dataframe
        y: Optional target series

    Returns:
        HTML report string
    """
    report = []
    report.append("<h1>Feature Engineering Report</h1>")

    # Basic statistics
    report.append("<h2>Dataset Overview</h2>")
    report.append(f"<p>Original shape: {df_original.shape}</p>")
    report.append(f"<p>Engineered shape: {df_engineered.shape}</p>")
    report.append(
        f"<p>Features added: {df_engineered.shape[1] - df_original.shape[1]}</p>"
    )

    # Feature types
    report.append("<h2>Feature Types</h2>")
    report.append("<h3>Original</h3>")
    report.append(
        f"<p>Numeric: {len(df_original.select_dtypes(include=[np.number]).columns)}</p>"
    )
    report.append(
        f"<p>Categorical: {len(df_original.select_dtypes(include=['object']).columns)}</p>"
    )

    report.append("<h3>Engineered</h3>")
    report.append(
        f"<p>Numeric: {len(df_engineered.select_dtypes(include=[np.number]).columns)}</p>"
    )
    report.append(
        f"<p>Categorical: {len(df_engineered.select_dtypes(include=['object']).columns)}</p>"
    )

    # Validation
    validator = FeatureValidator()
    validation_report = validator.validate(df_engineered, y)

    report.append("<h2>Validation Results</h2>")
    for check_name, check_result in validation_report["checks"].items():
        status = check_result.get("status", "unknown")
        color = (
            "green" if status == "ok" else "orange" if status == "warning" else "red"
        )
        report.append(
            f"<p><span style='color:{color}'>{check_name}: {status}</span></p>"
        )

    # Feature profile sample
    profiler = FeatureProfiler()
    profile = profiler.profile(df_engineered.iloc[:, :5])  # First 5 features

    report.append("<h2>Sample Feature Profiles</h2>")
    report.append(profile.to_html())

    return "\n".join(report)


def save_feature_engineering_config(config: Dict, filepath: str):
    """
    Save feature engineering configuration.

    Args:
        config: Configuration dictionary
        filepath: Path to save configuration
    """
    import json

    with open(filepath, "w") as f:
        json.dump(config, f, indent=2, default=str)

    logger.info(f"Configuration saved to {filepath}")


def load_feature_engineering_config(filepath: str) -> Dict:
    """
    Load feature engineering configuration.

    Args:
        filepath: Path to configuration file

    Returns:
        Configuration dictionary
    """
    import json

    with open(filepath, "r") as f:
        config = json.load(f)

    logger.info(f"Configuration loaded from {filepath}")
    return config
