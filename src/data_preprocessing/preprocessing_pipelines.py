"""Data Preprocessing Pipelines
Robust outlier detection, imputation strategies, transformations, and augmentation
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# Advanced imputation
from fancyimpute import SoftImpute
from imblearn.combine import SMOTEENN, SMOTETomek

# Data augmentation
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from scipy import stats
from scipy.stats import boxcox
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import (
    KBinsDiscretizer,
    MinMaxScaler,
    Normalizer,
    PolynomialFeatures,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from src.utils.logging_utils import configure_pipeline_logging, get_pipeline_logger

# Time series

# Configure centralized pipeline logging
configure_pipeline_logging()
logger = get_pipeline_logger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline"""

    outlier_method: str = (
        "isolation_forest"  # iqr, zscore, isolation_forest, lof, dbscan
    )
    outlier_threshold: float = 0.05
    imputation_strategy: str = "knn"  # mean, median, mode, knn, iterative, matrix
    scaling_method: str = "standard"  # standard, minmax, robust, quantile, power
    transformation_method: str | None = None  # log, sqrt, boxcox, yeo-johnson
    feature_engineering: bool = True
    handle_imbalance: bool = False
    imbalance_strategy: str = "smote"  # smote, adasyn, random_under, tomek
    n_jobs: int = -1


class OutlierDetector:
    """Advanced outlier detection methods"""

    def __init__(self, method: str = "isolation_forest", contamination: float = 0.05):
        self.method = method
        self.contamination = contamination
        self.detector = None
        self.outlier_indices = []

    def fit_detect(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Detect outliers in data"""
        if isinstance(X, pd.DataFrame):
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            X_numeric = X[numeric_cols].values
        else:
            X_numeric = X

        if self.method == "iqr":
            return self._detect_iqr(X_numeric)
        elif self.method == "zscore":
            return self._detect_zscore(X_numeric)
        elif self.method == "isolation_forest":
            return self._detect_isolation_forest(X_numeric)
        elif self.method == "lof":
            return self._detect_lof(X_numeric)
        elif self.method == "elliptic":
            return self._detect_elliptic_envelope(X_numeric)
        elif self.method == "dbscan":
            return self._detect_dbscan(X_numeric)
        else:
            raise ValueError(f"Unknown outlier detection method: {self.method}")

    def _detect_iqr(self, X: np.ndarray) -> np.ndarray:
        """IQR-based outlier detection"""
        outliers = np.zeros(len(X), dtype=bool)

        for col_idx in range(X.shape[1]):
            col = X[:, col_idx]
            Q1 = np.percentile(col, 25)
            Q3 = np.percentile(col, 75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers |= (col < lower) | (col > upper)

        return outliers

    def _detect_zscore(self, X: np.ndarray, threshold: float = 3) -> np.ndarray:
        """Z-score based outlier detection"""
        z_scores = np.abs(stats.zscore(X, axis=0, nan_policy="omit"))
        return np.any(z_scores > threshold, axis=1)

    def _detect_isolation_forest(self, X: np.ndarray) -> np.ndarray:
        """Isolation Forest outlier detection"""
        self.detector = IsolationForest(
            contamination=self.contamination, random_state=42, n_jobs=-1
        )
        predictions = self.detector.fit_predict(X)
        return predictions == -1

    def _detect_lof(self, X: np.ndarray) -> np.ndarray:
        """Local Outlier Factor detection"""
        self.detector = LocalOutlierFactor(
            contamination=self.contamination, novelty=False
        )
        predictions = self.detector.fit_predict(X)
        return predictions == -1

    def _detect_elliptic_envelope(self, X: np.ndarray) -> np.ndarray:
        """Elliptic Envelope outlier detection"""
        self.detector = EllipticEnvelope(
            contamination=self.contamination, random_state=42
        )
        predictions = self.detector.fit_predict(X)
        return predictions == -1

    def _detect_dbscan(
        self, X: np.ndarray, eps: float = 0.5, min_samples: int = 5
    ) -> np.ndarray:
        """DBSCAN-based outlier detection"""
        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        labels = clustering.fit_predict(X)
        return labels == -1  # Noise points are labeled as -1

    def remove_outliers(
        self, df: pd.DataFrame, outlier_mask: np.ndarray
    ) -> pd.DataFrame:
        """Remove detected outliers"""
        return df[~outlier_mask].reset_index(drop=True)

    def cap_outliers(
        self,
        df: pd.DataFrame,
        lower_percentile: float = 1,
        upper_percentile: float = 99,
    ) -> pd.DataFrame:
        """Cap outliers to percentile values"""
        df_capped = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            lower = df[col].quantile(lower_percentile / 100)
            upper = df[col].quantile(upper_percentile / 100)
            df_capped[col] = df[col].clip(lower, upper)

        return df_capped

    def transform_outliers(
        self, df: pd.DataFrame, outlier_mask: np.ndarray, strategy: str = "median"
    ) -> pd.DataFrame:
        """Transform outliers using various strategies"""
        df_transformed = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if strategy == "median":
                replacement = df.loc[~outlier_mask, col].median()
            elif strategy == "mean":
                replacement = df.loc[~outlier_mask, col].mean()
            elif strategy == "mode":
                replacement = df.loc[~outlier_mask, col].mode()[0]
            else:
                replacement = np.nan

            df_transformed.loc[outlier_mask, col] = replacement

        return df_transformed


class AdvancedImputer:
    """Advanced imputation strategies for missing data"""

    def __init__(self, strategy: str = "knn", **kwargs):
        self.strategy = strategy
        self.imputer = None
        self.kwargs = kwargs

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform data with imputation"""
        if self.strategy == "simple":
            return self._simple_impute(X)
        elif self.strategy == "knn":
            return self._knn_impute(X)
        elif self.strategy == "iterative":
            return self._iterative_impute(X)
        elif self.strategy == "matrix":
            return self._matrix_factorization_impute(X)
        elif self.strategy == "forward_fill":
            return self._forward_fill(X)
        elif self.strategy == "interpolate":
            return self._interpolate(X)
        elif self.strategy == "mice":
            return self._mice_impute(X)
        else:
            raise ValueError(f"Unknown imputation strategy: {self.strategy}")

    def _simple_impute(self, X: pd.DataFrame) -> pd.DataFrame:
        """Simple imputation (mean, median, mode)"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns

        X_imputed = X.copy()

        # Numeric columns
        if len(numeric_cols) > 0:
            imputer = SimpleImputer(
                strategy=self.kwargs.get("numeric_strategy", "median")
            )
            X_imputed[numeric_cols] = imputer.fit_transform(X[numeric_cols])

        # Categorical columns
        if len(categorical_cols) > 0:
            imputer = SimpleImputer(strategy="most_frequent")
            X_imputed[categorical_cols] = imputer.fit_transform(X[categorical_cols])

        return X_imputed

    def _knn_impute(self, X: pd.DataFrame) -> pd.DataFrame:
        """KNN-based imputation"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            return X

        X_imputed = X.copy()
        imputer = KNNImputer(
            n_neighbors=self.kwargs.get("n_neighbors", 5),
            weights=self.kwargs.get("weights", "distance"),
        )

        X_imputed[numeric_cols] = imputer.fit_transform(X[numeric_cols])
        return X_imputed

    def _iterative_impute(self, X: pd.DataFrame) -> pd.DataFrame:
        """Iterative imputation (MICE-like)"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            return X

        X_imputed = X.copy()
        imputer = IterativeImputer(
            estimator=RandomForestRegressor(
                n_estimators=self.kwargs.get("n_estimators", 10), random_state=42
            ),
            max_iter=self.kwargs.get("max_iter", 10),
            random_state=42,
        )

        X_imputed[numeric_cols] = imputer.fit_transform(X[numeric_cols])
        return X_imputed

    def _matrix_factorization_impute(self, X: pd.DataFrame) -> pd.DataFrame:
        """Matrix factorization imputation"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            return X

        X_imputed = X.copy()

        # Use SoftImpute for matrix factorization
        X_numeric = X[numeric_cols].values
        X_complete = SoftImpute(
            max_iters=self.kwargs.get("max_iters", 100), verbose=False
        ).fit_transform(X_numeric)

        X_imputed[numeric_cols] = X_complete
        return X_imputed

    def _forward_fill(self, X: pd.DataFrame) -> pd.DataFrame:
        """Forward fill imputation (for time series)"""
        return X.fillna(method="ffill").fillna(method="bfill")

    def _interpolate(self, X: pd.DataFrame) -> pd.DataFrame:
        """Interpolation imputation"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_imputed = X.copy()

        for col in numeric_cols:
            X_imputed[col] = X[col].interpolate(
                method=self.kwargs.get("method", "linear"), limit_direction="both"
            )

        return X_imputed

    def _mice_impute(self, X: pd.DataFrame) -> pd.DataFrame:
        """MICE (Multiple Imputation by Chained Equations)"""
        # This is similar to iterative impute but with multiple imputations
        return self._iterative_impute(X)

    def get_missing_indicators(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create binary indicators for missing values"""
        missing_mask = X.isnull()
        missing_indicators = missing_mask.astype(int)
        missing_indicators.columns = [
            f"{col}_missing" for col in missing_indicators.columns
        ]
        return missing_indicators


class DataTransformer:
    """Advanced data transformation techniques"""

    def __init__(self, method: str = "standard"):
        self.method = method
        self.transformer = None
        self.fitted_transformers = {}

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform data"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            return X

        X_transformed = X.copy()

        if self.method == "standard":
            self.transformer = StandardScaler()
        elif self.method == "minmax":
            self.transformer = MinMaxScaler()
        elif self.method == "robust":
            self.transformer = RobustScaler()
        elif self.method == "quantile":
            self.transformer = QuantileTransformer(output_distribution="normal")
        elif self.method == "power":
            self.transformer = PowerTransformer(method="yeo-johnson")
        elif self.method == "normalize":
            self.transformer = Normalizer()
        elif self.method == "log":
            return self._log_transform(X)
        elif self.method == "sqrt":
            return self._sqrt_transform(X)
        elif self.method == "boxcox":
            return self._boxcox_transform(X)
        elif self.method == "yeojohnson":
            return self._yeojohnson_transform(X)
        elif self.method == "rank":
            return self._rank_transform(X)
        else:
            raise ValueError(f"Unknown transformation method: {self.method}")

        if self.transformer:
            X_transformed[numeric_cols] = self.transformer.fit_transform(
                X[numeric_cols]
            )

        return X_transformed

    def _log_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Log transformation"""
        X_transformed = X.copy()
        numeric_cols = X.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            # Add small constant to avoid log(0)
            min_val = X[col].min()
            if min_val <= 0:
                offset = abs(min_val) + 1
                X_transformed[col] = np.log1p(X[col] + offset)
            else:
                X_transformed[col] = np.log1p(X[col])

        return X_transformed

    def _sqrt_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Square root transformation"""
        X_transformed = X.copy()
        numeric_cols = X.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            min_val = X[col].min()
            if min_val < 0:
                offset = abs(min_val)
                X_transformed[col] = np.sqrt(X[col] + offset)
            else:
                X_transformed[col] = np.sqrt(X[col])

        return X_transformed

    def _boxcox_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Box-Cox transformation"""
        X_transformed = X.copy()
        numeric_cols = X.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            # Box-Cox requires positive values
            min_val = X[col].min()
            if min_val <= 0:
                offset = abs(min_val) + 1
                data = X[col] + offset
            else:
                data = X[col]

            transformed, lambda_param = boxcox(data.dropna())
            X_transformed.loc[data.notna(), col] = transformed
            self.fitted_transformers[col] = lambda_param

        return X_transformed

    def _yeojohnson_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Yeo-Johnson transformation"""
        X_transformed = X.copy()
        numeric_cols = X.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            transformer = PowerTransformer(method="yeo-johnson")
            X_transformed[col] = transformer.fit_transform(X[[col]])
            self.fitted_transformers[col] = transformer

        return X_transformed

    def _rank_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Rank transformation"""
        X_transformed = X.copy()
        numeric_cols = X.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            X_transformed[col] = X[col].rank(method="average") / len(X)

        return X_transformed

    def apply_pca(self, X: pd.DataFrame, n_components: float = 0.95) -> pd.DataFrame:
        """Apply PCA transformation"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_cols]

        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_numeric)

        # Create column names
        n_components_kept = X_pca.shape[1]
        pca_cols = [f"PC{i + 1}" for i in range(n_components_kept)]

        # Create DataFrame
        X_pca_df = pd.DataFrame(X_pca, columns=pca_cols, index=X.index)

        # Add explained variance info
        self.pca_explained_variance = pca.explained_variance_ratio_

        return X_pca_df


class FeatureEngineer:
    """Advanced feature engineering techniques"""

    def __init__(self):
        self.generated_features = []

    def create_polynomial_features(
        self, X: pd.DataFrame, degree: int = 2, include_bias: bool = False
    ) -> pd.DataFrame:
        """Create polynomial features"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            return X

        poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
        X_poly = poly.fit_transform(X[numeric_cols])

        # Get feature names
        feature_names = poly.get_feature_names_out(numeric_cols)

        # Create DataFrame
        X_poly_df = pd.DataFrame(X_poly, columns=feature_names, index=X.index)

        return pd.concat([X, X_poly_df.iloc[:, len(numeric_cols) :]], axis=1)

    def create_interaction_features(
        self, X: pd.DataFrame, columns: list[str] | None = None
    ) -> pd.DataFrame:
        """Create interaction features between columns"""
        if columns is None:
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            columns = numeric_cols[
                : min(5, len(numeric_cols))
            ]  # Limit to prevent explosion

        X_interactions = X.copy()

        for i, col1 in enumerate(columns):
            for col2 in columns[i + 1 :]:
                # Multiplication interaction
                interaction_name = f"{col1}_x_{col2}"
                X_interactions[interaction_name] = X[col1] * X[col2]

                # Division interaction (with protection against division by zero)
                ratio_name = f"{col1}_div_{col2}"
                X_interactions[ratio_name] = X[col1] / (X[col2] + 1e-8)

        return X_interactions

    def create_binned_features(
        self,
        X: pd.DataFrame,
        columns: list[str] = None,
        n_bins: int = 5,
        strategy: str = "quantile",
    ) -> pd.DataFrame:
        """Create binned/discretized features"""
        if columns is None:
            columns = X.select_dtypes(include=[np.number]).columns

        X_binned = X.copy()

        for col in columns:
            binner = KBinsDiscretizer(
                n_bins=n_bins, encode="ordinal", strategy=strategy
            )
            X_binned[f"{col}_binned"] = binner.fit_transform(X[[col]])

        return X_binned

    def create_statistical_features(
        self, X: pd.DataFrame, window_sizes: list[int] = None
    ) -> pd.DataFrame:
        """Create statistical aggregation features"""
        if window_sizes is None:
            window_sizes = [3, 5, 10]

        X_stats = X.copy()
        numeric_cols = X.select_dtypes(include=[np.number]).columns

        for window in window_sizes:
            for col in numeric_cols:
                # Rolling statistics
                X_stats[f"{col}_rolling_mean_{window}"] = (
                    X[col].rolling(window, min_periods=1).mean()
                )
                X_stats[f"{col}_rolling_std_{window}"] = (
                    X[col].rolling(window, min_periods=1).std()
                )
                X_stats[f"{col}_rolling_max_{window}"] = (
                    X[col].rolling(window, min_periods=1).max()
                )
                X_stats[f"{col}_rolling_min_{window}"] = (
                    X[col].rolling(window, min_periods=1).min()
                )

        return X_stats

    def create_lag_features(
        self, X: pd.DataFrame, columns: list[str] = None, lags: list[int] = None
    ) -> pd.DataFrame:
        """Create lag features for time series"""
        if lags is None:
            lags = [1, 2, 3, 7, 14, 30]

        if columns is None:
            columns = X.select_dtypes(include=[np.number]).columns

        X_lagged = X.copy()

        for col in columns:
            for lag in lags:
                X_lagged[f"{col}_lag_{lag}"] = X[col].shift(lag)

        return X_lagged

    def create_datetime_features(
        self, X: pd.DataFrame, datetime_column: str
    ) -> pd.DataFrame:
        """Extract features from datetime column"""
        X_datetime = X.copy()

        if datetime_column not in X.columns:
            return X

        # Convert to datetime if not already
        X_datetime[datetime_column] = pd.to_datetime(X[datetime_column])

        # Extract components
        dt = X_datetime[datetime_column]
        X_datetime[f"{datetime_column}_year"] = dt.dt.year
        X_datetime[f"{datetime_column}_month"] = dt.dt.month
        X_datetime[f"{datetime_column}_day"] = dt.dt.day
        X_datetime[f"{datetime_column}_dayofweek"] = dt.dt.dayofweek
        X_datetime[f"{datetime_column}_dayofyear"] = dt.dt.dayofyear
        X_datetime[f"{datetime_column}_weekofyear"] = dt.dt.isocalendar().week
        X_datetime[f"{datetime_column}_quarter"] = dt.dt.quarter
        X_datetime[f"{datetime_column}_is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
        X_datetime[f"{datetime_column}_hour"] = dt.dt.hour
        X_datetime[f"{datetime_column}_minute"] = dt.dt.minute

        # Cyclical encoding for periodic features
        X_datetime[f"{datetime_column}_month_sin"] = np.sin(
            2 * np.pi * dt.dt.month / 12
        )
        X_datetime[f"{datetime_column}_month_cos"] = np.cos(
            2 * np.pi * dt.dt.month / 12
        )
        X_datetime[f"{datetime_column}_day_sin"] = np.sin(2 * np.pi * dt.dt.day / 31)
        X_datetime[f"{datetime_column}_day_cos"] = np.cos(2 * np.pi * dt.dt.day / 31)

        return X_datetime

    def create_text_features(self, X: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Extract features from text column"""
        X_text = X.copy()

        if text_column not in X.columns:
            return X

        text_series = X[text_column].fillna("")

        # Basic text statistics
        X_text[f"{text_column}_length"] = text_series.str.len()
        X_text[f"{text_column}_word_count"] = text_series.str.split().str.len()
        X_text[f"{text_column}_unique_words"] = text_series.str.split().apply(
            lambda x: len(set(x)) if x else 0
        )
        X_text[f"{text_column}_char_count"] = text_series.str.replace(" ", "").str.len()
        X_text[f"{text_column}_digit_count"] = text_series.str.count(r"\d")
        X_text[f"{text_column}_upper_count"] = text_series.str.count(r"[A-Z]")
        X_text[f"{text_column}_special_count"] = text_series.str.count(
            r"[^a-zA-Z0-9\s]"
        )

        return X_text


class DataAugmenter:
    """Data augmentation techniques for improving model robustness"""

    def __init__(self, strategy: str = "smote"):
        self.strategy = strategy
        self.augmenter = None

    def augment_data(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Augment data to handle class imbalance"""
        if self.strategy == "smote":
            self.augmenter = SMOTE(random_state=42)
        elif self.strategy == "adasyn":
            self.augmenter = ADASYN(random_state=42)
        elif self.strategy == "borderline_smote":
            self.augmenter = BorderlineSMOTE(random_state=42)
        elif self.strategy == "random_under":
            self.augmenter = RandomUnderSampler(random_state=42)
        elif self.strategy == "tomek":
            self.augmenter = TomekLinks()
        elif self.strategy == "smoteenn":
            self.augmenter = SMOTEENN(random_state=42)
        elif self.strategy == "smotetomek":
            self.augmenter = SMOTETomek(random_state=42)
        else:
            raise ValueError(f"Unknown augmentation strategy: {self.strategy}")

        X_resampled, y_resampled = self.augmenter.fit_resample(X, y)

        return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)

    def add_noise(self, X: pd.DataFrame, noise_level: float = 0.01) -> pd.DataFrame:
        """Add Gaussian noise to numeric features"""
        X_noisy = X.copy()
        numeric_cols = X.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            std = X[col].std()
            noise = np.random.normal(0, noise_level * std, size=len(X))
            X_noisy[col] = X[col] + noise

        return X_noisy

    def mixup(
        self, X: pd.DataFrame, y: pd.Series, alpha: float = 0.2, n_samples: int = None
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Mixup augmentation"""
        if n_samples is None:
            n_samples = len(X)

        X_mixup = []
        y_mixup = []

        for _ in range(n_samples):
            # Sample two random indices
            idx1, idx2 = np.random.choice(len(X), 2, replace=False)

            # Sample lambda from Beta distribution
            lam = np.random.beta(alpha, alpha)

            # Create mixed sample
            x_mixed = lam * X.iloc[idx1].values + (1 - lam) * X.iloc[idx2].values
            y_mixed = lam * y.iloc[idx1] + (1 - lam) * y.iloc[idx2]

            X_mixup.append(x_mixed)
            y_mixup.append(y_mixed)

        X_augmented = pd.DataFrame(X_mixup, columns=X.columns)
        y_augmented = pd.Series(y_mixup)

        return pd.concat([X, X_augmented]), pd.concat([y, y_augmented])


class PreprocessingPipeline:
    """Complete preprocessing pipeline orchestrator"""

    def __init__(self, config: PreprocessingConfig = None):
        self.config = config or PreprocessingConfig()
        self.outlier_detector = OutlierDetector(
            method=config.outlier_method, contamination=config.outlier_threshold
        )
        self.imputer = AdvancedImputer(strategy=config.imputation_strategy)
        self.transformer = DataTransformer(method=config.scaling_method)
        self.feature_engineer = FeatureEngineer()
        self.augmenter = DataAugmenter(strategy=config.imbalance_strategy)

        self.pipeline_steps = []
        self.fitted = False

    def fit_transform(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        """Fit and transform the complete preprocessing pipeline"""
        X_processed = X.copy()
        y_processed = y.copy() if y is not None else None

        # Step 1: Outlier detection and handling
        logger.info("Detecting and handling outliers...")
        outlier_mask = self.outlier_detector.fit_detect(X_processed)
        X_processed = self.outlier_detector.cap_outliers(X_processed)
        self.pipeline_steps.append(("outlier_detection", outlier_mask.sum()))

        # Step 2: Missing data imputation
        logger.info("Imputing missing values...")
        missing_before = X_processed.isnull().sum().sum()
        X_processed = self.imputer.fit_transform(X_processed)
        self.pipeline_steps.append(("imputation", missing_before))

        # Step 3: Feature engineering
        if self.config.feature_engineering:
            logger.info("Engineering features...")
            n_features_before = X_processed.shape[1]

            # Add polynomial features for top numeric columns
            numeric_cols = X_processed.select_dtypes(include=[np.number]).columns[:3]
            if len(numeric_cols) > 0:
                X_processed = self.feature_engineer.create_polynomial_features(
                    X_processed, degree=2
                )

            # Add interaction features
            X_processed = self.feature_engineer.create_interaction_features(X_processed)

            n_features_after = X_processed.shape[1]
            self.pipeline_steps.append(
                ("feature_engineering", n_features_after - n_features_before)
            )

        # Step 4: Data transformation
        logger.info(f"Applying {self.config.scaling_method} transformation...")
        X_processed = self.transformer.fit_transform(X_processed)
        self.pipeline_steps.append(("transformation", self.config.scaling_method))

        # Step 5: Handle class imbalance (if target provided)
        if y_processed is not None and self.config.handle_imbalance:
            logger.info("Handling class imbalance...")
            original_size = len(X_processed)
            X_processed, y_processed = self.augmenter.augment_data(
                X_processed, y_processed
            )
            self.pipeline_steps.append(
                ("augmentation", len(X_processed) - original_size)
            )

        self.fitted = True

        return X_processed, y_processed

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted pipeline"""
        if not self.fitted:
            raise ValueError("Pipeline must be fitted before transform")

        # Apply the same transformations (without fitting)
        # This is a simplified version - in production, you'd store fitted transformers
        X_processed = X.copy()

        # Apply outlier capping
        X_processed = self.outlier_detector.cap_outliers(X_processed)

        # Apply imputation
        X_processed = self.imputer.fit_transform(
            X_processed
        )  # Should use fitted imputer

        # Apply transformation
        X_processed = self.transformer.fit_transform(
            X_processed
        )  # Should use fitted transformer

        return X_processed

    def get_pipeline_summary(self) -> dict[str, Any]:
        """Get summary of preprocessing pipeline"""
        return {
            "config": self.config.__dict__,
            "steps": self.pipeline_steps,
            "fitted": self.fitted,
        }


# Example usage
if __name__ == "__main__":
    # Create sample data with issues
    np.random.seed(42)
    n_samples = 1000

    df = pd.DataFrame(
        {
            "numeric1": np.random.normal(100, 15, n_samples),
            "numeric2": np.random.exponential(50, n_samples),
            "numeric3": np.random.uniform(0, 100, n_samples),
            "categorical": np.random.choice(["A", "B", "C"], n_samples),
            "target": np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),  # Imbalanced
        }
    )

    # Add missing values
    df.loc[np.random.choice(df.index, 100, replace=False), "numeric1"] = np.nan
    df.loc[np.random.choice(df.index, 50, replace=False), "numeric2"] = np.nan

    # Add outliers
    df.loc[np.random.choice(df.index, 20, replace=False), "numeric1"] = 500

    # Create preprocessing pipeline
    config = PreprocessingConfig(
        outlier_method="isolation_forest",
        imputation_strategy="knn",
        scaling_method="robust",
        feature_engineering=True,
        handle_imbalance=True,
        imbalance_strategy="smote",
    )

    pipeline = PreprocessingPipeline(config)

    # Fit and transform
    X = df.drop("target", axis=1)
    y = df["target"]

    X_processed, y_processed = pipeline.fit_transform(X, y)

    print(f"Original shape: {X.shape}")
    print(f"Processed shape: {X_processed.shape}")
    print(f"Original target distribution: {y.value_counts().to_dict()}")
    print(f"Processed target distribution: {y_processed.value_counts().to_dict()}")
    print(f"Pipeline summary: {pipeline.get_pipeline_summary()}")
