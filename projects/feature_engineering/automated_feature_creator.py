"""Automated Feature Engineering System

Comprehensive automated feature creation with multiple strategies and techniques.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    SelectKBest,
    chi2,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.preprocessing import (
    PolynomialFeatures,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
)

warnings.filterwarnings("ignore")

# Optional imports with graceful fallback
try:
    from category_encoders import CatBoostEncoder, OneHotEncoder, TargetEncoder

    CATEGORY_ENCODERS_AVAILABLE = True
except ImportError:
    CATEGORY_ENCODERS_AVAILABLE = False

try:
    import featuretools as ft

    FEATURETOOLS_AVAILABLE = True
except ImportError:
    FEATURETOOLS_AVAILABLE = False

try:
    from tsfresh import extract_features, select_features
    from tsfresh.utilities.dataframe_functions import impute

    TSFRESH_AVAILABLE = True
except ImportError:
    TSFRESH_AVAILABLE = False

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutomatedFeatureEngineer:
    """Comprehensive automated feature engineering system."""

    def __init__(
        self,
        task_type: str = "classification",
        max_features: int | None = None,
        feature_selection_method: str = "mutual_info",
        verbosity: int = 1,
    ):
        """Initialize automated feature engineer.

        Args:
            task_type: 'classification' or 'regression'
            max_features: Maximum number of features to select
            feature_selection_method: Method for feature selection
            verbosity: Logging level (0=silent, 1=info, 2=debug)
        """
        self.task_type = task_type
        self.max_features = max_features
        self.feature_selection_method = feature_selection_method
        self.verbosity = verbosity
        self.feature_metadata = {}
        self.transformers = {}
        self.encoders = {}
        self.feature_importance = {}

    def engineer_features(
        self,
        df: pd.DataFrame,
        target: pd.Series | None = None,
        entity_col: str | None = None,
        time_col: str | None = None,
        include_interactions: bool = True,
        include_polynomial: bool = True,
        include_aggregates: bool = True,
        include_deep_features: bool = False,
    ) -> pd.DataFrame:
        """Main feature engineering pipeline.

        Args:
            df: Input dataframe
            target: Target variable for supervised feature engineering
            entity_col: Column identifying entities for aggregation
            time_col: Time column for temporal features
            include_interactions: Create interaction features
            include_polynomial: Create polynomial features
            include_aggregates: Create aggregate features
            include_deep_features: Use Featuretools for deep feature synthesis

        Returns:
            DataFrame with engineered features
        """
        if self.verbosity >= 1:
            logger.info("Starting automated feature engineering...")
            logger.info(f"Initial shape: {df.shape}")

        # Store original columns
        original_cols = df.columns.tolist()

        # Detect feature types
        feature_types = self._detect_feature_types(df)
        self.feature_metadata["feature_types"] = feature_types

        if self.verbosity >= 1:
            logger.info(
                f"Detected feature types: {self._summarize_types(feature_types)}"
            )

        # Basic features
        df_features = self._create_basic_features(df, feature_types)

        # Missing value features
        df_features = self._create_missing_value_features(df_features, original_cols)

        # Statistical features
        df_features = self._create_statistical_features(df_features, feature_types)

        # Interaction features
        if include_interactions and len(feature_types["numeric"]) > 1:
            df_features = self._create_interaction_features(df_features, feature_types)

        # Polynomial features
        if include_polynomial and len(feature_types["numeric"]) > 0:
            df_features = self._create_polynomial_features(df_features, feature_types)

        # Time-based features
        if time_col:
            df_features = self._create_temporal_features(df_features, time_col)

        # Text features
        if feature_types["text"]:
            df_features = self._create_text_features(df_features, feature_types["text"])

        # Aggregate features
        if include_aggregates and entity_col:
            df_features = self._create_aggregate_features(
                df_features, entity_col, feature_types
            )

        # Deep features using Featuretools
        if include_deep_features and FEATURETOOLS_AVAILABLE and entity_col:
            df_features = self._create_deep_features(df_features, entity_col, time_col)

        # Target encoding for categorical
        if target is not None and feature_types["categorical"]:
            df_features = self._apply_target_encoding(
                df_features, target, feature_types
            )

        # Transform features
        df_features = self._apply_transformations(df_features, feature_types)

        # Feature selection
        if self.max_features and target is not None:
            df_features = self._select_best_features(df_features, target)

        # Generate feature importance
        if target is not None:
            self._calculate_feature_importance(df_features, target)

        if self.verbosity >= 1:
            logger.info(f"Final shape: {df_features.shape}")
            logger.info(f"Created {df_features.shape[1] - df.shape[1]} new features")

        return df_features

    def _detect_feature_types(self, df: pd.DataFrame) -> dict:
        """Automatically detect feature types."""
        feature_types = {
            "numeric": [],
            "categorical": [],
            "datetime": [],
            "text": [],
            "binary": [],
            "constant": [],
            "id": [],
        }

        for col in df.columns:
            # Check for constant columns
            if df[col].nunique() <= 1:
                feature_types["constant"].append(col)
                continue

            # Check for ID columns
            if df[col].nunique() == len(df) and df[col].dtype in ["int64", "object"]:
                feature_types["id"].append(col)
                continue

            # Numeric types
            if pd.api.types.is_numeric_dtype(df[col]):
                if df[col].nunique() == 2:
                    feature_types["binary"].append(col)
                elif df[col].nunique() < 20 and df[col].dtype in ["int64", "int32"]:
                    feature_types["categorical"].append(col)
                else:
                    feature_types["numeric"].append(col)

            # Datetime types
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                feature_types["datetime"].append(col)

            # Object types
            elif df[col].dtype == "object":
                # Try to detect datetime strings
                sample = df[col].dropna().iloc[:5]
                try:
                    pd.to_datetime(sample)
                    feature_types["datetime"].append(col)
                except:
                    # Check if text or categorical
                    avg_length = df[col].astype(str).str.len().mean()
                    unique_ratio = df[col].nunique() / len(df)

                    if avg_length > 50 or unique_ratio > 0.9:
                        feature_types["text"].append(col)
                    else:
                        feature_types["categorical"].append(col)

        return feature_types

    def _create_basic_features(
        self, df: pd.DataFrame, feature_types: dict
    ) -> pd.DataFrame:
        """Create basic statistical features."""
        df = df.copy()

        # Numeric features
        for col in feature_types["numeric"]:
            # Log transformation
            if (df[col] > 0).all():
                df[f"{col}_log"] = np.log1p(df[col])

            # Square and square root
            df[f"{col}_squared"] = df[col] ** 2
            if (df[col] >= 0).all():
                df[f"{col}_sqrt"] = np.sqrt(df[col])

            # Reciprocal
            df[f"{col}_reciprocal"] = 1 / (df[col] + 1e-8)

            # Binning
            df[f"{col}_bin"] = pd.qcut(df[col], q=5, labels=False, duplicates="drop")

        return df

    def _create_missing_value_features(
        self, df: pd.DataFrame, original_cols: list[str]
    ) -> pd.DataFrame:
        """Create features from missing values."""
        df = df.copy()

        # Missing value indicators
        for col in original_cols:
            if df[col].isnull().any():
                df[f"{col}_is_missing"] = df[col].isnull().astype(int)

        # Count of missing values per row
        df["missing_count"] = df[original_cols].isnull().sum(axis=1)
        df["missing_ratio"] = df["missing_count"] / len(original_cols)

        return df

    def _create_statistical_features(
        self, df: pd.DataFrame, feature_types: dict
    ) -> pd.DataFrame:
        """Create statistical aggregation features."""
        df = df.copy()
        numeric_cols = feature_types["numeric"]

        if len(numeric_cols) > 1:
            # Row-wise statistics
            df["row_mean"] = df[numeric_cols].mean(axis=1)
            df["row_std"] = df[numeric_cols].std(axis=1)
            df["row_max"] = df[numeric_cols].max(axis=1)
            df["row_min"] = df[numeric_cols].min(axis=1)
            df["row_range"] = df["row_max"] - df["row_min"]
            df["row_median"] = df[numeric_cols].median(axis=1)
            df["row_sum"] = df[numeric_cols].sum(axis=1)

            # Coefficient of variation
            df["row_cv"] = df["row_std"] / (df["row_mean"] + 1e-8)

            # Skewness and kurtosis
            df["row_skew"] = df[numeric_cols].skew(axis=1)
            df["row_kurtosis"] = df[numeric_cols].kurtosis(axis=1)

        return df

    def _create_interaction_features(
        self, df: pd.DataFrame, feature_types: dict, max_interactions: int = 50
    ) -> pd.DataFrame:
        """Create interaction features between numeric variables."""
        df = df.copy()
        numeric_cols = feature_types["numeric"][:10]  # Limit to prevent explosion

        interaction_count = 0
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i + 1 :]:
                if interaction_count >= max_interactions:
                    break

                # Multiplication
                df[f"{col1}_times_{col2}"] = df[col1] * df[col2]

                # Division (with protection against division by zero)
                df[f"{col1}_div_{col2}"] = df[col1] / (df[col2].replace(0, np.nan))
                df[f"{col2}_div_{col1}"] = df[col2] / (df[col1].replace(0, np.nan))

                # Difference
                df[f"{col1}_minus_{col2}"] = df[col1] - df[col2]

                # Ratio
                df[f"{col1}_ratio_{col2}"] = df[col1] / (df[col1] + df[col2] + 1e-8)

                interaction_count += 5

        return df

    def _create_polynomial_features(
        self, df: pd.DataFrame, feature_types: dict, degree: int = 2
    ) -> pd.DataFrame:
        """Create polynomial features."""
        df = df.copy()
        numeric_cols = feature_types["numeric"][:5]  # Limit to prevent explosion

        if not numeric_cols:
            return df

        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly_features = poly.fit_transform(df[numeric_cols])

        # Get feature names
        feature_names = poly.get_feature_names_out(numeric_cols)

        # Add to dataframe
        poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df.index)

        # Only keep interaction terms (not original features)
        new_cols = [col for col in poly_df.columns if col not in numeric_cols]
        df = pd.concat([df, poly_df[new_cols]], axis=1)

        return df

    def _create_temporal_features(
        self, df: pd.DataFrame, time_col: str
    ) -> pd.DataFrame:
        """Create time-based features."""
        df = df.copy()

        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col])

        # Basic temporal features
        df[f"{time_col}_year"] = df[time_col].dt.year
        df[f"{time_col}_month"] = df[time_col].dt.month
        df[f"{time_col}_day"] = df[time_col].dt.day
        df[f"{time_col}_dayofweek"] = df[time_col].dt.dayofweek
        df[f"{time_col}_hour"] = df[time_col].dt.hour
        df[f"{time_col}_minute"] = df[time_col].dt.minute
        df[f"{time_col}_dayofyear"] = df[time_col].dt.dayofyear
        df[f"{time_col}_weekofyear"] = df[time_col].dt.isocalendar().week
        df[f"{time_col}_quarter"] = df[time_col].dt.quarter

        # Cyclical encoding
        df[f"{time_col}_month_sin"] = np.sin(2 * np.pi * df[time_col].dt.month / 12)
        df[f"{time_col}_month_cos"] = np.cos(2 * np.pi * df[time_col].dt.month / 12)
        df[f"{time_col}_day_sin"] = np.sin(2 * np.pi * df[time_col].dt.day / 31)
        df[f"{time_col}_day_cos"] = np.cos(2 * np.pi * df[time_col].dt.day / 31)
        df[f"{time_col}_hour_sin"] = np.sin(2 * np.pi * df[time_col].dt.hour / 24)
        df[f"{time_col}_hour_cos"] = np.cos(2 * np.pi * df[time_col].dt.hour / 24)
        df[f"{time_col}_dayofweek_sin"] = np.sin(
            2 * np.pi * df[time_col].dt.dayofweek / 7
        )
        df[f"{time_col}_dayofweek_cos"] = np.cos(
            2 * np.pi * df[time_col].dt.dayofweek / 7
        )

        # Boolean features
        df[f"{time_col}_is_weekend"] = (
            df[time_col].dt.dayofweek.isin([5, 6]).astype(int)
        )
        df[f"{time_col}_is_month_start"] = df[time_col].dt.is_month_start.astype(int)
        df[f"{time_col}_is_month_end"] = df[time_col].dt.is_month_end.astype(int)
        df[f"{time_col}_is_quarter_start"] = df[time_col].dt.is_quarter_start.astype(
            int
        )
        df[f"{time_col}_is_quarter_end"] = df[time_col].dt.is_quarter_end.astype(int)
        df[f"{time_col}_is_year_start"] = df[time_col].dt.is_year_start.astype(int)
        df[f"{time_col}_is_year_end"] = df[time_col].dt.is_year_end.astype(int)

        # Time since epoch
        df[f"{time_col}_timestamp"] = df[time_col].astype(np.int64) // 10**9

        # Lag features
        df = df.sort_values(time_col)
        df[f"{time_col}_diff"] = df[time_col].diff().dt.total_seconds()
        df[f"{time_col}_diff_hours"] = df[f"{time_col}_diff"] / 3600
        df[f"{time_col}_diff_days"] = df[f"{time_col}_diff"] / 86400

        return df

    def _create_text_features(
        self, df: pd.DataFrame, text_cols: list[str]
    ) -> pd.DataFrame:
        """Create features from text columns."""
        df = df.copy()

        for col in text_cols:
            # Basic text statistics
            df[f"{col}_length"] = df[col].astype(str).str.len()
            df[f"{col}_word_count"] = df[col].astype(str).str.split().str.len()
            df[f"{col}_unique_words"] = (
                df[col].astype(str).apply(lambda x: len(set(str(x).split())))
            )

            # Character counts
            df[f"{col}_digit_count"] = df[col].astype(str).str.count(r"\d")
            df[f"{col}_upper_count"] = df[col].astype(str).str.count(r"[A-Z]")
            df[f"{col}_lower_count"] = df[col].astype(str).str.count(r"[a-z]")
            df[f"{col}_space_count"] = df[col].astype(str).str.count(r"\s")
            df[f"{col}_punctuation_count"] = df[col].astype(str).str.count(r"[^\w\s]")

            # Ratios
            df[f"{col}_digit_ratio"] = df[f"{col}_digit_count"] / (
                df[f"{col}_length"] + 1
            )
            df[f"{col}_upper_ratio"] = df[f"{col}_upper_count"] / (
                df[f"{col}_length"] + 1
            )
            df[f"{col}_space_ratio"] = df[f"{col}_space_count"] / (
                df[f"{col}_length"] + 1
            )

            # Average word length
            df[f"{col}_avg_word_length"] = df[f"{col}_length"] / (
                df[f"{col}_word_count"] + 1
            )

        return df

    def _create_aggregate_features(
        self, df: pd.DataFrame, entity_col: str, feature_types: dict
    ) -> pd.DataFrame:
        """Create aggregate features grouped by entity."""
        df = df.copy()
        numeric_cols = feature_types["numeric"]

        if not numeric_cols:
            return df

        # Group statistics
        for col in numeric_cols[:10]:  # Limit to prevent explosion
            # Mean encoding
            group_mean = df.groupby(entity_col)[col].mean()
            df[f"{col}_by_{entity_col}_mean"] = df[entity_col].map(group_mean)

            # Std encoding
            group_std = df.groupby(entity_col)[col].std()
            df[f"{col}_by_{entity_col}_std"] = df[entity_col].map(group_std)

            # Count encoding
            group_count = df.groupby(entity_col)[col].count()
            df[f"{col}_by_{entity_col}_count"] = df[entity_col].map(group_count)

            # Min/Max encoding
            group_min = df.groupby(entity_col)[col].min()
            group_max = df.groupby(entity_col)[col].max()
            df[f"{col}_by_{entity_col}_min"] = df[entity_col].map(group_min)
            df[f"{col}_by_{entity_col}_max"] = df[entity_col].map(group_max)
            df[f"{col}_by_{entity_col}_range"] = (
                df[f"{col}_by_{entity_col}_max"] - df[f"{col}_by_{entity_col}_min"]
            )

            # Relative features
            df[f"{col}_to_group_mean_ratio"] = df[col] / (
                df[f"{col}_by_{entity_col}_mean"] + 1e-8
            )
            df[f"{col}_to_group_mean_diff"] = (
                df[col] - df[f"{col}_by_{entity_col}_mean"]
            )

        return df

    def _create_deep_features(
        self, df: pd.DataFrame, entity_col: str, time_col: str | None = None
    ) -> pd.DataFrame:
        """Create deep features using Featuretools."""
        if not FEATURETOOLS_AVAILABLE:
            logger.warning(
                "Featuretools not available. Skipping deep feature synthesis."
            )
            return df

        try:
            # Create entity set
            es = ft.EntitySet()

            # Add entity
            es = es.add_dataframe(
                dataframe_name="main",
                dataframe=df,
                index=entity_col,
                time_index=time_col if time_col else None,
            )

            # Run deep feature synthesis
            feature_matrix, feature_defs = ft.dfs(
                entityset=es,
                target_dataframe_name="main",
                agg_primitives=[
                    "sum",
                    "mean",
                    "max",
                    "min",
                    "std",
                    "count",
                    "percent_true",
                ],
                trans_primitives=[
                    "add_numeric",
                    "multiply_numeric",
                    "divide_numeric",
                    "subtract_numeric",
                    "percentile",
                ],
                max_depth=2,
                n_jobs=-1,
            )

            # Add feature definitions to metadata
            self.feature_metadata["deep_features"] = [str(f) for f in feature_defs]

            return feature_matrix

        except Exception as e:
            logger.warning(f"Deep feature synthesis failed: {e}")
            return df

    def _apply_target_encoding(
        self, df: pd.DataFrame, target: pd.Series, feature_types: dict
    ) -> pd.DataFrame:
        """Apply target encoding to categorical features."""
        df = df.copy()

        if not CATEGORY_ENCODERS_AVAILABLE:
            logger.warning(
                "category_encoders not available. Using simple mean encoding."
            )

            for col in feature_types["categorical"]:
                # Simple mean encoding
                mean_target = df.groupby(col)[target.name].mean()
                df[f"{col}_target_encoded"] = df[col].map(mean_target)

                # Count encoding
                count_values = df[col].value_counts()
                df[f"{col}_count_encoded"] = df[col].map(count_values)

            return df

        # Use advanced encoding
        for col in feature_types["categorical"]:
            # Target encoding
            encoder = TargetEncoder(cols=[col], return_df=True)
            df_encoded = encoder.fit_transform(df[[col]], target)
            df[f"{col}_target_encoded"] = df_encoded[col]

            # Store encoder for later use
            self.encoders[f"{col}_target"] = encoder

            # CatBoost encoding (handles high cardinality well)
            if df[col].nunique() > 20:
                cb_encoder = CatBoostEncoder(cols=[col], return_df=True)
                df_cb = cb_encoder.fit_transform(df[[col]], target)
                df[f"{col}_catboost_encoded"] = df_cb[col]
                self.encoders[f"{col}_catboost"] = cb_encoder

        return df

    def _apply_transformations(
        self, df: pd.DataFrame, feature_types: dict
    ) -> pd.DataFrame:
        """Apply various transformations to features."""
        df = df.copy()
        numeric_cols = feature_types["numeric"]

        if not numeric_cols:
            return df

        # Power transformation (Yeo-Johnson handles negative values)
        power_transformer = PowerTransformer(method="yeo-johnson", standardize=True)
        df_power = pd.DataFrame(
            power_transformer.fit_transform(df[numeric_cols]),
            columns=[f"{col}_power" for col in numeric_cols],
            index=df.index,
        )
        df = pd.concat([df, df_power], axis=1)
        self.transformers["power"] = power_transformer

        # Quantile transformation
        quantile_transformer = QuantileTransformer(output_distribution="uniform")
        df_quantile = pd.DataFrame(
            quantile_transformer.fit_transform(df[numeric_cols]),
            columns=[f"{col}_quantile" for col in numeric_cols],
            index=df.index,
        )
        df = pd.concat([df, df_quantile], axis=1)
        self.transformers["quantile"] = quantile_transformer

        # Robust scaling
        robust_scaler = RobustScaler()
        df_robust = pd.DataFrame(
            robust_scaler.fit_transform(df[numeric_cols]),
            columns=[f"{col}_robust" for col in numeric_cols],
            index=df.index,
        )
        df = pd.concat([df, df_robust], axis=1)
        self.transformers["robust"] = robust_scaler

        return df

    def _select_best_features(
        self, df: pd.DataFrame, target: pd.Series
    ) -> pd.DataFrame:
        """Select best features using various methods."""
        # Remove non-numeric columns for selection
        numeric_df = df.select_dtypes(include=[np.number])

        # Handle missing values
        numeric_df = numeric_df.fillna(numeric_df.median())

        # Replace infinite values
        numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
        numeric_df = numeric_df.fillna(numeric_df.median())

        if self.feature_selection_method == "mutual_info":
            if self.task_type == "classification":
                selector = SelectKBest(
                    mutual_info_classif, k=min(self.max_features, numeric_df.shape[1])
                )
            else:
                selector = SelectKBest(
                    mutual_info_regression,
                    k=min(self.max_features, numeric_df.shape[1]),
                )

        elif self.feature_selection_method == "f_test":
            if self.task_type == "classification":
                selector = SelectKBest(
                    f_classif, k=min(self.max_features, numeric_df.shape[1])
                )
            else:
                selector = SelectKBest(
                    f_regression, k=min(self.max_features, numeric_df.shape[1])
                )

        elif self.feature_selection_method == "chi2":
            # Chi2 requires non-negative features
            numeric_df_positive = numeric_df - numeric_df.min() + 1
            selector = SelectKBest(chi2, k=min(self.max_features, numeric_df.shape[1]))
            X_selected = selector.fit_transform(numeric_df_positive, target)

        elif self.feature_selection_method == "rfe":
            if self.task_type == "classification":
                estimator = RandomForestClassifier(
                    n_estimators=50, random_state=42, n_jobs=-1
                )
            else:
                estimator = RandomForestRegressor(
                    n_estimators=50, random_state=42, n_jobs=-1
                )

            selector = RFE(
                estimator,
                n_features_to_select=min(self.max_features, numeric_df.shape[1]),
            )

        elif self.feature_selection_method == "tree":
            if self.task_type == "classification":
                estimator = RandomForestClassifier(
                    n_estimators=100, random_state=42, n_jobs=-1
                )
            else:
                estimator = RandomForestRegressor(
                    n_estimators=100, random_state=42, n_jobs=-1
                )

            selector = SelectFromModel(estimator, max_features=self.max_features)

        else:
            raise ValueError(
                f"Unknown feature selection method: {self.feature_selection_method}"
            )

        # Fit selector
        if self.feature_selection_method != "chi2":
            X_selected = selector.fit_transform(numeric_df, target)

        # Get selected feature names
        selected_features = numeric_df.columns[selector.get_support()].tolist()

        # Store metadata
        self.feature_metadata["selected_features"] = selected_features
        self.feature_metadata["n_features_selected"] = len(selected_features)

        if hasattr(selector, "scores_"):
            self.feature_metadata["feature_scores"] = dict(
                zip(numeric_df.columns, selector.scores_, strict=False)
            )

        if self.verbosity >= 1:
            logger.info(
                f"Selected {len(selected_features)} features from {numeric_df.shape[1]}"
            )

        return df[selected_features]

    def _calculate_feature_importance(self, df: pd.DataFrame, target: pd.Series):
        """Calculate feature importance using tree-based model."""
        # Remove non-numeric columns
        numeric_df = df.select_dtypes(include=[np.number])

        # Handle missing values
        numeric_df = numeric_df.fillna(numeric_df.median())

        # Train model for feature importance
        if self.task_type == "classification":
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

        model.fit(numeric_df, target)

        # Store feature importance
        self.feature_importance = dict(
            zip(numeric_df.columns, model.feature_importances_, strict=False)
        )

        # Sort by importance
        self.feature_importance = dict(
            sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        )

    def _summarize_types(self, feature_types: dict) -> str:
        """Create summary string of feature types."""
        summary = []
        for type_name, features in feature_types.items():
            if features:
                summary.append(f"{type_name}={len(features)}")
        return ", ".join(summary)

    def get_feature_importance(self) -> dict:
        """Return feature importance scores."""
        return self.feature_importance

    def get_feature_metadata(self) -> dict:
        """Return feature engineering metadata."""
        return self.feature_metadata

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted encoders and transformers.

        Args:
            df: New dataframe to transform

        Returns:
            Transformed dataframe
        """
        df_transformed = df.copy()

        # Apply encoders
        for name, encoder in self.encoders.items():
            col = name.split("_")[0]
            if col in df.columns:
                if "target" in name:
                    df_transformed[f"{col}_target_encoded"] = encoder.transform(
                        df[[col]]
                    )[col]
                elif "catboost" in name:
                    df_transformed[f"{col}_catboost_encoded"] = encoder.transform(
                        df[[col]]
                    )[col]

        # Apply transformers
        feature_types = self.feature_metadata.get("feature_types", {})
        numeric_cols = feature_types.get("numeric", [])

        if "power" in self.transformers and numeric_cols:
            df_power = pd.DataFrame(
                self.transformers["power"].transform(df[numeric_cols]),
                columns=[f"{col}_power" for col in numeric_cols],
                index=df.index,
            )
            df_transformed = pd.concat([df_transformed, df_power], axis=1)

        if "quantile" in self.transformers and numeric_cols:
            df_quantile = pd.DataFrame(
                self.transformers["quantile"].transform(df[numeric_cols]),
                columns=[f"{col}_quantile" for col in numeric_cols],
                index=df.index,
            )
            df_transformed = pd.concat([df_transformed, df_quantile], axis=1)

        if "robust" in self.transformers and numeric_cols:
            df_robust = pd.DataFrame(
                self.transformers["robust"].transform(df[numeric_cols]),
                columns=[f"{col}_robust" for col in numeric_cols],
                index=df.index,
            )
            df_transformed = pd.concat([df_transformed, df_robust], axis=1)

        # Select same features if feature selection was applied
        if "selected_features" in self.feature_metadata:
            selected_features = self.feature_metadata["selected_features"]
            available_features = [
                f for f in selected_features if f in df_transformed.columns
            ]
            df_transformed = df_transformed[available_features]

        return df_transformed
