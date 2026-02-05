"""
Enhanced Feature Engineering for Bank Churn Prediction.
Comprehensive feature engineering with automated selection, interaction detection,
stability metrics, and proper target encoding.
"""

import warnings
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    SelectKBest,
    chi2,
    f_classif,
    mutual_info_classif,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler

warnings.filterwarnings("ignore")


@dataclass
class FeatureImportanceResult:
    """Container for comprehensive feature importance analysis."""

    method: str
    scores: pd.DataFrame
    selected_features: List[str]
    stability_score: float
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    feature_clusters: Dict[str, List[str]] = field(default_factory=dict)


class EnhancedFeatureSelector:
    """
    Enhanced automated feature selection with multiple methods,
    stability analysis, and ensemble voting.
    """

    def __init__(
        self,
        methods: List[str] = None,
        k: Union[int, float] = 0.5,
        importance_threshold: float = 0.01,
        stability_selection: bool = True,
        n_bootstrap: int = 100,
        consensus_threshold: float = 0.5,
    ):
        """
        Initialize enhanced feature selector.

        Args:
            methods: List of selection methods
            k: Number or proportion of features to select
            importance_threshold: Minimum importance threshold
            stability_selection: Whether to use stability selection
            n_bootstrap: Number of bootstrap samples for stability
            consensus_threshold: Threshold for consensus selection
        """
        if methods is None:
            methods = [
                "mutual_info",
                "chi2",
                "anova",
                "random_forest",
                "gradient_boost",
                "lasso",
                "recursive_elimination",
                "boruta",
            ]

        self.methods = methods
        self.k = k
        self.importance_threshold = importance_threshold
        self.stability_selection = stability_selection
        self.n_bootstrap = n_bootstrap
        self.consensus_threshold = consensus_threshold
        self.feature_scores_ = {}
        self.selected_features_ = {}
        self.stability_scores_ = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EnhancedFeatureSelector":
        """Fit feature selector with stability analysis."""

        # Determine number of features to select
        if isinstance(self.k, float) and 0 < self.k <= 1:
            n_features = int(self.k * X.shape[1])
        else:
            n_features = min(self.k, X.shape[1])

        # Run each selection method
        for method in self.methods:
            if self.stability_selection:
                scores, selected = self._stable_feature_selection(
                    X, y, method, n_features
                )
                self.stability_scores_[method] = self._calculate_stability(X, y, method)
            else:
                scores, selected = self._single_feature_selection(
                    X, y, method, n_features
                )
                self.stability_scores_[method] = 1.0  # Perfect stability for single run

            self.feature_scores_[method] = scores
            self.selected_features_[method] = selected

        # Ensemble selection based on consensus
        self._ensemble_selection()

        return self

    def _stable_feature_selection(
        self, X: pd.DataFrame, y: pd.Series, method: str, n_features: int
    ) -> Tuple[pd.Series, List[str]]:
        """Perform stable feature selection using bootstrap."""
        n_samples = len(X)
        feature_counts = pd.Series(0, index=X.columns)
        feature_scores_list = []

        for _ in range(self.n_bootstrap):
            # Bootstrap sample
            idx = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X.iloc[idx]
            y_boot = y.iloc[idx]

            # Get features from single selection
            scores, selected = self._single_feature_selection(
                X_boot, y_boot, method, n_features
            )

            # Update counts
            feature_counts[selected] += 1
            feature_scores_list.append(scores)

        # Average scores
        avg_scores = pd.concat(feature_scores_list, axis=1).mean(axis=1)

        # Select features that appear in at least threshold proportion of bootstraps
        selection_freq = feature_counts / self.n_bootstrap
        selected_features = selection_freq[
            selection_freq >= self.consensus_threshold
        ].index.tolist()

        return avg_scores, selected_features

    def _single_feature_selection(
        self, X: pd.DataFrame, y: pd.Series, method: str, n_features: int
    ) -> Tuple[pd.Series, List[str]]:
        """Perform single feature selection based on method."""

        if method == "mutual_info":
            scores = mutual_info_classif(X, y, random_state=42)
            scores_series = pd.Series(scores, index=X.columns)

        elif method == "chi2":
            # Only for non-negative features
            non_neg_cols = X.columns[(X >= 0).all()]
            if len(non_neg_cols) > 0:
                scores, _ = chi2(X[non_neg_cols], y)
                scores_series = pd.Series(scores, index=non_neg_cols)
            else:
                scores_series = pd.Series(0, index=X.columns)

        elif method == "anova":
            scores, _ = f_classif(X, y)
            scores_series = pd.Series(scores, index=X.columns)

        elif method == "random_forest":
            rf = RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1, max_depth=5
            )
            rf.fit(X, y)
            scores_series = pd.Series(rf.feature_importances_, index=X.columns)

        elif method == "gradient_boost":
            from sklearn.ensemble import GradientBoostingClassifier

            gb = GradientBoostingClassifier(
                n_estimators=100, random_state=42, max_depth=3
            )
            gb.fit(X, y)
            scores_series = pd.Series(gb.feature_importances_, index=X.columns)

        elif method == "lasso":
            from sklearn.linear_model import LassoCV
            from sklearn.preprocessing import StandardScaler

            X_scaled = StandardScaler().fit_transform(X)
            lasso = LassoCV(cv=5, random_state=42, n_jobs=-1)
            lasso.fit(X_scaled, y)
            scores_series = pd.Series(np.abs(lasso.coef_), index=X.columns)

        elif method == "recursive_elimination":
            rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            rfe = RFE(rf, n_features_to_select=n_features)
            rfe.fit(X, y)
            scores_series = pd.Series(rfe.ranking_, index=X.columns)
            # Invert ranking (lower is better in RFE)
            scores_series = 1 / scores_series

        elif method == "boruta":
            scores_series = self._boruta_selection(X, y)

        else:
            raise ValueError(f"Unknown method: {method}")

        # Select top features
        selected = scores_series.nlargest(n_features).index.tolist()

        return scores_series, selected

    def _boruta_selection(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Perform Boruta-like shadow feature selection."""
        # Create shadow features
        X_shadow = X.apply(np.random.permutation)
        X_shadow.columns = [f"shadow_{col}" for col in X.columns]
        X_combined = pd.concat([X, X_shadow], axis=1)

        # Train random forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_combined, y)

        # Get importance scores
        importance = pd.Series(rf.feature_importances_, index=X_combined.columns)

        # Compare with shadow max
        shadow_max = importance[
            [col for col in importance.index if "shadow_" in col]
        ].max()
        original_importance = importance[
            [col for col in importance.index if "shadow_" not in col]
        ]

        # Features that beat shadow max
        selected_mask = original_importance > shadow_max

        return original_importance

    def _calculate_stability(self, X: pd.DataFrame, y: pd.Series, method: str) -> float:
        """Calculate stability score for feature selection method."""
        selections = []
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(X, y):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]

            _, selected = self._single_feature_selection(
                X_train,
                y_train,
                method,
                int(self.k * X.shape[1]) if isinstance(self.k, float) else self.k,
            )
            selections.append(set(selected))

        # Calculate Jaccard similarity between all pairs
        similarities = []
        for i in range(len(selections)):
            for j in range(i + 1, len(selections)):
                intersection = len(selections[i] & selections[j])
                union = len(selections[i] | selections[j])
                if union > 0:
                    similarities.append(intersection / union)

        return np.mean(similarities) if similarities else 0.0

    def _ensemble_selection(self):
        """Perform ensemble selection based on multiple methods."""
        # Count how many methods selected each feature
        feature_votes = pd.Series(0, index=self.feature_scores_[self.methods[0]].index)

        for method in self.methods:
            if method in self.selected_features_:
                for feature in self.selected_features_[method]:
                    if feature in feature_votes.index:
                        feature_votes[feature] += 1

        # Select features based on consensus
        n_methods = len(self.methods)
        min_votes = int(n_methods * self.consensus_threshold)
        self.selected_features_["ensemble"] = feature_votes[
            feature_votes >= min_votes
        ].index.tolist()

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data to selected features."""
        return X[self.selected_features_["ensemble"]]

    def get_feature_report(self) -> pd.DataFrame:
        """Get comprehensive feature importance report."""
        report = pd.DataFrame()

        for method in self.methods:
            if method in self.feature_scores_:
                scores = self.feature_scores_[method]
                report[f"{method}_score"] = scores
                report[f"{method}_selected"] = scores.index.isin(
                    self.selected_features_[method]
                )

        report["ensemble_selected"] = report.index.isin(
            self.selected_features_["ensemble"]
        )

        # Add stability scores
        stability_df = pd.DataFrame.from_dict(
            self.stability_scores_, orient="index", columns=["stability"]
        )

        return report, stability_df


class FeatureInteractionDetector:
    """Detect and create feature interactions."""

    def __init__(
        self,
        max_interactions: int = 50,
        interaction_types: List[str] = None,
        correlation_threshold: float = 0.1,
        importance_threshold: float = 0.01,
    ):
        """
        Initialize feature interaction detector.

        Args:
            max_interactions: Maximum number of interactions to create
            interaction_types: Types of interactions to detect
            correlation_threshold: Minimum correlation for interaction
            importance_threshold: Minimum importance for interaction
        """
        if interaction_types is None:
            interaction_types = ["multiply", "divide", "add", "subtract", "polynomial"]

        self.max_interactions = max_interactions
        self.interaction_types = interaction_types
        self.correlation_threshold = correlation_threshold
        self.importance_threshold = importance_threshold
        self.interactions_ = []
        self.interaction_importance_ = {}

    def fit(self, X: pd.DataFrame, y: pd.Series, feature_list: List[str] = None):
        """
        Detect important interactions.

        Args:
            X: Feature DataFrame
            y: Target Series
            feature_list: List of features to consider (None for all)
        """
        if feature_list is None:
            feature_list = X.columns.tolist()

        # Limit to top features if too many
        if len(feature_list) > 20:
            # Use mutual information to select top features
            mi_scores = mutual_info_classif(X[feature_list], y, random_state=42)
            top_features_idx = np.argsort(mi_scores)[-20:]
            feature_list = [feature_list[i] for i in top_features_idx]

        interactions_scores = []

        # Test pairwise interactions
        for feat1, feat2 in combinations(feature_list, 2):
            if feat1 != feat2:
                interaction_score = self._test_interaction(X, y, feat1, feat2)
                if interaction_score > self.importance_threshold:
                    interactions_scores.append(
                        {
                            "feature1": feat1,
                            "feature2": feat2,
                            "score": interaction_score,
                            "type": self._determine_best_interaction(
                                X, y, feat1, feat2
                            ),
                        }
                    )

        # Sort by score and keep top interactions
        interactions_scores = sorted(
            interactions_scores, key=lambda x: x["score"], reverse=True
        )[: self.max_interactions]

        self.interactions_ = interactions_scores
        return self

    def _test_interaction(
        self, X: pd.DataFrame, y: pd.Series, feat1: str, feat2: str
    ) -> float:
        """Test importance of interaction between two features."""
        # Create interaction features
        interactions = pd.DataFrame()
        interactions["multiply"] = X[feat1] * X[feat2]

        # Avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            interactions["divide"] = X[feat1] / (X[feat2] + 1e-10)
            interactions["divide"].fillna(0, inplace=True)

        interactions["add"] = X[feat1] + X[feat2]
        interactions["subtract"] = X[feat1] - X[feat2]

        # Calculate mutual information for each interaction
        scores = []
        for col in interactions.columns:
            mi = mutual_info_classif(interactions[[col]], y, random_state=42)[0]
            scores.append(mi)

        return max(scores)

    def _determine_best_interaction(
        self, X: pd.DataFrame, y: pd.Series, feat1: str, feat2: str
    ) -> str:
        """Determine the best type of interaction."""
        interactions = {
            "multiply": X[feat1] * X[feat2],
            "divide": X[feat1] / (X[feat2] + 1e-10),
            "add": X[feat1] + X[feat2],
            "subtract": X[feat1] - X[feat2],
        }

        best_type = None
        best_score = -np.inf

        for int_type, int_values in interactions.items():
            mi = mutual_info_classif(
                int_values.values.reshape(-1, 1), y, random_state=42
            )[0]
            if mi > best_score:
                best_score = mi
                best_type = int_type

        return best_type

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features."""
        X_new = X.copy()

        for interaction in self.interactions_:
            feat1 = interaction["feature1"]
            feat2 = interaction["feature2"]
            int_type = interaction["type"]

            if int_type == "multiply":
                X_new[f"{feat1}*{feat2}"] = X[feat1] * X[feat2]
            elif int_type == "divide":
                X_new[f"{feat1}/{feat2}"] = X[feat1] / (X[feat2] + 1e-10)
            elif int_type == "add":
                X_new[f"{feat1}+{feat2}"] = X[feat1] + X[feat2]
            elif int_type == "subtract":
                X_new[f"{feat1}-{feat2}"] = X[feat1] - X[feat2]

        return X_new


class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Proper target encoding with regularization and cross-validation
    to prevent overfitting.
    """

    def __init__(
        self,
        columns: List[str] = None,
        smoothing: float = 1.0,
        min_samples_leaf: int = 1,
        cv: int = 5,
        hierarchy: Dict[str, str] = None,
    ):
        """
        Initialize target encoder.

        Args:
            columns: Columns to encode (None for all categorical)
            smoothing: Smoothing parameter for regularization
            min_samples_leaf: Minimum samples to trust category average
            cv: Number of CV folds for encoding
            hierarchy: Column hierarchy for hierarchical encoding
        """
        self.columns = columns
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.cv = cv
        self.hierarchy = hierarchy
        self.encodings_ = {}
        self.global_mean_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TargetEncoder":
        """Fit target encoder with cross-validation."""
        self.global_mean_ = y.mean()

        # Identify categorical columns if not specified
        if self.columns is None:
            self.columns = X.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

        # Fit encoding for each column
        for col in self.columns:
            self.encodings_[col] = self._fit_column(X[col], y)

        return self

    def _fit_column(self, X_col: pd.Series, y: pd.Series) -> Dict[Any, float]:
        """Fit encoding for a single column using CV."""
        encoding = {}

        # Use cross-validation to prevent overfitting
        kf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(X_col, y):
            X_train, y_train = X_col.iloc[train_idx], y.iloc[train_idx]
            X_val = X_col.iloc[val_idx]

            # Calculate encoding on training fold
            for category in X_val.unique():
                if category not in encoding:
                    mask = X_train == category
                    n = mask.sum()

                    if n >= self.min_samples_leaf:
                        # Regularized encoding
                        category_mean = y_train[mask].mean()
                        encoding[category] = self._regularize_encoding(
                            category_mean, n, self.global_mean_
                        )
                    else:
                        encoding[category] = self.global_mean_

        # Handle unseen categories
        all_categories = X_col.unique()
        for category in all_categories:
            if category not in encoding:
                encoding[category] = self.global_mean_

        return encoding

    def _regularize_encoding(
        self, category_mean: float, n: int, global_mean: float
    ) -> float:
        """Apply regularization to encoding."""
        # Empirical Bayes smoothing
        lambda_param = 1 / (1 + np.exp(-(n - self.min_samples_leaf) / self.smoothing))
        return lambda_param * category_mean + (1 - lambda_param) * global_mean

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical columns to target encodings."""
        X_encoded = X.copy()

        for col in self.columns:
            if col in X.columns:
                X_encoded[col] = X[col].map(self.encodings_[col])
                # Handle unseen categories
                X_encoded[col].fillna(self.global_mean_, inplace=True)

        return X_encoded

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform with proper CV to prevent leakage."""
        X_encoded = X.copy()
        self.global_mean_ = y.mean()

        if self.columns is None:
            self.columns = X.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

        # Use out-of-fold predictions
        for col in self.columns:
            if col in X.columns:
                X_encoded[col] = self._cv_encode_column(X[col], y)

        return X_encoded

    def _cv_encode_column(self, X_col: pd.Series, y: pd.Series) -> pd.Series:
        """Encode column using cross-validation to prevent leakage."""
        encoded = pd.Series(index=X_col.index, dtype=float)

        kf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(X_col, y):
            X_train, y_train = X_col.iloc[train_idx], y.iloc[train_idx]
            X_val = X_col.iloc[val_idx]

            # Fit on train
            encoding = {}
            for category in X_val.unique():
                mask = X_train == category
                n = mask.sum()

                if n >= self.min_samples_leaf:
                    category_mean = y_train[mask].mean()
                    encoding[category] = self._regularize_encoding(
                        category_mean, n, self.global_mean_
                    )
                else:
                    encoding[category] = self.global_mean_

            # Transform validation
            encoded.iloc[val_idx] = X_val.map(encoding).fillna(self.global_mean_)

        # Store final encodings for transform
        self.encodings_[X_col.name] = self._fit_column(X_col, y)

        return encoded


class FeatureStabilityAnalyzer:
    """Analyze feature stability across different data splits and time periods."""

    def __init__(self, n_splits: int = 10, test_size: float = 0.2):
        """
        Initialize stability analyzer.

        Args:
            n_splits: Number of splits for stability testing
            test_size: Size of test split
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.stability_scores_ = {}
        self.temporal_stability_ = {}

    def analyze_stability(
        self, X: pd.DataFrame, y: pd.Series, feature_list: List[str] = None
    ) -> pd.DataFrame:
        """
        Analyze feature stability across different data splits.

        Args:
            X: Feature DataFrame
            y: Target Series
            feature_list: Features to analyze

        Returns:
            DataFrame with stability metrics
        """
        if feature_list is None:
            feature_list = X.columns.tolist()

        stability_metrics = {
            "mean": [],
            "std": [],
            "cv": [],  # Coefficient of variation
            "iqr": [],  # Interquartile range
            "range": [],
            "stability_score": [],
        }

        for feature in feature_list:
            if feature not in X.columns:
                continue

            # Calculate feature importance across different splits
            importances = []

            for i in range(self.n_splits):
                # Random split
                split_idx = int(len(X) * (1 - self.test_size))
                idx = np.random.permutation(len(X))
                train_idx = idx[:split_idx]

                X_split = X.iloc[train_idx]
                y_split = y.iloc[train_idx]

                # Calculate importance (using mutual information)
                if X[feature].dtype in ["object", "category"]:
                    # Encode categorical
                    from sklearn.preprocessing import LabelEncoder

                    le = LabelEncoder()
                    feature_encoded = le.fit_transform(
                        X_split[feature].fillna("missing")
                    )
                    mi_score = mutual_info_classif(
                        feature_encoded.reshape(-1, 1), y_split, random_state=42
                    )[0]
                else:
                    mi_score = mutual_info_classif(
                        X_split[[feature]], y_split, random_state=42
                    )[0]

                importances.append(mi_score)

            # Calculate stability metrics
            importances = np.array(importances)

            stability_metrics["mean"].append(np.mean(importances))
            stability_metrics["std"].append(np.std(importances))
            stability_metrics["cv"].append(
                np.std(importances) / np.mean(importances)
                if np.mean(importances) > 0
                else np.inf
            )
            stability_metrics["iqr"].append(
                np.percentile(importances, 75) - np.percentile(importances, 25)
            )
            stability_metrics["range"].append(np.max(importances) - np.min(importances))

            # Overall stability score (lower CV is more stable)
            cv = stability_metrics["cv"][-1]
            stability_score = 1 / (1 + cv) if cv != np.inf else 0
            stability_metrics["stability_score"].append(stability_score)

        # Create DataFrame
        stability_df = pd.DataFrame(
            stability_metrics, index=feature_list[: len(stability_metrics["mean"])]
        )
        stability_df = stability_df.sort_values("stability_score", ascending=False)

        self.stability_scores_ = stability_df
        return stability_df

    def analyze_temporal_stability(
        self, X: pd.DataFrame, y: pd.Series, time_column: str, n_periods: int = 5
    ) -> pd.DataFrame:
        """
        Analyze feature stability over time.

        Args:
            X: Feature DataFrame with time column
            y: Target Series
            time_column: Name of time column
            n_periods: Number of time periods to analyze

        Returns:
            DataFrame with temporal stability metrics
        """
        # Split data into time periods
        time_values = pd.to_datetime(X[time_column])
        period_edges = pd.qcut(
            time_values.rank(method="first"), n_periods, duplicates="drop"
        )

        temporal_importance = {}

        for period in period_edges.unique():
            period_mask = period_edges == period
            X_period = X[period_mask].drop(columns=[time_column])
            y_period = y[period_mask]

            # Calculate feature importance for this period
            for feature in X_period.columns:
                if feature not in temporal_importance:
                    temporal_importance[feature] = []

                if X_period[feature].dtype in ["object", "category"]:
                    # Encode categorical
                    from sklearn.preprocessing import LabelEncoder

                    le = LabelEncoder()
                    feature_encoded = le.fit_transform(
                        X_period[feature].fillna("missing")
                    )
                    mi_score = mutual_info_classif(
                        feature_encoded.reshape(-1, 1), y_period, random_state=42
                    )[0]
                else:
                    mi_score = mutual_info_classif(
                        X_period[[feature]], y_period, random_state=42
                    )[0]

                temporal_importance[feature].append(mi_score)

        # Calculate temporal stability metrics
        temporal_stability = {}

        for feature, scores in temporal_importance.items():
            scores = np.array(scores)

            # Trend (correlation with time)
            time_indices = np.arange(len(scores))
            trend_corr, _ = spearmanr(time_indices, scores)

            # Variance over time
            temporal_cv = (
                np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else np.inf
            )

            temporal_stability[feature] = {
                "trend": trend_corr,
                "temporal_cv": temporal_cv,
                "temporal_stability": 1 / (1 + abs(trend_corr) + temporal_cv),
                "mean_importance": np.mean(scores),
                "importance_by_period": scores.tolist(),
            }

        temporal_df = pd.DataFrame.from_dict(temporal_stability, orient="index")
        temporal_df = temporal_df.sort_values("temporal_stability", ascending=False)

        self.temporal_stability_ = temporal_df
        return temporal_df


class FeatureEngineering:
    """Placeholder feature engineering pipeline."""

    pass
