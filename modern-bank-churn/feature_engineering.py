"""Advanced Feature Engineering for Bank Churn Prediction.
Implements automated feature selection, interaction detection, and stability metrics.
"""

import warnings
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    chi2,
    f_classif,
    mutual_info_classif,
)

warnings.filterwarnings("ignore")


@dataclass
class FeatureImportance:
    """Container for feature importance results."""

    method: str
    scores: pd.DataFrame
    selected_features: list[str]
    stability_score: float = None


class AutomatedFeatureSelector:
    """Automated feature selection using multiple methods."""

    def __init__(
        self, methods: list[str] = None, k: int = 20, importance_threshold: float = 0.01
    ):
        """Initialize feature selector.

        Args:
            methods: List of selection methods to use
            k: Number of top features to select
            importance_threshold: Minimum importance threshold
        """
        if methods is None:
            methods = ["mutual_info", "chi2", "anova", "random_forest", "l1"]

        self.methods = methods
        self.k = k
        self.importance_threshold = importance_threshold
        self.feature_scores_ = {}
        self.selected_features_ = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "AutomatedFeatureSelector":
        """Fit feature selector on data.

        Args:
            X: Feature DataFrame
            y: Target Series
        """
        # Mutual Information
        if "mutual_info" in self.methods:
            mi_scores = mutual_info_classif(X, y, random_state=42)
            self.feature_scores_["mutual_info"] = pd.Series(
                mi_scores, index=X.columns
            ).sort_values(ascending=False)

        # Chi-squared (for non-negative features)
        if "chi2" in self.methods:
            try:
                # Only use non-negative features
                non_neg_cols = X.columns[(X >= 0).all()]
                if len(non_neg_cols) > 0:
                    chi2_scores, _ = chi2(X[non_neg_cols], y)
                    self.feature_scores_["chi2"] = pd.Series(
                        chi2_scores, index=non_neg_cols
                    ).sort_values(ascending=False)
            except Exception:
                pass

        # ANOVA F-statistic
        if "anova" in self.methods:
            f_scores, _ = f_classif(X, y)
            self.feature_scores_["anova"] = pd.Series(
                f_scores, index=X.columns
            ).sort_values(ascending=False)

        # Random Forest importance
        if "random_forest" in self.methods:
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            self.feature_scores_["random_forest"] = pd.Series(
                rf.feature_importances_, index=X.columns
            ).sort_values(ascending=False)

        # L1-based selection (Lasso)
        if "l1" in self.methods:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler

            X_scaled = StandardScaler().fit_transform(X)
            lasso = LogisticRegression(penalty="l1", solver="liblinear", C=1.0)
            lasso.fit(X_scaled, y)

            # Use absolute coefficients as importance
            importance = np.abs(lasso.coef_[0])
            self.feature_scores_["l1"] = pd.Series(
                importance, index=X.columns
            ).sort_values(ascending=False)

        # Select top features for each method
        for method, scores in self.feature_scores_.items():
            # Normalize scores
            normalized_scores = scores / scores.max()

            # Select by threshold and top-k
            threshold_mask = normalized_scores > self.importance_threshold
            top_k_features = scores.head(self.k).index.tolist()
            threshold_features = scores[threshold_mask].index.tolist()

            # Combine both criteria
            self.selected_features_[method] = list(
                set(top_k_features) & set(threshold_features)
            )

        return self

    def get_consensus_features(self, min_methods: int = 2) -> list[str]:
        """Get features selected by multiple methods.

        Args:
            min_methods: Minimum number of methods that must select a feature

        Returns:
            List of consensus features
        """
        from collections import Counter

        all_features = []
        for features in self.selected_features_.values():
            all_features.extend(features)

        feature_counts = Counter(all_features)
        consensus_features = [
            feat for feat, count in feature_counts.items() if count >= min_methods
        ]

        return consensus_features

    def get_importance_summary(self) -> pd.DataFrame:
        """Get summary of feature importance across methods."""
        summary = pd.DataFrame(self.feature_scores_)

        # Add rank columns
        for col in summary.columns:
            summary[f"{col}_rank"] = summary[col].rank(ascending=False)

        # Add average rank
        rank_cols = [c for c in summary.columns if "_rank" in c]
        summary["avg_rank"] = summary[rank_cols].mean(axis=1)

        return summary.sort_values("avg_rank")


class FeatureInteractionDetector:
    """Detect and create feature interactions."""

    def __init__(
        self,
        max_interactions: int = 2,
        correlation_threshold: float = 0.7,
        top_k: int = 20,
    ):
        """Initialize interaction detector.

        Args:
            max_interactions: Maximum interaction order (2 for pairs, 3 for triples)
            correlation_threshold: Threshold for interaction strength
            top_k: Number of top interactions to keep
        """
        self.max_interactions = max_interactions
        self.correlation_threshold = correlation_threshold
        self.top_k = top_k
        self.interactions_ = []
        self.interaction_scores_ = {}

    def detect_interactions(
        self, X: pd.DataFrame, y: pd.Series, method: str = "mutual_info"
    ) -> pd.DataFrame:
        """Detect feature interactions.

        Args:
            X: Feature DataFrame
            y: Target Series
            method: Method for measuring interaction strength

        Returns:
            DataFrame with interaction scores
        """
        interactions_list = []

        # Generate pairs (or higher order)
        feature_combinations = combinations(X.columns, self.max_interactions)

        for features in feature_combinations:
            # Create interaction term
            interaction = X[list(features)].prod(axis=1)
            interaction_name = " * ".join(features)

            # Measure interaction strength
            if method == "mutual_info":
                # Mutual information with target
                score = mutual_info_classif(
                    interaction.values.reshape(-1, 1), y, random_state=42
                )[0]

                # Subtract individual contributions to get pure interaction
                individual_scores = [
                    mutual_info_classif(X[[feat]].values, y, random_state=42)[0]
                    for feat in features
                ]
                interaction_score = score - np.mean(individual_scores)

            elif method == "correlation":
                # Correlation with target
                score = np.abs(spearmanr(interaction, y)[0])
                interaction_score = score

            elif method == "variance":
                # Variance reduction when grouping by interaction
                groups = pd.qcut(interaction, q=5, duplicates="drop")
                group_variance = y.groupby(groups).var().mean()
                total_variance = y.var()
                interaction_score = 1 - (group_variance / total_variance)

            else:
                raise ValueError(f"Unknown method: {method}")

            interactions_list.append(
                {
                    "features": features,
                    "interaction_name": interaction_name,
                    "score": score,
                    "interaction_score": interaction_score,
                    "method": method,
                }
            )

        # Create DataFrame and sort by interaction score
        interactions_df = pd.DataFrame(interactions_list)
        interactions_df = interactions_df.sort_values(
            "interaction_score", ascending=False
        )

        # Store top interactions
        self.interactions_ = interactions_df.head(self.top_k)
        self.interaction_scores_ = dict(
            zip(
                self.interactions_["interaction_name"],
                self.interactions_["interaction_score"], strict=False,
            )
        )

        return self.interactions_

    def create_interactions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features.

        Args:
            X: Feature DataFrame

        Returns:
            DataFrame with original and interaction features
        """
        X_with_interactions = X.copy()

        for _, interaction in self.interactions_.iterrows():
            features = interaction["features"]
            name = f"interaction_{' * '.join(features)}"

            # Create interaction term
            X_with_interactions[name] = X[list(features)].prod(axis=1)

        return X_with_interactions


class TargetEncoder(BaseEstimator, TransformerMixin):
    """Target encoding with regularization and cross-validation."""

    def __init__(
        self,
        smoothing: float = 1.0,
        min_samples: int = 1,
        cv_folds: int = 5,
        noise_level: float = 0.01,
    ):
        """Initialize target encoder.

        Args:
            smoothing: Smoothing parameter for regularization
            min_samples: Minimum samples required for encoding
            cv_folds: Number of cross-validation folds
            noise_level: Noise to add to prevent overfitting
        """
        self.smoothing = smoothing
        self.min_samples = min_samples
        self.cv_folds = cv_folds
        self.noise_level = noise_level
        self.encodings_ = {}
        self.global_mean_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TargetEncoder":
        """Fit target encoder.

        Args:
            X: Feature DataFrame (categorical)
            y: Target Series
        """
        self.global_mean_ = y.mean()

        for col in X.columns:
            # Calculate target statistics for each category
            stats = y.groupby(X[col]).agg(["mean", "count"])

            # Apply smoothing (similar to Bayesian average)
            smoothed_mean = (
                stats["mean"] * stats["count"] + self.global_mean_ * self.smoothing
            ) / (stats["count"] + self.smoothing)

            # Only keep encodings with minimum samples
            valid_categories = stats[stats["count"] >= self.min_samples].index
            self.encodings_[col] = smoothed_mean[valid_categories].to_dict()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features using target encoding.

        Args:
            X: Feature DataFrame

        Returns:
            Encoded DataFrame
        """
        X_encoded = pd.DataFrame(index=X.index)

        for col in X.columns:
            if col in self.encodings_:
                # Map encodings
                encoded = X[col].map(self.encodings_[col])

                # Fill missing with global mean
                encoded = encoded.fillna(self.global_mean_)

                # Add noise to prevent overfitting
                if self.noise_level > 0:
                    noise = np.random.normal(0, self.noise_level, len(encoded))
                    encoded = encoded + noise

                X_encoded[f"{col}_target_encoded"] = encoded
            else:
                X_encoded[col] = X[col]

        return X_encoded

    def fit_transform_cv(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform with cross-validation to prevent overfitting.

        Args:
            X: Feature DataFrame
            y: Target Series

        Returns:
            Encoded DataFrame
        """
        from sklearn.model_selection import KFold

        X_encoded = pd.DataFrame(index=X.index)
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

        for col in X.columns:
            encoded_col = pd.Series(index=X.index, dtype=float)

            for train_idx, val_idx in kf.split(X):
                # Fit on train fold
                X_train_fold = X.iloc[train_idx]
                y_train_fold = y.iloc[train_idx]

                # Calculate encodings for this fold
                fold_global_mean = y_train_fold.mean()
                stats = y_train_fold.groupby(X_train_fold[col]).agg(["mean", "count"])

                smoothed_mean = (
                    stats["mean"] * stats["count"] + fold_global_mean * self.smoothing
                ) / (stats["count"] + self.smoothing)

                # Apply to validation fold
                X_val_fold = X.iloc[val_idx]
                fold_encoded = X_val_fold[col].map(smoothed_mean.to_dict())
                fold_encoded = fold_encoded.fillna(fold_global_mean)

                encoded_col.iloc[val_idx] = fold_encoded

            # Add noise
            if self.noise_level > 0:
                noise = np.random.normal(0, self.noise_level, len(encoded_col))
                encoded_col = encoded_col + noise

            X_encoded[f"{col}_target_encoded"] = encoded_col

        return X_encoded


class FeatureStabilityAnalyzer:
    """Analyze feature stability over time and across segments."""

    def __init__(self):
        self.stability_scores_ = {}
        self.drift_scores_ = {}

    def calculate_temporal_stability(
        self, X: pd.DataFrame, time_column: str, window_size: int = 30
    ) -> pd.DataFrame:
        """Calculate feature stability over time.

        Args:
            X: Feature DataFrame
            time_column: Name of time column
            window_size: Window size for stability calculation

        Returns:
            DataFrame with stability scores
        """
        stability_results = []

        # Sort by time
        X_sorted = X.sort_values(time_column)

        # Calculate rolling statistics
        for col in X.columns:
            if col == time_column:
                continue

            # Rolling mean and std
            rolling_mean = X_sorted[col].rolling(window=window_size).mean()
            rolling_std = X_sorted[col].rolling(window=window_size).std()

            # Coefficient of variation
            cv = rolling_std / (rolling_mean + 1e-10)

            # Population Stability Index (PSI)
            # Compare each window to the first window
            first_window = X_sorted[col].iloc[:window_size]
            psi_scores = []

            for i in range(window_size, len(X_sorted), window_size):
                current_window = X_sorted[col].iloc[i : i + window_size]
                psi = self._calculate_psi(first_window, current_window)
                psi_scores.append(psi)

            stability_results.append(
                {
                    "feature": col,
                    "mean_cv": cv.mean(),
                    "max_cv": cv.max(),
                    "mean_psi": np.mean(psi_scores) if psi_scores else 0,
                    "max_psi": np.max(psi_scores) if psi_scores else 0,
                }
            )

        stability_df = pd.DataFrame(stability_results)
        stability_df["stability_score"] = 1 / (
            1 + stability_df["mean_cv"] + stability_df["mean_psi"]
        )

        self.stability_scores_ = stability_df
        return stability_df

    def calculate_segment_stability(
        self, X: pd.DataFrame, segment_column: str
    ) -> pd.DataFrame:
        """Calculate feature stability across segments.

        Args:
            X: Feature DataFrame
            segment_column: Name of segment column

        Returns:
            DataFrame with stability scores
        """
        segment_results = []

        for col in X.columns:
            if col == segment_column:
                continue

            # Calculate statistics per segment
            segment_stats = X.groupby(segment_column)[col].agg(["mean", "std"])

            # Coefficient of variation across segments
            cv_across = segment_stats["std"].std() / (
                segment_stats["mean"].std() + 1e-10
            )

            # Maximum difference between segments
            max_diff = segment_stats["mean"].max() - segment_stats["mean"].min()
            range_normalized = max_diff / (X[col].std() + 1e-10)

            segment_results.append(
                {
                    "feature": col,
                    "cv_across_segments": cv_across,
                    "normalized_range": range_normalized,
                    "n_unique_segments": len(segment_stats),
                }
            )

        segment_df = pd.DataFrame(segment_results)
        segment_df["segment_stability"] = 1 / (1 + segment_df["cv_across_segments"])

        return segment_df

    def _calculate_psi(
        self, reference: pd.Series, current: pd.Series, n_bins: int = 10
    ) -> float:
        """Calculate Population Stability Index.

        Args:
            reference: Reference distribution
            current: Current distribution
            n_bins: Number of bins

        Returns:
            PSI value
        """
        # Create bins from reference
        _, bins = pd.qcut(reference, q=n_bins, retbins=True, duplicates="drop")

        # Calculate proportions
        ref_counts = pd.cut(reference, bins=bins, include_lowest=True).value_counts()
        ref_prop = ref_counts / len(reference)

        curr_counts = pd.cut(current, bins=bins, include_lowest=True).value_counts()
        curr_prop = curr_counts / len(current)

        # Align indices and fill zeros
        ref_prop = ref_prop.reindex(
            ref_prop.index.union(curr_prop.index), fill_value=0.0001
        )
        curr_prop = curr_prop.reindex(ref_prop.index, fill_value=0.0001)

        # Calculate PSI
        psi = np.sum((curr_prop - ref_prop) * np.log(curr_prop / ref_prop))

        return psi

    def detect_feature_drift(
        self, X_reference: pd.DataFrame, X_current: pd.DataFrame, method: str = "ks"
    ) -> pd.DataFrame:
        """Detect feature drift between reference and current data.

        Args:
            X_reference: Reference DataFrame
            X_current: Current DataFrame
            method: Statistical test method ('ks', 'psi', 'wasserstein')

        Returns:
            DataFrame with drift scores
        """
        drift_results = []

        for col in X_reference.columns:
            if col not in X_current.columns:
                continue

            if method == "ks":
                # Kolmogorov-Smirnov test
                statistic, p_value = stats.ks_2samp(X_reference[col], X_current[col])
                drift_score = statistic

            elif method == "psi":
                # Population Stability Index
                drift_score = self._calculate_psi(X_reference[col], X_current[col])
                p_value = None

            elif method == "wasserstein":
                # Wasserstein distance
                from scipy.stats import wasserstein_distance

                drift_score = wasserstein_distance(X_reference[col], X_current[col])
                p_value = None

            else:
                raise ValueError(f"Unknown method: {method}")

            drift_results.append(
                {
                    "feature": col,
                    "drift_score": drift_score,
                    "p_value": p_value if method == "ks" else None,
                    "method": method,
                    "has_drift": (
                        drift_score > 0.1
                        if method == "psi"
                        else (p_value < 0.05 if p_value else False)
                    ),
                }
            )

        drift_df = pd.DataFrame(drift_results)
        self.drift_scores_ = drift_df

        return drift_df


class AdvancedFeatureEngineering:
    """Comprehensive feature engineering pipeline."""

    def __init__(self):
        self.feature_selector = AutomatedFeatureSelector()
        self.interaction_detector = FeatureInteractionDetector()
        self.target_encoder = TargetEncoder()
        self.stability_analyzer = FeatureStabilityAnalyzer()
        self.feature_importance_history = []

    def fit_transform(
        self, X: pd.DataFrame, y: pd.Series, categorical_columns: list[str] = None
    ) -> pd.DataFrame:
        """Complete feature engineering pipeline.

        Args:
            X: Feature DataFrame
            y: Target Series
            categorical_columns: List of categorical column names

        Returns:
            Transformed DataFrame with engineered features
        """
        X_transformed = X.copy()

        # Separate numerical and categorical
        if categorical_columns is None:
            categorical_columns = X.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

        numerical_columns = [c for c in X.columns if c not in categorical_columns]

        # 1. Target encoding for categorical variables
        if categorical_columns:
            X_cat_encoded = self.target_encoder.fit_transform_cv(
                X[categorical_columns], y
            )
            X_transformed = pd.concat(
                [X_transformed.drop(columns=categorical_columns), X_cat_encoded], axis=1
            )

        # 2. Detect and create interactions for numerical features
        if numerical_columns:
            interactions = self.interaction_detector.detect_interactions(
                X[numerical_columns], y
            )
            X_with_interactions = self.interaction_detector.create_interactions(
                X_transformed
            )
            X_transformed = X_with_interactions

        # 3. Automated feature selection
        self.feature_selector.fit(X_transformed, y)
        selected_features = self.feature_selector.get_consensus_features()

        # 4. Feature importance summary
        importance_summary = self.feature_selector.get_importance_summary()

        # Store importance history
        self.feature_importance_history.append(
            {
                "timestamp": pd.Timestamp.now(),
                "importance": importance_summary,
                "selected_features": selected_features,
            }
        )

        return X_transformed[selected_features] if selected_features else X_transformed

    def get_feature_report(self) -> dict:
        """Get comprehensive feature engineering report.

        Returns:
            Dictionary with feature engineering results
        """
        report = {
            "selected_features": self.feature_selector.selected_features_,
            "consensus_features": self.feature_selector.get_consensus_features(),
            "feature_importance": self.feature_selector.get_importance_summary(),
            "top_interactions": self.interaction_detector.interactions_,
            "stability_scores": self.stability_analyzer.stability_scores_,
            "drift_scores": self.stability_analyzer.drift_scores_,
        }

        return report


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 1000

    X = pd.DataFrame(
        {
            "feature1": np.random.randn(n_samples),
            "feature2": np.random.randn(n_samples),
            "feature3": np.random.exponential(1, n_samples),
            "category1": np.random.choice(["A", "B", "C"], n_samples),
            "category2": np.random.choice(["X", "Y", "Z"], n_samples),
        }
    )

    y = pd.Series(np.random.randint(0, 2, n_samples))

    # Apply feature engineering
    fe = AdvancedFeatureEngineering()
    X_engineered = fe.fit_transform(X, y, ["category1", "category2"])

    print("Feature Engineering Complete!")
    print(f"Original features: {X.shape[1]}")
    print(f"Engineered features: {X_engineered.shape[1]}")

    # Get report
    report = fe.get_feature_report()
    print("\nTop Features by Importance:")
    print(report["feature_importance"].head())
