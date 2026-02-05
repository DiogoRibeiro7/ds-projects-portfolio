"""Enhanced Production Readiness for Bank Churn Prediction.
Model versioning, monitoring, drift detection, explainability,
and A/B testing framework.
"""

import hashlib
import json
import pickle
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


@dataclass
class ModelVersion:
    """Container for model version information."""

    version_id: str
    model_name: str
    created_at: datetime
    metrics: dict[str, float]
    parameters: dict[str, Any]
    feature_names: list[str]
    model_hash: str
    training_data_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftReport:
    """Container for drift detection results."""

    feature_drift: pd.DataFrame
    target_drift: float
    concept_drift: float
    data_quality_issues: list[str]
    drift_detected: bool
    drift_severity: str  # 'none', 'low', 'medium', 'high'
    recommendations: list[str]


@dataclass
class MonitoringMetrics:
    """Container for model monitoring metrics."""

    timestamp: datetime
    prediction_volume: int
    average_prediction: float
    prediction_distribution: dict[str, float]
    latency_ms: float
    error_rate: float
    feature_statistics: pd.DataFrame
    alerts: list[str]


class ModelVersionControl:
    """Model versioning and management system.
    """

    def __init__(self, base_path: str = "./model_registry"):
        """Initialize model version control.

        Args:
            base_path: Base directory for model storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.versions = {}
        self.current_version = None
        self._load_registry()

    def register_model(
        self,
        model: BaseEstimator,
        model_name: str,
        metrics: dict[str, float],
        parameters: dict[str, Any],
        feature_names: list[str],
        X_train: pd.DataFrame = None,
        metadata: dict[str, Any] = None,
    ) -> str:
        """Register a new model version.

        Args:
            model: Trained model
            model_name: Name of the model
            metrics: Model performance metrics
            parameters: Model hyperparameters
            feature_names: List of feature names
            X_train: Training data (for hash)
            metadata: Additional metadata

        Returns:
            Version ID
        """
        # Generate version ID
        timestamp = datetime.now()
        version_id = f"{model_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}"

        # Calculate hashes
        model_hash = self._calculate_model_hash(model)
        data_hash = self._calculate_data_hash(X_train) if X_train is not None else ""

        # Create version object
        version = ModelVersion(
            version_id=version_id,
            model_name=model_name,
            created_at=timestamp,
            metrics=metrics,
            parameters=parameters,
            feature_names=feature_names,
            model_hash=model_hash,
            training_data_hash=data_hash,
            metadata=metadata or {},
        )

        # Save model and version info
        self._save_model(model, version)
        self.versions[version_id] = version
        self._save_registry()

        return version_id

    def load_model(self, version_id: str) -> tuple[BaseEstimator, ModelVersion]:
        """Load a specific model version.

        Args:
            version_id: Version ID to load

        Returns:
            Tuple of (model, version_info)
        """
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")

        version = self.versions[version_id]
        model_path = self.base_path / f"{version_id}.pkl"

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        return model, version

    def compare_versions(
        self, version_ids: list[str], metrics: list[str] = None
    ) -> pd.DataFrame:
        """Compare multiple model versions.

        Args:
            version_ids: List of version IDs to compare
            metrics: Metrics to compare (None for all)

        Returns:
            DataFrame with comparison results
        """
        comparison_data = []

        for version_id in version_ids:
            if version_id not in self.versions:
                continue

            version = self.versions[version_id]
            row_data = {
                "version_id": version_id,
                "model_name": version.model_name,
                "created_at": version.created_at,
            }

            # Add metrics
            if metrics:
                for metric in metrics:
                    row_data[metric] = version.metrics.get(metric, np.nan)
            else:
                row_data.update(version.metrics)

            comparison_data.append(row_data)

        return pd.DataFrame(comparison_data)

    def promote_version(self, version_id: str, stage: str = "production"):
        """Promote a model version to a specific stage.

        Args:
            version_id: Version ID to promote
            stage: Target stage ('staging', 'production', etc.)
        """
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")

        version = self.versions[version_id]
        version.metadata["stage"] = stage
        version.metadata["promoted_at"] = datetime.now().isoformat()

        self.current_version = version_id
        self._save_registry()

    def _save_model(self, model: BaseEstimator, version: ModelVersion):
        """Save model to disk."""
        model_path = self.base_path / f"{version.version_id}.pkl"

        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Save version metadata
        meta_path = self.base_path / f"{version.version_id}_meta.json"
        with open(meta_path, "w") as f:
            version_dict = asdict(version)
            version_dict["created_at"] = version.created_at.isoformat()
            json.dump(version_dict, f, indent=2)

    def _load_registry(self):
        """Load model registry from disk."""
        registry_path = self.base_path / "registry.json"

        if registry_path.exists():
            with open(registry_path) as f:
                registry_data = json.load(f)

                for version_id, version_data in registry_data["versions"].items():
                    version_data["created_at"] = datetime.fromisoformat(
                        version_data["created_at"]
                    )
                    self.versions[version_id] = ModelVersion(**version_data)

                self.current_version = registry_data.get("current_version")

    def _save_registry(self):
        """Save model registry to disk."""
        registry_path = self.base_path / "registry.json"

        registry_data = {"versions": {}, "current_version": self.current_version}

        for version_id, version in self.versions.items():
            version_dict = asdict(version)
            version_dict["created_at"] = version.created_at.isoformat()
            registry_data["versions"][version_id] = version_dict

        with open(registry_path, "w") as f:
            json.dump(registry_data, f, indent=2)

    def _calculate_model_hash(self, model: BaseEstimator) -> str:
        """Calculate hash of model object."""
        model_bytes = pickle.dumps(model)
        return hashlib.md5(model_bytes).hexdigest()

    def _calculate_data_hash(self, data: pd.DataFrame) -> str:
        """Calculate hash of training data."""
        data_str = data.to_string()
        return hashlib.md5(data_str.encode()).hexdigest()


class DriftDetector:
    """Detect data and concept drift in production.
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        reference_predictions: np.ndarray = None,
        drift_threshold: float = 0.1,
        warning_threshold: float = 0.05,
    ):
        """Initialize drift detector.

        Args:
            reference_data: Reference (training) data
            reference_predictions: Reference model predictions
            drift_threshold: Threshold for drift detection
            warning_threshold: Threshold for drift warning
        """
        self.reference_data = reference_data
        self.reference_predictions = reference_predictions
        self.drift_threshold = drift_threshold
        self.warning_threshold = warning_threshold
        self.reference_stats = self._calculate_statistics(reference_data)

    def detect_drift(
        self,
        current_data: pd.DataFrame,
        current_predictions: np.ndarray = None,
        feature_subset: list[str] = None,
    ) -> DriftReport:
        """Detect drift in current data compared to reference.

        Args:
            current_data: Current production data
            current_predictions: Current model predictions
            feature_subset: Subset of features to check

        Returns:
            DriftReport object
        """
        if feature_subset is None:
            feature_subset = self.reference_data.columns.tolist()

        # Feature drift detection
        feature_drift_results = self._detect_feature_drift(current_data[feature_subset])

        # Target/prediction drift
        target_drift = 0
        if current_predictions is not None and self.reference_predictions is not None:
            target_drift = self._detect_target_drift(current_predictions)

        # Concept drift (if we have ground truth)
        concept_drift = self._detect_concept_drift(current_data, current_predictions)

        # Data quality checks
        quality_issues = self._check_data_quality(current_data)

        # Determine overall drift status
        max_feature_drift = feature_drift_results["drift_score"].max()
        drift_detected = (
            max_feature_drift > self.drift_threshold
            or target_drift > self.drift_threshold
            or concept_drift > self.drift_threshold
        )

        # Determine severity
        if not drift_detected:
            severity = "none"
        elif (
            max(max_feature_drift, target_drift, concept_drift)
            > self.drift_threshold * 2
        ):
            severity = "high"
        elif max(max_feature_drift, target_drift, concept_drift) > self.drift_threshold:
            severity = "medium"
        else:
            severity = "low"

        # Generate recommendations
        recommendations = self._generate_recommendations(
            feature_drift_results, target_drift, concept_drift, quality_issues
        )

        return DriftReport(
            feature_drift=feature_drift_results,
            target_drift=target_drift,
            concept_drift=concept_drift,
            data_quality_issues=quality_issues,
            drift_detected=drift_detected,
            drift_severity=severity,
            recommendations=recommendations,
        )

    def _detect_feature_drift(self, current_data: pd.DataFrame) -> pd.DataFrame:
        """Detect drift in individual features."""
        drift_results = []

        for column in current_data.columns:
            if column not in self.reference_stats:
                continue

            ref_stats = self.reference_stats[column]
            curr_stats = self._calculate_column_statistics(current_data[column])

            # Choose appropriate test based on data type
            if current_data[column].dtype in ["float64", "int64"]:
                # Kolmogorov-Smirnov test for numerical features
                if len(current_data[column].dropna()) > 0:
                    ks_stat, p_value = stats.ks_2samp(
                        self.reference_data[column].dropna(),
                        current_data[column].dropna(),
                    )
                else:
                    ks_stat, p_value = 0, 1

                drift_score = ks_stat

                # Also check distribution shift
                mean_shift = abs(curr_stats["mean"] - ref_stats["mean"]) / (
                    ref_stats["std"] + 1e-10
                )
                std_shift = abs(curr_stats["std"] - ref_stats["std"]) / (
                    ref_stats["std"] + 1e-10
                )

            else:
                # Chi-square test for categorical features
                ref_dist = self.reference_data[column].value_counts(normalize=True)
                curr_dist = current_data[column].value_counts(normalize=True)

                # Align categories
                all_categories = set(ref_dist.index) | set(curr_dist.index)
                ref_dist = ref_dist.reindex(all_categories, fill_value=0)
                curr_dist = curr_dist.reindex(all_categories, fill_value=0)

                # Calculate chi-square statistic
                chi2_stat = np.sum((ref_dist - curr_dist) ** 2 / (ref_dist + 1e-10))
                drift_score = chi2_stat
                p_value = 1 - stats.chi2.cdf(chi2_stat, len(all_categories) - 1)

                mean_shift = 0
                std_shift = 0

            drift_results.append(
                {
                    "feature": column,
                    "drift_score": drift_score,
                    "p_value": p_value,
                    "mean_shift": mean_shift,
                    "std_shift": std_shift,
                    "drift_detected": drift_score > self.drift_threshold,
                }
            )

        return pd.DataFrame(drift_results)

    def _detect_target_drift(self, current_predictions: np.ndarray) -> float:
        """Detect drift in prediction distribution."""
        # KS test for prediction distribution
        ks_stat, _ = stats.ks_2samp(self.reference_predictions, current_predictions)

        return ks_stat

    def _detect_concept_drift(
        self, current_data: pd.DataFrame, current_predictions: np.ndarray
    ) -> float:
        """Detect concept drift (change in P(y|X))."""
        # This would require ground truth labels
        # For now, return a proxy based on prediction confidence
        if current_predictions is None:
            return 0

        # Check if prediction confidence has changed
        ref_confidence = np.mean(np.abs(self.reference_predictions - 0.5))
        curr_confidence = np.mean(np.abs(current_predictions - 0.5))

        confidence_shift = abs(curr_confidence - ref_confidence) / (
            ref_confidence + 1e-10
        )

        return confidence_shift

    def _check_data_quality(self, current_data: pd.DataFrame) -> list[str]:
        """Check for data quality issues."""
        issues = []

        # Check for missing values
        missing_rate = current_data.isnull().sum() / len(current_data)
        high_missing = missing_rate[missing_rate > 0.1]

        if len(high_missing) > 0:
            issues.append(f"High missing rate in: {high_missing.index.tolist()}")

        # Check for new categories
        for column in current_data.select_dtypes(
            include=["object", "category"]
        ).columns:
            if column in self.reference_data.columns:
                new_categories = set(current_data[column].unique()) - set(
                    self.reference_data[column].unique()
                )
                if new_categories:
                    issues.append(f"New categories in {column}: {new_categories}")

        # Check for outliers in numerical features
        for column in current_data.select_dtypes(include=[np.number]).columns:
            if column in self.reference_stats:
                ref_stats = self.reference_stats[column]
                lower_bound = ref_stats["mean"] - 4 * ref_stats["std"]
                upper_bound = ref_stats["mean"] + 4 * ref_stats["std"]

                outliers = (
                    (current_data[column] < lower_bound)
                    | (current_data[column] > upper_bound)
                ).sum()

                if outliers > len(current_data) * 0.05:
                    issues.append(
                        f"High outlier rate in {column}: {outliers / len(current_data):.2%}"
                    )

        return issues

    def _calculate_statistics(self, data: pd.DataFrame) -> dict[str, dict[str, float]]:
        """Calculate statistics for all columns."""
        stats_dict = {}

        for column in data.columns:
            stats_dict[column] = self._calculate_column_statistics(data[column])

        return stats_dict

    def _calculate_column_statistics(self, series: pd.Series) -> dict[str, float]:
        """Calculate statistics for a single column."""
        if series.dtype in ["float64", "int64"]:
            return {
                "mean": series.mean(),
                "std": series.std(),
                "min": series.min(),
                "max": series.max(),
                "q25": series.quantile(0.25),
                "q50": series.quantile(0.50),
                "q75": series.quantile(0.75),
            }
        else:
            value_counts = series.value_counts()
            return {
                "unique_count": series.nunique(),
                "most_common": value_counts.index[0] if len(value_counts) > 0 else None,
                "most_common_freq": value_counts.iloc[0] / len(series)
                if len(value_counts) > 0
                else 0,
            }

    def _generate_recommendations(
        self,
        feature_drift: pd.DataFrame,
        target_drift: float,
        concept_drift: float,
        quality_issues: list[str],
    ) -> list[str]:
        """Generate recommendations based on drift detection."""
        recommendations = []

        # Feature drift recommendations
        drifted_features = feature_drift[feature_drift["drift_detected"]][
            "feature"
        ].tolist()
        if drifted_features:
            recommendations.append(
                f"Retrain model - features with drift: {drifted_features[:5]}"
            )

        # Target drift recommendations
        if target_drift > self.drift_threshold:
            recommendations.append("Investigate prediction distribution shift")
            recommendations.append("Consider adjusting decision threshold")

        # Concept drift recommendations
        if concept_drift > self.drift_threshold:
            recommendations.append(
                "Potential concept drift detected - validate with recent labels"
            )
            recommendations.append("Consider online learning or frequent retraining")

        # Data quality recommendations
        if quality_issues:
            recommendations.append("Address data quality issues before model inference")
            recommendations.extend(quality_issues[:3])

        if not recommendations:
            recommendations.append(
                "No significant issues detected - continue monitoring"
            )

        return recommendations


class ModelExplainer:
    """Model explainability using SHAP and other methods.
    """

    def __init__(self, model: BaseEstimator, X_background: pd.DataFrame = None):
        """Initialize model explainer.

        Args:
            model: Model to explain
            X_background: Background data for SHAP
        """
        self.model = model
        self.X_background = X_background
        self.explainer = None
        self._initialize_explainer()

    def _initialize_explainer(self):
        """Initialize SHAP explainer."""
        if self.X_background is not None:
            if hasattr(self.model, "predict_proba"):
                self.explainer = shap.Explainer(
                    lambda x: self.model.predict_proba(x)[:, 1],
                    self.X_background.sample(min(100, len(self.X_background))),
                )
            else:
                self.explainer = shap.Explainer(
                    self.model.predict,
                    self.X_background.sample(min(100, len(self.X_background))),
                )

    def explain_prediction(
        self, X_instance: pd.DataFrame, plot: bool = True
    ) -> dict[str, Any]:
        """Explain a single prediction.

        Args:
            X_instance: Single instance to explain
            plot: Whether to create visualization

        Returns:
            Dictionary with explanation details
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized")

        # Calculate SHAP values
        shap_values = self.explainer(X_instance)

        # Get feature importance for this instance
        feature_importance = pd.DataFrame(
            {
                "feature": X_instance.columns,
                "value": X_instance.iloc[0].values,
                "shap_value": shap_values.values[0],
                "abs_shap": np.abs(shap_values.values[0]),
            }
        ).sort_values("abs_shap", ascending=False)

        # Get prediction
        if hasattr(self.model, "predict_proba"):
            prediction = self.model.predict_proba(X_instance)[0, 1]
        else:
            prediction = self.model.predict(X_instance)[0]

        # Create explanation
        explanation = {
            "prediction": prediction,
            "base_value": shap_values.base_values[0]
            if hasattr(shap_values, "base_values")
            else 0,
            "feature_contributions": feature_importance,
            "top_positive_features": feature_importance[
                feature_importance["shap_value"] > 0
            ].head(5),
            "top_negative_features": feature_importance[
                feature_importance["shap_value"] < 0
            ].head(5),
        }

        if plot:
            self._plot_explanation(shap_values, X_instance)

        return explanation

    def explain_model_global(
        self, X_sample: pd.DataFrame, plot: bool = True
    ) -> dict[str, Any]:
        """Generate global model explanations.

        Args:
            X_sample: Sample of data for explanation
            plot: Whether to create visualizations

        Returns:
            Dictionary with global explanation details
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized")

        # Calculate SHAP values for sample
        shap_values = self.explainer(X_sample)

        # Calculate global feature importance
        global_importance = pd.DataFrame(
            {
                "feature": X_sample.columns,
                "importance": np.abs(shap_values.values).mean(axis=0),
            }
        ).sort_values("importance", ascending=False)

        # Calculate feature interactions
        if len(X_sample) < 1000:  # Limit for computational efficiency
            interaction_values = (
                shap.Interaction().values if hasattr(shap, "Interaction") else None
            )
        else:
            interaction_values = None

        explanation = {
            "global_importance": global_importance,
            "mean_abs_shap": global_importance["importance"].values,
            "feature_interactions": interaction_values,
        }

        if plot:
            self._plot_global_explanation(shap_values, X_sample)

        return explanation

    def _plot_explanation(self, shap_values, X_instance):
        """Create explanation plots."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Waterfall plot
        shap.plots.waterfall(shap_values[0], show=False)
        plt.sca(axes[0])

        # Force plot (as matplotlib plot)
        # Note: Force plot might not work in all environments
        try:
            shap.plots.force(shap_values[0], matplotlib=True, show=False)
            plt.sca(axes[1])
        except:
            pass

        plt.tight_layout()
        plt.show()

    def _plot_global_explanation(self, shap_values, X_sample):
        """Create global explanation plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Summary plot
        plt.sca(axes[0, 0])
        shap.summary_plot(shap_values.values, X_sample, show=False, plot_type="bar")

        # Summary plot (dot)
        plt.sca(axes[0, 1])
        shap.summary_plot(shap_values.values, X_sample, show=False)

        # Dependence plots for top features
        top_features = (
            pd.DataFrame(
                {
                    "feature": X_sample.columns,
                    "importance": np.abs(shap_values.values).mean(axis=0),
                }
            )
            .sort_values("importance", ascending=False)["feature"]
            .head(2)
        )

        for idx, feature in enumerate(top_features):
            plt.sca(axes[1, idx])
            feature_idx = X_sample.columns.get_loc(feature)
            shap.dependence_plot(
                feature_idx, shap_values.values, X_sample, show=False, ax=axes[1, idx]
            )

        plt.tight_layout()
        plt.show()


class ABTestingFramework:
    """A/B testing framework for model deployment.
    """

    def __init__(
        self,
        model_a: BaseEstimator,
        model_b: BaseEstimator,
        traffic_split: float = 0.5,
        min_sample_size: int = 1000,
        confidence_level: float = 0.95,
    ):
        """Initialize A/B testing framework.

        Args:
            model_a: Control model
            model_b: Treatment model
            traffic_split: Proportion of traffic for model B
            min_sample_size: Minimum sample size for significance
            confidence_level: Confidence level for tests
        """
        self.model_a = model_a
        self.model_b = model_b
        self.traffic_split = traffic_split
        self.min_sample_size = min_sample_size
        self.confidence_level = confidence_level
        self.results_a = []
        self.results_b = []

    def assign_variant(self, user_id: str | int) -> str:
        """Assign user to variant based on consistent hash.

        Args:
            user_id: User identifier

        Returns:
            'A' or 'B'
        """
        # Use consistent hashing for assignment
        hash_val = int(hashlib.md5(str(user_id).encode()).hexdigest(), 16)
        assignment_prob = (hash_val % 1000) / 1000

        return "B" if assignment_prob < self.traffic_split else "A"

    def predict(
        self, X: pd.DataFrame, user_id: str | int
    ) -> tuple[np.ndarray, str]:
        """Make prediction using assigned model.

        Args:
            X: Features
            user_id: User identifier

        Returns:
            Tuple of (prediction, variant)
        """
        variant = self.assign_variant(user_id)

        if variant == "A":
            model = self.model_a
        else:
            model = self.model_b

        if hasattr(model, "predict_proba"):
            prediction = model.predict_proba(X)[:, 1]
        else:
            prediction = model.predict(X)

        return prediction, variant

    def record_result(
        self,
        variant: str,
        prediction: float,
        actual: float,
        metadata: dict[str, Any] = None,
    ):
        """Record result for analysis.

        Args:
            variant: 'A' or 'B'
            prediction: Model prediction
            actual: Actual outcome
            metadata: Additional metadata
        """
        result = {
            "timestamp": datetime.now(),
            "prediction": prediction,
            "actual": actual,
            "error": abs(prediction - actual),
            "metadata": metadata or {},
        }

        if variant == "A":
            self.results_a.append(result)
        else:
            self.results_b.append(result)

    def analyze_results(self) -> dict[str, Any]:
        """Analyze A/B test results.

        Returns:
            Dictionary with test results
        """
        if (
            len(self.results_a) < self.min_sample_size
            or len(self.results_b) < self.min_sample_size
        ):
            return {
                "status": "insufficient_data",
                "samples_a": len(self.results_a),
                "samples_b": len(self.results_b),
                "required": self.min_sample_size,
            }

        # Calculate metrics for each variant
        metrics_a = self._calculate_metrics(self.results_a)
        metrics_b = self._calculate_metrics(self.results_b)

        # Statistical significance testing
        # Using bootstrap for confidence intervals
        diff_metrics = {}

        for metric in ["accuracy", "auc", "error"]:
            samples_a = [r.get(metric, r.get("error", 0)) for r in self.results_a]
            samples_b = [r.get(metric, r.get("error", 0)) for r in self.results_b]

            # Bootstrap difference
            diff_samples = []
            for _ in range(1000):
                sample_a = np.random.choice(samples_a, len(samples_a), replace=True)
                sample_b = np.random.choice(samples_b, len(samples_b), replace=True)
                diff_samples.append(np.mean(sample_b) - np.mean(sample_a))

            # Calculate confidence interval
            ci_lower = np.percentile(
                diff_samples, (1 - self.confidence_level) / 2 * 100
            )
            ci_upper = np.percentile(
                diff_samples, (1 + self.confidence_level) / 2 * 100
            )

            # Check significance
            significant = not (ci_lower <= 0 <= ci_upper)

            diff_metrics[metric] = {
                "difference": metrics_b[metric] - metrics_a[metric],
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "significant": significant,
                "relative_change": (metrics_b[metric] - metrics_a[metric])
                / metrics_a[metric]
                * 100,
            }

        # Determine winner
        if diff_metrics["error"]["significant"]:
            winner = "B" if diff_metrics["error"]["difference"] < 0 else "A"
        else:
            winner = "no_significant_difference"

        return {
            "status": "complete",
            "samples_a": len(self.results_a),
            "samples_b": len(self.results_b),
            "metrics_a": metrics_a,
            "metrics_b": metrics_b,
            "differences": diff_metrics,
            "winner": winner,
            "confidence_level": self.confidence_level,
        }

    def _calculate_metrics(self, results: list[dict]) -> dict[str, float]:
        """Calculate metrics for a variant."""
        predictions = np.array([r["prediction"] for r in results])
        actuals = np.array([r["actual"] for r in results])

        # Binary classification metrics
        pred_binary = (predictions >= 0.5).astype(int)
        accuracy = np.mean(pred_binary == actuals)

        # AUC if we have both classes
        if len(np.unique(actuals)) > 1:
            auc = roc_auc_score(actuals, predictions)
        else:
            auc = 0.5

        # Error metrics
        mae = np.mean(np.abs(predictions - actuals))

        return {
            "accuracy": accuracy,
            "auc": auc,
            "error": mae,
            "mean_prediction": np.mean(predictions),
            "mean_actual": np.mean(actuals),
        }


if __name__ == "__main__":
    print("Enhanced Production Module Loaded Successfully")
    print("=" * 60)
    print("Available Classes:")
    print("  - ModelVersionControl: Model versioning and management")
    print("  - DriftDetector: Data and concept drift detection")
    print("  - ModelExplainer: SHAP-based model explanations")
    print("  - ABTestingFramework: A/B testing for model deployment")
