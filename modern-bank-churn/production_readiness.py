"""
Production Readiness Components for Bank Churn Models.
Implements model versioning, monitoring, drift detection, explainability, and A/B testing.
"""

import hashlib
import json
import pickle
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from scipy import stats
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

warnings.filterwarnings("ignore")


@dataclass
class ModelVersion:
    """Container for model version information."""

    version_id: str
    model_name: str
    timestamp: datetime
    metrics: Dict[str, float]
    parameters: Dict[str, Any]
    feature_names: List[str]
    training_data_hash: str
    model_path: str
    metadata: Dict[str, Any]


@dataclass
class DriftReport:
    """Container for drift detection results."""

    timestamp: datetime
    feature_drift: Dict[str, float]
    prediction_drift: float
    performance_drift: Dict[str, float]
    has_drift: bool
    drift_severity: str
    recommendations: List[str]


@dataclass
class MonitoringMetrics:
    """Container for monitoring metrics."""

    timestamp: datetime
    prediction_volume: int
    avg_confidence: float
    performance_metrics: Dict[str, float]
    latency_ms: float
    error_rate: float
    data_quality_issues: List[str]


class ModelVersionControl:
    """Model versioning and registry system."""

    def __init__(self, base_path: str = "./model_registry"):
        """
        Initialize model version control.

        Args:
            base_path: Base path for model storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.registry = {}
        self.current_version = None
        self.load_registry()

    def save_model(
        self,
        model,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        metrics: Dict[str, float],
        parameters: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None,
    ) -> str:
        """
        Save model with versioning.

        Args:
            model: Trained model
            model_name: Name of the model
            X_train: Training features
            y_train: Training labels
            metrics: Model metrics
            parameters: Model parameters
            metadata: Additional metadata

        Returns:
            Version ID
        """
        # Generate version ID
        version_id = self._generate_version_id(model_name)

        # Create model directory
        model_dir = self.base_path / model_name / version_id
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = model_dir / "model.pkl"
        joblib.dump(model, model_path)

        # Calculate data hash
        data_hash = self._calculate_data_hash(X_train, y_train)

        # Create version info
        version_info = ModelVersion(
            version_id=version_id,
            model_name=model_name,
            timestamp=datetime.now(),
            metrics=metrics,
            parameters=parameters or {},
            feature_names=X_train.columns.tolist(),
            training_data_hash=data_hash,
            model_path=str(model_path),
            metadata=metadata or {},
        )

        # Save version info
        version_info_path = model_dir / "version_info.json"
        with open(version_info_path, "w") as f:
            json.dump(asdict(version_info), f, default=str, indent=2)

        # Update registry
        if model_name not in self.registry:
            self.registry[model_name] = {}
        self.registry[model_name][version_id] = version_info

        # Save registry
        self.save_registry()

        return version_id

    def load_model(self, model_name: str, version_id: str = None):
        """
        Load model by version.

        Args:
            model_name: Model name
            version_id: Version ID (latest if None)

        Returns:
            Loaded model and version info
        """
        if model_name not in self.registry:
            raise ValueError(f"Model {model_name} not found")

        if version_id is None:
            # Get latest version
            versions = self.registry[model_name]
            version_id = max(versions.keys())

        version_info = self.registry[model_name][version_id]
        model = joblib.load(version_info.model_path)

        return model, version_info

    def compare_versions(
        self, model_name: str, version1: str, version2: str
    ) -> pd.DataFrame:
        """
        Compare two model versions.

        Args:
            model_name: Model name
            version1: First version ID
            version2: Second version ID

        Returns:
            Comparison DataFrame
        """
        v1_info = self.registry[model_name][version1]
        v2_info = self.registry[model_name][version2]

        comparison = []

        # Compare metrics
        for metric in set(v1_info.metrics.keys()) | set(v2_info.metrics.keys()):
            comparison.append(
                {
                    "aspect": f"metric_{metric}",
                    "version1": v1_info.metrics.get(metric),
                    "version2": v2_info.metrics.get(metric),
                    "improvement": (
                        v2_info.metrics.get(metric, 0) - v1_info.metrics.get(metric, 0)
                        if metric in v1_info.metrics and metric in v2_info.metrics
                        else None
                    ),
                }
            )

        # Compare features
        v1_features = set(v1_info.feature_names)
        v2_features = set(v2_info.feature_names)
        comparison.append(
            {
                "aspect": "n_features",
                "version1": len(v1_features),
                "version2": len(v2_features),
                "improvement": len(v2_features) - len(v1_features),
            }
        )

        return pd.DataFrame(comparison)

    def _generate_version_id(self, model_name: str) -> str:
        """Generate unique version ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.md5(str(np.random.random()).encode()).hexdigest()[:6]
        return f"{model_name}_{timestamp}_{random_suffix}"

    def _calculate_data_hash(self, X: pd.DataFrame, y: pd.Series) -> str:
        """Calculate hash of training data."""
        data_str = (
            f"{X.shape}_{X.columns.tolist()}_{y.shape}_{y.value_counts().to_dict()}"
        )
        return hashlib.md5(data_str.encode()).hexdigest()

    def save_registry(self):
        """Save registry to file."""
        registry_path = self.base_path / "registry.json"
        registry_dict = {}
        for model_name, versions in self.registry.items():
            registry_dict[model_name] = {}
            for version_id, version_info in versions.items():
                registry_dict[model_name][version_id] = asdict(version_info)

        with open(registry_path, "w") as f:
            json.dump(registry_dict, f, default=str, indent=2)

    def load_registry(self):
        """Load registry from file."""
        registry_path = self.base_path / "registry.json"
        if registry_path.exists():
            with open(registry_path, "r") as f:
                registry_dict = json.load(f)

            for model_name, versions in registry_dict.items():
                self.registry[model_name] = {}
                for version_id, version_dict in versions.items():
                    # Convert dict back to ModelVersion
                    version_info = ModelVersion(**version_dict)
                    self.registry[model_name][version_id] = version_info


class ModelMonitor:
    """Real-time model monitoring."""

    def __init__(self, alert_thresholds: Dict[str, float] = None):
        """
        Initialize model monitor.

        Args:
            alert_thresholds: Thresholds for alerts
        """
        if alert_thresholds is None:
            alert_thresholds = {
                "error_rate": 0.05,
                "latency_ms": 1000,
                "performance_drop": 0.1,
                "drift_score": 0.15,
            }

        self.alert_thresholds = alert_thresholds
        self.monitoring_history = []
        self.alerts = []

    def log_prediction(
        self,
        features: pd.DataFrame,
        prediction: np.ndarray,
        latency_ms: float,
        error: bool = False,
    ):
        """
        Log single prediction for monitoring.

        Args:
            features: Input features
            prediction: Model prediction
            latency_ms: Prediction latency
            error: Whether prediction resulted in error
        """
        timestamp = datetime.now()

        # Check for data quality issues
        data_quality_issues = self._check_data_quality(features)

        # Store monitoring data
        self.monitoring_history.append(
            {
                "timestamp": timestamp,
                "prediction": prediction,
                "latency_ms": latency_ms,
                "error": error,
                "data_quality_issues": data_quality_issues,
            }
        )

        # Check for alerts
        self._check_alerts(latency_ms, error, data_quality_issues)

    def get_monitoring_metrics(self, window_size: int = 100) -> MonitoringMetrics:
        """
        Calculate monitoring metrics for recent predictions.

        Args:
            window_size: Number of recent predictions to analyze

        Returns:
            MonitoringMetrics object
        """
        if not self.monitoring_history:
            return None

        recent = self.monitoring_history[-window_size:]

        # Calculate metrics
        predictions = [h["prediction"] for h in recent if h["prediction"] is not None]
        latencies = [h["latency_ms"] for h in recent]
        errors = [h["error"] for h in recent]
        quality_issues = [h["data_quality_issues"] for h in recent]

        # Aggregate data quality issues
        all_issues = []
        for issues in quality_issues:
            all_issues.extend(issues)

        metrics = MonitoringMetrics(
            timestamp=datetime.now(),
            prediction_volume=len(recent),
            avg_confidence=np.mean(predictions) if predictions else 0,
            performance_metrics={},  # Would need ground truth
            latency_ms=np.mean(latencies),
            error_rate=np.mean(errors),
            data_quality_issues=list(set(all_issues)),
        )

        return metrics

    def _check_data_quality(self, features: pd.DataFrame) -> List[str]:
        """Check for data quality issues."""
        issues = []

        # Check for missing values
        if features.isnull().any().any():
            missing_cols = features.columns[features.isnull().any()].tolist()
            issues.append(f"Missing values in: {missing_cols}")

        # Check for extreme values (outside 5 std)
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            values = features[col].values
            mean = np.mean(values)
            std = np.std(values)
            if np.any(np.abs(values - mean) > 5 * std):
                issues.append(f"Extreme values in: {col}")

        return issues

    def _check_alerts(
        self, latency_ms: float, error: bool, data_quality_issues: List[str]
    ):
        """Check and trigger alerts."""
        alerts = []

        # Latency alert
        if latency_ms > self.alert_thresholds["latency_ms"]:
            alerts.append(f"High latency: {latency_ms:.0f}ms")

        # Error alert
        if error:
            alerts.append("Prediction error occurred")

        # Data quality alert
        if data_quality_issues:
            alerts.append(f"Data quality issues: {data_quality_issues}")

        if alerts:
            self.alerts.append({"timestamp": datetime.now(), "alerts": alerts})


class DriftDetector:
    """Detect data and concept drift."""

    def __init__(
        self, reference_data: pd.DataFrame, reference_predictions: np.ndarray = None
    ):
        """
        Initialize drift detector.

        Args:
            reference_data: Reference feature data
            reference_predictions: Reference predictions
        """
        self.reference_data = reference_data
        self.reference_predictions = reference_predictions
        self.drift_history = []

    def detect_drift(
        self,
        current_data: pd.DataFrame,
        current_predictions: np.ndarray = None,
        method: str = "ks",
    ) -> DriftReport:
        """
        Detect drift in data and predictions.

        Args:
            current_data: Current feature data
            current_predictions: Current predictions
            method: Statistical test method

        Returns:
            DriftReport object
        """
        feature_drift = {}
        has_drift = False
        severity = "none"
        recommendations = []

        # Feature drift
        for column in self.reference_data.columns:
            if column not in current_data.columns:
                continue

            ref_values = self.reference_data[column].values
            curr_values = current_data[column].values

            if method == "ks":
                # Kolmogorov-Smirnov test
                if np.issubdtype(ref_values.dtype, np.number):
                    statistic, p_value = stats.ks_2samp(ref_values, curr_values)
                    feature_drift[column] = {
                        "statistic": statistic,
                        "p_value": p_value,
                        "has_drift": p_value < 0.05,
                    }

                    if p_value < 0.05:
                        has_drift = True
                        recommendations.append(
                            f"Feature '{column}' shows significant drift (p={p_value:.4f})"
                        )

            elif method == "psi":
                # Population Stability Index
                psi = self._calculate_psi(ref_values, curr_values)
                feature_drift[column] = {"psi": psi, "has_drift": psi > 0.1}

                if psi > 0.2:
                    has_drift = True
                    severity = "high"
                    recommendations.append(
                        f"Feature '{column}' has high PSI ({psi:.3f})"
                    )
                elif psi > 0.1:
                    has_drift = True
                    if severity != "high":
                        severity = "medium"

        # Prediction drift
        prediction_drift = 0
        if self.reference_predictions is not None and current_predictions is not None:
            # Compare prediction distributions
            statistic, p_value = stats.ks_2samp(
                self.reference_predictions, current_predictions
            )
            prediction_drift = statistic

            if p_value < 0.05:
                has_drift = True
                recommendations.append(
                    f"Prediction distribution drift detected (p={p_value:.4f})"
                )

        # Performance drift (would need ground truth)
        performance_drift = {}

        # Severity assessment
        if not has_drift:
            severity = "none"
        elif len([f for f in feature_drift.values() if f.get("has_drift", False)]) > 5:
            severity = "high"

        # Recommendations
        if severity == "high":
            recommendations.append("Consider retraining the model immediately")
        elif severity == "medium":
            recommendations.append("Monitor closely and prepare for retraining")

        drift_report = DriftReport(
            timestamp=datetime.now(),
            feature_drift=feature_drift,
            prediction_drift=prediction_drift,
            performance_drift=performance_drift,
            has_drift=has_drift,
            drift_severity=severity,
            recommendations=recommendations,
        )

        self.drift_history.append(drift_report)
        return drift_report

    def _calculate_psi(
        self, reference: np.ndarray, current: np.ndarray, n_bins: int = 10
    ) -> float:
        """Calculate Population Stability Index."""
        # Create bins from reference
        if np.issubdtype(reference.dtype, np.number):
            _, bins = pd.qcut(reference, q=n_bins, retbins=True, duplicates="drop")

            # Calculate proportions
            ref_counts = pd.cut(
                reference, bins=bins, include_lowest=True
            ).value_counts()
            ref_prop = ref_counts / len(reference)

            curr_counts = pd.cut(current, bins=bins, include_lowest=True).value_counts()
            curr_prop = curr_counts / len(current)

            # Align and fill zeros
            ref_prop = ref_prop.reindex(
                ref_prop.index.union(curr_prop.index), fill_value=0.0001
            )
            curr_prop = curr_prop.reindex(ref_prop.index, fill_value=0.0001)

            # Calculate PSI
            psi = np.sum((curr_prop - ref_prop) * np.log(curr_prop / ref_prop))
        else:
            # For categorical variables
            ref_counts = pd.Series(reference).value_counts(normalize=True)
            curr_counts = pd.Series(current).value_counts(normalize=True)

            # Align categories
            all_categories = set(ref_counts.index) | set(curr_counts.index)
            ref_prop = ref_counts.reindex(all_categories, fill_value=0.0001)
            curr_prop = curr_counts.reindex(all_categories, fill_value=0.0001)

            psi = np.sum((curr_prop - ref_prop) * np.log(curr_prop / ref_prop))

        return psi

    def plot_drift_history(self) -> plt.Figure:
        """Plot drift history over time."""
        if not self.drift_history:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Extract drift scores over time
        timestamps = [d.timestamp for d in self.drift_history]
        prediction_drifts = [d.prediction_drift for d in self.drift_history]
        severities = [d.drift_severity for d in self.drift_history]

        # Prediction drift over time
        axes[0, 0].plot(timestamps, prediction_drifts, "o-")
        axes[0, 0].axhline(
            y=0.1, color="r", linestyle="--", alpha=0.5, label="Alert threshold"
        )
        axes[0, 0].set_xlabel("Time")
        axes[0, 0].set_ylabel("Prediction Drift")
        axes[0, 0].set_title("Prediction Drift Over Time")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Severity distribution
        severity_counts = pd.Series(severities).value_counts()
        axes[0, 1].bar(severity_counts.index, severity_counts.values)
        axes[0, 1].set_xlabel("Severity")
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].set_title("Drift Severity Distribution")
        axes[0, 1].grid(True, alpha=0.3)

        # Feature drift heatmap (latest)
        if self.drift_history:
            latest_drift = self.drift_history[-1]
            feature_scores = []
            feature_names = []

            for feat, drift_info in latest_drift.feature_drift.items():
                feature_names.append(feat)
                if "statistic" in drift_info:
                    feature_scores.append(drift_info["statistic"])
                elif "psi" in drift_info:
                    feature_scores.append(drift_info["psi"])

            if feature_scores:
                # Sort by drift score
                sorted_idx = np.argsort(feature_scores)[::-1][:10]  # Top 10
                top_features = [feature_names[i] for i in sorted_idx]
                top_scores = [feature_scores[i] for i in sorted_idx]

                axes[1, 0].barh(range(len(top_features)), top_scores)
                axes[1, 0].set_yticks(range(len(top_features)))
                axes[1, 0].set_yticklabels(top_features)
                axes[1, 0].set_xlabel("Drift Score")
                axes[1, 0].set_title("Top Drifting Features")
                axes[1, 0].grid(True, alpha=0.3)

        # Drift alerts over time
        alert_counts = [len(d.recommendations) for d in self.drift_history]
        axes[1, 1].plot(timestamps, alert_counts, "o-", color="red")
        axes[1, 1].fill_between(timestamps, alert_counts, alpha=0.3, color="red")
        axes[1, 1].set_xlabel("Time")
        axes[1, 1].set_ylabel("Number of Alerts")
        axes[1, 1].set_title("Drift Alerts Over Time")
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle("Drift Monitoring Dashboard", fontsize=16)
        plt.tight_layout()

        return fig


class ModelExplainer:
    """Model explainability using SHAP and LIME."""

    def __init__(self, model, X_train: pd.DataFrame, feature_names: List[str] = None):
        """
        Initialize explainer.

        Args:
            model: Trained model
            X_train: Training data for background
            feature_names: Feature names
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names or X_train.columns.tolist()

        # Initialize SHAP explainer
        try:
            self.shap_explainer = shap.TreeExplainer(model)
        except:
            # Fall back to kernel explainer for non-tree models
            self.shap_explainer = shap.KernelExplainer(
                (
                    model.predict_proba
                    if hasattr(model, "predict_proba")
                    else model.predict
                ),
                shap.sample(X_train, 100),
            )

        # Initialize LIME explainer
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train.values,
            feature_names=self.feature_names,
            class_names=["No Churn", "Churn"],
            mode="classification",
        )

    def explain_prediction_shap(self, X: pd.DataFrame, plot: bool = True) -> Dict:
        """
        Explain prediction using SHAP.

        Args:
            X: Features to explain
            plot: Whether to create plots

        Returns:
            SHAP values and explanation
        """
        shap_values = self.shap_explainer.shap_values(X)

        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            # Binary classification - take positive class
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        # Create explanation dict
        explanation = {
            "shap_values": shap_values,
            "base_value": self.shap_explainer.expected_value,
            "feature_importance": pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "importance": np.abs(shap_values).mean(axis=0),
                }
            ).sort_values("importance", ascending=False),
        }

        if plot and len(X) == 1:
            # Force plot for single prediction
            shap.force_plot(
                self.shap_explainer.expected_value,
                shap_values[0],
                X.iloc[0],
                feature_names=self.feature_names,
            )

        return explanation

    def explain_prediction_lime(self, X: pd.DataFrame, num_features: int = 10) -> Dict:
        """
        Explain prediction using LIME.

        Args:
            X: Features to explain (single row)
            num_features: Number of features to show

        Returns:
            LIME explanation
        """
        if len(X) != 1:
            raise ValueError("LIME explanation requires single instance")

        # Get explanation
        explanation = self.lime_explainer.explain_instance(
            X.values[0],
            (
                self.model.predict_proba
                if hasattr(self.model, "predict_proba")
                else self.model.predict
            ),
            num_features=num_features,
        )

        # Extract feature contributions
        feature_contributions = explanation.as_list()

        return {
            "explanation": explanation,
            "feature_contributions": feature_contributions,
            "prediction_proba": explanation.predict_proba,
        }

    def global_feature_importance(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Calculate global feature importance.

        Args:
            n_samples: Number of samples to use

        Returns:
            DataFrame with feature importance
        """
        # Sample data
        X_sample = self.X_train.sample(min(n_samples, len(self.X_train)))

        # Calculate SHAP values
        shap_values = self.shap_explainer.shap_values(X_sample)

        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        # Calculate importance
        importance = pd.DataFrame(
            {
                "feature": self.feature_names,
                "shap_importance": np.abs(shap_values).mean(axis=0),
                "shap_std": np.abs(shap_values).std(axis=0),
            }
        )

        # Add model's native importance if available
        if hasattr(self.model, "feature_importances_"):
            importance["model_importance"] = self.model.feature_importances_

        importance = importance.sort_values("shap_importance", ascending=False)

        return importance


class ABTestingFramework:
    """A/B testing for model deployment."""

    def __init__(self, model_a, model_b, traffic_split: float = 0.5):
        """
        Initialize A/B testing framework.

        Args:
            model_a: Control model
            model_b: Treatment model
            traffic_split: Proportion of traffic for model B
        """
        self.model_a = model_a
        self.model_b = model_b
        self.traffic_split = traffic_split
        self.test_results = {"model_a": [], "model_b": []}

    def route_traffic(self, X: pd.DataFrame) -> Tuple[str, np.ndarray]:
        """
        Route traffic to models based on split.

        Args:
            X: Features

        Returns:
            Model name and prediction
        """
        if np.random.random() < self.traffic_split:
            # Route to model B
            prediction = self.model_b.predict_proba(X)[:, 1]
            model_name = "model_b"
        else:
            # Route to model A
            prediction = self.model_a.predict_proba(X)[:, 1]
            model_name = "model_a"

        return model_name, prediction

    def log_result(
        self,
        model_name: str,
        features: pd.DataFrame,
        prediction: np.ndarray,
        actual: int = None,
        business_value: float = None,
    ):
        """
        Log A/B test result.

        Args:
            model_name: Which model was used
            features: Input features
            prediction: Model prediction
            actual: Actual outcome (if available)
            business_value: Business value of this prediction
        """
        result = {
            "timestamp": datetime.now(),
            "prediction": prediction,
            "actual": actual,
            "business_value": business_value,
        }

        self.test_results[model_name].append(result)

    def analyze_results(self, min_samples: int = 100) -> Dict:
        """
        Analyze A/B test results.

        Args:
            min_samples: Minimum samples for significance

        Returns:
            Analysis results
        """
        results_a = pd.DataFrame(self.test_results["model_a"])
        results_b = pd.DataFrame(self.test_results["model_b"])

        if len(results_a) < min_samples or len(results_b) < min_samples:
            return {
                "status": "insufficient_data",
                "samples_a": len(results_a),
                "samples_b": len(results_b),
                "min_required": min_samples,
            }

        analysis = {}

        # Compare performance (if actuals available)
        if "actual" in results_a.columns and results_a["actual"].notna().any():
            # Calculate metrics for both models
            metrics_a = self._calculate_metrics(
                results_a["actual"].dropna().values,
                results_a.loc[results_a["actual"].notna(), "prediction"].values,
            )
            metrics_b = self._calculate_metrics(
                results_b["actual"].dropna().values,
                results_b.loc[results_b["actual"].notna(), "prediction"].values,
            )

            analysis["metrics"] = {
                "model_a": metrics_a,
                "model_b": metrics_b,
                "improvement": {
                    k: metrics_b[k] - metrics_a[k] for k in metrics_a.keys()
                },
            }

            # Statistical significance test
            from scipy.stats import mannwhitneyu

            statistic, p_value = mannwhitneyu(
                results_a.loc[results_a["actual"].notna(), "prediction"],
                results_b.loc[results_b["actual"].notna(), "prediction"],
            )

            analysis["significance"] = {
                "statistic": statistic,
                "p_value": p_value,
                "significant": p_value < 0.05,
            }

        # Compare business value (if available)
        if "business_value" in results_a.columns:
            value_a = results_a["business_value"].mean()
            value_b = results_b["business_value"].mean()

            analysis["business_value"] = {
                "model_a": value_a,
                "model_b": value_b,
                "improvement": value_b - value_a,
                "relative_improvement": (value_b - value_a) / (value_a + 1e-10),
            }

        # Recommendation
        if "metrics" in analysis and "business_value" in analysis:
            # Weighted decision
            metric_improvement = np.mean(
                list(analysis["metrics"]["improvement"].values())
            )
            value_improvement = analysis["business_value"]["relative_improvement"]

            if metric_improvement > 0.05 and value_improvement > 0.05:
                analysis["recommendation"] = "deploy_model_b"
            elif metric_improvement < -0.05 or value_improvement < -0.05:
                analysis["recommendation"] = "keep_model_a"
            else:
                analysis["recommendation"] = "continue_testing"

        return analysis

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate performance metrics."""
        y_pred_binary = (y_pred >= 0.5).astype(int)

        return {
            "auc": roc_auc_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred_binary),
            "recall": recall_score(y_true, y_pred_binary),
            "f1": f1_score(y_true, y_pred_binary),
        }


# Example usage
if __name__ == "__main__":
    # Create sample data and model
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier

    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(20)])
    y = pd.Series(y)

    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 1. Model Versioning
    print("1. Model Versioning")
    version_control = ModelVersionControl()
    version_id = version_control.save_model(
        model,
        "churn_model",
        X_train,
        y_train,
        metrics={"auc": 0.85, "f1": 0.78},
        parameters={"n_estimators": 100},
    )
    print(f"Saved model with version: {version_id}")

    # 2. Drift Detection
    print("\n2. Drift Detection")
    drift_detector = DriftDetector(X_train)
    drift_report = drift_detector.detect_drift(X_test)
    print(f"Drift detected: {drift_report.has_drift}")
    print(f"Severity: {drift_report.drift_severity}")

    # 3. Model Explainability
    print("\n3. Model Explainability")
    explainer = ModelExplainer(model, X_train)
    importance = explainer.global_feature_importance(n_samples=100)
    print("Top 5 important features:")
    print(importance.head())

    # 4. Model Monitoring
    print("\n4. Model Monitoring")
    monitor = ModelMonitor()
    for i in range(10):
        pred = model.predict_proba(X_test.iloc[[i]])[:, 1]
        monitor.log_prediction(
            X_test.iloc[[i]], pred, latency_ms=np.random.uniform(10, 100)
        )

    metrics = monitor.get_monitoring_metrics()
    print(f"Average latency: {metrics.latency_ms:.2f}ms")
    print(f"Error rate: {metrics.error_rate:.2%}")

    print("\nProduction readiness components initialized successfully!")
