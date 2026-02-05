"""
Anomaly Detection Utilities Module

This module provides utility functions and classes for anomaly detection in time series data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from scipy import stats
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)
import warnings

warnings.filterwarnings("ignore")


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class AnomalyPattern:
    """Represents different types of anomaly patterns."""

    POINT = "point"  # Single point anomaly
    CONTEXTUAL = "contextual"  # Anomaly in context
    COLLECTIVE = "collective"  # Group of points forming anomaly
    SEASONAL = "seasonal"  # Seasonal anomaly
    TREND = "trend"  # Trend change anomaly
    LEVEL_SHIFT = "level_shift"  # Sudden level change
    VARIANCE_CHANGE = "variance_change"  # Change in variance


@dataclass
class AnomalyConfig:
    """Configuration for anomaly detection."""

    contamination: float = 0.1
    sensitivity: float = 0.95
    min_anomaly_duration: int = 1
    max_anomaly_duration: int = 100
    methods: List[str] = field(
        default_factory=lambda: ["zscore", "iqr", "isolation_forest"]
    )
    ensemble_voting: str = "majority"
    use_seasonal_decomposition: bool = True
    adaptive_threshold: bool = True


# ============================================================================
# Anomaly Generation
# ============================================================================


class AnomalyGenerator:
    """Generate synthetic anomalies for testing."""

    @staticmethod
    def inject_point_anomalies(
        data: np.ndarray, anomaly_rate: float = 0.05, magnitude: float = 3.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inject point anomalies into time series.

        Parameters:
        -----------
        data : np.ndarray
            Original time series
        anomaly_rate : float
            Proportion of points to make anomalous
        magnitude : float
            Anomaly magnitude in standard deviations

        Returns:
        --------
        tuple : (modified_data, anomaly_labels)
        """
        modified_data = data.copy()
        n_anomalies = int(len(data) * anomaly_rate)
        anomaly_indices = np.random.choice(len(data), n_anomalies, replace=False)

        std = np.std(data)
        for idx in anomaly_indices:
            # Random positive or negative anomaly
            sign = np.random.choice([-1, 1])
            modified_data[idx] += sign * magnitude * std

        anomaly_labels = np.zeros(len(data), dtype=bool)
        anomaly_labels[anomaly_indices] = True

        return modified_data, anomaly_labels

    @staticmethod
    def inject_collective_anomalies(
        data: np.ndarray,
        n_segments: int = 3,
        segment_length: int = 10,
        magnitude: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inject collective anomalies (anomalous segments).

        Parameters:
        -----------
        data : np.ndarray
            Original time series
        n_segments : int
            Number of anomalous segments
        segment_length : int
            Length of each segment
        magnitude : float
            Anomaly magnitude

        Returns:
        --------
        tuple : (modified_data, anomaly_labels)
        """
        modified_data = data.copy()
        anomaly_labels = np.zeros(len(data), dtype=bool)

        std = np.std(data)

        for _ in range(n_segments):
            start_idx = np.random.randint(0, len(data) - segment_length)
            end_idx = start_idx + segment_length

            # Apply collective anomaly pattern
            pattern = np.random.choice(["shift", "scale", "noise"])

            if pattern == "shift":
                modified_data[start_idx:end_idx] += magnitude * std
            elif pattern == "scale":
                modified_data[start_idx:end_idx] *= 1 + magnitude
            else:  # noise
                modified_data[start_idx:end_idx] += np.random.normal(
                    0, magnitude * std, segment_length
                )

            anomaly_labels[start_idx:end_idx] = True

        return modified_data, anomaly_labels

    @staticmethod
    def inject_contextual_anomalies(
        data: np.ndarray, context_window: int = 24, anomaly_rate: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inject contextual anomalies based on local context.

        Parameters:
        -----------
        data : np.ndarray
            Original time series
        context_window : int
            Size of context window
        anomaly_rate : float
            Proportion of anomalies

        Returns:
        --------
        tuple : (modified_data, anomaly_labels)
        """
        modified_data = data.copy()
        anomaly_labels = np.zeros(len(data), dtype=bool)

        n_anomalies = int(len(data) * anomaly_rate)
        anomaly_indices = np.random.choice(
            range(context_window, len(data) - context_window),
            n_anomalies,
            replace=False,
        )

        for idx in anomaly_indices:
            # Get local context
            context = data[idx - context_window : idx + context_window]
            local_mean = np.mean(context)
            local_std = np.std(context)

            # Make point anomalous relative to context
            modified_data[idx] = local_mean + np.random.choice([-1, 1]) * 3 * local_std
            anomaly_labels[idx] = True

        return modified_data, anomaly_labels


# ============================================================================
# Statistical Tests
# ============================================================================


class StatisticalTests:
    """Statistical tests for anomaly detection."""

    @staticmethod
    def dixon_test(data: np.ndarray, alpha: float = 0.05) -> np.ndarray:
        """
        Dixon's Q test for outliers.

        Parameters:
        -----------
        data : np.ndarray
            Data to test
        alpha : float
            Significance level

        Returns:
        --------
        np.ndarray : Boolean array of outliers
        """
        n = len(data)
        outliers = np.zeros(n, dtype=bool)

        if n < 3:
            return outliers

        sorted_data = np.sort(data)

        # Dixon's Q statistic
        if n <= 7:
            Q_low = (sorted_data[1] - sorted_data[0]) / (
                sorted_data[-1] - sorted_data[0]
            )
            Q_high = (sorted_data[-1] - sorted_data[-2]) / (
                sorted_data[-1] - sorted_data[0]
            )
        elif n <= 10:
            Q_low = (sorted_data[1] - sorted_data[0]) / (
                sorted_data[-2] - sorted_data[0]
            )
            Q_high = (sorted_data[-1] - sorted_data[-2]) / (
                sorted_data[-1] - sorted_data[1]
            )
        else:
            Q_low = (sorted_data[2] - sorted_data[0]) / (
                sorted_data[-2] - sorted_data[0]
            )
            Q_high = (sorted_data[-1] - sorted_data[-3]) / (
                sorted_data[-1] - sorted_data[2]
            )

        # Critical values (simplified)
        critical_values = {
            3: 0.941,
            4: 0.765,
            5: 0.642,
            6: 0.560,
            7: 0.507,
            8: 0.468,
            9: 0.437,
            10: 0.412,
        }

        Q_critical = critical_values.get(min(n, 10), 0.3)

        # Mark outliers
        if Q_low > Q_critical:
            outliers[np.argmin(data)] = True
        if Q_high > Q_critical:
            outliers[np.argmax(data)] = True

        return outliers

    @staticmethod
    def modified_zscore(data: np.ndarray, threshold: float = 3.5) -> np.ndarray:
        """
        Modified Z-score using median absolute deviation.

        Parameters:
        -----------
        data : np.ndarray
            Data to test
        threshold : float
            Modified Z-score threshold

        Returns:
        --------
        np.ndarray : Boolean array of outliers
        """
        median = np.median(data)
        mad = np.median(np.abs(data - median))

        if mad == 0:
            # Use mean absolute deviation if MAD is zero
            mad = np.mean(np.abs(data - median))

        if mad == 0:
            return np.zeros(len(data), dtype=bool)

        modified_z_scores = 0.6745 * (data - median) / mad
        return np.abs(modified_z_scores) > threshold

    @staticmethod
    def peirce_criterion(data: np.ndarray, n_outliers: int = None) -> np.ndarray:
        """
        Peirce's criterion for outlier detection.

        Parameters:
        -----------
        data : np.ndarray
            Data to test
        n_outliers : int
            Expected number of outliers

        Returns:
        --------
        np.ndarray : Boolean array of outliers
        """
        n = len(data)
        if n_outliers is None:
            n_outliers = max(1, int(n * 0.1))

        outliers = np.zeros(n, dtype=bool)

        for _ in range(n_outliers):
            mean = np.mean(data[~outliers])
            std = np.std(data[~outliers])

            if std == 0:
                break

            # Calculate deviations
            deviations = np.abs(data - mean)
            deviations[outliers] = 0

            # Find maximum deviation
            max_idx = np.argmax(deviations)
            max_dev = deviations[max_idx]

            # Peirce's criterion (simplified)
            threshold = std * np.sqrt(2 * np.log(n))

            if max_dev > threshold:
                outliers[max_idx] = True
            else:
                break

        return outliers


# ============================================================================
# Feature Engineering for Anomaly Detection
# ============================================================================


class AnomalyFeatures:
    """Feature engineering for anomaly detection."""

    @staticmethod
    def create_anomaly_features(
        data: pd.Series, window_sizes: List[int] = [10, 30, 100]
    ) -> pd.DataFrame:
        """
        Create features specifically for anomaly detection.

        Parameters:
        -----------
        data : pd.Series
            Time series data
        window_sizes : List[int]
            Window sizes for feature calculation

        Returns:
        --------
        pd.DataFrame : Feature matrix
        """
        features = pd.DataFrame(index=data.index)
        features["value"] = data.values

        for window in window_sizes:
            # Rolling statistics
            features[f"rolling_mean_{window}"] = data.rolling(
                window, center=True
            ).mean()
            features[f"rolling_std_{window}"] = data.rolling(window, center=True).std()
            features[f"rolling_min_{window}"] = data.rolling(window, center=True).min()
            features[f"rolling_max_{window}"] = data.rolling(window, center=True).max()

            # Distance from rolling mean
            features[f"dist_from_mean_{window}"] = np.abs(
                data - features[f"rolling_mean_{window}"]
            )

            # Z-score within window
            features[f"local_zscore_{window}"] = (
                data - features[f"rolling_mean_{window}"]
            ) / features[f"rolling_std_{window}"]

            # Percentile within window
            features[f"percentile_{window}"] = data.rolling(window, center=True).apply(
                lambda x: stats.percentileofscore(x, x.iloc[-1])
                if len(x) == window
                else np.nan
            )

        # Rate of change
        features["rate_of_change"] = data.pct_change()
        features["abs_rate_of_change"] = np.abs(features["rate_of_change"])

        # Acceleration (second derivative)
        features["acceleration"] = features["rate_of_change"].diff()

        # Distance from global statistics
        global_mean = data.mean()
        global_std = data.std()
        features["global_zscore"] = (data - global_mean) / global_std

        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            features[f"lag_{lag}"] = data.shift(lag)
            features[f"diff_lag_{lag}"] = data - features[f"lag_{lag}"]

        return features.fillna(method="bfill").fillna(method="ffill")

    @staticmethod
    def create_temporal_features(data: pd.Series) -> pd.DataFrame:
        """
        Create temporal features for anomaly detection.

        Parameters:
        -----------
        data : pd.Series
            Time series data with datetime index

        Returns:
        --------
        pd.DataFrame : Temporal features
        """
        features = pd.DataFrame(index=data.index)

        if isinstance(data.index, pd.DatetimeIndex):
            features["hour"] = data.index.hour
            features["day"] = data.index.day
            features["dayofweek"] = data.index.dayofweek
            features["month"] = data.index.month
            features["quarter"] = data.index.quarter
            features["dayofyear"] = data.index.dayofyear

            # Cyclical encoding
            features["hour_sin"] = np.sin(2 * np.pi * features["hour"] / 24)
            features["hour_cos"] = np.cos(2 * np.pi * features["hour"] / 24)
            features["day_sin"] = np.sin(2 * np.pi * features["day"] / 31)
            features["day_cos"] = np.cos(2 * np.pi * features["day"] / 31)
            features["month_sin"] = np.sin(2 * np.pi * features["month"] / 12)
            features["month_cos"] = np.cos(2 * np.pi * features["month"] / 12)

            # Binary features
            features["is_weekend"] = (features["dayofweek"] >= 5).astype(int)
            features["is_month_start"] = (features["day"] <= 3).astype(int)
            features["is_month_end"] = (features["day"] >= 28).astype(int)

        return features


# ============================================================================
# Evaluation and Metrics
# ============================================================================


class AnomalyMetrics:
    """Advanced metrics for anomaly detection evaluation."""

    @staticmethod
    def calculate_all_metrics(
        y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive metrics for anomaly detection.

        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        y_scores : np.ndarray
            Anomaly scores (optional)

        Returns:
        --------
        dict : All metrics
        """
        metrics = {}

        # Basic metrics
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
        metrics["f1_score"] = f1_score(y_true, y_pred, zero_division=0)

        # Confusion matrix metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics["true_positives"] = int(tp)
        metrics["true_negatives"] = int(tn)
        metrics["false_positives"] = int(fp)
        metrics["false_negatives"] = int(fn)

        # Additional rates
        metrics["tpr"] = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity
        metrics["tnr"] = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity
        metrics["fpr"] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics["fnr"] = fn / (fn + tp) if (fn + tp) > 0 else 0

        # Balanced accuracy
        metrics["balanced_accuracy"] = (metrics["tpr"] + metrics["tnr"]) / 2

        # Matthews correlation coefficient
        mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if mcc_denom == 0:
            metrics["mcc"] = 0
        else:
            metrics["mcc"] = (tp * tn - fp * fn) / mcc_denom

        # If scores are provided, calculate ranking metrics
        if y_scores is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_scores)
                metrics["pr_auc"] = average_precision_score(y_true, y_scores)
            except:
                metrics["roc_auc"] = 0.5
                metrics["pr_auc"] = np.mean(y_true)

        return metrics

    @staticmethod
    def calculate_segment_metrics(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate segment-based metrics for collective anomalies.

        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels

        Returns:
        --------
        dict : Segment-based metrics
        """

        def get_segments(labels):
            """Extract anomaly segments."""
            segments = []
            start = None

            for i, label in enumerate(labels):
                if label and start is None:
                    start = i
                elif not label and start is not None:
                    segments.append((start, i - 1))
                    start = None

            if start is not None:
                segments.append((start, len(labels) - 1))

            return segments

        true_segments = get_segments(y_true)
        pred_segments = get_segments(y_pred)

        # Calculate overlap-based metrics
        true_positives = 0
        for true_seg in true_segments:
            for pred_seg in pred_segments:
                # Check if segments overlap
                if not (pred_seg[1] < true_seg[0] or pred_seg[0] > true_seg[1]):
                    true_positives += 1
                    break

        false_positives = len(pred_segments) - true_positives
        false_negatives = len(true_segments) - true_positives

        # Calculate metrics
        precision = true_positives / len(pred_segments) if pred_segments else 0
        recall = true_positives / len(true_segments) if true_segments else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        return {
            "segment_precision": precision,
            "segment_recall": recall,
            "segment_f1": f1,
            "n_true_segments": len(true_segments),
            "n_pred_segments": len(pred_segments),
            "segment_tp": true_positives,
            "segment_fp": false_positives,
            "segment_fn": false_negatives,
        }


# ============================================================================
# Adaptive Thresholding
# ============================================================================


class AdaptiveThreshold:
    """Adaptive thresholding for anomaly detection."""

    def __init__(self, method: str = "quantile", window_size: int = 100):
        """
        Initialize adaptive threshold.

        Parameters:
        -----------
        method : str
            Thresholding method ('quantile', 'mad', 'ewma')
        window_size : int
            Window size for adaptation
        """
        self.method = method
        self.window_size = window_size
        self.threshold_history = []

    def compute_threshold(
        self, scores: np.ndarray, contamination: float = 0.1
    ) -> float:
        """
        Compute adaptive threshold.

        Parameters:
        -----------
        scores : np.ndarray
            Anomaly scores
        contamination : float
            Expected contamination rate

        Returns:
        --------
        float : Threshold value
        """
        if self.method == "quantile":
            threshold = np.percentile(scores, 100 * (1 - contamination))

        elif self.method == "mad":
            median = np.median(scores)
            mad = np.median(np.abs(scores - median))
            threshold = median + 3 * mad

        elif self.method == "ewma":
            # Exponentially weighted moving average
            alpha = 0.1
            if self.threshold_history:
                prev_threshold = self.threshold_history[-1]
                current_quantile = np.percentile(scores, 100 * (1 - contamination))
                threshold = alpha * current_quantile + (1 - alpha) * prev_threshold
            else:
                threshold = np.percentile(scores, 100 * (1 - contamination))

        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.threshold_history.append(threshold)
        return threshold

    def update(self, new_scores: np.ndarray, contamination: float = 0.1):
        """
        Update threshold with new scores.

        Parameters:
        -----------
        new_scores : np.ndarray
            New anomaly scores
        contamination : float
            Expected contamination rate
        """
        # Keep only recent scores
        if len(self.threshold_history) > self.window_size:
            self.threshold_history = self.threshold_history[-self.window_size :]

        # Compute new threshold
        self.compute_threshold(new_scores, contamination)


# ============================================================================
# Anomaly Explanation
# ============================================================================


class AnomalyExplainer:
    """Explain detected anomalies."""

    @staticmethod
    def explain_anomaly(
        data_point: float, context: np.ndarray, features: pd.DataFrame = None
    ) -> Dict[str, Any]:
        """
        Explain why a point is anomalous.

        Parameters:
        -----------
        data_point : float
            The anomalous point
        context : np.ndarray
            Surrounding context
        features : pd.DataFrame
            Feature values for the point

        Returns:
        --------
        dict : Explanation
        """
        explanation = {
            "value": data_point,
            "context_mean": np.mean(context),
            "context_std": np.std(context),
            "z_score": (data_point - np.mean(context)) / np.std(context)
            if np.std(context) > 0
            else 0,
            "percentile": stats.percentileofscore(context, data_point),
            "deviation": data_point - np.mean(context),
            "relative_deviation": (data_point - np.mean(context)) / np.mean(context)
            if np.mean(context) != 0
            else 0,
        }

        # Determine anomaly type
        if explanation["z_score"] > 3:
            explanation["type"] = "High outlier"
        elif explanation["z_score"] < -3:
            explanation["type"] = "Low outlier"
        elif explanation["percentile"] > 95:
            explanation["type"] = "Upper extreme"
        elif explanation["percentile"] < 5:
            explanation["type"] = "Lower extreme"
        else:
            explanation["type"] = "Contextual anomaly"

        # Add feature-based explanation if available
        if features is not None:
            explanation["contributing_features"] = (
                AnomalyExplainer._identify_contributing_features(features)
            )

        return explanation

    @staticmethod
    def _identify_contributing_features(features: pd.DataFrame) -> List[str]:
        """Identify features contributing most to anomaly."""
        contributing = []

        for col in features.columns:
            if "zscore" in col or "percentile" in col:
                value = (
                    features[col].iloc[0]
                    if len(features) == 1
                    else features[col].mean()
                )
                if abs(value) > 2:  # Significant deviation
                    contributing.append(f"{col}: {value:.2f}")

        return contributing[:5]  # Top 5 contributing features


# ============================================================================
# Export all classes and functions
# ============================================================================

__all__ = [
    "AnomalyPattern",
    "AnomalyConfig",
    "AnomalyGenerator",
    "StatisticalTests",
    "AnomalyFeatures",
    "AnomalyMetrics",
    "AdaptiveThreshold",
    "AnomalyExplainer",
]
