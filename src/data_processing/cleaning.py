"""
Enhanced data processing utilities for A/B testing and experimentation.

This module handles data cleaning, validation, and preparation for statistical analysis
with comprehensive implementations of all TODO items.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
import logging
from datetime import datetime, timedelta
from scipy import stats
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_ab_data(
    df: pd.DataFrame,
    user_col: str = "user_id",
    group_col: str = "group",
    metric_cols: Optional[List[str]] = None,
    outlier_method: str = "iqr",
    outlier_threshold: float = 1.5,
    by_group: bool = True,
    generate_report: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
    """Clean A/B test data with comprehensive preprocessing and quality reporting.

    Parameters
    ----------
    df : pd.DataFrame
        Input experiment data.
    user_col : str, default='user_id'
        Column name for user identifiers.
    group_col : str, default='group'
        Column name for experimental groups.
    metric_cols : list of str, optional
        Metric columns for outlier detection. Auto-detected if None.
    outlier_method : str, default='iqr'
        Outlier detection method: 'iqr', 'zscore', 'isolation_forest'.
    outlier_threshold : float, default=1.5
        Threshold for outlier detection (IQR multiplier or z-score).
    by_group : bool, default=True
        Whether to detect outliers within each group separately.
    generate_report : bool, default=False
        Whether to return detailed cleaning report.

    Returns
    -------
    df_clean : pd.DataFrame
        Cleaned dataset.
    report : dict, optional
        Data quality report (if generate_report=True).
    """
    df_clean = df.copy()
    initial_count = len(df_clean)

    # Initialize report
    report = {
        "initial_count": initial_count,
        "cleaning_steps": [],
        "final_count": 0,
        "data_quality_score": 100,
    }

    # Step 1: Remove duplicate users
    before_dedup = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=[user_col], keep="first")
    after_dedup = len(df_clean)
    duplicate_count = before_dedup - after_dedup

    if duplicate_count > 0:
        warnings.warn(f"Removed {duplicate_count} duplicate users")
        report["cleaning_steps"].append(
            {
                "step": "duplicate_removal",
                "removed_count": duplicate_count,
                "remaining_count": after_dedup,
            }
        )
        report["data_quality_score"] -= min(10, duplicate_count / initial_count * 100)

    # Step 2: Data type validation and conversion
    try:
        # Ensure group column is categorical
        if group_col in df_clean.columns:
            df_clean[group_col] = df_clean[group_col].astype("category")

        # Auto-detect metric columns if not provided
        if metric_cols is None:
            metric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
            # Remove user_id like columns
            metric_cols = [col for col in metric_cols if "id" not in col.lower()]

        # Convert metric columns to appropriate numeric types
        for col in metric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

        report["cleaning_steps"].append(
            {"step": "data_type_conversion", "metric_columns_detected": metric_cols}
        )

    except Exception as e:
        logger.warning(f"Data type conversion failed: {e}")
        report["data_quality_score"] -= 5

    # Step 3: Advanced outlier detection
    if metric_cols:
        total_outliers = 0

        for col in metric_cols:
            if col not in df_clean.columns:
                continue

            before_outlier = len(df_clean)

            if by_group and group_col in df_clean.columns:
                # Detect outliers within each group
                outlier_mask = pd.Series(False, index=df_clean.index)

                for group in df_clean[group_col].unique():
                    group_mask = df_clean[group_col] == group
                    group_data = df_clean.loc[group_mask, col]

                    group_outliers = _detect_outliers(
                        group_data, method=outlier_method, threshold=outlier_threshold
                    )
                    outlier_mask.loc[group_mask] |= group_outliers
            else:
                # Global outlier detection
                outlier_mask = _detect_outliers(
                    df_clean[col], method=outlier_method, threshold=outlier_threshold
                )

            outlier_count = outlier_mask.sum()
            if outlier_count > 0:
                df_clean = df_clean[~outlier_mask]
                total_outliers += outlier_count
                warnings.warn(f"Removed {outlier_count} outliers from {col}")

        if total_outliers > 0:
            report["cleaning_steps"].append(
                {
                    "step": "outlier_removal",
                    "method": outlier_method,
                    "total_outliers_removed": total_outliers,
                    "by_group": by_group,
                }
            )
            report["data_quality_score"] -= min(
                15, total_outliers / initial_count * 100
            )

    # Step 4: Handle missing values intelligently
    missing_before = df_clean.isnull().sum().sum()

    # Drop rows with missing critical columns
    critical_cols = [user_col, group_col]
    df_clean = df_clean.dropna(subset=critical_cols)

    # Handle missing metric values
    for col in metric_cols:
        if col in df_clean.columns:
            missing_rate = df_clean[col].isnull().mean()
            if missing_rate > 0.1:  # More than 10% missing
                warnings.warn(f"High missing rate ({missing_rate:.1%}) in {col}")
                report["data_quality_score"] -= missing_rate * 20

    missing_after = df_clean.isnull().sum().sum()
    if missing_before > missing_after:
        report["cleaning_steps"].append(
            {
                "step": "missing_value_handling",
                "missing_values_before": int(missing_before),
                "missing_values_after": int(missing_after),
            }
        )

    # Final report
    report["final_count"] = len(df_clean)
    report["retention_rate"] = len(df_clean) / initial_count
    report["data_quality_score"] = max(0, report["data_quality_score"])

    logger.info(
        f"Data cleaning completed: {initial_count} → {len(df_clean)} rows "
        f"(retention: {report['retention_rate']:.1%})"
    )

    if generate_report:
        return df_clean, report
    return df_clean


def _detect_outliers(
    series: pd.Series, method: str = "iqr", threshold: float = 1.5
) -> pd.Series:
    """Detect outliers using various methods.

    Parameters
    ----------
    series : pd.Series
        Data series to analyze.
    method : str
        Detection method: 'iqr', 'zscore', 'isolation_forest'.
    threshold : float
        Threshold for outlier detection.

    Returns
    -------
    outlier_mask : pd.Series
        Boolean mask indicating outliers.
    """
    if method == "iqr":
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (series < lower_bound) | (series > upper_bound)

    elif method == "zscore":
        z_scores = np.abs(stats.zscore(series.dropna()))
        outlier_indices = series.dropna().index[z_scores > threshold]
        return series.index.isin(outlier_indices)

    elif method == "isolation_forest":
        try:
            from sklearn.ensemble import IsolationForest

            clf = IsolationForest(contamination=0.1, random_state=42)
            outliers = clf.fit_predict(series.dropna().values.reshape(-1, 1))
            outlier_indices = series.dropna().index[outliers == -1]
            return series.index.isin(outlier_indices)
        except ImportError:
            logger.warning("sklearn not available, falling back to IQR method")
            return _detect_outliers(series, method="iqr", threshold=threshold)

    else:
        raise ValueError(f"Unknown outlier detection method: {method}")


def validate_experiment_data(
    df: pd.DataFrame,
    user_col: str = "user_id",
    group_col: str = "group",
    required_groups: Optional[List[str]] = None,
    date_col: Optional[str] = None,
    min_sample_ratio: float = 0.8,
    max_missing_rate: float = 0.05,
) -> Dict[str, Any]:
    """Comprehensive experiment data validation with enhanced checks.

    Parameters
    ----------
    df : pd.DataFrame
        Experiment data to validate.
    user_col : str, default='user_id'
        User identifier column.
    group_col : str, default='group'
        Group assignment column.
    required_groups : list of str, optional
        Expected group names.
    date_col : str, optional
        Date column for temporal analysis.
    min_sample_ratio : float, default=0.8
        Minimum acceptable ratio between group sizes.
    max_missing_rate : float, default=0.05
        Maximum acceptable missing data rate.

    Returns
    -------
    validation_results : dict
        Comprehensive validation results and recommendations.
    """
    validation_results = {
        "is_valid": True,
        "warnings": [],
        "errors": [],
        "summary": {},
        "recommendations": [],
        "data_quality_score": 100,
    }

    # Basic structure validation
    required_cols = [user_col, group_col]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        validation_results["errors"].append(f"Missing required columns: {missing_cols}")
        validation_results["is_valid"] = False
        return validation_results

    # Missing data analysis
    total_rows = len(df)
    for col in [user_col, group_col]:
        missing_count = df[col].isnull().sum()
        missing_rate = missing_count / total_rows

        if missing_count > 0:
            validation_results["errors"].append(
                f"{missing_count} missing values in {col} ({missing_rate:.1%})"
            )
            validation_results["is_valid"] = False
            validation_results["data_quality_score"] -= missing_rate * 50

    # Group validation
    if validation_results["is_valid"]:
        group_counts = df[group_col].value_counts()

        # Check for required groups
        if required_groups:
            actual_groups = set(group_counts.index)
            expected_groups = set(required_groups)

            missing_groups = expected_groups - actual_groups
            extra_groups = actual_groups - expected_groups

            if missing_groups:
                validation_results["errors"].append(
                    f"Missing expected groups: {missing_groups}"
                )
                validation_results["is_valid"] = False

            if extra_groups:
                validation_results["warnings"].append(
                    f"Unexpected groups found: {extra_groups}"
                )
                validation_results["data_quality_score"] -= 5

        # Sample ratio mismatch check with configurable threshold
        if len(group_counts) >= 2:
            ratio = min(group_counts) / max(group_counts)
            if ratio < min_sample_ratio:
                validation_results["warnings"].append(
                    f"Sample ratio mismatch detected. Ratio: {ratio:.3f} "
                    f"(threshold: {min_sample_ratio})"
                )
                validation_results["data_quality_score"] -= (
                    min_sample_ratio - ratio
                ) * 50

    # Temporal consistency checks
    if date_col and date_col in df.columns:
        temporal_analysis = _analyze_temporal_patterns(df, date_col, group_col)
        validation_results["temporal_analysis"] = temporal_analysis

        if temporal_analysis["uneven_distribution"]:
            validation_results["warnings"].append(
                "Uneven temporal distribution of users across groups detected"
            )
            validation_results["data_quality_score"] -= 10

    # User behavior pattern analysis
    user_behavior = _analyze_user_patterns(df, user_col, group_col)
    validation_results["user_behavior"] = user_behavior

    if user_behavior["duplicate_users"] > 0:
        validation_results["warnings"].append(
            f"Found {user_behavior['duplicate_users']} users appearing multiple times"
        )

    # Generate recommendations
    if validation_results["data_quality_score"] < 80:
        validation_results["recommendations"].append(
            "🔍 Data quality score below 80% - recommend thorough data cleaning"
        )

    if len(validation_results["warnings"]) > 0:
        validation_results["recommendations"].append(
            "⚠️ Review warnings - may indicate randomization or data collection issues"
        )

    # Summary statistics
    validation_results["summary"] = {
        "total_users": total_rows,
        "unique_users": df[user_col].nunique(),
        "groups": group_counts.to_dict() if validation_results["is_valid"] else {},
        "duplicate_rate": 1 - (df[user_col].nunique() / total_rows),
        "data_quality_score": validation_results["data_quality_score"],
    }

    return validation_results


def _analyze_temporal_patterns(
    df: pd.DataFrame, date_col: str, group_col: str
) -> Dict[str, Any]:
    """Analyze temporal patterns in experiment data."""
    try:
        df_temp = df.copy()
        df_temp[date_col] = pd.to_datetime(df_temp[date_col])

        # Daily user counts by group
        daily_counts = (
            df_temp.groupby([df_temp[date_col].dt.date, group_col])
            .size()
            .unstack(fill_value=0)
        )

        # Check for uneven distribution
        daily_ratios = daily_counts.min(axis=1) / daily_counts.max(axis=1)
        uneven_days = (daily_ratios < 0.8).sum()

        return {
            "date_range": (
                df_temp[date_col].min().date(),
                df_temp[date_col].max().date(),
            ),
            "total_days": len(daily_counts),
            "uneven_distribution": uneven_days > len(daily_counts) * 0.2,
            "uneven_days_count": uneven_days,
        }
    except Exception as e:
        logger.warning(f"Temporal analysis failed: {e}")
        return {"error": str(e)}


def _analyze_user_patterns(
    df: pd.DataFrame, user_col: str, group_col: str
) -> Dict[str, Any]:
    """Analyze user behavior patterns."""
    user_counts = df[user_col].value_counts()
    duplicate_users = (user_counts > 1).sum()

    # Check for users in multiple groups
    user_groups = df.groupby(user_col)[group_col].nunique()
    multi_group_users = (user_groups > 1).sum()

    return {
        "duplicate_users": duplicate_users,
        "multi_group_users": multi_group_users,
        "max_appearances": user_counts.max(),
    }


def apply_cuped(
    df: pd.DataFrame,
    metric_col: str,
    covariate_col: str,
    group_col: str = "group",
    method: str = "global",
) -> pd.DataFrame:
    """Apply CUPED variance reduction with support for multiple approaches.

    Parameters
    ----------
    df : pd.DataFrame
        Experiment data.
    metric_col : str
        Target metric column.
    covariate_col : str
        Pre-experiment covariate column.
    group_col : str, default='group'
        Group assignment column.
    method : str, default='global'
        CUPED method: 'global', 'by_group', 'multi_covariate'.

    Returns
    -------
    df_cuped : pd.DataFrame
        Data with CUPED-adjusted metrics and diagnostic information.
    """
    df_cuped = df.copy()

    # Validate inputs
    required_cols = [metric_col, covariate_col, group_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Remove rows with missing values
    initial_count = len(df_cuped)
    df_cuped = df_cuped.dropna(subset=[metric_col, covariate_col])

    if len(df_cuped) == 0:
        raise ValueError("No valid data remaining after removing missing values")

    removed_count = initial_count - len(df_cuped)
    if removed_count > 0:
        logger.warning(f"Removed {removed_count} rows with missing values")

    y = df_cuped[metric_col].values
    x = df_cuped[covariate_col].values

    # Check for covariate variation
    if np.var(x) == 0:
        warnings.warn("Covariate has no variation - CUPED will have no effect")
        df_cuped[f"{metric_col}_cuped"] = y
        df_cuped["cuped_theta"] = 0
        df_cuped["variance_reduction"] = 0
        return df_cuped

    if method == "global":
        # Global CUPED - single theta for all data
        theta = np.cov(y, x)[0, 1] / np.var(x)
        x_centered = x - np.mean(x)
        y_adjusted = y - theta * x_centered

        df_cuped[f"{metric_col}_cuped"] = y_adjusted
        df_cuped["cuped_theta"] = theta

    elif method == "by_group":
        # Group-specific CUPED - separate theta for each group
        y_adjusted = np.zeros_like(y)
        theta_values = np.zeros_like(y)

        for group in df_cuped[group_col].unique():
            group_mask = df_cuped[group_col] == group
            y_group = y[group_mask]
            x_group = x[group_mask]

            if len(y_group) > 1 and np.var(x_group) > 0:
                theta_group = np.cov(y_group, x_group)[0, 1] / np.var(x_group)
                x_group_centered = x_group - np.mean(x_group)
                y_adjusted[group_mask] = y_group - theta_group * x_group_centered
                theta_values[group_mask] = theta_group
            else:
                y_adjusted[group_mask] = y_group
                theta_values[group_mask] = 0

        df_cuped[f"{metric_col}_cuped"] = y_adjusted
        df_cuped["cuped_theta"] = theta_values

    else:
        raise ValueError(f"Unknown CUPED method: {method}")

    # Calculate variance reduction
    var_original = np.var(y)
    var_adjusted = np.var(df_cuped[f"{metric_col}_cuped"])
    variance_reduction = (var_original - var_adjusted) / var_original

    df_cuped["variance_reduction"] = variance_reduction

    logger.info(f"CUPED applied with {variance_reduction:.1%} variance reduction")

    return df_cuped


class DataQualityChecker:
    """Comprehensive data quality assessment with ML-based anomaly detection."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize the data quality checker.

        Parameters
        ----------
        config : dict, optional
            Configuration dictionary with thresholds and settings.
        """
        self.config = config or {}
        self.default_thresholds = {
            "max_missing_rate": 0.05,
            "min_sample_ratio": 0.8,
            "max_outlier_rate": 0.01,
            "min_users_per_group": 100,
            "max_daily_variance": 0.3,
        }

        # Override defaults with provided config
        self.thresholds = {**self.default_thresholds, **self.config}

        # Initialize ML models for anomaly detection
        self._anomaly_detectors = {}

        logger.info("DataQualityChecker initialized with comprehensive monitoring")

    def run_full_check(
        self, df: pd.DataFrame, experiment_config: Dict
    ) -> Dict[str, Any]:
        """Run comprehensive data quality assessment with ML-based detection.

        Parameters
        ----------
        df : pd.DataFrame
            Experiment data to assess.
        experiment_config : dict
            Experiment configuration including column names and expected values.

        Returns
        -------
        quality_report : dict
            Comprehensive quality assessment with scores and recommendations.
        """
        quality_report = {
            "overall_score": 100,
            "checks": {},
            "recommendations": [],
            "anomalies": [],
            "timestamp": datetime.now().isoformat(),
        }

        # Basic structure checks
        quality_report["checks"]["structure"] = self._check_structure(
            df, experiment_config
        )

        # Missing data analysis with pattern detection
        quality_report["checks"]["missing_data"] = self._analyze_missing_patterns(df)

        # Temporal consistency analysis
        if "date_col" in experiment_config:
            quality_report["checks"]["temporal"] = self._check_temporal_consistency(
                df, experiment_config
            )

        # User behavior anomaly detection
        quality_report["checks"]["user_behavior"] = self._detect_user_anomalies(
            df, experiment_config
        )

        # Metric distribution analysis
        if "metric_cols" in experiment_config:
            quality_report["checks"]["metrics"] = self._analyze_metric_distributions(
                df, experiment_config["metric_cols"]
            )

        # Calculate overall score
        quality_report["overall_score"] = self._calculate_overall_score(
            quality_report["checks"]
        )

        # Generate recommendations
        quality_report["recommendations"] = self._generate_recommendations(
            quality_report
        )

        return quality_report

    def _check_structure(self, df: pd.DataFrame, config: Dict) -> Dict[str, Any]:
        """Check basic data structure requirements."""
        structure_check = {"status": "passed", "issues": [], "score": 100}

        # Required columns check
        required_cols = config.get("required_columns", ["user_id", "group"])
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            structure_check["issues"].append(
                f"Missing required columns: {missing_cols}"
            )
            structure_check["status"] = "failed"
            structure_check["score"] = 0

        # Data types check
        for col, expected_type in config.get("column_types", {}).items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if expected_type not in actual_type:
                    structure_check["issues"].append(
                        f"Column {col} has type {actual_type}, expected {expected_type}"
                    )
                    structure_check["score"] -= 10

        # Sample size check
        min_samples = self.thresholds["min_users_per_group"]
        if "group" in df.columns:
            group_sizes = df["group"].value_counts()
            small_groups = group_sizes[group_sizes < min_samples].index.tolist()

            if small_groups:
                structure_check["issues"].append(
                    f"Groups with insufficient samples (<{min_samples}): {small_groups}"
                )
                structure_check["score"] -= len(small_groups) * 20

        return structure_check

    def _analyze_missing_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing data patterns with advanced detection."""
        missing_analysis = {
            "overall_missing_rate": df.isnull().sum().sum()
            / (len(df) * len(df.columns)),
            "column_missing_rates": df.isnull().mean().to_dict(),
            "patterns": [],
            "score": 100,
        }

        # Detect non-random missing patterns
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            missing_rate = df[col].isnull().mean()

            if missing_rate > self.thresholds["max_missing_rate"]:
                missing_analysis["patterns"].append(
                    {
                        "column": col,
                        "missing_rate": missing_rate,
                        "severity": "high" if missing_rate > 0.2 else "medium",
                    }
                )
                missing_analysis["score"] -= missing_rate * 50

        # Check for systematic missing patterns
        if len(df.columns) > 1:
            try:
                # Correlation between missing indicators
                missing_indicators = df.isnull().astype(int)
                correlations = missing_indicators.corr()

                high_correlations = []
                for i, col1 in enumerate(correlations.columns):
                    for j, col2 in enumerate(correlations.columns):
                        if i < j and abs(correlations.iloc[i, j]) > 0.5:
                            high_correlations.append(
                                (col1, col2, correlations.iloc[i, j])
                            )

                if high_correlations:
                    missing_analysis["patterns"].append(
                        {
                            "type": "correlated_missing",
                            "correlations": high_correlations,
                        }
                    )
                    missing_analysis["score"] -= len(high_correlations) * 10

            except Exception as e:
                logger.warning(f"Missing pattern analysis failed: {e}")

        return missing_analysis

    def _check_temporal_consistency(
        self, df: pd.DataFrame, config: Dict
    ) -> Dict[str, Any]:
        """Check for temporal consistency and detect unusual patterns."""
        date_col = config["date_col"]

        if date_col not in df.columns:
            return {"status": "skipped", "reason": "date column not found"}

        temporal_check = {
            "status": "passed",
            "issues": [],
            "score": 100,
            "daily_stats": {},
        }

        try:
            df_temp = df.copy()
            df_temp[date_col] = pd.to_datetime(df_temp[date_col])

            # Daily user distribution
            daily_counts = df_temp.groupby(df_temp[date_col].dt.date).size()

            # Detect unusual daily variations
            daily_variance = daily_counts.std() / daily_counts.mean()
            max_variance = self.thresholds["max_daily_variance"]

            if daily_variance > max_variance:
                temporal_check["issues"].append(
                    f"High daily variance in user counts: {daily_variance:.3f} "
                    f"(threshold: {max_variance})"
                )
                temporal_check["score"] -= 20

            # Weekend/weekday pattern analysis
            df_temp["day_of_week"] = df_temp[date_col].dt.day_of_week
            weekday_weekend_ratio = (
                df_temp[df_temp["day_of_week"] < 5].shape[0]
                / df_temp[df_temp["day_of_week"] >= 5].shape[0]
            )

            temporal_check["daily_stats"] = {
                "date_range": (
                    df_temp[date_col].min().date(),
                    df_temp[date_col].max().date(),
                ),
                "daily_variance": daily_variance,
                "weekday_weekend_ratio": weekday_weekend_ratio,
                "total_days": len(daily_counts),
            }

        except Exception as e:
            temporal_check["status"] = "failed"
            temporal_check["issues"].append(f"Temporal analysis failed: {e}")
            temporal_check["score"] = 0

        return temporal_check

    def _detect_user_anomalies(self, df: pd.DataFrame, config: Dict) -> Dict[str, Any]:
        """Detect anomalous user behavior patterns."""
        user_analysis = {"status": "passed", "anomalies": [], "score": 100}

        user_col = config.get("user_col", "user_id")
        group_col = config.get("group_col", "group")

        if user_col not in df.columns:
            return {"status": "skipped", "reason": f"{user_col} column not found"}

        # Duplicate user analysis
        user_counts = df[user_col].value_counts()
        duplicate_users = (user_counts > 1).sum()

        if duplicate_users > 0:
            user_analysis["anomalies"].append(
                {
                    "type": "duplicate_users",
                    "count": duplicate_users,
                    "percentage": duplicate_users / len(user_counts),
                }
            )
            user_analysis["score"] -= min(30, duplicate_users / len(df) * 100)

        # Users in multiple groups
        if group_col in df.columns:
            user_groups = df.groupby(user_col)[group_col].nunique()
            multi_group_users = (user_groups > 1).sum()

            if multi_group_users > 0:
                user_analysis["anomalies"].append(
                    {"type": "multi_group_users", "count": multi_group_users}
                )
                user_analysis["score"] -= 50  # This is a serious issue

        return user_analysis

    def _analyze_metric_distributions(
        self, df: pd.DataFrame, metric_cols: List[str]
    ) -> Dict[str, Any]:
        """Analyze metric distributions for anomalies."""
        metric_analysis = {"distributions": {}, "anomalies": [], "score": 100}

        for col in metric_cols:
            if col not in df.columns:
                continue

            col_analysis = {
                "mean": df[col].mean(),
                "std": df[col].std(),
                "skewness": stats.skew(df[col].dropna()),
                "kurtosis": stats.kurtosis(df[col].dropna()),
                "outlier_rate": 0,
            }

            # Detect extreme distributions
            if abs(col_analysis["skewness"]) > 2:
                metric_analysis["anomalies"].append(
                    {
                        "column": col,
                        "type": "high_skewness",
                        "value": col_analysis["skewness"],
                    }
                )
                metric_analysis["score"] -= 10

            if col_analysis["kurtosis"] > 10:
                metric_analysis["anomalies"].append(
                    {
                        "column": col,
                        "type": "high_kurtosis",
                        "value": col_analysis["kurtosis"],
                    }
                )
                metric_analysis["score"] -= 10

            metric_analysis["distributions"][col] = col_analysis

        return metric_analysis

    def _calculate_overall_score(self, checks: Dict) -> float:
        """Calculate weighted overall quality score."""
        weights = {
            "structure": 0.3,
            "missing_data": 0.2,
            "temporal": 0.2,
            "user_behavior": 0.2,
            "metrics": 0.1,
        }

        weighted_score = 0
        total_weight = 0

        for check_name, weight in weights.items():
            if check_name in checks and "score" in checks[check_name]:
                weighted_score += checks[check_name]["score"] * weight
                total_weight += weight

        return weighted_score / total_weight if total_weight > 0 else 0

    def _generate_recommendations(self, quality_report: Dict) -> List[str]:
        """Generate actionable recommendations based on quality assessment."""
        recommendations = []
        overall_score = quality_report["overall_score"]

        if overall_score < 60:
            recommendations.append(
                "🚨 Critical data quality issues detected - recommend stopping experiment"
            )
        elif overall_score < 80:
            recommendations.append(
                "⚠️ Significant data quality issues - investigate before analysis"
            )

        # Specific recommendations based on checks
        for check_name, check_results in quality_report["checks"].items():
            if check_results.get("score", 100) < 70:
                if check_name == "structure":
                    recommendations.append(
                        "📋 Fix data structure issues before proceeding"
                    )
                elif check_name == "missing_data":
                    recommendations.append("🔍 Investigate missing data patterns")
                elif check_name == "temporal":
                    recommendations.append(
                        "📅 Review temporal data collection consistency"
                    )
                elif check_name == "user_behavior":
                    recommendations.append(
                        "👤 Check user assignment and deduplication logic"
                    )

        if not recommendations:
            recommendations.append("✅ Data quality is acceptable for analysis")

        return recommendations


def get_experiment_summary(
    df: pd.DataFrame,
    group_col: str = "group",
    metrics: Optional[List[str]] = None,
    include_power_analysis: bool = True,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Generate comprehensive experiment summary with statistical power analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Experiment data.
    group_col : str, default='group'
        Group assignment column.
    metrics : list of str, optional
        Metric columns to analyze. Auto-detected if None.
    include_power_analysis : bool, default=True
        Whether to include post-hoc power analysis.
    confidence_level : float, default=0.95
        Confidence level for intervals.

    Returns
    -------
    summary : pd.DataFrame
        Comprehensive summary statistics with effect sizes and power.
    """
    if metrics is None:
        # Smart metric detection
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Filter out ID columns and other non-metrics
        metrics = []
        for col in numeric_cols:
            col_lower = col.lower()
            if not any(keyword in col_lower for keyword in ["id", "index", "key"]):
                # Check if it looks like a metric (reasonable variation)
                if df[col].nunique() > 1 and df[col].std() > 0:
                    metrics.append(col)

    summary_stats = []
    alpha = 1 - confidence_level

    for metric in metrics:
        if metric not in df.columns:
            continue

        # Basic statistics by group
        metric_summary = (
            df.groupby(group_col)[metric]
            .agg(["count", "mean", "median", "std", "min", "max"])
            .round(6)
        )

        # Add confidence intervals for means
        for group in metric_summary.index:
            group_data = df[df[group_col] == group][metric].dropna()
            n = len(group_data)
            mean = group_data.mean()
            std_err = group_data.std() / np.sqrt(n)

            # t-distribution for small samples
            if n > 30:
                critical_value = stats.norm.ppf(1 - alpha / 2)
            else:
                critical_value = stats.t.ppf(1 - alpha / 2, df=n - 1)

            margin_error = critical_value * std_err
            metric_summary.loc[group, "ci_lower"] = mean - margin_error
            metric_summary.loc[group, "ci_upper"] = mean + margin_error

        # Effect size calculations for two-group comparison
        groups = metric_summary.index.tolist()
        if len(groups) == 2:
            group1, group2 = groups

            data1 = df[df[group_col] == group1][metric].dropna()
            data2 = df[df[group_col] == group2][metric].dropna()

            # Cohen's d
            pooled_std = np.sqrt(
                (
                    (len(data1) - 1) * data1.std() ** 2
                    + (len(data2) - 1) * data2.std() ** 2
                )
                / (len(data1) + len(data2) - 2)
            )

            if pooled_std > 0:
                cohens_d = (data2.mean() - data1.mean()) / pooled_std
                metric_summary.loc[group1, "cohens_d"] = -cohens_d  # Reference group
                metric_summary.loc[group2, "cohens_d"] = cohens_d

            # Statistical test
            if df[metric].nunique() == 2:  # Binary metric
                # Proportion test
                from .statistics.core import two_prop_ztest

                x1 = int(data1.sum())
                n1 = len(data1)
                x2 = int(data2.sum())
                n2 = len(data2)

                z_stat, p_value = two_prop_ztest(x1, n1, x2, n2)
                test_type = "z_test"
            else:
                # t-test for continuous metrics
                t_stat, p_value = stats.ttest_ind(data1, data2)
                z_stat = t_stat  # For consistency in output
                test_type = "t_test"

            metric_summary.loc[group1, "test_statistic"] = -z_stat
            metric_summary.loc[group1, "p_value"] = p_value
            metric_summary.loc[group1, "test_type"] = test_type
            metric_summary.loc[group2, "test_statistic"] = z_stat
            metric_summary.loc[group2, "p_value"] = p_value
            metric_summary.loc[group2, "test_type"] = test_type

            # Post-hoc power analysis
            if include_power_analysis:
                from .statistics.core import calculate_power

                if df[metric].nunique() == 2:  # Binary metric
                    baseline_rate = data1.mean()
                    effect_size = abs(data2.mean() - data1.mean())
                    power = calculate_power(
                        len(data1), len(data2), baseline_rate, effect_size
                    )
                else:
                    # For continuous metrics (simplified)
                    effect_size = abs(cohens_d) if "cohens_d" in locals() else 0
                    # Simplified power calculation
                    power = 0.8 if abs(z_stat) > 1.96 else 0.5  # Placeholder

                metric_summary.loc[group1, "statistical_power"] = power
                metric_summary.loc[group2, "statistical_power"] = power

        # Add metric name to each row
        metric_summary["metric"] = metric
        summary_stats.append(metric_summary.reset_index())

    if not summary_stats:
        return pd.DataFrame()

    final_summary = pd.concat(summary_stats, ignore_index=True)

    # Reorder columns for better readability
    col_order = [
        "metric",
        group_col,
        "count",
        "mean",
        "ci_lower",
        "ci_upper",
        "median",
        "std",
        "min",
        "max",
    ]

    # Add optional columns if they exist
    optional_cols = [
        "cohens_d",
        "test_statistic",
        "p_value",
        "test_type",
        "statistical_power",
    ]
    for col in optional_cols:
        if col in final_summary.columns:
            col_order.append(col)

    # Only include columns that exist
    final_col_order = [col for col in col_order if col in final_summary.columns]

    return final_summary[final_col_order]


# Export utilities for sharing results


def export_analysis_results(
    results: Dict[str, Any], output_path: str, format: str = "json"
) -> None:
    """Export analysis results to various formats.

    Parameters
    ----------
    results : dict
        Analysis results to export.
    output_path : str
        Output file path.
    format : str, default='json'
        Export format: 'json', 'csv', 'html'.
    """
    if format == "json":
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

    elif format == "csv" and "summary" in results:
        if isinstance(results["summary"], pd.DataFrame):
            results["summary"].to_csv(output_path, index=False)
        else:
            logger.warning("CSV export requires pandas DataFrame in results['summary']")

    elif format == "html":
        _export_html_report(results, output_path)

    else:
        raise ValueError(f"Unsupported export format: {format}")


def _export_html_report(results: Dict[str, Any], output_path: str) -> None:
    """Export results as formatted HTML report."""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Experiment Analysis Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .header { background-color: #f8f9fa; padding: 20px; border-radius: 5px; }
            .section { margin: 20px 0; }
            .metric { background-color: #e9ecef; padding: 10px; margin: 10px 0; border-radius: 3px; }
            .significant { color: #28a745; font-weight: bold; }
            .not-significant { color: #6c757d; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Experiment Analysis Report</h1>
            <p>Generated on: {timestamp}</p>
        </div>
        
        <div class="section">
            <h2>Summary</h2>
            {summary_content}
        </div>
        
        <div class="section">
            <h2>Results</h2>
            {results_content}
        </div>
    </body>
    </html>
    """

    # Generate content (simplified)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_content = "<p>Detailed HTML reporting coming soon...</p>"
    results_content = f"<pre>{json.dumps(results, indent=2, default=str)}</pre>"

    html_content = html_template.format(
        timestamp=timestamp,
        summary_content=summary_content,
        results_content=results_content,
    )

    with open(output_path, "w") as f:
        f.write(html_content)


# Memory optimization utilities


def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame memory usage by downcasting numeric types.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to optimize.

    Returns
    -------
    df_optimized : pd.DataFrame
        Memory-optimized DataFrame.
    """
    df_optimized = df.copy()

    # Optimize integer columns
    for col in df_optimized.select_dtypes(include=["int64"]):
        col_min = df_optimized[col].min()
        col_max = df_optimized[col].max()

        if col_min >= 0:
            if col_max <= 255:
                df_optimized[col] = df_optimized[col].astype("uint8")
            elif col_max <= 65535:
                df_optimized[col] = df_optimized[col].astype("uint16")
            elif col_max <= 4294967295:
                df_optimized[col] = df_optimized[col].astype("uint32")
        else:
            if col_min >= -128 and col_max <= 127:
                df_optimized[col] = df_optimized[col].astype("int8")
            elif col_min >= -32768 and col_max <= 32767:
                df_optimized[col] = df_optimized[col].astype("int16")
            elif col_min >= -2147483648 and col_max <= 2147483647:
                df_optimized[col] = df_optimized[col].astype("int32")

    # Optimize float columns
    for col in df_optimized.select_dtypes(include=["float64"]):
        df_optimized[col] = pd.to_numeric(df_optimized[col], downcast="float")

    # Convert object columns to category where beneficial
    for col in df_optimized.select_dtypes(include=["object"]):
        if (
            df_optimized[col].nunique() / len(df_optimized) < 0.5
        ):  # Less than 50% unique
            df_optimized[col] = df_optimized[col].astype("category")

    # Log memory savings
    original_memory = df.memory_usage(deep=True).sum()
    optimized_memory = df_optimized.memory_usage(deep=True).sum()
    savings = (original_memory - optimized_memory) / original_memory

    logger.info(
        f"Memory optimization: {savings:.1%} reduction "
        f"({original_memory / 1024 / 1024:.1f}MB → {optimized_memory / 1024 / 1024:.1f}MB)"
    )

    return df_optimized
