"""
Data processing utilities for A/B testing and experimentation.

This module handles data cleaning, validation, and preparation for statistical analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings


def clean_ab_data(df: pd.DataFrame, 
                  user_col: str = 'user_id',
                  group_col: str = 'group',
                  metric_cols: List[str] = None) -> pd.DataFrame:
    """Clean A/B test data with standard preprocessing steps.
    
    # TODO: Add automated data quality report generation
    # Should include missing data patterns, outlier detection, etc.
    # ASSIGNEE: @diogoribeiro7
    # LABELS: enhancement, data-quality
    # PRIORITY: high
    """
    df_clean = df.copy()
    
    # Remove duplicate users
    initial_count = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=[user_col], keep='first')
    duplicate_count = initial_count - len(df_clean)
    
    if duplicate_count > 0:
        warnings.warn(f"Removed {duplicate_count} duplicate users")
    
    # TODO: Implement sophisticated outlier detection
    # Current approach is too simplistic for real-world data
    # LABELS: enhancement, outlier-detection
    # PRIORITY: medium
    
    # FIXME: Hard-coded outlier threshold should be configurable
    # Different metrics may need different outlier definitions
    if metric_cols:
        for col in metric_cols:
            if col in df_clean.columns:
                # Simple IQR-based outlier removal
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # BUG: This removes outliers globally, but we should consider
                # removing outliers within each experimental group separately
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
                outlier_count = outlier_mask.sum()
                
                if outlier_count > 0:
                    warnings.warn(f"Removed {outlier_count} outliers from {col}")
                    df_clean = df_clean[~outlier_mask]
    
    # TODO: Add data type validation and conversion
    # Ensure metric columns are numeric, group columns are categorical, etc.
    # ASSIGNEE: @diogoribeiro7
    # LABELS: validation, data-types
    # PRIORITY: medium
    
    return df_clean


def validate_experiment_data(df: pd.DataFrame,
                           user_col: str = 'user_id',
                           group_col: str = 'group',
                           required_groups: List[str] = None) -> Dict[str, any]:
    """Validate A/B test data for common issues.
    
    # TODO: Add validation for proper randomization
    # Check for systematic differences in user characteristics between groups
    # LABELS: validation, randomization
    # PRIORITY: high
    """
    validation_results = {
        'is_valid': True,
        'warnings': [],
        'errors': [],
        'summary': {}
    }
    
    # Check for required columns
    required_cols = [user_col, group_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        validation_results['errors'].append(f"Missing required columns: {missing_cols}")
        validation_results['is_valid'] = False
        return validation_results
    
    # Check for missing values in key columns
    # NOTE: Some missing values in metric columns might be acceptable
    # but missing group assignments are always problematic
    for col in [user_col, group_col]:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            validation_results['errors'].append(f"{missing_count} missing values in {col}")
            validation_results['is_valid'] = False
    
    # TODO: Implement checks for temporal consistency
    # If timestamp data is available, check for proper randomization over time
    # ASSIGNEE: @diogoribeiro7
    # LABELS: validation, temporal
    # PRIORITY: medium
    
    # Check group distribution
    if required_groups:
        actual_groups = set(df[group_col].unique())
        expected_groups = set(required_groups)
        
        missing_groups = expected_groups - actual_groups
        extra_groups = actual_groups - expected_groups
        
        if missing_groups:
            validation_results['errors'].append(f"Missing expected groups: {missing_groups}")
            validation_results['is_valid'] = False
        
        if extra_groups:
            validation_results['warnings'].append(f"Unexpected groups found: {extra_groups}")
    
    # FIXME: Sample ratio mismatch check is too strict
    # Should allow for configurable tolerance based on randomization method
    group_counts = df[group_col].value_counts()
    if len(group_counts) == 2:
        ratio = min(group_counts) / max(group_counts)
        if ratio < 0.8:  # Hard-coded threshold
            validation_results['warnings'].append(
                f"Potential sample ratio mismatch. Ratio: {ratio:.3f}"
            )
    
    # TODO: Add checks for consistent user behavior patterns
    # Users appearing in multiple experiments, unusual activity patterns, etc.
    # LABELS: validation, user-behavior
    # PRIORITY: low
    
    validation_results['summary'] = {
        'total_users': len(df),
        'unique_users': df[user_col].nunique(),
        'groups': group_counts.to_dict(),
        'duplicate_rate': 1 - (df[user_col].nunique() / len(df))
    }
    
    return validation_results


def apply_cuped(df: pd.DataFrame,
                metric_col: str,
                covariate_col: str,
                group_col: str = 'group') -> pd.DataFrame:
    """Apply CUPED (Controlled-experiment Using Pre-Experiment Data) variance reduction.
    
    # TODO: Implement generalized CUPED for multiple covariates
    # Current implementation only supports single covariate
    # ASSIGNEE: @diogoribeiro7
    # LABELS: enhancement, variance-reduction
    # PRIORITY: high
    """
    df_cuped = df.copy()
    
    # Check if covariate exists and has variation
    if covariate_col not in df.columns:
        raise ValueError(f"Covariate column '{covariate_col}' not found")
    
    # FIXME: Should handle missing values in covariate more gracefully
    # Current implementation drops all rows with missing covariates
    df_cuped = df_cuped.dropna(subset=[covariate_col, metric_col])
    
    if df_cuped.empty:
        raise ValueError("No valid data remaining after removing missing values")
    
    # Calculate CUPED adjustment
    y = df_cuped[metric_col].values
    x = df_cuped[covariate_col].values
    
    # Check for sufficient covariate variation
    if np.var(x) == 0:
        warnings.warn("Covariate has no variation - CUPED will have no effect")
        df_cuped[f'{metric_col}_cuped'] = y
        return df_cuped
    
    # TODO: Add option to calculate theta separately by group
    # This can be more appropriate when treatment affects the covariate relationship
    # LABELS: enhancement, advanced-cuped
    # PRIORITY: medium
    
    # Calculate optimal theta (coefficient)
    covariance = np.cov(y, x)[0, 1]
    variance_x = np.var(x)
    theta = covariance / variance_x
    
    # Apply CUPED adjustment
    x_centered = x - np.mean(x)
    y_adjusted = y - theta * x_centered
    
    df_cuped[f'{metric_col}_cuped'] = y_adjusted
    df_cuped['cuped_theta'] = theta
    
    # TODO: Add diagnostic plots for CUPED effectiveness
    # Show variance reduction, correlation plots, etc.
    # ASSIGNEE: @diogoribeiro7
    # LABELS: visualization, diagnostics
    # PRIORITY: low
    
    return df_cuped


class DataQualityChecker:
    """Comprehensive data quality assessment for experiments.
    
    # TODO: Implement real-time data quality monitoring
    # Should be able to detect issues as data comes in, not just post-hoc
    # LABELS: monitoring, real-time
    # PRIORITY: high
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Default thresholds - TODO: make these configurable
        # LABELS: configuration, flexibility
        self.default_thresholds = {
            'max_missing_rate': 0.05,  # 5% missing data threshold
            'min_sample_ratio': 0.8,   # Minimum ratio for balanced groups
            'max_outlier_rate': 0.01   # 1% outlier threshold
        }
    
    def run_full_check(self, df: pd.DataFrame, 
                      experiment_config: Dict) -> Dict[str, any]:
        """Run comprehensive data quality assessment.
        
        # TODO: Add machine learning-based anomaly detection
        # Current rule-based approach may miss subtle data quality issues
        # ASSIGNEE: @diogoribeiro7
        # LABELS: enhancement, ml-detection
        # PRIORITY: medium
        """
        results = {
            'overall_score': 100,  # Start with perfect score
            'checks': {},
            'recommendations': []
        }
        
        # Basic structure checks
        results['checks']['structure'] = self._check_structure(df, experiment_config)
        
        # Missing data analysis
        results['checks']['missing_data'] = self._check_missing_data(df)
        
        # TODO: Add checks for data consistency over time
        # Detect sudden changes in metric distributions, user behavior, etc.
        # LABELS: temporal-analysis, consistency
        # PRIORITY: medium
        
        # BUG: Overall score calculation is overly simplistic
        # Should weight different types of issues differently
        
        return results
    
    def _check_structure(self, df: pd.DataFrame, config: Dict) -> Dict:
        """Check basic data structure requirements."""
        # TODO: Implement this method
        # LABELS: implementation, basic-checks
        return {'status': 'not_implemented'}
    
    def _check_missing_data(self, df: pd.DataFrame) -> Dict:
        """Analyze missing data patterns."""
        # TODO: Implement sophisticated missing data analysis
        # Should detect non-random missing patterns that could bias results
        # LABELS: implementation, missing-data
        return {'status': 'not_implemented'}


# HACK: Global variable for caching processed data - should use proper caching
_processed_data_cache = {}

def get_experiment_summary(df: pd.DataFrame, 
                          group_col: str = 'group',
                          metrics: List[str] = None) -> pd.DataFrame:
    """Generate experiment summary statistics.
    
    # TODO: Add statistical power calculation for observed effect sizes
    # This would help determine if non-significant results are due to 
    # insufficient power or truly no effect
    # ASSIGNEE: @diogoribeiro7
    # LABELS: power-analysis, summary
    # PRIORITY: high
    """
    if metrics is None:
        # Try to auto-detect numeric columns as metrics
        metrics = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # NOTE: This auto-detection might include ID columns or other 
        # non-metric numeric data - should be more intelligent
        # TODO: Implement smart metric detection based on column names/patterns
        # LABELS: auto-detection, smart-defaults
    
    summary_stats = []
    
    for metric in metrics:
        if metric not in df.columns:
            continue
            
        metric_summary = df.groupby(group_col)[metric].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(4)
        
        metric_summary['metric'] = metric
        summary_stats.append(metric_summary.reset_index())
    
    if not summary_stats:
        return pd.DataFrame()
    
    # TODO: Add confidence intervals for means
    # TODO: Add effect size calculations (Cohen's d, etc.)
    # LABELS: effect-size, confidence-intervals
    # PRIORITY: medium
    
    return pd.concat(summary_stats, ignore_index=True)


# TODO: Create data export utilities for sharing results
# Should support multiple formats (CSV, JSON, HTML reports)
# ASSIGNEE: @diogoribeiro7
# LABELS: export, reporting
# PRIORITY: low

# NOTE: Consider adding support for federated analysis
# where data cannot be centrally combined due to privacy constraints
