"""
Statistical utilities module for data science portfolio.

This module contains core statistical functions used across the portfolio projects.
"""

from typing import Tuple, Optional, List
import numpy as np
import pandas as pd
import math


def two_prop_ztest(x1: int, n1: int, x2: int, n2: int, two_sided: bool = True) -> Tuple[float, float]:
    """Two-sample z-test for proportions.
    
    # TODO: Add support for one-sided tests with proper directional hypotheses
    # ASSIGNEE: @diogoribeiro7
    # LABELS: enhancement, statistics
    # PRIORITY: medium
    
    Parameters
    ----------
    x1, n1 : int
        Success count and sample size for group 1.
    x2, n2 : int
        Success count and sample size for group 2.
    two_sided : bool, default=True
        Whether to perform two-sided test.
        
    Returns
    -------
    z_stat : float
        Z-statistic value.
    p_value : float
        P-value for the test.
    """
    # TODO: Implement continuity correction for small sample sizes
    # LABELS: enhancement, edge-cases
    
    if n1 <= 0 or n2 <= 0:
        raise ValueError("Sample sizes must be positive")
    
    p1, p2 = x1/n1, x2/n2
    p_pool = (x1 + x2) / (n1 + n2)
    
    # FIXME: Handle edge case when pooled proportion is 0 or 1
    # This currently causes division by zero in standard error calculation
    se = math.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    
    if se == 0:
        raise ZeroDivisionError("Standard error is zero")
    
    z = (p2 - p1) / se
    
    if two_sided:
        p_val = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    else:
        p_val = 1 - 0.5 * (1 + math.erf(z / math.sqrt(2)))
    
    return float(z), float(p_val)


def bootstrap_ci_diff(pA: float, pB: float, nA: int, nB: int, 
                     B: int = 5000, alpha: float = 0.05) -> Tuple[float, float]:
    """Bootstrap confidence interval for difference in proportions.
    
    # TODO: Add bias-corrected and accelerated (BCa) bootstrap method
    # ASSIGNEE: @diogoribeiro7
    # LABELS: enhancement, advanced-methods
    # PRIORITY: high
    """
    # BUG: Random seed should be configurable for reproducibility
    # Current implementation uses global random state
    
    diffs = np.empty(B, dtype=float)
    for b in range(B):
        xA = np.random.binomial(nA, pA)
        xB = np.random.binomial(nB, pB)
        diffs[b] = xB/nB - xA/nA
    
    # TODO: Add support for different confidence levels beyond 95%
    # LABELS: enhancement, flexibility
    
    lo = float(np.quantile(diffs, alpha/2))
    hi = float(np.quantile(diffs, 1 - alpha/2))
    return lo, hi


class ExperimentAnalyzer:
    """Main class for A/B test analysis.
    
    # TODO: Implement automated experiment monitoring with alerts
    # ASSIGNEE: @diogoribeiro7
    # LABELS: feature, monitoring
    # PRIORITY: high
    """
    
    def __init__(self, alpha: float = 0.05, power: float = 0.8):
        self.alpha = alpha
        self.power = power
        
        # HACK: Using hard-coded critical values instead of computing them
        # Should use proper inverse normal function
        self._z_alpha = 1.96  # For alpha = 0.05, two-sided
    
    def check_srm(self, df: pd.DataFrame, group_col: str = 'group') -> dict:
        """Check for Sample Ratio Mismatch.
        
        # TODO: Add support for non-50/50 expected ratios
        # Current implementation assumes equal allocation
        # LABELS: enhancement, experimental-design
        """
        group_counts = df[group_col].value_counts()
        
        # NOTE: This assumes only two groups - extend for multi-arm tests
        # TODO: Generalize for arbitrary number of treatment arms
        # ASSIGNEE: @diogoribeiro7
        # PRIORITY: medium
        
        if len(group_counts) != 2:
            raise ValueError("SRM check currently only supports two groups")
        
        nA, nB = group_counts.iloc[0], group_counts.iloc[1]
        n_total = nA + nB
        expected = n_total / 2
        
        chi2 = ((nA - expected)**2 + (nB - expected)**2) / expected
        
        # FIXME: Using approximation instead of exact chi-square distribution
        # Should use scipy.stats.chi2 for proper p-value calculation
        p_value = 2 * (1 - 0.5 * (1 + math.erf(math.sqrt(chi2) / math.sqrt(2))))
        
        return {
            'chi2_stat': chi2,
            'p_value': p_value,
            'is_srm': p_value < 0.05,
            'group_counts': group_counts.to_dict()
        }
    
    def analyze_conversion(self, df: pd.DataFrame, 
                          conversion_col: str = 'converted',
                          group_col: str = 'group') -> dict:
        """Analyze conversion rate differences between groups.
        
        # TODO: Add support for multiple metrics in single analysis
        # LABELS: enhancement, multi-metric
        # PRIORITY: medium
        """
        results = {}
        
        # Basic conversion analysis
        conv_by_group = df.groupby(group_col)[conversion_col].agg(['sum', 'count', 'mean'])
        
        # TODO: Implement proper multiple testing correction
        # Current analysis doesn't account for multiple comparisons
        # ASSIGNEE: @diogoribeiro7
        # LABELS: statistics, multiple-testing
        # PRIORITY: high
        
        groups = conv_by_group.index.tolist()
        if len(groups) == 2:
            x1 = int(conv_by_group.iloc[0]['sum'])
            n1 = int(conv_by_group.iloc[0]['count'])
            x2 = int(conv_by_group.iloc[1]['sum'])
            n2 = int(conv_by_group.iloc[1]['count'])
            
            z_stat, p_val = two_prop_ztest(x1, n1, x2, n2)
            
            results.update({
                'z_statistic': z_stat,
                'p_value': p_val,
                'significant': p_val < self.alpha,
                'conversion_rates': conv_by_group['mean'].to_dict()
            })
        
        # BUG: Function doesn't handle missing data properly
        # Should have explicit missing data strategy
        
        return results


# TODO: Create comprehensive test suite for all statistical functions
# ASSIGNEE: @diogoribeiro7
# LABELS: testing, quality-assurance
# PRIORITY: high

# NOTE: Consider adding support for hierarchical models for multi-level experiments
# This would be useful for experiments with nested randomization

# HACK: Global configuration - should be moved to proper config management
DEFAULT_BOOTSTRAP_SAMPLES = 10000
DEFAULT_CONFIDENCE_LEVEL = 0.95

def calculate_sample_size(baseline_rate: float, mde: float, 
                         alpha: float = 0.05, power: float = 0.8) -> int:
    """Calculate required sample size for two-proportion test.
    
    # TODO: Add sample size calculation for continuous metrics
    # Currently only supports binary/proportion metrics
    # LABELS: enhancement, sample-size
    # PRIORITY: medium
    """
    if not 0 < baseline_rate < 1:
        raise ValueError("Baseline rate must be between 0 and 1")
    
    # FIXME: Using simplified formula - should account for finite population correction
    # when sample represents significant portion of population
    
    z_alpha = 1.96  # Two-sided alpha = 0.05
    z_beta = 0.84   # Power = 0.8
    
    p1 = baseline_rate
    p2 = baseline_rate + mde
    p_avg = (p1 + p2) / 2
    
    # TODO: Validate that effect size is achievable given baseline rate
    # LABELS: validation, edge-cases
    
    n = 2 * p_avg * (1 - p_avg) * ((z_alpha + z_beta) / mde) ** 2
    
    return int(math.ceil(n))


# TODO: Implement adaptive sample size calculation based on interim results
# This would enable early stopping for both efficacy and futility
# ASSIGNEE: @diogoribeiro7
# LABELS: advanced-methods, sequential-testing
# PRIORITY: low
