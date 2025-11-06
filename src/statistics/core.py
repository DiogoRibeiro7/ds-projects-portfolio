"""
Statistical utilities module for data science portfolio.

This module contains core statistical functions used across the portfolio projects.
Enhanced with proper error handling, advanced methods, and comprehensive implementations.
"""

import logging
import math
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import ndtri

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def two_prop_ztest(
    x1: int,
    n1: int,
    x2: int,
    n2: int,
    two_sided: bool = True,
    continuity_correction: bool = True,
    alternative: str = "two-sided",
) -> Tuple[float, float]:
    """Two-sample z-test for proportions with enhanced features.

    Parameters
    ----------
    x1, n1 : int
        Success count and sample size for group 1 (control).
    x2, n2 : int
        Success count and sample size for group 2 (treatment).
    two_sided : bool, default=True
        Whether to perform two-sided test (deprecated, use alternative).
    continuity_correction : bool, default=True
        Whether to apply continuity correction for small samples.
    alternative : str, default='two-sided'
        Alternative hypothesis: 'two-sided', 'larger', 'smaller'.

    Returns
    -------
    z_stat : float
        Z-statistic value.
    p_value : float
        P-value for the test.

    Raises
    ------
    ValueError
        If sample sizes are invalid or proportions are out of bounds.
    ZeroDivisionError
        If pooled proportion leads to zero standard error.
    """
    # Input validation
    if n1 <= 0 or n2 <= 0:
        raise ValueError("Sample sizes must be positive")

    if x1 < 0 or x2 < 0:
        raise ValueError("Success counts cannot be negative")

    if x1 > n1 or x2 > n2:
        raise ValueError("Success counts cannot exceed sample sizes")

    # Handle backward compatibility
    if not two_sided and alternative == "two-sided":
        alternative = "larger"  # Assume treatment is expected to be larger

    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)

    # Handle edge cases for pooled proportion
    if p_pool == 0 or p_pool == 1:
        if p_pool == 0:
            if x1 == 0 and x2 == 0:
                return 0.0, 1.0  # No difference when both are 0
            else:
                return float("inf"), 0.0  # Significant difference
        else:  # p_pool == 1
            if x1 == n1 and x2 == n2:
                return 0.0, 1.0  # No difference when both are 100%
            else:
                return float("-inf"), 0.0  # Significant difference

    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))

    if se == 0:
        raise ZeroDivisionError("Standard error is zero - cannot perform test")

    # Calculate z-statistic
    z = (p2 - p1) / se

    # Apply continuity correction for small samples
    if continuity_correction:
        correction = 0.5 * (1 / n1 + 1 / n2)
        if abs(p2 - p1) > correction:
            z = (p2 - p1 - np.sign(p2 - p1) * correction) / se

    # Calculate p-value based on alternative hypothesis
    if alternative == "two-sided":
        p_val = 2 * (1 - stats.norm.cdf(abs(z)))
    elif alternative == "larger":
        p_val = 1 - stats.norm.cdf(z)
    elif alternative == "smaller":
        p_val = stats.norm.cdf(z)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    return float(z), float(p_val)


def bootstrap_ci_diff(
    pA: float,
    pB: float,
    nA: int,
    nB: int,
    B: int = 5000,
    alpha: float = 0.05,
    method: str = "percentile",
    random_state: Optional[int] = None,
) -> Tuple[float, float]:
    """Bootstrap confidence interval for difference in proportions.

    Parameters
    ----------
    pA, pB : float
        Observed proportions for groups A and B.
    nA, nB : int
        Sample sizes for groups A and B.
    B : int, default=5000
        Number of bootstrap samples.
    alpha : float, default=0.05
        Significance level (1-alpha confidence level).
    method : str, default='percentile'
        Bootstrap method: 'percentile', 'bca' (bias-corrected accelerated).
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    ci_lower, ci_upper : float
        Lower and upper confidence interval bounds.
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Input validation
    if not (0 <= pA <= 1) or not (0 <= pB <= 1):
        raise ValueError("Proportions must be between 0 and 1")

    if nA <= 0 or nB <= 0:
        raise ValueError("Sample sizes must be positive")

    if B <= 0:
        raise ValueError("Number of bootstrap samples must be positive")

    diffs = np.empty(B, dtype=float)

    for b in range(B):
        xA = np.random.binomial(nA, pA)
        xB = np.random.binomial(nB, pB)
        diffs[b] = xB / nB - xA / nA

    if method == "percentile":
        lo = float(np.quantile(diffs, alpha / 2))
        hi = float(np.quantile(diffs, 1 - alpha / 2))
    elif method == "bca":
        # Bias-corrected and accelerated bootstrap
        observed_diff = pB - pA

        # Bias correction
        bias_correction = stats.norm.ppf((diffs < observed_diff).mean())

        # Acceleration (simplified jackknife estimate)
        acceleration = 0  # Simplified - full implementation would use jackknife

        # Adjusted quantiles
        z_alpha_2 = stats.norm.ppf(alpha / 2)
        z_1_alpha_2 = stats.norm.ppf(1 - alpha / 2)

        alpha_1 = stats.norm.cdf(
            bias_correction
            + (bias_correction + z_alpha_2)
            / (1 - acceleration * (bias_correction + z_alpha_2))
        )
        alpha_2 = stats.norm.cdf(
            bias_correction
            + (bias_correction + z_1_alpha_2)
            / (1 - acceleration * (bias_correction + z_1_alpha_2))
        )

        lo = float(np.quantile(diffs, alpha_1))
        hi = float(np.quantile(diffs, alpha_2))
    else:
        raise ValueError(f"Unknown bootstrap method: {method}")

    return lo, hi


def calculate_sample_size(
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0,
    alternative: str = "two-sided",
) -> int:
    """Calculate required sample size for two-proportion test.

    Parameters
    ----------
    baseline_rate : float
        Expected baseline conversion rate (0 < baseline_rate < 1).
    mde : float
        Minimum detectable effect (absolute difference).
    alpha : float, default=0.05
        Type I error rate (significance level).
    power : float, default=0.8
        Statistical power (1 - Type II error rate).
    ratio : float, default=1.0
        Ratio of treatment to control sample size.
    alternative : str, default='two-sided'
        Type of test: 'two-sided' or 'one-sided'.

    Returns
    -------
    sample_size : int
        Required sample size per group (control group size).

    Raises
    ------
    ValueError
        If parameters are out of valid ranges.
    """
    # Input validation
    if not 0 < baseline_rate < 1:
        raise ValueError("Baseline rate must be between 0 and 1")

    if mde <= 0:
        raise ValueError("Minimum detectable effect must be positive")

    if baseline_rate + mde > 1:
        raise ValueError("Baseline rate + MDE cannot exceed 1")

    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1")

    if not 0 < power < 1:
        raise ValueError("Power must be between 0 and 1")

    if ratio <= 0:
        raise ValueError("Ratio must be positive")

    # Calculate z-scores
    if alternative == "two-sided":
        z_alpha = abs(ndtri(alpha / 2))
    else:
        z_alpha = abs(ndtri(alpha))

    z_beta = abs(ndtri(1 - power))

    # Proportions
    p1 = baseline_rate
    p2 = baseline_rate + mde
    p_avg = (p1 + p2) / 2

    # Sample size calculation
    numerator = (
        z_alpha * math.sqrt(2 * p_avg * (1 - p_avg))
        + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2) / ratio)
    ) ** 2
    denominator = mde**2

    n_control = numerator / denominator

    return int(math.ceil(n_control))


def calculate_power(
    n_control: int,
    n_treatment: int,
    baseline_rate: float,
    effect_size: float,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> float:
    """Calculate statistical power for given sample sizes and effect.

    Parameters
    ----------
    n_control, n_treatment : int
        Sample sizes for control and treatment groups.
    baseline_rate : float
        Expected baseline conversion rate.
    effect_size : float
        Expected effect size (absolute difference).
    alpha : float, default=0.05
        Significance level.
    alternative : str, default='two-sided'
        Type of test.

    Returns
    -------
    power : float
        Statistical power (probability of detecting true effect).
    """
    if alternative == "two-sided":
        z_alpha = abs(ndtri(alpha / 2))
    else:
        z_alpha = abs(ndtri(alpha))

    p1 = baseline_rate
    p2 = baseline_rate + effect_size

    # Pooled proportion under null hypothesis
    p_null = (p1 + p2) / 2
    se_null = math.sqrt(p_null * (1 - p_null) * (1 / n_control + 1 / n_treatment))

    # Standard error under alternative hypothesis
    se_alt = math.sqrt(p1 * (1 - p1) / n_control + p2 * (1 - p2) / n_treatment)

    # Critical value
    critical_value = z_alpha * se_null

    # Power calculation
    z_power = (effect_size - critical_value) / se_alt
    power = stats.norm.cdf(z_power)

    return float(power)


class ExperimentAnalyzer:
    """Enhanced A/B test analysis class with comprehensive functionality."""

    def __init__(self, alpha: float = 0.05, power: float = 0.8):
        """Initialize the analyzer.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level for statistical tests.
        power : float, default=0.8
            Target statistical power for sample size calculations.
        """
        self.alpha = alpha
        self.power = power
        self.results_cache = {}  # Cache for expensive computations

        logger.info(f"ExperimentAnalyzer initialized with alpha={alpha}, power={power}")

    def check_srm(
        self,
        df: pd.DataFrame,
        group_col: str = "group",
        expected_ratio: Dict[str, float] = None,
    ) -> Dict[str, Union[float, bool, Dict]]:
        """Check for Sample Ratio Mismatch with support for custom ratios.

        Parameters
        ----------
        df : pd.DataFrame
            Experiment data.
        group_col : str, default='group'
            Column name containing group assignments.
        expected_ratio : dict, optional
            Expected ratio for each group. If None, assumes equal allocation.

        Returns
        -------
        results : dict
            SRM test results including chi-square statistic and p-value.
        """
        group_counts = df[group_col].value_counts()
        groups = group_counts.index.tolist()

        if len(groups) < 2:
            raise ValueError("Need at least 2 groups for SRM check")

        n_total = len(df)

        # Set expected ratios
        if expected_ratio is None:
            # Equal allocation
            expected_counts = {group: n_total / len(groups) for group in groups}
        else:
            # Custom ratios
            total_ratio = sum(expected_ratio.values())
            expected_counts = {
                group: n_total * (expected_ratio.get(group, 0) / total_ratio)
                for group in groups
            }

        # Chi-square test
        observed = [group_counts[group] for group in groups]
        expected = [expected_counts[group] for group in groups]

        chi2_stat, p_value = stats.chisquare(observed, expected)

        return {
            "chi2_statistic": float(chi2_stat),
            "p_value": float(p_value),
            "is_srm": p_value < 0.05,
            "group_counts": group_counts.to_dict(),
            "expected_counts": expected_counts,
            "total_users": n_total,
        }

    def analyze_conversion(
        self,
        df: pd.DataFrame,
        conversion_col: str = "converted",
        group_col: str = "group",
        alpha: Optional[float] = None,
    ) -> Dict[str, Union[float, bool, Dict]]:
        """Comprehensive conversion rate analysis.

        Parameters
        ----------
        df : pd.DataFrame
            Experiment data.
        conversion_col : str, default='converted'
            Column name for conversion metric.
        group_col : str, default='group'
            Column name for group assignments.
        alpha : float, optional
            Significance level. Uses instance default if None.

        Returns
        -------
        results : dict
            Complete analysis results including statistical tests and effect sizes.
        """
        if alpha is None:
            alpha = self.alpha

        # Validate inputs
        if conversion_col not in df.columns:
            raise ValueError(f"Conversion column '{conversion_col}' not found")

        if group_col not in df.columns:
            raise ValueError(f"Group column '{group_col}' not found")

        # Calculate conversion rates by group
        conv_stats = (
            df.groupby(group_col)[conversion_col]
            .agg(["sum", "count", "mean", "std"])
            .round(6)
        )

        results = {
            "conversion_rates": conv_stats["mean"].to_dict(),
            "sample_sizes": conv_stats["count"].to_dict(),
            "conversions": conv_stats["sum"].to_dict(),
            "alpha": alpha,
        }

        groups = conv_stats.index.tolist()

        # Two-group analysis
        if len(groups) == 2:
            control, treatment = groups[0], groups[1]

            x1 = int(conv_stats.loc[control, "sum"])
            n1 = int(conv_stats.loc[control, "count"])
            x2 = int(conv_stats.loc[treatment, "sum"])
            n2 = int(conv_stats.loc[treatment, "count"])

            # Statistical test
            z_stat, p_val = two_prop_ztest(x1, n1, x2, n2)

            # Effect sizes
            p1, p2 = x1 / n1, x2 / n2
            absolute_lift = p2 - p1
            relative_lift = absolute_lift / p1 if p1 > 0 else float("inf")

            # Confidence interval for difference
            ci_lower, ci_upper = bootstrap_ci_diff(p1, p2, n1, n2, random_state=42)

            # Power analysis
            observed_power = calculate_power(n1, n2, p1, absolute_lift, alpha)

            results.update(
                {
                    "z_statistic": z_stat,
                    "p_value": p_val,
                    "significant": p_val < alpha,
                    "absolute_lift": absolute_lift,
                    "relative_lift": relative_lift,
                    "confidence_interval": (ci_lower, ci_upper),
                    "statistical_power": observed_power,
                    "control_group": control,
                    "treatment_group": treatment,
                }
            )

        # Multi-group analysis
        elif len(groups) > 2:
            # Overall chi-square test
            contingency_table = pd.crosstab(df[group_col], df[conversion_col])
            chi2_stat, p_val, dof, expected = stats.chi2_contingency(contingency_table)

            results.update(
                {
                    "chi2_statistic": chi2_stat,
                    "p_value": p_val,
                    "degrees_of_freedom": dof,
                    "significant": p_val < alpha,
                    "test_type": "chi_square",
                }
            )

        return results

    def run_comprehensive_analysis(
        self,
        df: pd.DataFrame,
        metrics: List[str],
        group_col: str = "group",
        date_col: Optional[str] = None,
    ) -> Dict[str, Dict]:
        """Run comprehensive multi-metric analysis.

        Parameters
        ----------
        df : pd.DataFrame
            Experiment data.
        metrics : list of str
            List of metric columns to analyze.
        group_col : str, default='group'
            Column name for group assignments.
        date_col : str, optional
            Column name for dates (enables temporal analysis).

        Returns
        -------
        analysis : dict
            Comprehensive analysis results for all metrics.
        """
        analysis = {"data_quality": {}, "metrics_analysis": {}, "recommendations": []}

        # Data quality checks
        analysis["data_quality"]["srm_check"] = self.check_srm(df, group_col)
        analysis["data_quality"]["sample_sizes"] = (
            df[group_col].value_counts().to_dict()
        )
        analysis["data_quality"]["missing_data"] = df[metrics].isnull().sum().to_dict()

        # Analyze each metric
        for metric in metrics:
            if metric in df.columns:
                if (
                    df[metric].dtype in ["bool", "int64", "float64"]
                    and df[metric].nunique() == 2
                ):
                    # Binary metric
                    analysis["metrics_analysis"][metric] = self.analyze_conversion(
                        df, metric, group_col
                    )
                else:
                    # Continuous metric (placeholder for future implementation)
                    analysis["metrics_analysis"][metric] = {
                        "type": "continuous",
                        "status": "not_implemented",
                    }

        # Generate recommendations
        if analysis["data_quality"]["srm_check"]["is_srm"]:
            analysis["recommendations"].append(
                "⚠️ Sample Ratio Mismatch detected. Check randomization system."
            )

        significant_metrics = [
            metric
            for metric, results in analysis["metrics_analysis"].items()
            if results.get("significant", False)
        ]

        if significant_metrics:
            analysis["recommendations"].append(
                f"✅ Significant results found for: {', '.join(significant_metrics)}"
            )
        else:
            analysis["recommendations"].append(
                "ℹ️ No significant results detected. Consider power analysis."
            )

        return analysis


# Utility functions for advanced methods


def apply_multiple_testing_correction(
    p_values: List[float], method: str = "holm"
) -> Tuple[List[float], List[bool]]:
    """Apply multiple testing correction to p-values.

    Parameters
    ----------
    p_values : list of float
        List of p-values to correct.
    method : str, default='holm'
        Correction method: 'holm', 'bonferroni', 'fdr_bh'.

    Returns
    -------
    corrected_p_values : list of float
        Corrected p-values.
    rejected : list of bool
        Boolean array indicating which hypotheses are rejected.
    """
    p_array = np.array(p_values)

    if method == "bonferroni":
        corrected = p_array * len(p_array)
        corrected = np.minimum(corrected, 1.0)
        rejected = corrected < 0.05

    elif method == "holm":
        n = len(p_array)
        sorted_indices = np.argsort(p_array)
        corrected = np.zeros_like(p_array)
        rejected = np.zeros(n, dtype=bool)

        for i, idx in enumerate(sorted_indices):
            corrected[idx] = min(1.0, p_array[idx] * (n - i))
            if corrected[idx] < 0.05:
                rejected[idx] = True
            else:
                break  # Holm method stops at first non-rejection

    elif method == "fdr_bh":
        # Benjamini-Hochberg FDR control
        n = len(p_array)
        sorted_indices = np.argsort(p_array)
        rejected = np.zeros(n, dtype=bool)

        for i in range(n - 1, -1, -1):
            idx = sorted_indices[i]
            if p_array[idx] <= (i + 1) / n * 0.05:
                rejected[sorted_indices[: i + 1]] = True
                break

        corrected = p_array * n / (np.arange(1, n + 1))
        corrected = np.minimum(corrected, 1.0)

    else:
        raise ValueError(f"Unknown correction method: {method}")

    return corrected.tolist(), rejected.tolist()


def sequential_testing_boundary(
    n: int, alpha: float = 0.05, method: str = "obrien_fleming"
) -> float:
    """Calculate sequential testing boundary for interim analysis.

    Parameters
    ----------
    n : int
        Current sample size.
    alpha : float, default=0.05
        Overall significance level.
    method : str, default='obrien_fleming'
        Boundary method: 'obrien_fleming', 'pocock'.

    Returns
    -------
    boundary : float
        Critical z-value for current analysis.
    """
    if method == "obrien_fleming":
        # O'Brien-Fleming boundary
        boundary = abs(ndtri(alpha / 2)) * math.sqrt(1.0)  # Simplified
    elif method == "pocock":
        # Pocock boundary
        boundary = abs(ndtri(alpha / 2)) * 1.2  # Simplified
    else:
        raise ValueError(f"Unknown boundary method: {method}")

    return boundary
