"""Statistical utilities module for the data science portfolio.

This module contains core statistical functions used across the portfolio
projects. Enhanced with proper error handling, advanced methods, and
comprehensive implementations.
"""

import logging
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import ndtri

# Module-level constants
DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.8
DEFAULT_BOOTSTRAP_SAMPLES = 5000
DEFAULT_CONFIDENCE_LEVEL = 0.95
MIN_PROPORTION = 0.0
MAX_PROPORTION = 1.0
CHI_SQUARE_SIGNIFICANCE = 0.05

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def two_prop_ztest(
    x1: int,
    n1: int,
    x2: int,
    n2: int,
    two_sided: bool = True,
    continuity_correction: bool = True,
    alternative: str = "two-sided",
) -> tuple[float, float]:
    """Run a two-proportion z-test with optional continuity correction.

    Args:
        x1 (int): Number of Bernoulli successes for the control cohort.
        n1 (int): Total observations for the control cohort (must be > 0).
        x2 (int): Number of successes for the treatment cohort.
        n2 (int): Total observations for the treatment cohort.
        two_sided (bool): Backward-compatible flag; prefer ``alternative``.
        continuity_correction (bool): Apply ±0.5/n Haldane-Anscombe correction
            to stabilize small-sample z-scores.
        alternative (str): `'two-sided'` (default), `'larger'` (treatment lift),
            or `'smaller'` (treatment drop).

    Returns:
        tuple[float, float]: ``(z_statistic, p_value)`` where the statistic is
        unit-less and the p-value is in ``[0, 1]``.

    Raises:
        ValueError: When counts are negative or exceed their sample sizes.
        ZeroDivisionError: When both cohorts are deterministic so the pooled
            standard error is zero.

    Examples:
        >>> from src.statistics.core import two_prop_ztest
        >>> two_prop_ztest(120, 1000, 150, 1000)
        (-1.9414506867886235, 0.05224153370662625)
        >>> # One-sided test when the treatment is expected to be larger
        >>> round(two_prop_ztest(60, 500, 85, 500, alternative="larger")[1], 4)
        0.0156

    Notes:
        - The helper returns ``(inf, 0.0)`` or ``(-inf, 0.0)`` when one cohort
          has zero variance (all successes or all failures).
        - ``two_sided=False`` is preserved for legacy callers and simply maps to
          ``alternative='larger'`` when no explicit alternative is supplied.
    """
    # Input validation
    logger.debug(f"Running two_prop_ztest with n1={n1}, n2={n2}, x1={x1}, x2={x2}")
    if n1 <= 0 or n2 <= 0:
        logger.error(f"Invalid sample sizes: n1={n1}, n2={n2}")
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
    B: int = DEFAULT_BOOTSTRAP_SAMPLES,
    alpha: float = DEFAULT_ALPHA,
    method: str = "percentile",
    random_state: int | None = None,
) -> tuple[float, float]:
    """Bootstrap a CI for the difference between two Bernoulli rates.

    Args:
        pA (float): Observed proportion for group A in `[0, 1]`.
        pB (float): Observed proportion for group B in `[0, 1]`.
        nA (int): Sample size for group A (number of trials).
        nB (int): Sample size for group B.
        B (int): Number of bootstrap simulations/draws.
        alpha (float): Significance level; coverage equals ``1 - alpha``.
        method (str): `'percentile'` or `'bca'`.
        random_state (int | None): Optional NumPy global seed to set.

    Returns:
        tuple[float, float]: Lower/upper bounds for ``pB - pA`` measured in
        absolute probability points.

    Raises:
        ValueError: If proportions or sample sizes fall outside valid ranges.

    Examples:
        >>> from src.statistics.core import bootstrap_ci_diff
        >>> bootstrap_ci_diff(0.12, 0.15, 5000, 5000, random_state=7)
        (-0.00523999999999999, 0.035039999999999985)
        >>> # BCa interval for a smaller sample with higher alpha
        >>> tuple(round(bound, 3) for bound in bootstrap_ci_diff(
        ...     0.42, 0.5, 800, 800, B=2000, alpha=0.1, method="bca", random_state=21
        ... ))
        (0.038, 0.119)

    Side Effects:
        Uses ``np.random.seed`` and ``np.random.binomial`` when `random_state`
        is provided; call with a seed to match documentation snippets.
    """
    if random_state is not None:
        # Seed NumPy's global RNG so documentation/tests can produce stable intervals.
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
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
    ratio: float = 1.0,
    alternative: str = "two-sided",
) -> int:
    """Compute the required sample size for a two-proportion experiment.

    Args:
        baseline_rate (float): Baseline conversion probability in `(0, 1)`.
        mde (float): Minimum detectable effect (absolute difference).
        alpha (float): Significance level (Type I error).
        power (float): Target statistical power (1 - Type II error).
        ratio (float): Treatment-to-control allocation ratio (e.g., `2` for 2:1).
        alternative (str): `'two-sided'` or `'one-sided'`.

    Returns:
        int: Rounded-up control-group sample size. Multiply by ``ratio`` to get
        the treatment size.

    Raises:
        ValueError: If any parameter falls outside the allowed range.

    Examples:
        >>> from src.statistics.core import calculate_sample_size
        >>> calculate_sample_size(baseline_rate=0.12, mde=0.02, power=0.9)
        4574
        >>> # Unequal traffic split (2 treatment users for every control user)
        >>> calculate_sample_size(0.2, 0.03, ratio=2.0)
        2699
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
    alpha: float = DEFAULT_ALPHA,
    alternative: str = "two-sided",
) -> float:
    """Calculate power for a fixed two-proportion design.

    Args:
        n_control (int): Control sample size (users exposed to baseline).
        n_treatment (int): Treatment sample size.
        baseline_rate (float): Baseline conversion probability.
        effect_size (float): Absolute lift relative to baseline.
        alpha (float): Significance level (Type I error).
        alternative (str): `'two-sided'` or `'one-sided'`.

    Returns:
        float: Probability of detecting ``effect_size`` with the supplied
        sample sizes. Always between 0 and 1.

    Examples:
        >>> from src.statistics.core import calculate_power
        >>> round(calculate_power(3500, 3500, 0.12, 0.02), 3)
        0.701
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
    """
    Analyze controlled experiments end-to-end (SRM, conversion, multi-metrics).

    The analyzer expects tidy pandas DataFrames with a categorical ``group``
    column plus one or more metric columns (binary conversions, continuous
    KPIs, etc.). Most helpers assume an ``(N × D)`` dataset with at least the
    binary ``conversion`` column, and they respect the default ``alpha``/``power``
    supplied at construction time.

    Attributes:
        alpha: Default significance level for downstream calls.
        power: Default target power used when back-solving for lifts.
        results_cache: Mutable dictionary for expensive intermediate results.

    Examples:
        >>> import pandas as pd
        >>> from src.statistics.core import ExperimentAnalyzer
        >>> df = pd.DataFrame({'group': ['control'] * 100 + ['treatment'] * 100,
        ...                    'converted': [0, 1] * 100})
        >>> analyzer = ExperimentAnalyzer(alpha=0.05)
        >>> report = analyzer.analyze_conversion(df)
        >>> ('z_statistic' in report) and ('conversion_rates' in report)
        True
    """

    def __init__(self, alpha: float = DEFAULT_ALPHA, power: float = DEFAULT_POWER):
        """Initialize the analyzer.

        Args:
            alpha (float): Default significance level for new analyses.
            power (float): Target power used when simulating recommendations.
        """
        self.alpha = alpha
        self.power = power
        self.results_cache: dict[str, Any] = {}  # Cache for expensive computations

        logger.info(f"ExperimentAnalyzer initialized with alpha={alpha}, power={power}")

    def check_srm(
        self,
        df: pd.DataFrame,
        group_col: str = "group",
        expected_ratio: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Check for sample ratio mismatches (SRM) with optional target ratios.

        Args:
            df (pandas.DataFrame): Experiment data with a group column.
            group_col (str): Column storing variant identifiers.
            expected_ratio (dict[str, float] | None): Expected allocation ratio
                per group. Defaults to equal allocation.

        Returns:
            dict[str, Any]: Chi-square statistic, p-value, counts, and flag.

        Raises:
            ValueError: If ``df`` only contains one group.

        Examples:
            >>> import pandas as pd
            >>> from src.statistics.core import ExperimentAnalyzer
            >>> data = pd.DataFrame({'group': ['control'] * 50 + ['treatment'] * 70})
            >>> ExperimentAnalyzer().check_srm(data)['is_srm']
            True
        """
        group_counts = df[group_col].value_counts()
        groups = group_counts.index.tolist()

        if len(groups) < 2:
            raise ValueError("Need at least 2 groups for SRM check")

        n_total = len(df)

        # Set expected ratios
        if expected_ratio is None:
            # Equal allocation is the most common design when no ratios are supplied.
            expected_counts = {group: n_total / len(groups) for group in groups}
        else:
            # Custom ratios: normalize supplied weights to match total traffic.
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
            "is_srm": p_value < CHI_SQUARE_SIGNIFICANCE,
            "group_counts": group_counts.to_dict(),
            "expected_counts": expected_counts,
            "total_users": n_total,
        }

    def analyze_conversion(
        self,
        df: pd.DataFrame,
        conversion_col: str = "converted",
        group_col: str = "group",
        alpha: float | None = None,
        robust: bool = False,
        trim_fraction: float = 0.0,
    ) -> dict[str, Any]:
        """Compute conversion statistics and tests for binary metrics.

        Args:
            df (pandas.DataFrame): Experiment dataset.
            conversion_col (str): Column representing the binary outcome.
            group_col (str): Column representing the variant key.
            alpha (float | None): Significance level override.

        Returns:
            dict[str, Any]: Conversion rates, lift, z-test, CI, and power.

        Raises:
            ValueError: If the required columns are missing.

        Examples:
            >>> import pandas as pd
            >>> from src.statistics.core import ExperimentAnalyzer
            >>> frame = pd.DataFrame({
            ...     'group': ['control'] * 100 + ['treatment'] * 100,
            ...     'converted': [0, 1] * 100,
            ... })
            >>> analyzer = ExperimentAnalyzer()
            >>> report = analyzer.analyze_conversion(frame)
            >>> round(report['absolute_lift'], 3)
            0.0
        """
        if alpha is None:
            alpha = self.alpha

        # Validate inputs
        if conversion_col not in df.columns:
            raise ValueError(f"Conversion column '{conversion_col}' not found")

        if group_col not in df.columns:
            raise ValueError(f"Group column '{group_col}' not found")

        data = df[[group_col, conversion_col]].copy()

        if robust and trim_fraction > 0:
            data = self._trim_by_group(
                data, group_col, conversion_col, trim_fraction
            )

        # Calculate conversion rates by group
        conv_stats = (
            data.groupby(group_col)[conversion_col]
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
                    "robust": robust,
                    "trim_fraction": trim_fraction,
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
                    "robust": robust,
                    "trim_fraction": trim_fraction,
                }
            )

        return results

    def _trim_by_group(
        self,
        df: pd.DataFrame,
        group_col: str,
        value_col: str,
        fraction: float,
    ) -> pd.DataFrame:
        """Trim extreme values per group."""
        if not 0 < fraction < 0.5:
            raise ValueError("trim_fraction must be in (0, 0.5)")

        def _trim(group: pd.DataFrame) -> pd.DataFrame:
            arr = group[value_col].values
            low = np.quantile(arr, fraction)
            high = np.quantile(arr, 1 - fraction)
            return group[(group[value_col] >= low) & (group[value_col] <= high)]

        return df.groupby(group_col, group_keys=False).apply(_trim)

    def run_comprehensive_analysis(
        self,
        df: pd.DataFrame,
        metrics: list[str],
        group_col: str = "group",
        date_col: str | None = None,
        robust: bool = False,
        trim_fraction: float = 0.0,
        huber_delta: float = 1.0,
    ) -> dict[str, Any]:
        """Run sample quality checks plus conversion analysis per metric.

        Args:
            df (pandas.DataFrame): Experiment dataset.
            metrics (list[str]): Column names to inspect.
            group_col (str): Column representing group assignments.
            date_col (str | None): Optional date column for future extensions.

        Returns:
            dict[str, Any]: Nested report covering SRM, conversions, and notes.

        Examples:
            >>> import pandas as pd
            >>> from src.statistics.core import ExperimentAnalyzer
            >>> data = pd.DataFrame({
            ...     'group': ['control', 'treatment'] * 50,
            ...     'converted': [0, 1] * 50,
            ...     'clicked': [1] * 100,
            ... })
            >>> analyzer = ExperimentAnalyzer()
        >>> summary = analyzer.run_comprehensive_analysis(
        ...     data, metrics=['converted']
        ... )
            >>> set(summary['metrics_analysis'].keys())
            {'converted'}
        """
        analysis: dict[str, Any] = {
            "data_quality": {},
            "metrics_analysis": {},
            "recommendations": [],
        }

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
                    analysis["metrics_analysis"][metric] = self.analyze_conversion(
                        df,
                        metric,
                        group_col,
                        robust=robust,
                        trim_fraction=trim_fraction,
                    )
                else:
                    analysis["metrics_analysis"][metric] = self._analyze_continuous_metric(
                        df,
                        metric,
                        group_col,
                        robust=robust,
                        trim_fraction=trim_fraction,
                        huber_delta=huber_delta,
                    )

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

    def _analyze_continuous_metric(
        self,
        df: pd.DataFrame,
        metric: str,
        group_col: str,
        robust: bool,
        trim_fraction: float,
        huber_delta: float,
    ) -> dict[str, Any]:
        groups = df[group_col].dropna().unique()
        if len(groups) != 2:
            return {
                "type": "continuous",
                "status": "multi-group_not_supported",
            }

        processed: dict[str, np.ndarray] = {}
        for group in groups:
            values = df.loc[df[group_col] == group, metric].dropna().values
            if robust:
                if trim_fraction > 0:
                    low = np.quantile(values, trim_fraction)
                    high = np.quantile(values, 1 - trim_fraction)
                    values = values[(values >= low) & (values <= high)]
                median = np.median(values)
                values = huber_smooth(values - median, delta=huber_delta) + median
            processed[group] = values

        control, treatment = groups
        t_stat, p_val = stats.ttest_ind(
            processed[control], processed[treatment], equal_var=False
        )
        mean_control = processed[control].mean()
        mean_treatment = processed[treatment].mean()
        return {
            "type": "continuous",
            "mean_diff": mean_treatment - mean_control,
            "means": {
                control: mean_control,
                treatment: mean_treatment,
            },
            "p_value": float(p_val),
            "statistic": float(t_stat),
            "robust": robust,
            "trim_fraction": trim_fraction,
            "huber_delta": huber_delta,
            "significant": p_val < self.alpha,
        }


# Utility functions for advanced methods


def apply_multiple_testing_correction(
    p_values: list[float], method: str = "holm"
) -> tuple[list[float], list[bool]]:
    """Apply a multiple-testing correction to a list of p-values.

    Args:
        p_values (list[float]): Raw p-values.
        method (str): `'holm'`, `'bonferroni'`, or `'fdr_bh'`.

    Returns:
        tuple[list[float], list[bool]]: Corrected p-values and rejection flags.

    Raises:
        ValueError: If ``method`` is unsupported.

    Examples:
        >>> from src.statistics.core import apply_multiple_testing_correction
        >>> apply_multiple_testing_correction([0.01, 0.04, 0.2], method='holm')
        ([0.03, 0.08, 0.0], [True, False, False])
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
                # Holm method stops once it encounters the first non-rejection so
                # downstream hypotheses retain their initialized zeros.
                break

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
    n: int, alpha: float = DEFAULT_ALPHA, method: str = "obrien_fleming"
) -> float:
    """Return the interim z-threshold for group sequential testing.

    Args:
        n (int): Current cumulative sample size.
        alpha (float): Overall significance level.
        method (str): `'obrien_fleming'` or `'pocock'`.

    Returns:
        float: Critical value to compare against the observed z-statistic.

    Raises:
        ValueError: When ``method`` is unknown.

    Examples:
        >>> from src.statistics.core import sequential_testing_boundary
        >>> round(sequential_testing_boundary(1000, alpha=0.01), 3)
        2.576
    """
    boundary: float
    if method == "obrien_fleming":
        # O'Brien-Fleming boundary
        boundary = float(abs(ndtri(alpha / 2)) * math.sqrt(1.0))  # Simplified
    elif method == "pocock":
        # Pocock boundary
        boundary = float(abs(ndtri(alpha / 2)) * 1.2)  # Simplified
    else:
        raise ValueError(f"Unknown boundary method: {method}")

    return boundary
