"""
Advanced statistical tests including non-parametric methods, multiple testing corrections,
bootstrap methods, and confidence intervals.
"""

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import bootstrap
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import NormalIndPower, TTestPower
from statsmodels.stats.proportion import proportion_confint

warnings.filterwarnings("ignore")


@dataclass
class TestResult:
    """Container for statistical test results."""

    statistic: float
    p_value: float
    confidence_interval: Tuple[float, float]
    effect_size: float
    power: Optional[float] = None
    method: str = ""
    alternative: str = "two-sided"
    n_samples: Optional[int] = None
    additional_info: Optional[Dict[str, Any]] = None


class NonParametricTests:
    """Non-parametric statistical tests."""

    @staticmethod
    def mann_whitney_u(
        group1: np.ndarray,
        group2: np.ndarray,
        alternative: str = "two-sided",
        confidence_level: float = 0.95,
    ) -> TestResult:
        """
        Mann-Whitney U test (Wilcoxon rank-sum test).

        Args:
            group1: First group data
            group2: Second group data
            alternative: 'two-sided', 'greater', or 'less'
            confidence_level: Confidence level for CI

        Returns:
            TestResult with test statistics
        """
        statistic, p_value = stats.mannwhitneyu(group1, group2, alternative=alternative)

        # Calculate effect size (rank-biserial correlation)
        n1, n2 = len(group1), len(group2)
        r = 1 - (2 * statistic) / (n1 * n2)

        # Bootstrap confidence interval for difference in medians
        def median_diff(x, y):
            return np.median(x) - np.median(y)

        boot_samples = []
        for _ in range(10000):
            sample1 = np.random.choice(group1, size=n1, replace=True)
            sample2 = np.random.choice(group2, size=n2, replace=True)
            boot_samples.append(median_diff(sample1, sample2))

        alpha = 1 - confidence_level
        ci = tuple(
            np.percentile(boot_samples, [alpha / 2 * 100, (1 - alpha / 2) * 100])
        )

        return TestResult(
            statistic=float(statistic),
            p_value=float(p_value),
            confidence_interval=ci,
            effect_size=float(r),
            method="Mann-Whitney U",
            alternative=alternative,
            n_samples=n1 + n2,
        )

    @staticmethod
    def wilcoxon_signed_rank(
        group1: np.ndarray, group2: np.ndarray, alternative: str = "two-sided"
    ) -> TestResult:
        """
        Wilcoxon signed-rank test for paired samples.

        Args:
            group1: First group data (paired)
            group2: Second group data (paired)
            alternative: 'two-sided', 'greater', or 'less'

        Returns:
            TestResult with test statistics
        """
        differences = group1 - group2
        statistic, p_value = stats.wilcoxon(differences, alternative=alternative)

        # Effect size (matched pairs rank-biserial correlation)
        n = len(differences)
        r = 1 - (2 * statistic) / (n * (n + 1) / 2)

        # CI for median difference
        sorted_diff = np.sort(differences)
        alpha = 0.05
        k = stats.binom.ppf(alpha / 2, n, 0.5)
        ci_lower = sorted_diff[int(k)] if k >= 0 else sorted_diff[0]
        ci_upper = sorted_diff[int(n - k - 1)] if n - k - 1 < n else sorted_diff[-1]

        return TestResult(
            statistic=float(statistic),
            p_value=float(p_value),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            effect_size=float(r),
            method="Wilcoxon signed-rank",
            alternative=alternative,
            n_samples=n,
        )

    @staticmethod
    def kruskal_wallis(*groups) -> TestResult:
        """
        Kruskal-Wallis H test for multiple groups.

        Args:
            *groups: Variable number of group arrays

        Returns:
            TestResult with test statistics
        """
        statistic, p_value = stats.kruskal(*groups)

        # Effect size (epsilon-squared)
        n_total = sum(len(g) for g in groups)
        k = len(groups)
        epsilon_squared = (statistic - k + 1) / (n_total - k)

        return TestResult(
            statistic=float(statistic),
            p_value=float(p_value),
            confidence_interval=(np.nan, np.nan),  # Not applicable
            effect_size=float(epsilon_squared),
            method="Kruskal-Wallis",
            n_samples=n_total,
            additional_info={"n_groups": k},
        )

    @staticmethod
    def friedman(*groups) -> TestResult:
        """
        Friedman test for repeated measures.

        Args:
            *groups: Variable number of group arrays (must be same length)

        Returns:
            TestResult with test statistics
        """
        statistic, p_value = stats.friedmanchisquare(*groups)

        # Effect size (Kendall's W)
        k = len(groups)
        n = len(groups[0])
        W = statistic / (n * (k - 1))

        return TestResult(
            statistic=float(statistic),
            p_value=float(p_value),
            confidence_interval=(np.nan, np.nan),
            effect_size=float(W),
            method="Friedman",
            n_samples=n,
            additional_info={"n_groups": k},
        )

    @staticmethod
    def permutation_test(
        group1: np.ndarray,
        group2: np.ndarray,
        statistic_func: Callable = np.mean,
        n_permutations: int = 10000,
    ) -> TestResult:
        """
        Permutation test for any statistic.

        Args:
            group1: First group data
            group2: Second group data
            statistic_func: Function to calculate statistic
            n_permutations: Number of permutations

        Returns:
            TestResult with test statistics
        """
        observed_diff = statistic_func(group1) - statistic_func(group2)
        combined = np.concatenate([group1, group2])
        n1, n2 = len(group1), len(group2)

        perm_diffs = []
        for _ in range(n_permutations):
            np.random.shuffle(combined)
            perm_g1 = combined[:n1]
            perm_g2 = combined[n1:]
            perm_diffs.append(statistic_func(perm_g1) - statistic_func(perm_g2))

        perm_diffs = np.array(perm_diffs)
        p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))

        # CI from permutation distribution
        ci = tuple(np.percentile(perm_diffs, [2.5, 97.5]))

        # Effect size (standardized difference)
        pooled_std = np.sqrt((np.std(group1) ** 2 + np.std(group2) ** 2) / 2)
        effect_size = observed_diff / pooled_std if pooled_std > 0 else 0

        return TestResult(
            statistic=float(observed_diff),
            p_value=float(p_value),
            confidence_interval=ci,
            effect_size=float(effect_size),
            method="Permutation test",
            n_samples=n1 + n2,
            additional_info={"n_permutations": n_permutations},
        )


class MultipleTestingCorrections:
    """Methods for multiple testing corrections."""

    @staticmethod
    def apply_corrections(
        p_values: np.ndarray, alpha: float = 0.05, methods: List[str] = None
    ) -> pd.DataFrame:
        """
        Apply multiple testing corrections.

        Args:
            p_values: Array of p-values
            alpha: Significance level
            methods: List of correction methods to apply

        Returns:
            DataFrame with corrected p-values and decisions
        """
        if methods is None:
            methods = ["bonferroni", "holm", "fdr_bh", "fdr_by"]

        results = pd.DataFrame({"original_pvalue": p_values})

        for method in methods:
            reject, p_corrected, _, _ = multipletests(
                p_values, alpha=alpha, method=method
            )
            results[f"pvalue_{method}"] = p_corrected
            results[f"reject_{method}"] = reject

        # Add Šidák correction
        n = len(p_values)
        sidak_alpha = 1 - (1 - alpha) ** (1 / n)
        results["pvalue_sidak"] = 1 - (1 - p_values) ** n
        results["reject_sidak"] = p_values < sidak_alpha

        return results

    @staticmethod
    def false_discovery_rate(
        p_values: np.ndarray, alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Benjamini-Hochberg FDR control.

        Args:
            p_values: Array of p-values
            alpha: FDR level

        Returns:
            Tuple of (rejected hypotheses, adjusted p-values)
        """
        reject, p_adjusted, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")
        return reject, p_adjusted

    @staticmethod
    def family_wise_error_rate(
        p_values: np.ndarray, alpha: float = 0.05, method: str = "holm"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Control family-wise error rate.

        Args:
            p_values: Array of p-values
            alpha: FWER level
            method: Correction method ('bonferroni', 'holm', 'holm-sidak')

        Returns:
            Tuple of (rejected hypotheses, adjusted p-values)
        """
        reject, p_adjusted, _, _ = multipletests(p_values, alpha=alpha, method=method)
        return reject, p_adjusted


class BootstrapMethods:
    """Bootstrap methods for confidence intervals and hypothesis testing."""

    @staticmethod
    def bootstrap_ci(
        data: np.ndarray,
        statistic: Callable = np.mean,
        confidence_level: float = 0.95,
        n_bootstrap: int = 10000,
        method: str = "percentile",
    ) -> Tuple[float, Tuple[float, float]]:
        """
        Calculate bootstrap confidence interval.

        Args:
            data: Input data
            statistic: Statistic function
            confidence_level: Confidence level
            n_bootstrap: Number of bootstrap samples
            method: CI method ('percentile', 'basic', 'bca')

        Returns:
            Tuple of (point estimate, confidence interval)
        """
        # Point estimate
        point_estimate = statistic(data)

        # Bootstrap samples
        boot_statistics = []
        n = len(data)

        for _ in range(n_bootstrap):
            boot_sample = np.random.choice(data, size=n, replace=True)
            boot_statistics.append(statistic(boot_sample))

        boot_statistics = np.array(boot_statistics)
        alpha = 1 - confidence_level

        if method == "percentile":
            ci = tuple(
                np.percentile(boot_statistics, [alpha / 2 * 100, (1 - alpha / 2) * 100])
            )
        elif method == "basic":
            lower = 2 * point_estimate - np.percentile(
                boot_statistics, (1 - alpha / 2) * 100
            )
            upper = 2 * point_estimate - np.percentile(boot_statistics, alpha / 2 * 100)
            ci = (lower, upper)
        elif method == "bca":
            # BCa (Bias-Corrected and Accelerated)
            ci = BootstrapMethods._bca_interval(
                data, boot_statistics, point_estimate, statistic, confidence_level
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        return float(point_estimate), ci

    @staticmethod
    def _bca_interval(
        data: np.ndarray,
        boot_stats: np.ndarray,
        point_estimate: float,
        statistic: Callable,
        confidence_level: float,
    ) -> Tuple[float, float]:
        """Calculate BCa confidence interval."""
        n = len(data)
        alpha = 1 - confidence_level

        # Bias correction
        z0 = stats.norm.ppf(np.mean(boot_stats < point_estimate))

        # Acceleration
        jackknife_stats = []
        for i in range(n):
            jack_sample = np.delete(data, i)
            jackknife_stats.append(statistic(jack_sample))

        jackknife_stats = np.array(jackknife_stats)
        jack_mean = np.mean(jackknife_stats)
        numerator = np.sum((jack_mean - jackknife_stats) ** 3)
        denominator = 6 * (np.sum((jack_mean - jackknife_stats) ** 2) ** 1.5)
        a = numerator / denominator if denominator != 0 else 0

        # Adjusted percentiles
        z_alpha = stats.norm.ppf(alpha / 2)
        z_1alpha = stats.norm.ppf(1 - alpha / 2)

        p_lower = stats.norm.cdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)))
        p_upper = stats.norm.cdf(z0 + (z0 + z_1alpha) / (1 - a * (z0 + z_1alpha)))

        return tuple(np.percentile(boot_stats, [p_lower * 100, p_upper * 100]))

    @staticmethod
    def bootstrap_hypothesis_test(
        group1: np.ndarray,
        group2: np.ndarray,
        statistic: Callable = np.mean,
        n_bootstrap: int = 10000,
    ) -> TestResult:
        """
        Bootstrap hypothesis test for difference between groups.

        Args:
            group1: First group data
            group2: Second group data
            statistic: Statistic function
            n_bootstrap: Number of bootstrap samples

        Returns:
            TestResult with test statistics
        """
        observed_diff = statistic(group1) - statistic(group2)
        n1, n2 = len(group1), len(group2)

        # Bootstrap under null hypothesis (no difference)
        combined = np.concatenate([group1, group2])
        combined_mean = statistic(combined)

        # Center both groups at combined mean
        centered_g1 = group1 - statistic(group1) + combined_mean
        centered_g2 = group2 - statistic(group2) + combined_mean

        boot_diffs = []
        for _ in range(n_bootstrap):
            boot_g1 = np.random.choice(centered_g1, size=n1, replace=True)
            boot_g2 = np.random.choice(centered_g2, size=n2, replace=True)
            boot_diffs.append(statistic(boot_g1) - statistic(boot_g2))

        boot_diffs = np.array(boot_diffs)
        p_value = np.mean(np.abs(boot_diffs) >= np.abs(observed_diff))

        # Confidence interval for difference
        boot_diffs_ci = []
        for _ in range(n_bootstrap):
            boot_g1 = np.random.choice(group1, size=n1, replace=True)
            boot_g2 = np.random.choice(group2, size=n2, replace=True)
            boot_diffs_ci.append(statistic(boot_g1) - statistic(boot_g2))

        ci = tuple(np.percentile(boot_diffs_ci, [2.5, 97.5]))

        # Effect size
        pooled_std = np.sqrt((np.std(group1) ** 2 + np.std(group2) ** 2) / 2)
        effect_size = observed_diff / pooled_std if pooled_std > 0 else 0

        return TestResult(
            statistic=float(observed_diff),
            p_value=float(p_value),
            confidence_interval=ci,
            effect_size=float(effect_size),
            method="Bootstrap test",
            n_samples=n1 + n2,
            additional_info={"n_bootstrap": n_bootstrap},
        )

    @staticmethod
    def bootstrap_correlation(
        x: np.ndarray, y: np.ndarray, method: str = "pearson", n_bootstrap: int = 10000
    ) -> Tuple[float, Tuple[float, float], float]:
        """
        Bootstrap confidence interval for correlation.

        Args:
            x: First variable
            y: Second variable
            method: Correlation method ('pearson', 'spearman', 'kendall')
            n_bootstrap: Number of bootstrap samples

        Returns:
            Tuple of (correlation, CI, p-value)
        """
        n = len(x)

        if method == "pearson":
            corr_func = lambda a, b: stats.pearsonr(a, b)[0]
        elif method == "spearman":
            corr_func = lambda a, b: stats.spearmanr(a, b)[0]
        elif method == "kendall":
            corr_func = lambda a, b: stats.kendalltau(a, b)[0]
        else:
            raise ValueError(f"Unknown correlation method: {method}")

        # Observed correlation
        observed_corr = corr_func(x, y)

        # Bootstrap
        boot_corrs = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(n, size=n, replace=True)
            boot_corrs.append(corr_func(x[idx], y[idx]))

        boot_corrs = np.array(boot_corrs)
        ci = tuple(np.percentile(boot_corrs, [2.5, 97.5]))

        # Test for zero correlation
        p_value = np.mean(np.abs(boot_corrs) >= np.abs(observed_corr))

        return float(observed_corr), ci, float(p_value)


class PowerAnalysis:
    """Power analysis and sample size calculations."""

    @staticmethod
    def t_test_power(
        effect_size: float, n: int, alpha: float = 0.05, alternative: str = "two-sided"
    ) -> float:
        """
        Calculate power for t-test.

        Args:
            effect_size: Cohen's d
            n: Sample size per group
            alpha: Significance level
            alternative: 'two-sided', 'larger', or 'smaller'

        Returns:
            Statistical power
        """
        power_analysis = TTestPower()
        power = power_analysis.solve_power(
            effect_size=effect_size, nobs1=n, alpha=alpha, alternative=alternative
        )
        return float(power)

    @staticmethod
    def t_test_sample_size(
        effect_size: float,
        power: float = 0.8,
        alpha: float = 0.05,
        alternative: str = "two-sided",
    ) -> int:
        """
        Calculate required sample size for t-test.

        Args:
            effect_size: Cohen's d
            power: Desired power
            alpha: Significance level
            alternative: 'two-sided', 'larger', or 'smaller'

        Returns:
            Required sample size per group
        """
        power_analysis = TTestPower()
        n = power_analysis.solve_power(
            effect_size=effect_size, power=power, alpha=alpha, alternative=alternative
        )
        return int(np.ceil(n))

    @staticmethod
    def proportion_test_power(
        p1: float, p2: float, n: int, alpha: float = 0.05
    ) -> float:
        """
        Calculate power for proportion test.

        Args:
            p1: Proportion in group 1
            p2: Proportion in group 2
            n: Sample size per group
            alpha: Significance level

        Returns:
            Statistical power
        """
        effect_size = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))
        power_analysis = NormalIndPower()
        power = power_analysis.solve_power(
            effect_size=effect_size, nobs1=n, alpha=alpha
        )
        return float(power)

    @staticmethod
    def simulation_power(
        test_func: Callable,
        effect_size: float,
        n: int,
        n_simulations: int = 1000,
        alpha: float = 0.05,
    ) -> float:
        """
        Estimate power through simulation.

        Args:
            test_func: Function that performs the test
            effect_size: Effect size to simulate
            n: Sample size
            n_simulations: Number of simulations
            alpha: Significance level

        Returns:
            Estimated power
        """
        significant_results = 0

        for _ in range(n_simulations):
            # Generate data with effect
            group1 = np.random.normal(0, 1, n)
            group2 = np.random.normal(effect_size, 1, n)

            # Perform test
            p_value = test_func(group1, group2)

            if p_value < alpha:
                significant_results += 1

        return significant_results / n_simulations


class EffectSizeCalculations:
    """Calculate various effect size measures."""

    @staticmethod
    def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d."""
        mean_diff = np.mean(group1) - np.mean(group2)
        pooled_std = np.sqrt((np.var(group1, ddof=1) + np.var(group2, ddof=1)) / 2)
        return float(mean_diff / pooled_std) if pooled_std > 0 else 0.0

    @staticmethod
    def glass_delta(group1: np.ndarray, control_group: np.ndarray) -> float:
        """Calculate Glass's delta (uses control group SD)."""
        mean_diff = np.mean(group1) - np.mean(control_group)
        control_std = np.std(control_group, ddof=1)
        return float(mean_diff / control_std) if control_std > 0 else 0.0

    @staticmethod
    def hedges_g(group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Hedges' g (corrected Cohen's d)."""
        n1, n2 = len(group1), len(group2)
        cohens_d = EffectSizeCalculations.cohens_d(group1, group2)

        # Hedges' correction factor
        df = n1 + n2 - 2
        correction = 1 - 3 / (4 * df - 1)

        return float(cohens_d * correction)

    @staticmethod
    def eta_squared(groups: List[np.ndarray]) -> float:
        """Calculate eta-squared for ANOVA."""
        all_data = np.concatenate(groups)
        grand_mean = np.mean(all_data)

        # Between-group sum of squares
        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)

        # Total sum of squares
        ss_total = np.sum((all_data - grand_mean) ** 2)

        return float(ss_between / ss_total) if ss_total > 0 else 0.0

    @staticmethod
    def omega_squared(groups: List[np.ndarray]) -> float:
        """Calculate omega-squared (less biased than eta-squared)."""
        k = len(groups)
        n_total = sum(len(g) for g in groups)

        # Perform ANOVA
        f_stat, _ = stats.f_oneway(*groups)

        # Calculate omega-squared
        df_between = k - 1
        df_within = n_total - k
        ms_between = f_stat * (
            np.sum([np.var(g, ddof=1) * len(g) for g in groups]) / df_within
        )

        omega_sq = (df_between * (ms_between - 1)) / (df_between * ms_between + n_total)

        return float(max(0, omega_sq))  # Can't be negative


if __name__ == "__main__":
    # Example usage
    print("Advanced Statistical Tests Module")
    print("=" * 60)

    # Generate sample data
    np.random.seed(42)
    group1 = np.random.normal(100, 15, 50)
    group2 = np.random.normal(105, 15, 50)

    # Non-parametric test
    print("\nNon-parametric Test (Mann-Whitney U):")
    result = NonParametricTests.mann_whitney_u(group1, group2)
    print(f"  Statistic: {result.statistic:.3f}")
    print(f"  P-value: {result.p_value:.4f}")
    print(f"  Effect size (r): {result.effect_size:.3f}")
    print(
        f"  CI: [{result.confidence_interval[0]:.2f}, {result.confidence_interval[1]:.2f}]"
    )

    # Bootstrap test
    print("\nBootstrap Test:")
    boot_result = BootstrapMethods.bootstrap_hypothesis_test(group1, group2)
    print(f"  Difference: {boot_result.statistic:.3f}")
    print(f"  P-value: {boot_result.p_value:.4f}")
    print(
        f"  CI: [{boot_result.confidence_interval[0]:.2f}, {boot_result.confidence_interval[1]:.2f}]"
    )

    # Multiple testing correction
    print("\nMultiple Testing Corrections:")
    p_values = np.array([0.01, 0.04, 0.03, 0.05, 0.20])
    corrections = MultipleTestingCorrections.apply_corrections(p_values)
    print(
        corrections[["original_pvalue", "pvalue_bonferroni", "pvalue_fdr_bh"]].round(4)
    )

    # Power analysis
    print("\nPower Analysis:")
    power = PowerAnalysis.t_test_power(effect_size=0.5, n=50)
    print(f"  Power for d=0.5, n=50: {power:.3f}")
    required_n = PowerAnalysis.t_test_sample_size(effect_size=0.5, power=0.8)
    print(f"  Required n for d=0.5, power=0.8: {required_n}")

    # Effect sizes
    print("\nEffect Size Calculations:")
    print(f"  Cohen's d: {EffectSizeCalculations.cohens_d(group1, group2):.3f}")
    print(f"  Hedges' g: {EffectSizeCalculations.hedges_g(group1, group2):.3f}")
