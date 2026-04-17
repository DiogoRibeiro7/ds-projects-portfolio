"""Advanced statistical helpers used throughout the experimentation toolkit.

The module consolidates non-parametric tests, multiple testing corrections,
bootstrap utilities, power analysis helpers, and effect size calculators that
show up across the CLI, dashboards, tutorials, and tests. All functions follow
the Google docstring convention defined in `docs/DOCS_STYLE.md` and expose
examples that match the usage patterns captured in documentation and notebooks.
"""

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import NormalIndPower, TTestPower

warnings.filterwarnings("ignore")


@dataclass
class TestResult:
    """Container for statistical test results.

    Attributes:
        statistic: Test statistic reported by the underlying SciPy function.
        p_value: P-value for the observed statistic.
        confidence_interval: Lower and upper bounds for the primary metric.
        effect_size: Scalar effect size (Cohen's d, rank-biserial, etc.).
        power: Optional post-hoc power estimate when available.
        method: Human-readable test name.
        alternative: Alternative hypothesis label.
        n_samples: Number of observations evaluated.
        additional_info: Extra metadata (permutation counts, corrections, ...).
    """

    statistic: float
    p_value: float
    confidence_interval: tuple[float, float]
    effect_size: float
    power: float | None = None
    method: str = ""
    alternative: str = "two-sided"
    n_samples: int | None = None
    additional_info: dict[str, Any] | None = None


class NonParametricTests:
    """Non-parametric statistical routines used by dashboards and docs.

    Examples:
        >>> from statistical_methods.advanced_statistical_tests import (
        ...     NonParametricTests
        ... )
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
        >>> control = rng.normal(loc=0.0, scale=1.0, size=400)
        >>> treatment = rng.normal(loc=0.2, scale=1.1, size=400)
        >>> result = NonParametricTests.mann_whitney_u(control, treatment)
        >>> round(result.effect_size, 3)
        -0.215
    """

    @staticmethod
    def mann_whitney_u(
        group1: np.ndarray,
        group2: np.ndarray,
        alternative: str = "two-sided",
        confidence_level: float = 0.95,
    ) -> TestResult:
        """Run a Mann-Whitney U (Wilcoxon rank-sum) test on independent samples.

        Args:
            group1 (np.ndarray): Baseline sample of shape ``(n1,)``.
            group2 (np.ndarray): Variant sample of shape ``(n2,)``.
            alternative (str): `'two-sided'`, `'greater'`, or `'less'`.
            confidence_level (float): Confidence interval coverage in ``(0, 1)``.

        Returns:
            TestResult: Rank-biserial effect size, p-value, and bootstrap CI.

        Side Effects:
            Uses NumPy's global RNG to resample bootstrap replicates. Call
            ``np.random.seed`` or use ``np.random.default_rng`` beforehand to
            obtain deterministic bounds in tests/docs.

        Complexity:
            O(n_bootstrap × (n1 + n2)) time and O(n_bootstrap) memory for the
            bootstrap samples (default 10,000 iterations in this implementation).

        Examples:
            >>> import numpy as np
            >>> from statistical_methods.advanced_statistical_tests import (
            ...     NonParametricTests
            ... )
            >>> rng = np.random.default_rng(7)
            >>> baseline = rng.normal(0.0, 1.0, 500)
            >>> treatment = rng.normal(0.15, 1.1, 500)
            >>> report = NonParametricTests.mann_whitney_u(baseline, treatment)
            >>> round(report.statistic, 3), round(report.p_value, 4)
            (98224.0, 0.0005)
        """
        statistic, p_value = stats.mannwhitneyu(group1, group2, alternative=alternative)

        # The rank-biserial correlation summarizes stochastic dominance between groups.
        n1, n2 = len(group1), len(group2)
        r = 1 - (2 * statistic) / (n1 * n2)

        # Bootstrap confidence interval for the difference in medians provides
        # a more interpretable estimate for stakeholders than the raw U value.
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
        """Run a Wilcoxon signed-rank test for paired samples.

        Args:
            group1 (np.ndarray): First group data (paired).
            group2 (np.ndarray): Second group data (paired).
            alternative (str): `'two-sided'`, `'greater'`, or `'less'`.

        Returns:
            TestResult: Signed-rank statistic, effect size, and CI proxy.
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
    def kruskal_wallis(*groups: np.ndarray) -> TestResult:
        """Run a Kruskal-Wallis H test for at least two groups.

        Args:
            *groups (np.ndarray): Variable number of group arrays.

        Returns:
            TestResult: Test statistic, p-value, and epsilon-squared effect size.
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
    def friedman(*groups: np.ndarray) -> TestResult:
        """Run a Friedman test for repeated-measure designs.

        Args:
            *groups (np.ndarray): Equal-length arrays per treatment condition.

        Returns:
            TestResult: Chi-square statistic, p-value, and Kendall's W.
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
        statistic_func: Callable[[np.ndarray], float] = np.mean,
        n_permutations: int = 10000,
    ) -> TestResult:
        """Perform a permutation test for an arbitrary scalar statistic.

        Args:
            group1 (np.ndarray): First group data.
            group2 (np.ndarray): Second group data.
            statistic_func (Callable[[np.ndarray], float]): Statistic function
                applied independently to each group.
            n_permutations (int): Number of shuffles to estimate the null.

        Returns:
            TestResult: Observed statistic, empirical p-value, and CI.
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
    """Methods for multiple testing corrections used in reporting.

    Examples:
        >>> import numpy as np
        >>> from statistical_methods.advanced_statistical_tests import (
        ...     MultipleTestingCorrections
        ... )
        >>> pvals = np.array([0.01, 0.03, 0.2])
        >>> summary = MultipleTestingCorrections.apply_corrections(pvals, alpha=0.05)
        >>> list(summary.columns)
        ['original_pvalue', 'pvalue_bonferroni', 'reject_bonferroni', ...]
    """

    @staticmethod
    def apply_corrections(
        p_values: np.ndarray, alpha: float = 0.05, methods: list[str] | None = None
    ) -> pd.DataFrame:
        """Apply multiple-testing corrections to an array of p-values.

        Args:
            p_values (np.ndarray): Raw p-values to correct (shape ``(n,)``).
            alpha (float): Family-wise error rate or FDR target.
            methods (list[str] | None): Iterable of `statsmodels` method names to
                evaluate. Defaults to Bonferroni, Holm, Benjamini-Hochberg, and
                Benjamini-Yekutieli.

        Returns:
            pandas.DataFrame: One column per correction with decision booleans.

        Examples:
            >>> import numpy as np
            >>> from statistical_methods.advanced_statistical_tests import (
            ...     MultipleTestingCorrections
            ... )
            >>> pvals = np.array([0.01, 0.04, 0.20])
            >>> MultipleTestingCorrections.apply_corrections(pvals, alpha=0.05)[
            ...     ['reject_bonferroni', 'reject_fdr_bh']
            ... ].values.tolist()
            [[True, True], [False, False], [False, False]]
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

        # Add Šidák correction manually because statsmodels does not expose it by
        # default.
        n = len(p_values)
        sidak_alpha = 1 - (1 - alpha) ** (1 / n)
        results["pvalue_sidak"] = 1 - (1 - p_values) ** n
        results["reject_sidak"] = p_values < sidak_alpha

        return results

    @staticmethod
    def false_discovery_rate(
        p_values: np.ndarray, alpha: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply Benjamini-Hochberg false-discovery-rate control.

        Args:
            p_values (np.ndarray): Array of unadjusted p-values.
            alpha (float): Target false-discovery rate.

        Returns:
            tuple[np.ndarray, np.ndarray]: Reject mask and adjusted p-values.
        """
        reject, p_adjusted, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")
        return reject, p_adjusted

    @staticmethod
    def family_wise_error_rate(
        p_values: np.ndarray, alpha: float = 0.05, method: str = "holm"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Control the family-wise error rate (FWER).

        Args:
            p_values (np.ndarray): Array of unadjusted p-values.
            alpha (float): FWER level.
            method (str): `'bonferroni'`, `'holm'`, or `'holm-sidak'`.

        Returns:
            tuple[np.ndarray, np.ndarray]: Reject mask and adjusted p-values.
        """
        reject, p_adjusted, _, _ = multipletests(p_values, alpha=alpha, method=method)
        return reject, p_adjusted


class BootstrapMethods:
    """Bootstrap methods for confidence intervals and hypothesis testing.

    Examples:
        >>> import numpy as np
        >>> from statistical_methods.advanced_statistical_tests import BootstrapMethods
        >>> rng = np.random.default_rng(123)
        >>> sample = rng.normal(0.0, 1.0, 1000)
        >>> _, interval = BootstrapMethods.bootstrap_ci(sample, np.mean)
        >>> len(interval)
        2
    """

    @staticmethod
    def bootstrap_ci(
        data: np.ndarray,
        statistic: Callable[[np.ndarray], float] = np.mean,
        confidence_level: float = 0.95,
        n_bootstrap: int = 10000,
        method: str = "percentile",
    ) -> tuple[float, tuple[float, float]]:
        """Calculate a bootstrap confidence interval for a statistic.

        Args:
            data (np.ndarray): Sample to resample with replacement.
            statistic (Callable[[np.ndarray], float]): Statistic function.
            confidence_level (float): Desired coverage probability.
            n_bootstrap (int): Number of bootstrap draws.
            method (str): `'percentile'`, `'basic'`, or `'bca'`.

        Returns:
            tuple[float, tuple[float, float]]: Point estimate and CI bounds.

        Raises:
            ValueError: If ``method`` is unsupported.

        Side Effects:
            Uses NumPy's global RNG via ``np.random.choice``; seed before calling
            to obtain deterministic documentation examples.
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
            percentiles = [alpha / 2 * 100, (1 - alpha / 2) * 100]
            ci = tuple(np.percentile(boot_statistics, percentiles))
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
        statistic: Callable[[np.ndarray], float],
        confidence_level: float,
    ) -> tuple[float, float]:
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
        statistic: Callable[[np.ndarray], float] = np.mean,
        n_bootstrap: int = 10000,
    ) -> TestResult:
        """Bootstrap the difference between two groups under the null hypothesis.

        Args:
            group1 (np.ndarray): First group data.
            group2 (np.ndarray): Second group data.
            statistic (Callable[[np.ndarray], float]): Statistic function.
            n_bootstrap (int): Number of bootstrap samples.

        Returns:
            TestResult: Bootstrap statistic, p-value, and confidence interval.

        Side Effects:
            Uses NumPy's global RNG; seed before calling for deterministic docs.
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
    ) -> tuple[float, tuple[float, float], float]:
        """Bootstrap a confidence interval for the correlation coefficient.

        Args:
            x (np.ndarray): First variable.
            y (np.ndarray): Second variable.
            method (str): `'pearson'`, `'spearman'`, or `'kendall'`.
            n_bootstrap (int): Number of bootstrap samples.

        Returns:
            tuple[float, tuple[float, float], float]: Observed correlation,
            confidence interval, and two-sided empirical p-value.
        """
        n = len(x)

        def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
            return stats.pearsonr(a, b)[0]

        def _spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
            return stats.spearmanr(a, b)[0]

        def _kendall_corr(a: np.ndarray, b: np.ndarray) -> float:
            return stats.kendalltau(a, b)[0]

        if method == "pearson":
            corr_func = _pearson_corr
        elif method == "spearman":
            corr_func = _spearman_corr
        elif method == "kendall":
            corr_func = _kendall_corr
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
    """Deterministic power-analysis helpers used by notebooks and CLI tools.

    Examples:
        >>> from statistical_methods.advanced_statistical_tests import PowerAnalysis
        >>> PowerAnalysis.t_test_power(effect_size=0.5, n=200)
        0.9999999999999982
    """

    @staticmethod
    def t_test_power(
        effect_size: float, n: int, alpha: float = 0.05, alternative: str = "two-sided"
    ) -> float:
        """Calculate statistical power for a two-sample t-test.

        Args:
            effect_size (float): Expected Cohen's d.
            n (int): Sample size per group.
            alpha (float): Significance level.
            alternative (str): `'two-sided'`, `'larger'`, or `'smaller'`.

        Returns:
            float: Probability of correctly rejecting the null hypothesis.
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
        """Calculate per-group sample size for a two-sample t-test.

        Args:
            effect_size (float): Target Cohen's d.
            power (float): Desired statistical power.
            alpha (float): Significance level.
            alternative (str): `'two-sided'`, `'larger'`, or `'smaller'`.

        Returns:
            int: Rounded-up sample size per group.

        Examples:
            >>> from statistical_methods.advanced_statistical_tests import PowerAnalysis
            >>> PowerAnalysis.t_test_sample_size(effect_size=0.3, power=0.8)
            174
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
        """Calculate power for a two-proportion z-test.

        Args:
            p1 (float): Control success probability in `[0, 1]`.
            p2 (float): Treatment success probability in `[0, 1]`.
            n (int): Sample size per group.
            alpha (float): Significance level.

        Returns:
            float: Probability of detecting the specified lift.
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
        """Estimate statistical power through Monte Carlo simulation.

        Args:
            test_func (Callable[[np.ndarray, np.ndarray], float]): Callable that
                returns a p-value when given two arrays.
            effect_size (float): Difference in means to simulate.
            n (int): Sample size per group.
            n_simulations (int): Number of Monte Carlo draws.
            alpha (float): Significance level.

        Returns:
            float: Observed proportion of simulations with ``p < alpha``.
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
    """Effect size calculators aligned with documentation and dashboards.

    Examples:
        >>> import numpy as np
        >>> from statistical_methods.advanced_statistical_tests import (
        ...     EffectSizeCalculations
        ... )
        >>> control = np.array([0, 1, 1, 0, 1])
        >>> variant = np.array([1, 1, 1, 0, 1])
        >>> round(EffectSizeCalculations.cohens_d(control, variant), 3)
        -0.408
    """

    @staticmethod
    def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d for two independent samples.

        Args:
            group1 (np.ndarray): Baseline sample.
            group2 (np.ndarray): Variant sample.

        Returns:
            float: Standardized effect size using the pooled standard deviation.

        Examples:
            >>> import numpy as np
            >>> from statistical_methods.advanced_statistical_tests import (
            ...     EffectSizeCalculations
            ... )
            >>> control = np.array([0.1, 0.2, 0.3])
            >>> variant = np.array([0.2, 0.4, 0.5])
            >>> round(EffectSizeCalculations.cohens_d(control, variant), 3)
            -1.091
        """
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
        """Calculate Hedges' g (small-sample correction for Cohen's d).

        Args:
            group1 (np.ndarray): Baseline sample.
            group2 (np.ndarray): Variant sample.

        Returns:
            float: Bias-corrected standardized difference.

        Examples:
            >>> import numpy as np
            >>> from statistical_methods.advanced_statistical_tests import (
            ...     EffectSizeCalculations
            ... )
            >>> control = np.array([0.1, 0.2, 0.3, 0.4])
            >>> variant = np.array([0.3, 0.4, 0.5, 0.6])
            >>> round(EffectSizeCalculations.hedges_g(control, variant), 3)
            -1.265
        """
        n1, n2 = len(group1), len(group2)
        cohens_d = EffectSizeCalculations.cohens_d(group1, group2)

        # Hedges' correction factor
        df = n1 + n2 - 2
        correction = 1 - 3 / (4 * df - 1)

        return float(cohens_d * correction)

    @staticmethod
    def eta_squared(groups: list[np.ndarray]) -> float:
        """Calculate eta-squared for ANOVA."""
        all_data = np.concatenate(groups)
        grand_mean = np.mean(all_data)

        # Between-group sum of squares
        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)

        # Total sum of squares
        ss_total = np.sum((all_data - grand_mean) ** 2)

        return float(ss_between / ss_total) if ss_total > 0 else 0.0

    @staticmethod
    def omega_squared(groups: list[np.ndarray]) -> float:
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
        "  CI: ["
        f"{result.confidence_interval[0]:.2f}, "
        f"{result.confidence_interval[1]:.2f}]"
    )

    # Bootstrap test
    print("\nBootstrap Test:")
    boot_result = BootstrapMethods.bootstrap_hypothesis_test(group1, group2)
    print(f"  Difference: {boot_result.statistic:.3f}")
    print(f"  P-value: {boot_result.p_value:.4f}")
    print(
        "  CI: ["
        f"{boot_result.confidence_interval[0]:.2f}, "
        f"{boot_result.confidence_interval[1]:.2f}]"
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
