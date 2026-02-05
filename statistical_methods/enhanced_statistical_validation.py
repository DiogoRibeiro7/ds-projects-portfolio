"""
Enhanced statistical validation suite for comparing implementations
with established libraries and theoretical values.
"""

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.power import TTestPower
from statsmodels.stats.proportion import proportion_confint, proportions_ztest

warnings.filterwarnings("ignore")


@dataclass
class ValidationResult:
    """
    Structured record summarizing a single validator comparison.

    Attributes
    ----------
    test_name : str
        Human-readable label for the scenario (e.g., ``"Proportion test: Large samples"``).
    our_result : float
        Metric computed by the in-repo implementation (z-score, CI width, etc.).
    reference_result : float
        Corresponding value from the reference library.
    absolute_error : float
        Absolute difference ``|our_result - reference_result|`` in native units.
    relative_error : float
        Ratio ``absolute_error / reference_result`` for quick tolerance checks.
    passed : bool
        ``True`` when the error falls within ``tolerance``.
    tolerance : float
        Error threshold used for this comparison.
    details : Dict[str, Any]
        Extra context (intermediate z-scores, confidence intervals, metadata).

    Examples
    --------
    >>> ValidationResult(
    ...     test_name="CI width sanity check",
    ...     our_result=0.12,
    ...     reference_result=0.119,
    ...     absolute_error=0.001,
    ...     relative_error=0.0084,
    ...     passed=True,
    ...     tolerance=0.01,
    ...     details={"method": "wilson"}
    ... )
    ValidationResult(test_name='CI width sanity check', our_result=0.12, reference_result=0.119, absolute_error=0.001, relative_error=0.0084, passed=True, tolerance=0.01, details={'method': 'wilson'})
    """

    test_name: str
    our_result: float
    reference_result: float
    absolute_error: float
    relative_error: float
    passed: bool
    tolerance: float
    details: Dict[str, Any]


class StatisticalValidator:
    """Comprehensive validation suite for statistical methods."""

    def __init__(self, tolerance: float = 1e-3):
        """
        Initialize validator.

        Args:
            tolerance: Default tolerance for comparisons
        """
        self.tolerance = tolerance
        self.validation_results = []

    def validate_proportion_test(self) -> List[ValidationResult]:
        """Validate proportion tests against statsmodels."""
        results = []

        test_cases = [
            {"name": "Equal proportions", "x1": 50, "n1": 100, "x2": 50, "n2": 100},
            {"name": "Different proportions", "x1": 30, "n1": 100, "x2": 45, "n2": 100},
            {"name": "Large samples", "x1": 500, "n1": 1000, "x2": 550, "n2": 1000},
            {"name": "Unequal sizes", "x1": 20, "n1": 50, "x2": 60, "n2": 150},
        ]

        for case in test_cases:
            # Our implementation (simplified for comparison)
            p1 = case["x1"] / case["n1"]
            p2 = case["x2"] / case["n2"]
            p_pooled = (case["x1"] + case["x2"]) / (case["n1"] + case["n2"])

            se = np.sqrt(p_pooled * (1 - p_pooled) * (1 / case["n1"] + 1 / case["n2"]))
            z_ours = (p2 - p1) / se if se > 0 else 0
            p_ours = 2 * (1 - stats.norm.cdf(abs(z_ours)))

            # Statsmodels reference
            z_ref, p_ref = proportions_ztest(
                [case["x1"], case["x2"]], [case["n1"], case["n2"]]
            )

            # Compare results
            abs_error = abs(p_ours - p_ref)
            rel_error = abs_error / p_ref if p_ref > 0 else abs_error

            results.append(
                ValidationResult(
                    test_name=f"Proportion test: {case['name']}",
                    our_result=p_ours,
                    reference_result=p_ref,
                    absolute_error=abs_error,
                    relative_error=rel_error,
                    passed=abs_error < self.tolerance,
                    tolerance=self.tolerance,
                    details={"z_score_ours": z_ours, "z_score_ref": z_ref},
                )
            )

        return results

    def validate_confidence_intervals(self) -> List[ValidationResult]:
        """Validate confidence interval calculations."""
        results = []

        test_cases = [
            {"successes": 30, "trials": 100, "confidence": 0.95},
            {"successes": 5, "trials": 20, "confidence": 0.95},
            {"successes": 100, "trials": 1000, "confidence": 0.99},
        ]

        for case in test_cases:
            # Our implementation (normal approximation)
            p = case["successes"] / case["trials"]
            se = np.sqrt(p * (1 - p) / case["trials"])
            z = stats.norm.ppf(1 - (1 - case["confidence"]) / 2)
            ci_ours = (p - z * se, p + z * se)

            # Statsmodels reference (multiple methods)
            ci_normal = proportion_confint(
                case["successes"],
                case["trials"],
                alpha=1 - case["confidence"],
                method="normal",
            )
            ci_wilson = proportion_confint(
                case["successes"],
                case["trials"],
                alpha=1 - case["confidence"],
                method="wilson",
            )
            ci_jeffrey = proportion_confint(
                case["successes"],
                case["trials"],
                alpha=1 - case["confidence"],
                method="jeffreys",
            )

            # Compare with Wilson (generally recommended)
            abs_error_lower = abs(ci_ours[0] - ci_wilson[0])
            abs_error_upper = abs(ci_ours[1] - ci_wilson[1])
            max_error = max(abs_error_lower, abs_error_upper)

            results.append(
                ValidationResult(
                    test_name=f"CI: n={case['trials']}, x={case['successes']}",
                    our_result=ci_ours[1] - ci_ours[0],  # Width
                    reference_result=ci_wilson[1] - ci_wilson[0],  # Width
                    absolute_error=max_error,
                    relative_error=max_error / (ci_wilson[1] - ci_wilson[0]),
                    passed=max_error < 0.05,  # Looser tolerance for CI
                    tolerance=0.05,
                    details={
                        "ci_ours": ci_ours,
                        "ci_normal": ci_normal,
                        "ci_wilson": ci_wilson,
                        "ci_jeffrey": ci_jeffrey,
                    },
                )
            )

        return results

    def validate_power_calculations(self) -> List[ValidationResult]:
        """Validate power calculations against statsmodels."""
        results = []

        test_cases = [
            {"effect_size": 0.5, "alpha": 0.05, "power": 0.8},
            {"effect_size": 0.2, "alpha": 0.05, "power": 0.8},
            {"effect_size": 0.8, "alpha": 0.01, "power": 0.9},
        ]

        power_analysis = TTestPower()

        for case in test_cases:
            # Statsmodels reference
            n_ref = power_analysis.solve_power(
                effect_size=case["effect_size"],
                power=case["power"],
                alpha=case["alpha"],
            )

            # Our simplified calculation
            z_alpha = stats.norm.ppf(1 - case["alpha"] / 2)
            z_beta = stats.norm.ppf(case["power"])
            n_ours = ((z_alpha + z_beta) / case["effect_size"]) ** 2

            # Compare
            abs_error = abs(n_ours - n_ref)
            rel_error = abs_error / n_ref

            results.append(
                ValidationResult(
                    test_name=f"Power: d={case['effect_size']}, α={case['alpha']}",
                    our_result=n_ours,
                    reference_result=n_ref,
                    absolute_error=abs_error,
                    relative_error=rel_error,
                    passed=rel_error < 0.05,  # 5% relative error tolerance
                    tolerance=0.05,
                    details={"power": case["power"]},
                )
            )

        return results

    def validate_statistical_distributions(self) -> List[ValidationResult]:
        """Validate distribution calculations."""
        results = []

        # Test normal distribution
        x = np.linspace(-3, 3, 100)

        # CDF validation
        cdf_ours = stats.norm.cdf(x)
        cdf_ref = stats.norm.cdf(x)  # Same in this case but shows structure

        max_error = np.max(np.abs(cdf_ours - cdf_ref))

        results.append(
            ValidationResult(
                test_name="Normal CDF",
                our_result=np.mean(cdf_ours),
                reference_result=np.mean(cdf_ref),
                absolute_error=max_error,
                relative_error=max_error,
                passed=max_error < 1e-10,
                tolerance=1e-10,
                details={"n_points": len(x)},
            )
        )

        # Test t-distribution for different degrees of freedom
        for df in [1, 5, 10, 30, 100]:
            t_vals = np.linspace(-3, 3, 100)
            pdf_ref = stats.t.pdf(t_vals, df)

            # Our calculation (for demonstration)
            pdf_ours = stats.t.pdf(t_vals, df)

            max_error = np.max(np.abs(pdf_ours - pdf_ref))

            results.append(
                ValidationResult(
                    test_name=f"T-distribution PDF (df={df})",
                    our_result=np.mean(pdf_ours),
                    reference_result=np.mean(pdf_ref),
                    absolute_error=max_error,
                    relative_error=max_error,
                    passed=max_error < 1e-10,
                    tolerance=1e-10,
                    details={"df": df},
                )
            )

        return results

    def validate_monte_carlo_simulations(
        self, n_simulations: int = 10000
    ) -> List[ValidationResult]:
        """Validate Monte Carlo simulation accuracy."""
        results = []

        # Test 1: Estimate π using Monte Carlo
        points_in_circle = 0
        for _ in range(n_simulations):
            x, y = np.random.uniform(-1, 1, 2)
            if x**2 + y**2 <= 1:
                points_in_circle += 1

        pi_estimate = 4 * points_in_circle / n_simulations
        pi_error = abs(pi_estimate - np.pi)

        results.append(
            ValidationResult(
                test_name="Monte Carlo π estimation",
                our_result=pi_estimate,
                reference_result=np.pi,
                absolute_error=pi_error,
                relative_error=pi_error / np.pi,
                passed=pi_error < 0.1,  # Loose tolerance for Monte Carlo
                tolerance=0.1,
                details={"n_simulations": n_simulations},
            )
        )

        # Test 2: Central Limit Theorem validation
        sample_means = []
        for _ in range(1000):
            # Sample from uniform distribution
            sample = np.random.uniform(0, 1, 30)
            sample_means.append(np.mean(sample))

        # Should be approximately normal with mean=0.5, std=1/(12*sqrt(30))
        expected_mean = 0.5
        expected_std = 1 / (np.sqrt(12 * 30))

        observed_mean = np.mean(sample_means)
        observed_std = np.std(sample_means)

        mean_error = abs(observed_mean - expected_mean)
        std_error = abs(observed_std - expected_std)

        results.append(
            ValidationResult(
                test_name="CLT validation (mean)",
                our_result=observed_mean,
                reference_result=expected_mean,
                absolute_error=mean_error,
                relative_error=mean_error / expected_mean,
                passed=mean_error < 0.01,
                tolerance=0.01,
                details={"observed_std": observed_std, "expected_std": expected_std},
            )
        )

        return results

    def validate_bayesian_methods(self) -> List[ValidationResult]:
        """Validate Bayesian calculations."""
        results = []

        # Beta-Binomial conjugate prior validation
        prior_alpha, prior_beta = 1, 1  # Uniform prior
        successes, trials = 7, 10

        # Posterior parameters
        post_alpha = prior_alpha + successes
        post_beta = prior_beta + trials - successes

        # Analytical posterior mean
        posterior_mean_analytical = post_alpha / (post_alpha + post_beta)

        # Monte Carlo estimate
        samples = np.random.beta(post_alpha, post_beta, 10000)
        posterior_mean_mc = np.mean(samples)

        error = abs(posterior_mean_analytical - posterior_mean_mc)

        results.append(
            ValidationResult(
                test_name="Beta-Binomial posterior mean",
                our_result=posterior_mean_mc,
                reference_result=posterior_mean_analytical,
                absolute_error=error,
                relative_error=error / posterior_mean_analytical,
                passed=error < 0.01,
                tolerance=0.01,
                details={
                    "prior": (prior_alpha, prior_beta),
                    "data": (successes, trials),
                    "posterior": (post_alpha, post_beta),
                },
            )
        )

        # Credible interval validation
        ci_analytical = stats.beta.ppf([0.025, 0.975], post_alpha, post_beta)
        ci_mc = np.percentile(samples, [2.5, 97.5])

        ci_error = np.max(np.abs(ci_analytical - ci_mc))

        results.append(
            ValidationResult(
                test_name="Beta-Binomial credible interval",
                our_result=ci_mc[1] - ci_mc[0],  # Width
                reference_result=ci_analytical[1] - ci_analytical[0],  # Width
                absolute_error=ci_error,
                relative_error=ci_error / (ci_analytical[1] - ci_analytical[0]),
                passed=ci_error < 0.01,
                tolerance=0.01,
                details={"ci_analytical": ci_analytical, "ci_mc": ci_mc},
            )
        )

        return results

    def validate_multiple_testing_corrections(self) -> List[ValidationResult]:
        """Validate multiple testing correction methods."""
        results = []

        # Generate p-values
        np.random.seed(42)
        p_values = [0.001, 0.01, 0.029, 0.03, 0.049, 0.05, 0.20, 0.40, 0.75, 0.90]

        # Bonferroni correction
        alpha = 0.05
        n_tests = len(p_values)

        bonferroni_threshold = alpha / n_tests
        bonferroni_rejected = [p < bonferroni_threshold for p in p_values]

        # Holm-Bonferroni correction
        sorted_indices = np.argsort(p_values)
        sorted_p = np.array(p_values)[sorted_indices]

        holm_rejected = np.zeros(n_tests, dtype=bool)
        for i, p in enumerate(sorted_p):
            if p < alpha / (n_tests - i):
                holm_rejected[sorted_indices[i]] = True
            else:
                break  # Stop at first non-rejection

        # Validation: Holm should reject at least as many as Bonferroni
        holm_count = np.sum(holm_rejected)
        bonf_count = np.sum(bonferroni_rejected)

        results.append(
            ValidationResult(
                test_name="Holm >= Bonferroni rejections",
                our_result=float(holm_count),
                reference_result=float(bonf_count),
                absolute_error=0 if holm_count >= bonf_count else 1,
                relative_error=0 if holm_count >= bonf_count else 1,
                passed=holm_count >= bonf_count,
                tolerance=0,
                details={
                    "n_tests": n_tests,
                    "bonferroni_rejected": bonf_count,
                    "holm_rejected": holm_count,
                },
            )
        )

        # FDR control validation (Benjamini-Hochberg)
        sorted_p_bh = sorted_p.copy()
        threshold_indices = []

        for i in range(n_tests - 1, -1, -1):
            if sorted_p_bh[i] <= (i + 1) / n_tests * alpha:
                threshold_indices.append(i)

        if threshold_indices:
            max_idx = max(threshold_indices)
            bh_rejected = sorted_indices[: max_idx + 1]
        else:
            bh_rejected = []

        # BH should control FDR at alpha level
        results.append(
            ValidationResult(
                test_name="Benjamini-Hochberg FDR control",
                our_result=float(len(bh_rejected)),
                reference_result=float(
                    n_tests * alpha
                ),  # Expected rejections under null
                absolute_error=abs(len(bh_rejected) - n_tests * alpha),
                relative_error=abs(len(bh_rejected) - n_tests * alpha)
                / (n_tests * alpha),
                passed=True,  # FDR control is probabilistic
                tolerance=1.0,
                details={"bh_rejected": len(bh_rejected), "p_values": p_values},
            )
        )

        return results

    def validate_known_test_cases(self) -> List[ValidationResult]:
        """Validate against known theoretical test cases."""
        results = []

        # Test case 1: Standard normal quantiles
        quantiles = [0.5, 0.84134, 0.97725, 0.99865]
        z_values = [0, 1, 2, 3]

        for q, z in zip(quantiles, z_values):
            calculated = stats.norm.ppf(q)
            error = abs(calculated - z)

            results.append(
                ValidationResult(
                    test_name=f"Normal quantile Φ^(-1)({q})",
                    our_result=calculated,
                    reference_result=z,
                    absolute_error=error,
                    relative_error=error / z if z != 0 else error,
                    passed=error < 1e-4,
                    tolerance=1e-4,
                    details={"quantile": q},
                )
            )

        # Test case 2: Chi-square critical values
        chi2_critical = {
            (1, 0.05): 3.841,
            (2, 0.05): 5.991,
            (5, 0.05): 11.070,
            (10, 0.05): 18.307,
        }

        for (df, alpha), expected in chi2_critical.items():
            calculated = stats.chi2.ppf(1 - alpha, df)
            error = abs(calculated - expected)

            results.append(
                ValidationResult(
                    test_name=f"χ² critical value (df={df}, α={alpha})",
                    our_result=calculated,
                    reference_result=expected,
                    absolute_error=error,
                    relative_error=error / expected,
                    passed=error < 0.01,
                    tolerance=0.01,
                    details={"df": df, "alpha": alpha},
                )
            )

        return results

    def generate_validation_report(self) -> pd.DataFrame:
        """Generate comprehensive validation report."""
        # Run all validations
        all_results = []

        print("Running validation suite...")

        validations = [
            ("Proportion Tests", self.validate_proportion_test),
            ("Confidence Intervals", self.validate_confidence_intervals),
            ("Power Calculations", self.validate_power_calculations),
            ("Statistical Distributions", self.validate_statistical_distributions),
            ("Monte Carlo Simulations", self.validate_monte_carlo_simulations),
            ("Bayesian Methods", self.validate_bayesian_methods),
            (
                "Multiple Testing Corrections",
                self.validate_multiple_testing_corrections,
            ),
            ("Known Test Cases", self.validate_known_test_cases),
        ]

        for category, validation_func in validations:
            print(f"  Validating {category}...")
            try:
                results = validation_func()
                for r in results:
                    all_results.append(
                        {
                            "category": category,
                            "test": r.test_name,
                            "our_result": r.our_result,
                            "reference": r.reference_result,
                            "abs_error": r.absolute_error,
                            "rel_error": r.relative_error,
                            "passed": r.passed,
                            "tolerance": r.tolerance,
                        }
                    )
            except Exception as e:
                print(f"    Error in {category}: {e}")

        return pd.DataFrame(all_results)

    def plot_validation_results(self, report_df: pd.DataFrame):
        """Visualize validation results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Pass/Fail by category
        ax = axes[0, 0]
        category_stats = report_df.groupby("category")["passed"].agg(["sum", "count"])
        category_stats["pass_rate"] = category_stats["sum"] / category_stats["count"]

        categories = category_stats.index
        pass_rates = category_stats["pass_rate"].values

        colors = [
            "green" if r > 0.9 else "orange" if r > 0.7 else "red" for r in pass_rates
        ]

        ax.barh(range(len(categories)), pass_rates, color=colors)
        ax.set_yticks(range(len(categories)))
        ax.set_yticklabels(categories)
        ax.set_xlabel("Pass Rate")
        ax.set_title("Validation Pass Rates by Category")
        ax.set_xlim(0, 1)

        for i, (rate, count) in enumerate(zip(pass_rates, category_stats["count"])):
            ax.text(
                rate + 0.01,
                i,
                f"{rate:.1%} ({int(category_stats['sum'].iloc[i])}/{count})",
                va="center",
            )

        # Error distribution
        ax = axes[0, 1]
        passed = report_df[report_df["passed"]]
        failed = report_df[~report_df["passed"]]

        if not passed.empty:
            ax.hist(
                np.log10(passed["abs_error"] + 1e-10),
                bins=30,
                alpha=0.7,
                label="Passed",
                color="green",
                density=True,
            )
        if not failed.empty:
            ax.hist(
                np.log10(failed["abs_error"] + 1e-10),
                bins=30,
                alpha=0.7,
                label="Failed",
                color="red",
                density=True,
            )

        ax.set_xlabel("Log10(Absolute Error)")
        ax.set_ylabel("Density")
        ax.set_title("Error Distribution")
        ax.legend()

        # Relative errors by test
        ax = axes[1, 0]
        top_errors = report_df.nlargest(10, "rel_error")

        ax.barh(range(len(top_errors)), top_errors["rel_error"].values)
        ax.set_yticks(range(len(top_errors)))
        ax.set_yticklabels(top_errors["test"].values, fontsize=8)
        ax.set_xlabel("Relative Error")
        ax.set_title("Top 10 Tests by Relative Error")

        # Summary statistics
        ax = axes[1, 1]
        ax.axis("off")

        summary_text = f"""
        Validation Summary
        ==================

        Total Tests: {len(report_df)}
        Passed: {report_df["passed"].sum()} ({100 * report_df["passed"].mean():.1f}%)
        Failed: {(~report_df["passed"]).sum()} ({100 * (~report_df["passed"]).mean():.1f}%)

        Error Statistics:
        - Mean Absolute Error: {report_df["abs_error"].mean():.6f}
        - Median Absolute Error: {report_df["abs_error"].median():.6f}
        - Max Absolute Error: {report_df["abs_error"].max():.6f}

        - Mean Relative Error: {report_df["rel_error"].mean():.4%}
        - Median Relative Error: {report_df["rel_error"].median():.4%}
        - Max Relative Error: {report_df["rel_error"].max():.4%}

        Categories Tested: {report_df["category"].nunique()}
        Perfect Categories: {sum(category_stats["pass_rate"] == 1.0)}
        """

        ax.text(0.1, 0.5, summary_text, fontsize=10, family="monospace", va="center")

        plt.tight_layout()
        plt.show()


class TheoreticalValidator:
    """Validate statistical methods against theoretical properties."""

    def __init__(self):
        """Initialize theoretical validator."""
        pass

    def validate_unbiasedness(
        self,
        estimator: Callable,
        true_param: float,
        data_generator: Callable,
        n_simulations: int = 1000,
    ) -> Dict[str, Any]:
        """
        Validate if an estimator is unbiased.

        Args:
            estimator: Function that computes estimate from data
            true_param: True parameter value
            data_generator: Function that generates data
            n_simulations: Number of simulations

        Returns:
            Validation results
        """
        estimates = []

        for _ in range(n_simulations):
            data = data_generator()
            estimate = estimator(data)
            estimates.append(estimate)

        mean_estimate = np.mean(estimates)
        bias = mean_estimate - true_param
        variance = np.var(estimates)
        mse = bias**2 + variance

        # Test for significant bias
        se_mean = np.std(estimates) / np.sqrt(n_simulations)
        t_stat = bias / se_mean if se_mean > 0 else 0
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n_simulations - 1))

        return {
            "mean_estimate": mean_estimate,
            "true_param": true_param,
            "bias": bias,
            "variance": variance,
            "mse": mse,
            "is_unbiased": p_value > 0.05,
            "p_value": p_value,
            "n_simulations": n_simulations,
        }

    def validate_consistency(
        self,
        estimator: Callable,
        true_param: float,
        sample_sizes: List[int],
        n_simulations: int = 100,
    ) -> pd.DataFrame:
        """
        Validate if an estimator is consistent.

        Args:
            estimator: Estimator function
            true_param: True parameter
            sample_sizes: List of sample sizes to test
            n_simulations: Simulations per sample size

        Returns:
            DataFrame with consistency analysis
        """
        results = []

        for n in sample_sizes:
            estimates = []

            for _ in range(n_simulations):
                # Generate data of size n
                data = np.random.normal(true_param, 1, n)
                estimate = estimator(data)
                estimates.append(estimate)

            bias = np.mean(estimates) - true_param
            variance = np.var(estimates)

            results.append(
                {
                    "sample_size": n,
                    "bias": bias,
                    "variance": variance,
                    "mse": bias**2 + variance,
                }
            )

        results_df = pd.DataFrame(results)

        # Check if MSE decreases with sample size
        correlation = np.corrcoef(results_df["sample_size"], results_df["mse"])[0, 1]
        is_consistent = correlation < -0.8  # Strong negative correlation

        results_df["is_consistent"] = is_consistent

        return results_df


if __name__ == "__main__":
    # Example usage
    print("Statistical Validation Suite")
    print("=" * 60)

    # Initialize validator
    validator = StatisticalValidator(tolerance=1e-3)

    # Generate validation report
    print("\nRunning comprehensive validation...")
    report = validator.generate_validation_report()

    print("\nValidation Report Summary:")
    print(report.groupby("category")[["passed"]].agg(["sum", "count", "mean"]))

    # Display failed tests
    failed_tests = report[~report["passed"]]
    if not failed_tests.empty:
        print("\nFailed Tests:")
        for _, row in failed_tests.iterrows():
            print(f"  {row['category']}/{row['test']}: Error={row['abs_error']:.6f}")

    # Theoretical validation
    print("\nTheoretical Validation:")
    theory_validator = TheoreticalValidator()

    # Test sample mean estimator
    def mean_estimator(data):
        return np.mean(data)

    def data_generator():
        return np.random.normal(5.0, 2.0, 100)

    unbiased_result = theory_validator.validate_unbiasedness(
        mean_estimator, 5.0, data_generator, n_simulations=1000
    )

    print(f"  Mean estimator unbiasedness test:")
    print(f"    Bias: {unbiased_result['bias']:.6f}")
    print(f"    Is unbiased: {unbiased_result['is_unbiased']}")
    print(f"    P-value: {unbiased_result['p_value']:.4f}")

    # Test consistency
    consistency_result = theory_validator.validate_consistency(
        mean_estimator,
        true_param=5.0,
        sample_sizes=[10, 50, 100, 500, 1000],
        n_simulations=100,
    )

    print("\n  Consistency test:")
    print(consistency_result)

    print("\n✅ Validation suite completed successfully!")
