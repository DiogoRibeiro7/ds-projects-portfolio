"""
Bayesian A/B testing methods with advanced capabilities.
"""

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.special import betaln, gammaln

warnings.filterwarnings("ignore")


@dataclass
class BayesianTestResult:
    """Results from Bayesian A/B test."""

    probability_b_better: float
    expected_loss_a: float
    expected_loss_b: float
    credible_interval_a: Tuple[float, float]
    credible_interval_b: Tuple[float, float]
    effect_size: float
    effect_credible_interval: Tuple[float, float]
    rope_probability: float  # Region of Practical Equivalence
    samples_a: np.ndarray
    samples_b: np.ndarray


class BayesianABTesting:
    """Advanced Bayesian A/B testing implementation."""

    def __init__(self, prior_alpha: float = 1, prior_beta: float = 1):
        """
        Initialize Bayesian A/B testing.

        Args:
            prior_alpha: Alpha parameter for Beta prior
            prior_beta: Beta parameter for Beta prior
        """
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

    def test_conversion(
        self,
        conversions_a: int,
        visitors_a: int,
        conversions_b: int,
        visitors_b: int,
        n_samples: int = 100000,
        rope: Tuple[float, float] = (-0.01, 0.01),
    ) -> BayesianTestResult:
        """
        Perform Bayesian A/B test for conversion rates.

        Args:
            conversions_a: Number of conversions in variant A
            visitors_a: Number of visitors in variant A
            conversions_b: Number of conversions in variant B
            visitors_b: Number of visitors in variant B
            n_samples: Number of posterior samples
            rope: Region of Practical Equivalence (min_effect, max_effect)

        Returns:
            BayesianTestResult with test outcomes
        """
        # Calculate posterior parameters
        alpha_a = self.prior_alpha + conversions_a
        beta_a = self.prior_beta + visitors_a - conversions_a
        alpha_b = self.prior_alpha + conversions_b
        beta_b = self.prior_beta + visitors_b - conversions_b

        # Generate samples from posterior distributions
        samples_a = np.random.beta(alpha_a, beta_a, n_samples)
        samples_b = np.random.beta(alpha_b, beta_b, n_samples)

        # Calculate probability B is better than A
        prob_b_better = np.mean(samples_b > samples_a)

        # Calculate expected loss
        loss_a = np.maximum(samples_b - samples_a, 0).mean()
        loss_b = np.maximum(samples_a - samples_b, 0).mean()

        # Calculate credible intervals
        ci_a = np.percentile(samples_a, [2.5, 97.5])
        ci_b = np.percentile(samples_b, [2.5, 97.5])

        # Calculate effect size (relative lift)
        effect_samples = (samples_b - samples_a) / samples_a
        effect_size = np.mean(effect_samples)
        effect_ci = np.percentile(effect_samples, [2.5, 97.5])

        # Calculate ROPE probability
        diff_samples = samples_b - samples_a
        rope_prob = np.mean((diff_samples > rope[0]) & (diff_samples < rope[1]))

        return BayesianTestResult(
            probability_b_better=prob_b_better,
            expected_loss_a=loss_a,
            expected_loss_b=loss_b,
            credible_interval_a=tuple(ci_a),
            credible_interval_b=tuple(ci_b),
            effect_size=effect_size,
            effect_credible_interval=tuple(effect_ci),
            rope_probability=rope_prob,
            samples_a=samples_a,
            samples_b=samples_b,
        )

    def test_continuous(
        self, data_a: np.ndarray, data_b: np.ndarray, n_samples: int = 100000
    ) -> BayesianTestResult:
        """
        Bayesian A/B test for continuous metrics (e.g., revenue).

        Uses BEST (Bayesian Estimation Supersedes t-test) method.

        Args:
            data_a: Observations from variant A
            data_b: Observations from variant B
            n_samples: Number of posterior samples

        Returns:
            BayesianTestResult with test outcomes
        """
        from scipy import stats as st

        # Fit t-distributions to data (more robust than normal)
        # Using method of moments for initial estimates
        mean_a, std_a = np.mean(data_a), np.std(data_a)
        mean_b, std_b = np.mean(data_b), np.std(data_b)

        # Generate posterior samples using MCMC-like approach
        # Simplified version - in production use PyMC3 or Stan
        samples_mean_a = st.t.rvs(
            df=len(data_a) - 1,
            loc=mean_a,
            scale=std_a / np.sqrt(len(data_a)),
            size=n_samples,
        )
        samples_mean_b = st.t.rvs(
            df=len(data_b) - 1,
            loc=mean_b,
            scale=std_b / np.sqrt(len(data_b)),
            size=n_samples,
        )

        # Calculate metrics
        prob_b_better = np.mean(samples_mean_b > samples_mean_a)

        loss_a = np.maximum(samples_mean_b - samples_mean_a, 0).mean()
        loss_b = np.maximum(samples_mean_a - samples_mean_b, 0).mean()

        ci_a = np.percentile(samples_mean_a, [2.5, 97.5])
        ci_b = np.percentile(samples_mean_b, [2.5, 97.5])

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)
        effect_samples = (samples_mean_b - samples_mean_a) / pooled_std
        effect_size = np.mean(effect_samples)
        effect_ci = np.percentile(effect_samples, [2.5, 97.5])

        # ROPE for standardized effect
        rope = (-0.1, 0.1)  # Small effect threshold
        diff_samples = (samples_mean_b - samples_mean_a) / pooled_std
        rope_prob = np.mean((diff_samples > rope[0]) & (diff_samples < rope[1]))

        return BayesianTestResult(
            probability_b_better=prob_b_better,
            expected_loss_a=loss_a,
            expected_loss_b=loss_b,
            credible_interval_a=tuple(ci_a),
            credible_interval_b=tuple(ci_b),
            effect_size=effect_size,
            effect_credible_interval=tuple(effect_ci),
            rope_probability=rope_prob,
            samples_a=samples_mean_a,
            samples_b=samples_mean_b,
        )

    def sequential_test(
        self,
        conversions: List[Tuple[int, int]],
        visitors: List[Tuple[int, int]],
        threshold: float = 0.95,
    ) -> List[Dict[str, Any]]:
        """
        Sequential Bayesian testing with early stopping.

        Args:
            conversions: List of (conversions_a, conversions_b) over time
            visitors: List of (visitors_a, visitors_b) over time
            threshold: Probability threshold for early stopping

        Returns:
            List of test results at each time point
        """
        results = []
        cumul_conv_a, cumul_conv_b = 0, 0
        cumul_vis_a, cumul_vis_b = 0, 0

        for (conv_a, conv_b), (vis_a, vis_b) in zip(conversions, visitors):
            cumul_conv_a += conv_a
            cumul_conv_b += conv_b
            cumul_vis_a += vis_a
            cumul_vis_b += vis_b

            if cumul_vis_a > 0 and cumul_vis_b > 0:
                result = self.test_conversion(
                    cumul_conv_a,
                    cumul_vis_a,
                    cumul_conv_b,
                    cumul_vis_b,
                    n_samples=10000,  # Fewer samples for speed
                )

                results.append(
                    {
                        "time": len(results),
                        "prob_b_better": result.probability_b_better,
                        "should_stop": result.probability_b_better > threshold
                        or result.probability_b_better < (1 - threshold),
                        "cumulative_visitors": cumul_vis_a + cumul_vis_b,
                    }
                )

                if results[-1]["should_stop"]:
                    break

        return results

    def plot_posterior(
        self, result: BayesianTestResult, metric_name: str = "Conversion Rate"
    ):
        """
        Plot posterior distributions and difference.

        Args:
            result: BayesianTestResult from test
            metric_name: Name of the metric for labels
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot posterior distributions
        axes[0].hist(
            result.samples_a, bins=50, alpha=0.7, label="Variant A", density=True
        )
        axes[0].hist(
            result.samples_b, bins=50, alpha=0.7, label="Variant B", density=True
        )
        axes[0].axvline(
            np.mean(result.samples_a), color="blue", linestyle="--", alpha=0.8
        )
        axes[0].axvline(
            np.mean(result.samples_b), color="orange", linestyle="--", alpha=0.8
        )
        axes[0].set_xlabel(metric_name)
        axes[0].set_ylabel("Density")
        axes[0].set_title("Posterior Distributions")
        axes[0].legend()

        # Plot difference distribution
        diff = result.samples_b - result.samples_a
        axes[1].hist(diff, bins=50, alpha=0.7, color="green", density=True)
        axes[1].axvline(
            0, color="red", linestyle="--", alpha=0.8, label="No difference"
        )
        axes[1].axvline(
            np.mean(diff),
            color="green",
            linestyle="--",
            alpha=0.8,
            label="Mean difference",
        )
        axes[1].set_xlabel(f"Difference in {metric_name}")
        axes[1].set_ylabel("Density")
        axes[1].set_title(
            f"Posterior of Difference\nP(B > A) = {result.probability_b_better:.3f}"
        )
        axes[1].legend()

        # Plot cumulative probabilities
        sorted_diff = np.sort(diff)
        cumprob = np.arange(1, len(sorted_diff) + 1) / len(sorted_diff)
        axes[2].plot(sorted_diff, cumprob)
        axes[2].axvline(0, color="red", linestyle="--", alpha=0.8)
        axes[2].axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        axes[2].fill_between(
            sorted_diff,
            0,
            cumprob,
            where=(sorted_diff > 0),
            alpha=0.3,
            color="green",
            label="B better",
        )
        axes[2].fill_between(
            sorted_diff,
            0,
            cumprob,
            where=(sorted_diff <= 0),
            alpha=0.3,
            color="red",
            label="A better",
        )
        axes[2].set_xlabel(f"Difference in {metric_name}")
        axes[2].set_ylabel("Cumulative Probability")
        axes[2].set_title("Cumulative Distribution of Difference")
        axes[2].legend()

        plt.tight_layout()
        plt.show()


class BayesianMultivariateTest:
    """Bayesian testing for multiple metrics simultaneously."""

    def __init__(self):
        """Initialize multivariate Bayesian testing."""
        pass

    def test_multiple_metrics(
        self,
        metrics_a: Dict[str, np.ndarray],
        metrics_b: Dict[str, np.ndarray],
        weights: Optional[Dict[str, float]] = None,
        n_samples: int = 100000,
    ) -> Dict[str, Any]:
        """
        Test multiple metrics with optional weighting.

        Args:
            metrics_a: Dictionary of metric arrays for variant A
            metrics_b: Dictionary of metric arrays for variant B
            weights: Optional weights for each metric
            n_samples: Number of posterior samples

        Returns:
            Combined test results
        """
        if weights is None:
            weights = {k: 1.0 / len(metrics_a) for k in metrics_a.keys()}

        results = {}
        combined_score_a = np.zeros(n_samples)
        combined_score_b = np.zeros(n_samples)

        for metric_name in metrics_a.keys():
            # Test individual metric
            data_a = metrics_a[metric_name]
            data_b = metrics_b[metric_name]

            # Standardize and sample
            mean_a, std_a = np.mean(data_a), np.std(data_a)
            mean_b, std_b = np.mean(data_b), np.std(data_b)

            samples_a = np.random.normal(
                mean_a, std_a / np.sqrt(len(data_a)), n_samples
            )
            samples_b = np.random.normal(
                mean_b, std_b / np.sqrt(len(data_b)), n_samples
            )

            # Standardize for combination
            pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)
            if pooled_std > 0:
                std_samples_a = samples_a / pooled_std
                std_samples_b = samples_b / pooled_std
            else:
                std_samples_a = samples_a
                std_samples_b = samples_b

            # Add to combined score
            weight = weights.get(metric_name, 1.0)
            combined_score_a += weight * std_samples_a
            combined_score_b += weight * std_samples_b

            # Store individual results
            results[metric_name] = {
                "prob_b_better": np.mean(samples_b > samples_a),
                "mean_a": mean_a,
                "mean_b": mean_b,
                "ci_a": tuple(np.percentile(samples_a, [2.5, 97.5])),
                "ci_b": tuple(np.percentile(samples_b, [2.5, 97.5])),
            }

        # Overall comparison
        results["overall"] = {
            "prob_b_better": np.mean(combined_score_b > combined_score_a),
            "expected_gain": np.mean(combined_score_b - combined_score_a),
            "ci_gain": tuple(
                np.percentile(combined_score_b - combined_score_a, [2.5, 97.5])
            ),
        }

        return results


class HierarchicalBayesianTest:
    """Hierarchical Bayesian models for testing with multiple segments."""

    def __init__(self):
        """Initialize hierarchical Bayesian testing."""
        pass

    def test_with_segments(
        self,
        data: pd.DataFrame,
        segment_col: str,
        metric_col: str,
        treatment_col: str,
        n_samples: int = 10000,
    ) -> Dict[str, Any]:
        """
        Hierarchical Bayesian test accounting for segment effects.

        Args:
            data: DataFrame with test data
            segment_col: Column identifying segments
            metric_col: Metric column to test
            treatment_col: Treatment indicator column
            n_samples: Number of posterior samples

        Returns:
            Test results by segment and overall
        """
        segments = data[segment_col].unique()
        results = {"segments": {}, "overall": {}}

        # Hyperpriors for population-level effects
        pop_mean_prior = np.mean(data[metric_col])
        pop_std_prior = np.std(data[metric_col])

        all_effects = []

        for segment in segments:
            segment_data = data[data[segment_col] == segment]

            control = segment_data[segment_data[treatment_col] == 0][metric_col].values
            treatment = segment_data[segment_data[treatment_col] == 1][
                metric_col
            ].values

            if len(control) > 0 and len(treatment) > 0:
                # Segment-specific posterior
                mean_c, std_c = np.mean(control), np.std(control)
                mean_t, std_t = np.mean(treatment), np.std(treatment)

                # Sample from posterior with shrinkage toward population mean
                shrinkage = min(len(control), len(treatment)) / (
                    min(len(control), len(treatment)) + 10
                )

                posterior_mean_c = shrinkage * mean_c + (1 - shrinkage) * pop_mean_prior
                posterior_mean_t = shrinkage * mean_t + (1 - shrinkage) * pop_mean_prior

                samples_c = np.random.normal(
                    posterior_mean_c, std_c / np.sqrt(len(control)), n_samples
                )
                samples_t = np.random.normal(
                    posterior_mean_t, std_t / np.sqrt(len(treatment)), n_samples
                )

                effect_samples = samples_t - samples_c
                all_effects.extend(effect_samples)

                results["segments"][segment] = {
                    "effect_mean": np.mean(effect_samples),
                    "effect_ci": tuple(np.percentile(effect_samples, [2.5, 97.5])),
                    "prob_positive": np.mean(effect_samples > 0),
                    "sample_size": len(control) + len(treatment),
                }

        # Population-level effect
        if all_effects:
            results["overall"] = {
                "effect_mean": np.mean(all_effects),
                "effect_ci": tuple(np.percentile(all_effects, [2.5, 97.5])),
                "prob_positive": np.mean(np.array(all_effects) > 0),
                "heterogeneity": np.std(
                    [r["effect_mean"] for r in results["segments"].values()]
                ),
            }

        return results


# Utility functions for Bayesian testing
def calculate_sample_size_bayesian(
    baseline_rate: float,
    minimum_effect: float,
    confidence: float = 0.95,
    power: float = 0.8,
) -> int:
    """
    Calculate required sample size for Bayesian A/B test.

    Args:
        baseline_rate: Baseline conversion rate
        minimum_effect: Minimum detectable effect (relative)
        confidence: Desired confidence level
        power: Desired statistical power

    Returns:
        Required sample size per variant
    """
    # Use approximation based on frequentist formula
    # adjusted for Bayesian context
    from scipy.stats import norm

    p1 = baseline_rate
    p2 = baseline_rate * (1 + minimum_effect)

    pooled_p = (p1 + p2) / 2
    z_alpha = norm.ppf(confidence)
    z_beta = norm.ppf(power)

    n = ((z_alpha + z_beta) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2))) / (p1 - p2) ** 2

    # Adjust for Bayesian (typically needs ~80% of frequentist sample)
    n_bayesian = int(n * 0.8)

    return n_bayesian


def simulate_bayesian_test(
    true_rate_a: float, true_rate_b: float, sample_size: int, n_simulations: int = 1000
) -> Dict[str, float]:
    """
    Simulate Bayesian A/B test to validate performance.

    Args:
        true_rate_a: True conversion rate for A
        true_rate_b: True conversion rate for B
        sample_size: Sample size per variant
        n_simulations: Number of simulations

    Returns:
        Dictionary with simulation results
    """
    tester = BayesianABTesting()

    correct_decisions = 0
    false_positives = 0
    false_negatives = 0

    for _ in range(n_simulations):
        # Generate data
        conversions_a = np.random.binomial(sample_size, true_rate_a)
        conversions_b = np.random.binomial(sample_size, true_rate_b)

        # Run test
        result = tester.test_conversion(
            conversions_a, sample_size, conversions_b, sample_size, n_samples=10000
        )

        # Check decision
        decision_b_better = result.probability_b_better > 0.95
        truth_b_better = true_rate_b > true_rate_a

        if decision_b_better == truth_b_better:
            correct_decisions += 1
        elif decision_b_better and not truth_b_better:
            false_positives += 1
        elif not decision_b_better and truth_b_better:
            false_negatives += 1

    return {
        "accuracy": correct_decisions / n_simulations,
        "false_positive_rate": false_positives / n_simulations,
        "false_negative_rate": false_negatives / n_simulations,
        "power": (
            1 - false_negatives / n_simulations if true_rate_b != true_rate_a else None
        ),
    }


if __name__ == "__main__":
    # Example usage
    print("Bayesian A/B Testing Module")
    print("=" * 60)

    # Create tester
    tester = BayesianABTesting()

    # Test conversion rates
    result = tester.test_conversion(
        conversions_a=120, visitors_a=1000, conversions_b=150, visitors_b=1000
    )

    print(f"\nConversion Test Results:")
    print(f"Probability B is better: {result.probability_b_better:.3f}")
    print(f"Expected loss if choosing A: {result.expected_loss_a:.4f}")
    print(f"Expected loss if choosing B: {result.expected_loss_b:.4f}")
    print(
        f"Effect size: {result.effect_size:.3f} [{result.effect_credible_interval[0]:.3f}, {result.effect_credible_interval[1]:.3f}]"
    )
    print(f"ROPE probability: {result.rope_probability:.3f}")

    # Validate with simulation
    print("\nValidation Simulation:")
    validation = simulate_bayesian_test(0.12, 0.15, 1000, n_simulations=100)
    print(f"Accuracy: {validation['accuracy']:.3f}")
    print(f"Power: {validation['power']:.3f}" if validation["power"] else "Power: N/A")
