"""
Advanced power analysis and simulation tools for experiment design.

This module provides comprehensive power analysis simulations for various
experimental designs, including adaptive designs, sequential testing,
and complex effect structures.
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from scipy import stats
from scipy.optimize import brentq, minimize_scalar
from statsmodels.stats.power import NormalIndPower, TTestPower

warnings.filterwarnings("ignore")


@dataclass
class PowerAnalysisResult:
    """Container for power analysis results."""

    power: float
    sample_size: int
    effect_size: float
    alpha: float
    minimum_detectable_effect: float
    confidence_interval: Tuple[float, float]
    simulation_details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class SimulationResult:
    """Container for simulation results."""

    true_effect: float
    observed_power: float
    type_i_error: float
    type_ii_error: float
    bias: float
    mean_squared_error: float
    coverage: float
    average_ci_width: float
    convergence_time: Optional[int] = None
    early_stopping_rate: Optional[float] = None


class PowerAnalysisSimulator:
    """Advanced power analysis through simulation."""

    def __init__(self, n_simulations: int = 10000, n_jobs: int = -1):
        """
        Initialize power analysis simulator.

        Args:
            n_simulations: Number of simulations to run
            n_jobs: Number of parallel jobs (-1 for all CPUs)
        """
        self.n_simulations = n_simulations
        self.n_jobs = n_jobs

    def simulate_ab_test_power(
        self,
        baseline_rate: float,
        effect_sizes: Union[float, List[float]],
        sample_sizes: Union[int, List[int]],
        alpha: float = 0.05,
        test_type: str = "proportion",
        alternative: str = "two-sided",
        variance_ratio: float = 1.0,
    ) -> pd.DataFrame:
        """
        Simulate power for A/B tests across effect sizes and sample sizes.

        Args:
            baseline_rate: Baseline conversion/mean
            effect_sizes: Effect size(s) to test
            sample_sizes: Sample size(s) per group to test
            alpha: Significance level
            test_type: 'proportion' or 'continuous'
            alternative: 'two-sided', 'greater', or 'less'
            variance_ratio: Ratio of variances for continuous outcomes

        Returns:
            DataFrame with power analysis results
        """
        if isinstance(effect_sizes, (int, float)):
            effect_sizes = [effect_sizes]
        if isinstance(sample_sizes, int):
            sample_sizes = [sample_sizes]

        results = []

        for effect_size in effect_sizes:
            for sample_size in sample_sizes:
                if test_type == "proportion":
                    power = self._simulate_proportion_test(
                        baseline_rate, effect_size, sample_size, alpha, alternative
                    )
                else:
                    power = self._simulate_continuous_test(
                        baseline_rate,
                        effect_size,
                        sample_size,
                        alpha,
                        alternative,
                        variance_ratio,
                    )

                results.append(
                    {
                        "effect_size": effect_size,
                        "sample_size": sample_size,
                        "power": power,
                        "alpha": alpha,
                        "test_type": test_type,
                        "alternative": alternative,
                    }
                )

        return pd.DataFrame(results)

    def _simulate_proportion_test(
        self, p1: float, effect_size: float, n: int, alpha: float, alternative: str
    ) -> float:
        """Simulate power for proportion test."""
        p2 = p1 + effect_size

        # Clip to valid range
        p2 = np.clip(p2, 0.001, 0.999)

        significant_tests = 0

        for _ in range(self.n_simulations):
            # Generate data
            x1 = np.random.binomial(n, p1)
            x2 = np.random.binomial(n, p2)

            # Perform test
            p1_hat = x1 / n
            p2_hat = x2 / n
            p_pooled = (x1 + x2) / (2 * n)

            se = np.sqrt(p_pooled * (1 - p_pooled) * (2 / n))
            z = (p2_hat - p1_hat) / se if se > 0 else 0

            # Determine significance
            if alternative == "two-sided":
                p_value = 2 * (1 - stats.norm.cdf(abs(z)))
            elif alternative == "greater":
                p_value = 1 - stats.norm.cdf(z)
            else:  # less
                p_value = stats.norm.cdf(z)

            if p_value < alpha:
                significant_tests += 1

        return significant_tests / self.n_simulations

    def _simulate_continuous_test(
        self,
        mean1: float,
        effect_size: float,
        n: int,
        alpha: float,
        alternative: str,
        variance_ratio: float,
    ) -> float:
        """Simulate power for continuous test."""
        mean2 = mean1 + effect_size
        std1 = 1.0  # Standardized
        std2 = std1 * np.sqrt(variance_ratio)

        significant_tests = 0

        for _ in range(self.n_simulations):
            # Generate data
            data1 = np.random.normal(mean1, std1, n)
            data2 = np.random.normal(mean2, std2, n)

            # Perform t-test
            _, p_value = stats.ttest_ind(data1, data2, alternative=alternative)

            if p_value < alpha:
                significant_tests += 1

        return significant_tests / self.n_simulations

    def simulate_sequential_testing(
        self,
        true_effect: float,
        max_sample_size: int,
        check_points: List[int],
        alpha: float = 0.05,
        spending_function: str = "obrien_fleming",
        futility_bounds: bool = True,
    ) -> SimulationResult:
        """
        Simulate sequential testing with alpha spending.

        Args:
            true_effect: True treatment effect
            max_sample_size: Maximum sample size per group
            check_points: Sample sizes at which to check
            alpha: Overall significance level
            spending_function: 'obrien_fleming' or 'pocock'
            futility_bounds: Whether to include futility stopping

        Returns:
            SimulationResult with sequential testing outcomes
        """
        n_checks = len(check_points)
        alpha_spent = self._calculate_alpha_spending(
            check_points, max_sample_size, alpha, spending_function
        )

        decisions = []
        stopping_times = []

        for sim in range(self.n_simulations):
            stopped = False
            decision = None
            stopping_n = max_sample_size

            # Generate all data upfront
            if true_effect > 0:
                all_data_control = np.random.binomial(1, 0.1, max_sample_size)
                all_data_treatment = np.random.binomial(
                    1, 0.1 + true_effect, max_sample_size
                )
            else:
                all_data_control = np.random.binomial(1, 0.1, max_sample_size)
                all_data_treatment = np.random.binomial(1, 0.1, max_sample_size)

            for i, n in enumerate(check_points):
                if stopped:
                    break

                # Get data up to current checkpoint
                data_c = all_data_control[:n]
                data_t = all_data_treatment[:n]

                # Calculate test statistic
                p_c = data_c.mean()
                p_t = data_t.mean()
                p_pooled = (data_c.sum() + data_t.sum()) / (2 * n)

                se = (
                    np.sqrt(p_pooled * (1 - p_pooled) * 2 / n)
                    if p_pooled > 0
                    else 1e-10
                )
                z = (p_t - p_c) / se

                # Check efficacy boundary
                z_boundary = stats.norm.ppf(1 - alpha_spent[i])
                if abs(z) > z_boundary:
                    stopped = True
                    decision = "reject"
                    stopping_n = n

                # Check futility boundary
                if futility_bounds and i < n_checks - 1:
                    conditional_power = self._calculate_conditional_power(
                        z, n, max_sample_size, true_effect
                    )
                    if conditional_power < 0.2:  # Futility threshold
                        stopped = True
                        decision = "futile"
                        stopping_n = n

            if decision is None:
                decision = "fail_to_reject"

            decisions.append(decision)
            stopping_times.append(stopping_n)

        # Calculate metrics
        power = sum(d == "reject" for d in decisions) / len(decisions)
        type_i = (
            sum(d == "reject" for d in decisions) / len(decisions)
            if true_effect == 0
            else None
        )
        type_ii = 1 - power if true_effect > 0 else None
        early_stop_rate = sum(s < max_sample_size for s in stopping_times) / len(
            stopping_times
        )
        avg_stopping_time = np.mean(stopping_times)

        return SimulationResult(
            true_effect=true_effect,
            observed_power=power,
            type_i_error=type_i if type_i else 0,
            type_ii_error=type_ii if type_ii else 0,
            bias=0,  # Not calculated for sequential
            mean_squared_error=0,  # Not calculated for sequential
            coverage=0,  # Not calculated for sequential
            average_ci_width=0,  # Not calculated for sequential
            convergence_time=int(avg_stopping_time),
            early_stopping_rate=early_stop_rate,
        )

    def _calculate_alpha_spending(
        self, check_points: List[int], max_n: int, alpha: float, method: str
    ) -> List[float]:
        """Calculate alpha spending at each checkpoint."""
        info_fractions = [n / max_n for n in check_points]
        alpha_spent = []

        for i, t in enumerate(info_fractions):
            if method == "obrien_fleming":
                # O'Brien-Fleming spending function
                if i == 0:
                    spent = 2 * (
                        1 - stats.norm.cdf(stats.norm.ppf(1 - alpha / 2) / np.sqrt(t))
                    )
                else:
                    prev_t = info_fractions[i - 1]
                    spent = 2 * (
                        1 - stats.norm.cdf(stats.norm.ppf(1 - alpha / 2) / np.sqrt(t))
                    )
                    spent -= 2 * (
                        1
                        - stats.norm.cdf(
                            stats.norm.ppf(1 - alpha / 2) / np.sqrt(prev_t)
                        )
                    )
            else:  # Pocock
                spent = alpha * t

            alpha_spent.append(spent)

        return alpha_spent

    def _calculate_conditional_power(
        self, current_z: float, current_n: int, max_n: int, assumed_effect: float
    ) -> float:
        """Calculate conditional power for futility assessment."""
        remaining_n = max_n - current_n
        if remaining_n <= 0:
            return 0

        # Information fraction
        info_current = current_n / max_n
        info_remaining = remaining_n / max_n

        # Expected future z-score under assumed effect
        future_z_mean = assumed_effect * np.sqrt(remaining_n)

        # Combined z-score
        combined_z_mean = current_z * np.sqrt(info_current) + future_z_mean * np.sqrt(
            info_remaining
        )

        # Critical value at final analysis
        z_critical = stats.norm.ppf(0.975)  # Two-sided test

        # Conditional power
        cond_power = 1 - stats.norm.cdf(z_critical - combined_z_mean)

        return cond_power

    def simulate_adaptive_design(
        self,
        initial_sample: int,
        max_sample: int,
        true_effects: Dict[str, float],
        adaptation_rule: str = "thompson_sampling",
        n_arms: int = 3,
    ) -> Dict[str, Any]:
        """
        Simulate adaptive randomization design.

        Args:
            initial_sample: Initial equal randomization phase
            max_sample: Maximum total sample size
            true_effects: True effects for each arm
            adaptation_rule: 'thompson_sampling' or 'response_adaptive'
            n_arms: Number of treatment arms

        Returns:
            Simulation results for adaptive design
        """
        arm_names = list(true_effects.keys())[:n_arms]
        results = []

        for sim in range(min(self.n_simulations, 1000)):  # Limit for speed
            # Initialize counts
            arm_counts = {arm: 0 for arm in arm_names}
            arm_successes = {arm: 0 for arm in arm_names}

            # Initial equal randomization
            for _ in range(initial_sample):
                arm = np.random.choice(arm_names)
                success = np.random.binomial(1, true_effects[arm])
                arm_counts[arm] += 1
                arm_successes[arm] += success

            # Adaptive phase
            for _ in range(initial_sample, max_sample):
                if adaptation_rule == "thompson_sampling":
                    # Thompson sampling for arm selection
                    samples = {}
                    for arm in arm_names:
                        alpha = 1 + arm_successes[arm]
                        beta = 1 + arm_counts[arm] - arm_successes[arm]
                        samples[arm] = np.random.beta(alpha, beta)

                    selected_arm = max(samples, key=samples.get)

                else:  # response_adaptive
                    # Response-adaptive randomization (Play-the-Winner)
                    probs = []
                    for arm in arm_names:
                        if arm_counts[arm] > 0:
                            success_rate = arm_successes[arm] / arm_counts[arm]
                        else:
                            success_rate = 0.5
                        probs.append(success_rate)

                    probs = np.array(probs)
                    probs = probs / probs.sum()
                    selected_arm = np.random.choice(arm_names, p=probs)

                # Generate outcome
                success = np.random.binomial(1, true_effects[selected_arm])
                arm_counts[selected_arm] += 1
                arm_successes[selected_arm] += success

            # Calculate final estimates
            estimates = {}
            for arm in arm_names:
                if arm_counts[arm] > 0:
                    estimates[arm] = arm_successes[arm] / arm_counts[arm]
                else:
                    estimates[arm] = 0

            # Determine best arm
            best_arm_true = max(true_effects, key=true_effects.get)
            best_arm_selected = max(estimates, key=estimates.get)

            results.append(
                {
                    "correct_selection": best_arm_selected == best_arm_true,
                    "allocation_proportions": {
                        arm: arm_counts[arm] / max_sample for arm in arm_names
                    },
                    "estimates": estimates,
                    "sample_sizes": arm_counts,
                }
            )

        # Aggregate results
        correct_selection_rate = np.mean([r["correct_selection"] for r in results])
        avg_allocations = {
            arm: np.mean([r["allocation_proportions"][arm] for r in results])
            for arm in arm_names
        }

        return {
            "correct_selection_probability": correct_selection_rate,
            "average_allocations": avg_allocations,
            "adaptation_rule": adaptation_rule,
            "initial_sample": initial_sample,
            "max_sample": max_sample,
            "true_effects": true_effects,
            "simulation_details": {
                "n_simulations": len(results),
                "convergence": self._check_convergence(results),
            },
        }

    def _check_convergence(self, results: List[Dict]) -> Dict[str, Any]:
        """Check if adaptive design has converged."""
        if len(results) < 100:
            return {"converged": False, "reason": "Too few simulations"}

        # Check if correct selection rate has stabilized
        selection_rates = [r["correct_selection"] for r in results]
        window_size = min(50, len(results) // 4)

        recent_rate = np.mean(selection_rates[-window_size:])
        previous_rate = np.mean(selection_rates[-2 * window_size : -window_size])

        rate_change = abs(recent_rate - previous_rate)

        converged = rate_change < 0.01

        return {
            "converged": converged,
            "recent_rate": recent_rate,
            "rate_change": rate_change,
        }

    def calculate_sample_size(
        self,
        power: float,
        effect_size: float,
        alpha: float = 0.05,
        test_type: str = "proportion",
        baseline: float = 0.1,
        method: str = "analytical",
    ) -> PowerAnalysisResult:
        """
        Calculate required sample size for desired power.

        Args:
            power: Desired statistical power
            effect_size: Effect size to detect
            alpha: Significance level
            test_type: 'proportion' or 'continuous'
            baseline: Baseline rate/mean
            method: 'analytical' or 'simulation'

        Returns:
            PowerAnalysisResult with sample size calculation
        """
        if method == "analytical":
            if test_type == "proportion":
                n = self._analytical_sample_size_proportion(
                    baseline, effect_size, alpha, power
                )
            else:
                n = self._analytical_sample_size_continuous(effect_size, alpha, power)
        else:
            # Simulation-based sample size calculation
            n = self._simulation_sample_size(
                power, effect_size, alpha, test_type, baseline
            )

        # Calculate MDE at this sample size
        mde = self.calculate_mde(n, power, alpha, test_type, baseline)

        # Confidence interval for power
        se_power = np.sqrt(power * (1 - power) / self.n_simulations)
        ci_power = (power - 1.96 * se_power, power + 1.96 * se_power)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            n, power, effect_size, test_type
        )

        return PowerAnalysisResult(
            power=power,
            sample_size=n,
            effect_size=effect_size,
            alpha=alpha,
            minimum_detectable_effect=mde,
            confidence_interval=ci_power,
            simulation_details={
                "method": method,
                "test_type": test_type,
                "baseline": baseline,
            },
            recommendations=recommendations,
        )

    def _analytical_sample_size_proportion(
        self, p1: float, effect_size: float, alpha: float, power: float
    ) -> int:
        """Analytical sample size for proportion test."""
        p2 = p1 + effect_size
        p2 = np.clip(p2, 0.001, 0.999)

        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)

        p_bar = (p1 + p2) / 2
        n = ((z_alpha + z_beta) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2))) / (p1 - p2) ** 2

        return int(np.ceil(n))

    def _analytical_sample_size_continuous(
        self, effect_size: float, alpha: float, power: float
    ) -> int:
        """Analytical sample size for continuous test."""
        power_analysis = TTestPower()
        n = power_analysis.solve_power(
            effect_size=effect_size, power=power, alpha=alpha, alternative="two-sided"
        )
        return int(np.ceil(n))

    def _simulation_sample_size(
        self,
        target_power: float,
        effect_size: float,
        alpha: float,
        test_type: str,
        baseline: float,
    ) -> int:
        """Find sample size through binary search with simulation."""

        def power_at_n(n):
            df = self.simulate_ab_test_power(
                baseline, effect_size, int(n), alpha, test_type
            )
            return df["power"].iloc[0]

        # Binary search
        n_min, n_max = 10, 100000

        while n_max - n_min > 1:
            n_mid = (n_min + n_max) // 2
            power_mid = power_at_n(n_mid)

            if power_mid < target_power:
                n_min = n_mid
            else:
                n_max = n_mid

        return n_max

    def calculate_mde(
        self,
        sample_size: int,
        power: float,
        alpha: float = 0.05,
        test_type: str = "proportion",
        baseline: float = 0.1,
    ) -> float:
        """
        Calculate minimum detectable effect given sample size.

        Args:
            sample_size: Sample size per group
            power: Desired power
            alpha: Significance level
            test_type: 'proportion' or 'continuous'
            baseline: Baseline rate/mean

        Returns:
            Minimum detectable effect
        """
        if test_type == "proportion":
            z_alpha = stats.norm.ppf(1 - alpha / 2)
            z_beta = stats.norm.ppf(power)

            # Solve for effect size
            def mde_equation(delta):
                p2 = baseline + delta
                if p2 <= 0 or p2 >= 1:
                    return 1e10

                p_bar = (baseline + p2) / 2
                se = np.sqrt(2 * p_bar * (1 - p_bar) / sample_size)
                return abs((z_alpha + z_beta) * se - abs(delta))

            try:
                mde = minimize_scalar(
                    mde_equation, bounds=(0.001, 1 - baseline), method="bounded"
                ).x
            except:
                mde = 0.1  # Default if optimization fails

        else:
            power_analysis = TTestPower()
            mde = power_analysis.solve_power(
                nobs=sample_size, power=power, alpha=alpha, alternative="two-sided"
            )

        return mde

    def _generate_recommendations(
        self, sample_size: int, power: float, effect_size: float, test_type: str
    ) -> List[str]:
        """Generate recommendations based on power analysis."""
        recommendations = []

        # Sample size recommendations
        if sample_size > 10000:
            recommendations.append(
                f"Large sample size required (n={sample_size}). Consider:"
            )
            recommendations.append("  - Using sequential testing to stop early")
            recommendations.append("  - Increasing acceptable effect size")
            recommendations.append("  - Using more efficient test designs")

        if sample_size < 100:
            recommendations.append(f"Small sample size (n={sample_size}) may lead to:")
            recommendations.append("  - High variance in estimates")
            recommendations.append("  - Sensitivity to outliers")
            recommendations.append("  - Consider exact tests for small samples")

        # Power recommendations
        if power < 0.8:
            recommendations.append(
                f"Power is below 80% ({power:.1%}). Risk of Type II error."
            )
            recommendations.append("  - Consider increasing sample size")
            recommendations.append("  - Or accept larger effect sizes")

        # Effect size recommendations
        if test_type == "proportion" and effect_size < 0.01:
            recommendations.append(
                "Very small effect size (<1%). Consider practical significance."
            )

        # General recommendations
        recommendations.append("\nConsider:")
        recommendations.append("  - Running A/A test to validate setup")
        recommendations.append("  - Implementing sample ratio mismatch checks")
        recommendations.append("  - Planning for multiple testing if needed")

        return recommendations

    def plot_power_curves(
        self, results_df: pd.DataFrame, variable: str = "sample_size"
    ):
        """
        Plot power curves from simulation results.

        Args:
            results_df: DataFrame from simulate_ab_test_power
            variable: 'sample_size' or 'effect_size' for x-axis
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Power curve
        ax = axes[0, 0]
        if variable == "sample_size":
            for effect in results_df["effect_size"].unique():
                data = results_df[results_df["effect_size"] == effect]
                ax.plot(
                    data["sample_size"],
                    data["power"],
                    marker="o",
                    label=f"Effect={effect}",
                )
            ax.set_xlabel("Sample Size per Group")
        else:
            for n in results_df["sample_size"].unique():
                data = results_df[results_df["sample_size"] == n]
                ax.plot(data["effect_size"], data["power"], marker="o", label=f"n={n}")
            ax.set_xlabel("Effect Size")

        ax.set_ylabel("Statistical Power")
        ax.set_title("Power Curves")
        ax.axhline(0.8, color="red", linestyle="--", alpha=0.5, label="80% power")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Sample size requirements
        ax = axes[0, 1]
        effect_sizes = results_df["effect_size"].unique()
        required_n = []

        for effect in effect_sizes:
            data = results_df[results_df["effect_size"] == effect]
            # Find minimum n for 80% power
            high_power = data[data["power"] >= 0.8]
            if not high_power.empty:
                required_n.append(high_power["sample_size"].min())
            else:
                required_n.append(np.nan)

        ax.bar(range(len(effect_sizes)), required_n)
        ax.set_xticks(range(len(effect_sizes)))
        ax.set_xticklabels([f"{e:.3f}" for e in effect_sizes], rotation=45)
        ax.set_xlabel("Effect Size")
        ax.set_ylabel("Required Sample Size (80% power)")
        ax.set_title("Sample Size Requirements")
        ax.grid(True, alpha=0.3, axis="y")

        # MDE curve
        ax = axes[1, 0]
        sample_sizes = results_df["sample_size"].unique()
        mdes = []

        for n in sample_sizes:
            mde = self.calculate_mde(n, 0.8, 0.05, results_df["test_type"].iloc[0], 0.1)
            mdes.append(mde)

        ax.plot(sample_sizes, mdes, "g-", marker="s")
        ax.set_xlabel("Sample Size per Group")
        ax.set_ylabel("Minimum Detectable Effect")
        ax.set_title("MDE vs Sample Size")
        ax.grid(True, alpha=0.3)

        # Summary statistics
        ax = axes[1, 1]
        ax.axis("off")

        summary_text = f"""
        Power Analysis Summary
        =====================

        Test Type: {results_df["test_type"].iloc[0]}
        Alpha: {results_df["alpha"].iloc[0]}
        Alternative: {results_df["alternative"].iloc[0]}

        Effect Sizes Tested: {len(results_df["effect_size"].unique())}
        Sample Sizes Tested: {len(results_df["sample_size"].unique())}

        Key Findings:
        - Min sample for 80% power: {min(n for n in required_n if not np.isnan(n)):.0f}
        - Max power achieved: {results_df["power"].max():.3f}
        - Min MDE at n=1000: {self.calculate_mde(1000, 0.8, 0.05, results_df["test_type"].iloc[0], 0.1):.4f}
        """

        ax.text(0.1, 0.5, summary_text, fontsize=10, family="monospace", va="center")

        plt.tight_layout()
        plt.show()


class SensitivityAnalysis:
    """Sensitivity analysis for experimental results."""

    def __init__(self):
        """Initialize sensitivity analysis."""
        pass

    def analyze_assumption_sensitivity(
        self,
        base_effect: float,
        base_se: float,
        assumptions: Dict[str, Tuple[float, float]],
        n_simulations: int = 1000,
    ) -> pd.DataFrame:
        """
        Analyze sensitivity to assumption violations.

        Args:
            base_effect: Base treatment effect estimate
            base_se: Standard error of base estimate
            assumptions: Dict of assumption: (min_value, max_value)
            n_simulations: Number of simulations

        Returns:
            DataFrame with sensitivity analysis results
        """
        results = []

        for assumption, (min_val, max_val) in assumptions.items():
            for value in np.linspace(min_val, max_val, 20):
                # Adjust effect based on assumption violation
                adjusted_effect = self._adjust_for_assumption(
                    base_effect, assumption, value
                )

                # Calculate adjusted statistics
                z_score = adjusted_effect / base_se
                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

                results.append(
                    {
                        "assumption": assumption,
                        "violation_level": value,
                        "adjusted_effect": adjusted_effect,
                        "p_value": p_value,
                        "significant": p_value < 0.05,
                        "relative_change": (adjusted_effect - base_effect)
                        / base_effect,
                    }
                )

        return pd.DataFrame(results)

    def _adjust_for_assumption(
        self, base_effect: float, assumption: str, violation_level: float
    ) -> float:
        """Adjust effect estimate for assumption violation."""
        adjustments = {
            "spillover": base_effect * (1 - violation_level),  # Reduce by spillover
            "non_compliance": base_effect * (1 - violation_level),  # ITT dilution
            "attrition_bias": base_effect + violation_level,  # Bias from attrition
            "measurement_error": base_effect * (1 - violation_level**2),  # Attenuation
            "selection_bias": base_effect + violation_level,  # Confounding
        }

        return adjustments.get(assumption, base_effect)

    def conduct_bounds_analysis(
        self,
        observed_effect: float,
        n_treated: int,
        n_control: int,
        attrition_treated: float,
        attrition_control: float,
        outcome_range: Tuple[float, float],
    ) -> Dict[str, Tuple[float, float]]:
        """
        Calculate bounds for treatment effects under missing data.

        Args:
            observed_effect: Observed treatment effect
            n_treated: Number in treatment group
            n_control: Number in control group
            attrition_treated: Attrition rate in treatment
            attrition_control: Attrition rate in control
            outcome_range: (min, max) possible outcome values

        Returns:
            Dictionary with different bounds
        """
        y_min, y_max = outcome_range

        # Lee bounds (for differential attrition)
        if attrition_treated > attrition_control:
            # Trim from treatment group
            trim_fraction = (attrition_treated - attrition_control) / (
                1 - attrition_control
            )
            lee_lower = observed_effect - trim_fraction * (y_max - y_min)
            lee_upper = observed_effect
        else:
            # Trim from control group
            trim_fraction = (attrition_control - attrition_treated) / (
                1 - attrition_treated
            )
            lee_lower = observed_effect
            lee_upper = observed_effect + trim_fraction * (y_max - y_min)

        # Manski bounds (worst-case)
        manski_lower = observed_effect - max(attrition_treated, attrition_control) * (
            y_max - y_min
        )
        manski_upper = observed_effect + max(attrition_treated, attrition_control) * (
            y_max - y_min
        )

        # Horowitz-Manski bounds (tighter worst-case)
        hm_lower = observed_effect - (
            attrition_treated * y_max - attrition_control * y_min
        )
        hm_upper = observed_effect + (
            attrition_control * y_max - attrition_treated * y_min
        )

        return {
            "lee_bounds": (lee_lower, lee_upper),
            "manski_bounds": (manski_lower, manski_upper),
            "horowitz_manski_bounds": (hm_lower, hm_upper),
            "point_estimate": (observed_effect, observed_effect),
        }


if __name__ == "__main__":
    # Example usage
    print("Power Analysis and Simulations")
    print("=" * 60)

    # Initialize simulator
    simulator = PowerAnalysisSimulator(n_simulations=1000)

    # Simulate power across effect sizes and sample sizes
    print("\nPower Analysis Simulation:")
    power_results = simulator.simulate_ab_test_power(
        baseline_rate=0.1,
        effect_sizes=[0.01, 0.02, 0.03, 0.05],
        sample_sizes=[500, 1000, 2000, 5000],
        alpha=0.05,
        test_type="proportion",
    )

    print(power_results.head(10))

    # Calculate required sample size
    print("\nSample Size Calculation:")
    sample_result = simulator.calculate_sample_size(
        power=0.8, effect_size=0.02, alpha=0.05, test_type="proportion", baseline=0.1
    )

    print(f"  Required sample size: {sample_result.sample_size}")
    print(f"  MDE: {sample_result.minimum_detectable_effect:.4f}")
    print("\n  Recommendations:")
    for rec in sample_result.recommendations:
        print(f"    {rec}")

    # Simulate sequential testing
    print("\nSequential Testing Simulation:")
    seq_result = simulator.simulate_sequential_testing(
        true_effect=0.02,
        max_sample_size=2000,
        check_points=[500, 1000, 1500, 2000],
        alpha=0.05,
    )

    print(f"  Power: {seq_result.observed_power:.3f}")
    print(f"  Early stopping rate: {seq_result.early_stopping_rate:.3f}")
    print(f"  Average sample size: {seq_result.convergence_time}")

    # Simulate adaptive design
    print("\nAdaptive Design Simulation:")
    adaptive_result = simulator.simulate_adaptive_design(
        initial_sample=100,
        max_sample=1000,
        true_effects={"control": 0.1, "treatment_a": 0.12, "treatment_b": 0.15},
        adaptation_rule="thompson_sampling",
    )

    print(
        f"  Correct selection probability: {adaptive_result['correct_selection_probability']:.3f}"
    )
    print("  Average allocations:")
    for arm, prop in adaptive_result["average_allocations"].items():
        print(f"    {arm}: {prop:.3f}")

    # Sensitivity analysis
    print("\nSensitivity Analysis:")
    sensitivity = SensitivityAnalysis()

    bounds = sensitivity.conduct_bounds_analysis(
        observed_effect=0.05,
        n_treated=1000,
        n_control=1000,
        attrition_treated=0.1,
        attrition_control=0.05,
        outcome_range=(0, 1),
    )

    print("  Effect bounds under missing data:")
    for bound_type, (lower, upper) in bounds.items():
        print(f"    {bound_type}: [{lower:.4f}, {upper:.4f}]")
