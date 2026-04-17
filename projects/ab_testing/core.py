"""Minimal A/B testing utilities used by integration tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import stats


class ABTestRunner:
    """Run basic A/B tests for conversion or continuous metrics."""

    def __init__(
        self,
        control_name: str = "Control",
        treatment_name: str = "Treatment",
        metric_type: str = "conversion",
        alpha: float = 0.05,
    ) -> None:
        self.control_name = control_name
        self.treatment_name = treatment_name
        self.metric_type = metric_type
        self.alpha = alpha

    def run_test(
        self, data, metric_column: str, group_column: str
    ) -> dict[str, float | bool | tuple[float, float]]:
        control = data[data[group_column] == "control"][metric_column].dropna()
        treatment = data[data[group_column] == "treatment"][metric_column].dropna()
        n1, n2 = len(control), len(treatment)
        if n1 == 0 or n2 == 0:
            raise ValueError("Both control and treatment must have data.")

        if self.metric_type == "conversion":
            p1, p2 = control.mean(), treatment.mean()
            p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
            se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
            z = (p2 - p1) / se if se > 0 else 0.0
            p_value = 2 * (1 - stats.norm.cdf(abs(z)))
            z_alpha = stats.norm.ppf(1 - self.alpha / 2)
            se_diff = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
            ci = (p2 - p1 - z_alpha * se_diff, p2 - p1 + z_alpha * se_diff)
            power = self._approx_power(p1, p2, n1, n2, self.alpha)
            return {
                "p_value": float(p_value),
                "confidence_interval": (float(ci[0]), float(ci[1])),
                "effect_size": float(p2 - p1),
                "statistical_power": float(power),
                "is_significant": bool(p_value < self.alpha),
            }

        # Continuous metric
        t_stat, p_value = stats.ttest_ind(treatment, control, equal_var=False)
        return {
            "mean_control": float(control.mean()),
            "mean_treatment": float(treatment.mean()),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
        }

    def calculate_mde(
        self, baseline_rate: float, sample_size: int, alpha: float = 0.05, power: float = 0.8
    ) -> float:
        """Approximate minimum detectable effect for two-proportion tests."""
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)
        p = baseline_rate
        se = np.sqrt(2 * p * (1 - p) / sample_size)
        return float((z_alpha + z_beta) * se)

    @staticmethod
    def _approx_power(p1: float, p2: float, n1: int, n2: int, alpha: float) -> float:
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        if se == 0:
            return 0.0
        z_effect = abs(p2 - p1) / se
        return float(stats.norm.cdf(z_effect - z_alpha))


class BayesianABTest:
    """Simple Bayesian A/B testing utilities."""

    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> None:
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

    def analyze_conversion(
        self, successes_a: int, trials_a: int, successes_b: int, trials_b: int, draws: int = 20000
    ) -> dict[str, float | tuple[float, float]]:
        a1, b1 = self.prior_alpha + successes_a, self.prior_beta + (trials_a - successes_a)
        a2, b2 = self.prior_alpha + successes_b, self.prior_beta + (trials_b - successes_b)
        samples_a = np.random.beta(a1, b1, size=draws)
        samples_b = np.random.beta(a2, b2, size=draws)
        prob_b_better = float(np.mean(samples_b > samples_a))
        expected_loss_a = float(np.mean(np.maximum(samples_b - samples_a, 0)))
        expected_loss_b = float(np.mean(np.maximum(samples_a - samples_b, 0)))
        ci_a = (float(np.quantile(samples_a, 0.025)), float(np.quantile(samples_a, 0.975)))
        ci_b = (float(np.quantile(samples_b, 0.025)), float(np.quantile(samples_b, 0.975)))
        return {
            "probability_b_better": prob_b_better,
            "expected_loss_a": expected_loss_a,
            "expected_loss_b": expected_loss_b,
            "credible_interval_a": ci_a,
            "credible_interval_b": ci_b,
        }

    def analyze_continuous(self, data_a: Iterable[float], data_b: Iterable[float]) -> dict[str, float | tuple[float, float]]:
        a = np.asarray(list(data_a), dtype=float)
        b = np.asarray(list(data_b), dtype=float)
        mean_a, mean_b = a.mean(), b.mean()
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        diff = mean_b - mean_a
        z = diff / se if se > 0 else 0.0
        prob_b_better = float(stats.norm.cdf(z))
        ci = (float(diff - 1.96 * se), float(diff + 1.96 * se))
        return {"probability_b_better": prob_b_better, "difference_credible_interval": ci}

    def check_early_stopping(
        self, successes_a: int, trials_a: int, successes_b: int, trials_b: int, threshold: float = 0.95
    ) -> dict[str, object]:
        results = self.analyze_conversion(successes_a, trials_a, successes_b, trials_b, draws=5000)
        prob = float(results["probability_b_better"])
        stop = prob >= threshold or (1 - prob) >= threshold
        winner = "B" if prob >= 0.5 else "A"
        return {"stop": stop, "winner": winner, "probability_b_better": prob}

    def recommend_sample_size(self, baseline_rate: float, minimum_effect: float, confidence_threshold: float = 0.95) -> int:
        z = stats.norm.ppf(confidence_threshold)
        p1 = baseline_rate
        p2 = baseline_rate + minimum_effect
        se = np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
        n = (z * se / max(minimum_effect, 1e-6)) ** 2
        return int(max(100, np.ceil(n)))


class MultiArmedBandit:
    """Basic multi-armed bandit algorithms."""

    def __init__(self, n_arms: int, algorithm: str = "epsilon_greedy", epsilon: float = 0.1) -> None:
        self.n_arms = n_arms
        self.algorithm = algorithm
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms, dtype=int)
        self.values = np.zeros(n_arms, dtype=float)
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)
        self.total_pulls = 0

    def select_arm(self) -> int:
        if self.algorithm == "epsilon_greedy":
            if np.random.rand() < self.epsilon:
                return int(np.random.randint(self.n_arms))
            return int(np.argmax(self.values))
        if self.algorithm == "thompson":
            samples = np.random.beta(self.alpha, self.beta)
            return int(np.argmax(samples))
        if self.algorithm == "ucb":
            self.total_pulls += 1
            ucb_values = np.zeros(self.n_arms)
            for i in range(self.n_arms):
                if self.counts[i] == 0:
                    return i
                bonus = np.sqrt(2 * np.log(self.total_pulls) / self.counts[i])
                ucb_values[i] = self.values[i] + bonus
            return int(np.argmax(ucb_values))
        raise ValueError("Unknown algorithm")

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        self.total_pulls += 1
        n = self.counts[arm]
        self.values[arm] = ((n - 1) / n) * self.values[arm] + (1 / n) * reward
        if self.algorithm in ("thompson", "epsilon_greedy", "ucb"):
            if reward >= 1:
                self.alpha[arm] += 1
            else:
                self.beta[arm] += 1

    def get_arm_statistics(self) -> dict[int, dict[str, float]]:
        stats_map: dict[int, dict[str, float]] = {}
        for i in range(self.n_arms):
            stats_map[i] = {
                "n_selections": float(self.counts[i]),
                "mean_reward": float(self.values[i]),
            }
        return stats_map


class ContextualBandit:
    """Simple linear contextual bandit (ridge regression per arm)."""

    def __init__(self, n_arms: int, n_features: int, alpha: float = 1.0) -> None:
        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha
        self.A = [np.eye(n_features) for _ in range(n_arms)]
        self.b = [np.zeros(n_features) for _ in range(n_arms)]

    def select_arm(self, context: np.ndarray) -> int:
        ctx = np.asarray(context, dtype=float)
        scores = []
        for i in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[i])
            theta = A_inv @ self.b[i]
            p = theta @ ctx + self.alpha * np.sqrt(ctx @ A_inv @ ctx)
            scores.append(p)
        return int(np.argmax(scores))

    def update(self, arm: int, reward: float, context: np.ndarray) -> None:
        ctx = np.asarray(context, dtype=float)
        self.A[arm] += np.outer(ctx, ctx)
        self.b[arm] += reward * ctx


class SequentialTest:
    """Sequential testing utilities."""

    def __init__(self, alpha: float, beta: float, effect_size: float) -> None:
        self.alpha = alpha
        self.beta = beta
        self.effect_size = effect_size
        self.log_lr = 0.0

    def update_sprt(self, control: int, treatment: int) -> str:
        p0 = 0.10
        p1 = min(0.99, p0 + self.effect_size)
        llr = (
            treatment * np.log(p1 / p0)
            + (1 - treatment) * np.log((1 - p1) / (1 - p0))
        )
        self.log_lr += llr
        A = np.log((1 - self.beta) / self.alpha)
        B = np.log(self.beta / (1 - self.alpha))
        if self.log_lr >= A:
            return "reject_null"
        if self.log_lr <= B:
            return "accept_null"
        return "continue"

    @staticmethod
    def get_obrien_fleming_boundary(information_fraction: float, alpha: float) -> float:
        z = stats.norm.ppf(1 - alpha / 2)
        return float(z / np.sqrt(information_fraction))

    @staticmethod
    def calculate_z_score(x1: int, n1: int, x2: int, n2: int) -> float:
        p1, p2 = x1 / n1, x2 / n2
        p_pool = (x1 + x2) / (n1 + n2)
        se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
        return float((p2 - p1) / se) if se > 0 else 0.0

    @staticmethod
    def reestimate_sample_size(observed_effect: float, current_n: int, target_power: float) -> int:
        if observed_effect <= 0:
            return current_n * 2
        scale = (0.1 / observed_effect) ** 2
        return int(np.ceil(current_n * scale))


class PowerAnalysis:
    """Power analysis helpers."""

    @staticmethod
    def calculate_power(effect_size: float, sample_size: int, alpha: float = 0.05) -> float:
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z = np.sqrt(sample_size / 2) * effect_size
        return float(stats.norm.cdf(z - z_alpha))

    @staticmethod
    def calculate_cohens_d(mean1: float, std1: float, mean2: float, std2: float) -> float:
        pooled = np.sqrt((std1**2 + std2**2) / 2)
        return float((mean2 - mean1) / pooled)

    @staticmethod
    def calculate_cohens_h(p1: float, p2: float) -> float:
        return float(2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1))))

    def generate_power_curve(self, effect_size: float, sample_sizes: Iterable[int], alpha: float = 0.05) -> list[float]:
        return [self.calculate_power(effect_size, n, alpha) for n in sample_sizes]

    def find_minimum_sample_size(self, effect_size: float, desired_power: float, alpha: float = 0.05) -> int:
        n = 50
        while self.calculate_power(effect_size, n, alpha) < desired_power and n < 100000:
            n += 50
        return n


class SampleSizeCalculator:
    """Sample size calculators."""

    def two_proportion_sample_size(self, p1: float, p2: float, alpha: float = 0.05, power: float = 0.8) -> int:
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)
        p_bar = (p1 + p2) / 2
        se = np.sqrt(2 * p_bar * (1 - p_bar))
        n = (z_alpha + z_beta) ** 2 * se**2 / (p2 - p1) ** 2
        return int(np.ceil(n))

    def continuous_sample_size(self, mean_diff: float, std_dev: float, alpha: float = 0.05, power: float = 0.8) -> int:
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)
        n = 2 * (z_alpha + z_beta) ** 2 * (std_dev**2) / (mean_diff**2)
        return int(np.ceil(n))

    def unequal_allocation_size(
        self, p1: float, p2: float, ratio: float = 1.0, alpha: float = 0.05, power: float = 0.8
    ) -> tuple[int, int]:
        n_total = self.two_proportion_sample_size(p1, p2, alpha, power)
        n_treatment = int(np.ceil(n_total / (1 + ratio)))
        n_control = int(np.ceil(n_treatment * ratio))
        return n_control, n_treatment

    @staticmethod
    def adjust_for_multiple_comparisons(
        base_sample_size: int, n_comparisons: int, method: str = "bonferroni"
    ) -> int:
        if method == "bonferroni":
            return int(np.ceil(base_sample_size * n_comparisons))
        return base_sample_size

