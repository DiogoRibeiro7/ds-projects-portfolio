"""Core statistical tests for controlled experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import stats

Alternative = Literal["two-sided", "larger", "smaller"]


@dataclass(frozen=True)
class TestResult:
    """A hypothesis-test result with an estimated effect."""

    statistic: float
    p_value: float
    effect: float
    confidence_interval: tuple[float, float] | None = None


@dataclass(frozen=True)
class BootstrapInterval:
    """Bootstrap interval for an observed effect."""

    lower: float
    upper: float
    observed_effect: float


def two_proportion_z_test(
    successes_control: int,
    n_control: int,
    successes_treatment: int,
    n_treatment: int,
    *,
    alternative: Alternative = "two-sided",
    continuity_correction: bool = True,
) -> TestResult:
    """Run a two-proportion z-test for treatment minus control lift."""

    _validate_counts(successes_control, n_control, "control")
    _validate_counts(successes_treatment, n_treatment, "treatment")

    p_control = successes_control / n_control
    p_treatment = successes_treatment / n_treatment
    effect = p_treatment - p_control
    pooled = (successes_control + successes_treatment) / (n_control + n_treatment)

    if pooled in {0.0, 1.0}:
        statistic = 0.0 if effect == 0 else math.copysign(math.inf, effect)
        p_value = 1.0 if effect == 0 else 0.0
        return TestResult(statistic=statistic, p_value=p_value, effect=effect)

    standard_error = math.sqrt(pooled * (1 - pooled) * (1 / n_control + 1 / n_treatment))
    adjusted_effect = effect
    if continuity_correction:
        correction = 0.5 * (1 / n_control + 1 / n_treatment)
        if abs(effect) > correction:
            adjusted_effect = effect - math.copysign(correction, effect)

    statistic = adjusted_effect / standard_error
    p_value = _p_value_from_z(statistic, alternative)
    z_alpha = stats.norm.ppf(0.975)
    effect_se = math.sqrt(
        p_control * (1 - p_control) / n_control + p_treatment * (1 - p_treatment) / n_treatment
    )
    confidence_interval = (effect - z_alpha * effect_se, effect + z_alpha * effect_se)
    return TestResult(
        statistic=float(statistic),
        p_value=float(p_value),
        effect=float(effect),
        confidence_interval=(float(confidence_interval[0]), float(confidence_interval[1])),
    )


def welch_t_test(
    control: list[float] | NDArray[np.float64],
    treatment: list[float] | NDArray[np.float64],
) -> TestResult:
    """Run Welch's t-test for treatment minus control mean difference."""

    control_array = _as_non_empty_array(control, "control")
    treatment_array = _as_non_empty_array(treatment, "treatment")
    statistic, p_value = stats.ttest_ind(treatment_array, control_array, equal_var=False)
    effect = float(np.mean(treatment_array) - np.mean(control_array))
    standard_error = math.sqrt(
        float(np.var(control_array, ddof=1)) / len(control_array)
        + float(np.var(treatment_array, ddof=1)) / len(treatment_array)
    )
    dof = _welch_degrees_of_freedom(control_array, treatment_array)
    margin = stats.t.ppf(0.975, dof) * standard_error
    return TestResult(
        statistic=float(statistic),
        p_value=float(p_value),
        effect=effect,
        confidence_interval=(effect - float(margin), effect + float(margin)),
    )


def bootstrap_ci_diff(
    control_rate: float,
    treatment_rate: float,
    n_control: int,
    n_treatment: int,
    *,
    n_bootstrap: int = 5000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> BootstrapInterval:
    """Bootstrap a percentile interval for treatment minus control conversion lift."""

    if not 0 <= control_rate <= 1 or not 0 <= treatment_rate <= 1:
        raise ValueError("rates must be in [0, 1]")
    if n_control <= 0 or n_treatment <= 0:
        raise ValueError("sample sizes must be positive")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    generator = np.random.default_rng(seed)
    control_draws = generator.binomial(n_control, control_rate, size=n_bootstrap) / n_control
    treatment_draws = (
        generator.binomial(n_treatment, treatment_rate, size=n_bootstrap) / n_treatment
    )
    diffs = treatment_draws - control_draws
    return BootstrapInterval(
        lower=float(np.quantile(diffs, alpha / 2)),
        upper=float(np.quantile(diffs, 1 - alpha / 2)),
        observed_effect=float(treatment_rate - control_rate),
    )


def _validate_counts(successes: int, sample_size: int, label: str) -> None:
    if sample_size <= 0:
        raise ValueError(f"{label} sample size must be positive")
    if successes < 0:
        raise ValueError(f"{label} successes cannot be negative")
    if successes > sample_size:
        raise ValueError(f"{label} successes cannot exceed sample size")


def _p_value_from_z(z_value: float, alternative: Alternative) -> float:
    if alternative == "two-sided":
        return float(2 * (1 - stats.norm.cdf(abs(z_value))))
    if alternative == "larger":
        return float(1 - stats.norm.cdf(z_value))
    if alternative == "smaller":
        return float(stats.norm.cdf(z_value))
    raise ValueError(f"unknown alternative: {alternative}")


def _as_non_empty_array(
    values: list[float] | NDArray[np.float64], label: str
) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    if array.size < 2:
        raise ValueError(f"{label} must contain at least two observations")
    if np.isnan(array).any():
        raise ValueError(f"{label} cannot contain NaN values")
    return array


def _welch_degrees_of_freedom(
    control: NDArray[np.float64], treatment: NDArray[np.float64]
) -> float:
    control_term = float(np.var(control, ddof=1)) / len(control)
    treatment_term = float(np.var(treatment, ddof=1)) / len(treatment)
    numerator = (control_term + treatment_term) ** 2
    denominator = (control_term**2 / (len(control) - 1)) + (
        treatment_term**2 / (len(treatment) - 1)
    )
    return numerator / denominator
