"""Power and sample-size calculations for two-proportion experiments."""

from __future__ import annotations

import math

from scipy import stats


def _validate_probability(value: float, name: str) -> None:
    if not 0 < value < 1:
        raise ValueError(f"{name} must be between 0 and 1")


def sample_size_two_proportions(
    baseline_rate: float,
    minimum_detectable_effect: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    allocation_ratio: float = 1.0,
    two_sided: bool = True,
) -> int:
    """Return the required control-group size for a two-proportion test."""

    _validate_probability(baseline_rate, "baseline_rate")
    _validate_probability(alpha, "alpha")
    _validate_probability(power, "power")
    if minimum_detectable_effect <= 0:
        raise ValueError("minimum_detectable_effect must be positive")
    if baseline_rate + minimum_detectable_effect >= 1:
        raise ValueError("baseline_rate + minimum_detectable_effect must be below 1")
    if allocation_ratio <= 0:
        raise ValueError("allocation_ratio must be positive")

    p_control = baseline_rate
    p_treatment = baseline_rate + minimum_detectable_effect
    p_average = (p_control + p_treatment) / 2
    z_alpha = stats.norm.ppf(1 - alpha / (2 if two_sided else 1))
    z_beta = stats.norm.ppf(power)

    numerator = (
        z_alpha * math.sqrt(2 * p_average * (1 - p_average))
        + z_beta
        * math.sqrt(
            p_control * (1 - p_control) + p_treatment * (1 - p_treatment) / allocation_ratio
        )
    ) ** 2
    return int(math.ceil(numerator / minimum_detectable_effect**2))


def power_two_proportions(
    n_control: int,
    n_treatment: int,
    baseline_rate: float,
    effect_size: float,
    *,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> float:
    """Approximate power for a fixed two-proportion design."""

    if n_control <= 0 or n_treatment <= 0:
        raise ValueError("sample sizes must be positive")
    _validate_probability(baseline_rate, "baseline_rate")
    _validate_probability(alpha, "alpha")
    if effect_size <= 0:
        raise ValueError("effect_size must be positive")
    if baseline_rate + effect_size >= 1:
        raise ValueError("baseline_rate + effect_size must be below 1")

    p_control = baseline_rate
    p_treatment = baseline_rate + effect_size
    p_null = (p_control + p_treatment) / 2
    z_alpha = stats.norm.ppf(1 - alpha / (2 if two_sided else 1))
    se_null = math.sqrt(p_null * (1 - p_null) * (1 / n_control + 1 / n_treatment))
    se_alt = math.sqrt(
        p_control * (1 - p_control) / n_control + p_treatment * (1 - p_treatment) / n_treatment
    )
    critical_effect = z_alpha * se_null
    return float(stats.norm.cdf((effect_size - critical_effect) / se_alt))


def cohens_h(p_control: float, p_treatment: float) -> float:
    """Return Cohen's h for two proportions."""

    if not 0 <= p_control <= 1 or not 0 <= p_treatment <= 1:
        raise ValueError("proportions must be in [0, 1]")
    return float(2 * (math.asin(math.sqrt(p_treatment)) - math.asin(math.sqrt(p_control))))
