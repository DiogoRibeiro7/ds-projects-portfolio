"""Experiment validation checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class SampleRatioResult:
    """Sample-ratio mismatch result."""

    statistic: float
    p_value: float
    is_mismatch: bool
    observed: dict[str, int]
    expected: dict[str, float]


def sample_ratio_mismatch(
    observed: dict[str, int],
    expected_proportions: dict[str, float] | None = None,
    *,
    alpha: float = 0.001,
) -> SampleRatioResult:
    """Run a chi-square sample-ratio mismatch check."""

    if len(observed) < 2:
        raise ValueError("at least two groups are required")
    if any(count < 0 for count in observed.values()):
        raise ValueError("observed counts cannot be negative")
    total = sum(observed.values())
    if total <= 0:
        raise ValueError("total observed count must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    groups = list(observed)
    if expected_proportions is None:
        expected_proportions = {group: 1 / len(groups) for group in groups}
    missing = set(groups) - set(expected_proportions)
    if missing:
        raise ValueError(f"missing expected proportions for groups: {sorted(missing)}")

    expected_sum = sum(expected_proportions[group] for group in groups)
    if expected_sum <= 0:
        raise ValueError("expected proportions must sum to a positive value")

    expected = {group: total * expected_proportions[group] / expected_sum for group in groups}
    statistic, p_value = stats.chisquare(
        [observed[group] for group in groups],
        [expected[group] for group in groups],
    )
    return SampleRatioResult(
        statistic=float(statistic),
        p_value=float(p_value),
        is_mismatch=bool(p_value < alpha),
        observed=dict(observed),
        expected=expected,
    )


def validate_binary_metric(values: list[int] | np.ndarray) -> None:
    """Raise if a metric vector is not binary."""

    array = np.asarray(values)
    unique_values = set(array[~np.isnan(array.astype(float))].tolist())
    if not unique_values.issubset({0, 1}):
        raise ValueError("binary metric values must be 0/1 or bool")
