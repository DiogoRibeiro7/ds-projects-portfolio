"""Off-policy evaluation primitives for contextual bandit data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OverlapDiagnostics:
    """Importance-weight overlap diagnostics."""

    n: int
    ess: float
    ess_fraction: float
    propensity_min: float
    propensity_median: float
    propensity_max: float
    weight_p50: float
    weight_p90: float
    weight_p95: float
    weight_p99: float
    weight_max: float


def importance_weights(
    target_probability_logged_action: np.ndarray,
    logging_propensity: np.ndarray,
    *,
    cap: float | None = None,
) -> np.ndarray:
    """Return target/logging importance weights with hard support checks."""
    target = np.asarray(target_probability_logged_action, dtype=float)
    logging = np.asarray(logging_propensity, dtype=float)
    if target.shape != logging.shape:
        raise ValueError("target probabilities and logging propensities must align.")
    if target.ndim != 1:
        raise ValueError("importance-weight inputs must be one-dimensional.")
    if np.any(~np.isfinite(target)) or np.any(target < 0.0) or np.any(target > 1.0):
        raise ValueError("target probabilities must be finite and lie in [0, 1].")
    if np.any(~np.isfinite(logging)) or np.any(logging <= 0.0) or np.any(logging > 1.0):
        raise ValueError("logging propensities must be finite and lie in (0, 1].")
    weights = target / logging
    if cap is not None:
        if not np.isfinite(cap) or cap <= 0.0:
            raise ValueError("importance-weight cap must be finite and positive.")
        weights = np.minimum(weights, float(cap))
    return weights


def ips(reward: np.ndarray, weights: np.ndarray) -> float:
    """Inverse-propensity-score policy value estimate."""
    reward_arr, weight_arr = _aligned(reward, weights)
    return float(np.mean(weight_arr * reward_arr))


def snips(reward: np.ndarray, weights: np.ndarray) -> float:
    """Self-normalized inverse-propensity-score policy value estimate."""
    reward_arr, weight_arr = _aligned(reward, weights)
    denominator = float(np.sum(weight_arr))
    if denominator <= 0.0:
        raise ValueError("SNIPS requires a positive total importance weight.")
    return float(np.sum(weight_arr * reward_arr) / denominator)


def direct_method(expected_reward_target: np.ndarray) -> float:
    """Direct-method value estimate from per-row target-policy expected rewards."""
    values = np.asarray(expected_reward_target, dtype=float)
    if values.ndim != 1 or values.size == 0 or np.any(~np.isfinite(values)):
        raise ValueError("direct-method expected rewards must be a finite non-empty vector.")
    return float(np.mean(values))


def doubly_robust(
    reward: np.ndarray,
    weights: np.ndarray,
    q_logged_action: np.ndarray,
    expected_reward_target: np.ndarray,
) -> float:
    """Doubly robust policy-value estimate."""
    reward_arr, weight_arr = _aligned(reward, weights)
    q_logged = np.asarray(q_logged_action, dtype=float)
    dm = np.asarray(expected_reward_target, dtype=float)
    if q_logged.shape != reward_arr.shape or dm.shape != reward_arr.shape:
        raise ValueError("DR inputs must have identical one-dimensional shapes.")
    if np.any(~np.isfinite(q_logged)) or np.any(~np.isfinite(dm)):
        raise ValueError("DR reward-model inputs must be finite.")
    return float(np.mean(dm + weight_arr * (reward_arr - q_logged)))


def overlap_diagnostics(logging_propensity: np.ndarray, weights: np.ndarray) -> OverlapDiagnostics:
    """Compute effective sample size and frozen weight quantiles."""
    logging = np.asarray(logging_propensity, dtype=float)
    weight_arr = np.asarray(weights, dtype=float)
    if logging.shape != weight_arr.shape or logging.ndim != 1 or logging.size == 0:
        raise ValueError("propensities and weights must be aligned non-empty vectors.")
    if np.any(~np.isfinite(logging)) or np.any(logging <= 0.0):
        raise ValueError("logging propensities must be finite and positive.")
    if np.any(~np.isfinite(weight_arr)) or np.any(weight_arr < 0.0):
        raise ValueError("importance weights must be finite and non-negative.")
    sum_w = float(np.sum(weight_arr))
    sum_w2 = float(np.sum(weight_arr**2))
    if sum_w2 <= 0.0:
        raise ValueError("ESS is undefined when every importance weight is zero.")
    ess = sum_w * sum_w / sum_w2
    return OverlapDiagnostics(
        n=int(weight_arr.size),
        ess=ess,
        ess_fraction=ess / float(weight_arr.size),
        propensity_min=float(np.min(logging)),
        propensity_median=float(np.median(logging)),
        propensity_max=float(np.max(logging)),
        weight_p50=float(np.quantile(weight_arr, 0.50)),
        weight_p90=float(np.quantile(weight_arr, 0.90)),
        weight_p95=float(np.quantile(weight_arr, 0.95)),
        weight_p99=float(np.quantile(weight_arr, 0.99)),
        weight_max=float(np.max(weight_arr)),
    )


def promotion_decision(
    bootstrap_lower_95: float,
    ess_fraction: float,
    *,
    minimum_ess_fraction: float = 0.10,
) -> str:
    """Apply the frozen lower-confidence-bound promotion rule."""
    if not np.isfinite(bootstrap_lower_95) or not np.isfinite(ess_fraction):
        raise ValueError("promotion inputs must be finite.")
    if minimum_ess_fraction <= 0.0 or minimum_ess_fraction > 1.0:
        raise ValueError("minimum ESS fraction must lie in (0, 1].")
    if bootstrap_lower_95 > 0.0 and ess_fraction >= minimum_ess_fraction:
        return "promote_challenger"
    return "do_not_promote"


def _aligned(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)
    if left_arr.shape != right_arr.shape or left_arr.ndim != 1 or left_arr.size == 0:
        raise ValueError("inputs must be aligned non-empty one-dimensional vectors.")
    if np.any(~np.isfinite(left_arr)) or np.any(~np.isfinite(right_arr)):
        raise ValueError("inputs must be finite.")
    return left_arr, right_arr
