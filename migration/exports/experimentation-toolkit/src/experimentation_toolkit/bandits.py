"""Small multi-armed bandit policy helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BanditState:
    """Observed state for a finite-arm bandit."""

    pulls: NDArray[np.float64]
    rewards: NDArray[np.float64]

    @classmethod
    def from_sequences(cls, pulls: list[float], rewards: list[float]) -> BanditState:
        if len(pulls) != len(rewards):
            raise ValueError("pulls and rewards must have the same length")
        if not pulls:
            raise ValueError("at least one arm is required")
        pulls_array = np.asarray(pulls, dtype=float)
        rewards_array = np.asarray(rewards, dtype=float)
        if np.any(pulls_array < 0):
            raise ValueError("pull counts cannot be negative")
        return cls(pulls=pulls_array, rewards=rewards_array)

    @property
    def values(self) -> NDArray[np.float64]:
        values = np.zeros_like(self.rewards, dtype=float)
        observed = self.pulls > 0
        values[observed] = self.rewards[observed] / self.pulls[observed]
        return values


def epsilon_greedy_arm(
    values: list[float] | NDArray[np.float64],
    epsilon: float = 0.1,
    *,
    rng: np.random.Generator | None = None,
) -> int:
    """Select an arm using epsilon-greedy exploration."""

    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be in [0, 1]")
    value_array = np.asarray(values, dtype=float)
    if value_array.size == 0:
        raise ValueError("at least one arm is required")

    generator = rng or np.random.default_rng()
    if generator.random() < epsilon:
        return int(generator.integers(value_array.size))
    return int(np.argmax(value_array))


def thompson_beta_arm(
    successes: list[float] | NDArray[np.float64],
    failures: list[float] | NDArray[np.float64],
    *,
    rng: np.random.Generator | None = None,
) -> int:
    """Select an arm using beta-binomial Thompson sampling."""

    success_array = np.asarray(successes, dtype=float)
    failure_array = np.asarray(failures, dtype=float)
    if success_array.shape != failure_array.shape:
        raise ValueError("successes and failures must have the same shape")
    if success_array.size == 0:
        raise ValueError("at least one arm is required")
    if np.any(success_array < 0) or np.any(failure_array < 0):
        raise ValueError("successes and failures must be non-negative")

    generator = rng or np.random.default_rng()
    samples = generator.beta(success_array + 1, failure_array + 1)
    return int(np.argmax(samples))


def ucb1_arm(state: BanditState, exploration: float = 2.0) -> int:
    """Select an arm with the UCB1 rule."""

    if exploration <= 0:
        raise ValueError("exploration must be positive")
    unobserved = np.flatnonzero(state.pulls == 0)
    if unobserved.size:
        return int(unobserved[0])

    total_pulls = float(np.sum(state.pulls))
    bonus = np.sqrt(exploration * np.log(total_pulls) / state.pulls)
    return int(np.argmax(state.values + bonus))
