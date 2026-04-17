"""Multi-armed bandit algorithms for optimization and experimentation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class BanditArm:
    """Represents a single arm in a multi-armed bandit."""

    id: str
    pulls: int = 0
    rewards: list[float] = field(default_factory=list)
    total_reward: float = 0.0

    @property
    def mean_reward(self) -> float:
        """Calculate mean reward for this arm."""
        return self.total_reward / self.pulls if self.pulls > 0 else 0.0

    @property
    def conversion_rate(self) -> float:
        """For binary rewards, calculate conversion rate."""
        if not self.rewards:
            return 0.0
        return sum(1 for r in self.rewards if r > 0) / len(self.rewards)

    def update(self, reward: float):
        """Update arm with new reward."""
        self.pulls += 1
        self.rewards.append(reward)
        self.total_reward += reward


class BanditAlgorithm(ABC):
    """Abstract base class for bandit algorithms."""

    def __init__(self, n_arms: int, arm_names: list[str] | None = None):
        """Initialize bandit algorithm.

        Args:
            n_arms: Number of arms
            arm_names: Optional names for arms
        """
        self.n_arms = n_arms
        if arm_names is None:
            arm_names = [f"arm_{i}" for i in range(n_arms)]
        self.arms = [BanditArm(id=name) for name in arm_names]
        self.total_pulls = 0
        self.history = []

    @abstractmethod
    def select_arm(self) -> int:
        """Select which arm to pull next."""
        pass

    def update(self, arm_idx: int, reward: float):
        """Update algorithm after observing reward.

        Args:
            arm_idx: Index of pulled arm
            reward: Observed reward
        """
        self.arms[arm_idx].update(reward)
        self.total_pulls += 1
        self.history.append(
            {
                "pull": self.total_pulls,
                "arm": arm_idx,
                "reward": reward,
                "cumulative_reward": sum(arm.total_reward for arm in self.arms),
            }
        )

    def get_statistics(self) -> pd.DataFrame:
        """Get current statistics for all arms."""
        stats = []
        for i, arm in enumerate(self.arms):
            stats.append(
                {
                    "arm": arm.id,
                    "pulls": arm.pulls,
                    "mean_reward": arm.mean_reward,
                    "total_reward": arm.total_reward,
                    "conversion_rate": arm.conversion_rate,
                    "pull_proportion": (
                        arm.pulls / self.total_pulls if self.total_pulls > 0 else 0
                    ),
                }
            )
        return pd.DataFrame(stats)


class EpsilonGreedy(BanditAlgorithm):
    """Epsilon-greedy bandit algorithm."""

    def __init__(
        self,
        n_arms: int,
        epsilon: float = 0.1,
        decay_rate: float = 0.99,
        min_epsilon: float = 0.01,
        arm_names: list[str] | None = None,
    ):
        """Initialize epsilon-greedy algorithm.

        Args:
            n_arms: Number of arms
            epsilon: Exploration probability
            decay_rate: Epsilon decay rate per pull
            min_epsilon: Minimum epsilon value
            arm_names: Optional names for arms
        """
        super().__init__(n_arms, arm_names)
        self.initial_epsilon = epsilon
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.min_epsilon = min_epsilon

    def select_arm(self) -> int:
        """Select arm using epsilon-greedy strategy."""
        if np.random.random() < self.epsilon:
            # Explore: random selection
            return np.random.randint(self.n_arms)
        else:
            # Exploit: select best arm
            mean_rewards = [arm.mean_reward for arm in self.arms]
            # Break ties randomly
            best_arms = np.where(mean_rewards == np.max(mean_rewards))[0]
            return np.random.choice(best_arms)

    def update(self, arm_idx: int, reward: float):
        """Update and decay epsilon."""
        super().update(arm_idx, reward)
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)


class ThompsonSampling(BanditAlgorithm):
    """Thompson Sampling (Bayesian) bandit algorithm."""

    def __init__(
        self,
        n_arms: int,
        prior_alpha: float = 1,
        prior_beta: float = 1,
        arm_names: list[str] | None = None,
    ):
        """Initialize Thompson Sampling.

        Args:
            n_arms: Number of arms
            prior_alpha: Alpha parameter for Beta prior
            prior_beta: Beta parameter for Beta prior
            arm_names: Optional names for arms
        """
        super().__init__(n_arms, arm_names)
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.alpha = [prior_alpha] * n_arms
        self.beta = [prior_beta] * n_arms

    def select_arm(self) -> int:
        """Select arm using Thompson Sampling."""
        # Sample from posterior distributions
        samples = [
            np.random.beta(self.alpha[i], self.beta[i]) for i in range(self.n_arms)
        ]
        return np.argmax(samples)

    def update(self, arm_idx: int, reward: float):
        """Update posterior parameters."""
        super().update(arm_idx, reward)
        # For binary rewards
        if reward > 0:
            self.alpha[arm_idx] += 1
        else:
            self.beta[arm_idx] += 1

    def get_posterior_stats(self) -> pd.DataFrame:
        """Get posterior statistics for each arm."""
        stats = []
        for i, arm in enumerate(self.arms):
            # Calculate posterior mean and credible interval
            posterior_mean = self.alpha[i] / (self.alpha[i] + self.beta[i])
            posterior_samples = np.random.beta(self.alpha[i], self.beta[i], 10000)
            ci_lower, ci_upper = np.percentile(posterior_samples, [2.5, 97.5])

            stats.append(
                {
                    "arm": arm.id,
                    "posterior_mean": posterior_mean,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "alpha": self.alpha[i],
                    "beta": self.beta[i],
                }
            )
        return pd.DataFrame(stats)


class UCB(BanditAlgorithm):
    """Upper Confidence Bound (UCB) bandit algorithm."""

    def __init__(
        self,
        n_arms: int,
        confidence: float = 2.0,
        arm_names: list[str] | None = None,
    ):
        """Initialize UCB algorithm.

        Args:
            n_arms: Number of arms
            confidence: Confidence parameter (controls exploration)
            arm_names: Optional names for arms
        """
        super().__init__(n_arms, arm_names)
        self.confidence = confidence

    def select_arm(self) -> int:
        """Select arm using UCB strategy."""
        # Initialize: pull each arm once
        for i in range(self.n_arms):
            if self.arms[i].pulls == 0:
                return i

        # Calculate upper confidence bounds
        ucb_values = []
        for arm in self.arms:
            avg_reward = arm.mean_reward
            confidence_width = self.confidence * np.sqrt(
                2 * np.log(self.total_pulls) / arm.pulls
            )
            ucb_values.append(avg_reward + confidence_width)

        return np.argmax(ucb_values)


class LinUCB(BanditAlgorithm):
    """Contextual Linear Upper Confidence Bound algorithm."""

    def __init__(
        self,
        n_arms: int,
        n_features: int,
        alpha: float = 1.0,
        arm_names: list[str] | None = None,
    ):
        """Initialize LinUCB for contextual bandits.

        Args:
            n_arms: Number of arms
            n_features: Number of context features
            alpha: Exploration parameter
            arm_names: Optional names for arms
        """
        super().__init__(n_arms, arm_names)
        self.n_features = n_features
        self.alpha = alpha

        # Initialize parameters for each arm
        self.A = [np.identity(n_features) for _ in range(n_arms)]
        self.b = [np.zeros((n_features, 1)) for _ in range(n_arms)]
        self.theta = [np.zeros((n_features, 1)) for _ in range(n_arms)]

    def select_arm(self, context: np.ndarray) -> int:
        """Select arm based on context.

        Args:
            context: Feature vector for current context

        Returns:
            Selected arm index
        """
        context = context.reshape(-1, 1)
        p = np.zeros(self.n_arms)

        for i in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[i])
            self.theta[i] = A_inv @ self.b[i]

            # Calculate UCB
            p[i] = float(
                self.theta[i].T @ context
                + self.alpha * np.sqrt(context.T @ A_inv @ context)
            )

        return np.argmax(p)

    def update_contextual(self, arm_idx: int, reward: float, context: np.ndarray):
        """Update model with contextual information.

        Args:
            arm_idx: Index of pulled arm
            reward: Observed reward
            context: Context vector
        """
        context = context.reshape(-1, 1)
        self.A[arm_idx] += context @ context.T
        self.b[arm_idx] += reward * context
        super().update(arm_idx, reward)


class BanditSimulator:
    """Simulate and compare bandit algorithms."""

    def __init__(self, true_probs: list[float], n_rounds: int = 1000):
        """Initialize bandit simulator.

        Args:
            true_probs: True reward probabilities for each arm
            n_rounds: Number of rounds to simulate
        """
        self.true_probs = true_probs
        self.n_arms = len(true_probs)
        self.n_rounds = n_rounds
        self.optimal_arm = np.argmax(true_probs)

    def simulate(self, algorithm: BanditAlgorithm) -> dict[str, Any]:
        """Simulate bandit algorithm.

        Args:
            algorithm: Bandit algorithm to test

        Returns:
            Simulation results
        """
        cumulative_regret = []
        cumulative_reward = 0
        optimal_pulls = 0

        for round in range(self.n_rounds):
            # Select arm
            arm = algorithm.select_arm()

            # Generate reward
            reward = np.random.binomial(1, self.true_probs[arm])

            # Update algorithm
            algorithm.update(arm, reward)

            # Track metrics
            cumulative_reward += reward
            optimal_reward = self.true_probs[self.optimal_arm]
            regret = optimal_reward - self.true_probs[arm]
            cumulative_regret.append(
                cumulative_regret[-1] + regret if cumulative_regret else regret
            )

            if arm == self.optimal_arm:
                optimal_pulls += 1

        return {
            "algorithm": algorithm.__class__.__name__,
            "cumulative_reward": cumulative_reward,
            "cumulative_regret": cumulative_regret[-1],
            "average_reward": cumulative_reward / self.n_rounds,
            "optimal_arm_proportion": optimal_pulls / self.n_rounds,
            "regret_history": cumulative_regret,
            "arm_statistics": algorithm.get_statistics(),
        }

    def compare_algorithms(self, algorithms: list[BanditAlgorithm]) -> pd.DataFrame:
        """Compare multiple bandit algorithms.

        Args:
            algorithms: List of algorithms to compare

        Returns:
            Comparison results DataFrame
        """
        results = []

        for algo in algorithms:
            result = self.simulate(algo)
            results.append(
                {
                    "algorithm": result["algorithm"],
                    "total_reward": result["cumulative_reward"],
                    "total_regret": result["cumulative_regret"],
                    "avg_reward": result["average_reward"],
                    "optimal_proportion": result["optimal_arm_proportion"],
                }
            )

        return pd.DataFrame(results)

    def plot_comparison(self, algorithms: list[BanditAlgorithm]):
        """Plot comparison of algorithms.

        Args:
            algorithms: List of algorithms to compare
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        for algo in algorithms:
            result = self.simulate(algo)
            algo_name = result["algorithm"]

            # Cumulative regret
            axes[0, 0].plot(result["regret_history"], label=algo_name)

            # Average reward over time
            history_df = pd.DataFrame(algo.history)
            if not history_df.empty:
                rolling_avg = history_df["reward"].expanding().mean()
                axes[0, 1].plot(rolling_avg.values, label=algo_name)

                # Arm selection over time
                for arm_idx in range(self.n_arms):
                    arm_pulls = history_df[history_df["arm"] == arm_idx]["pull"].values
                    axes[1, 0].scatter(
                        arm_pulls,
                        [arm_idx] * len(arm_pulls),
                        alpha=0.5,
                        s=1,
                        label=f"{algo_name}_arm_{arm_idx}",
                    )

        axes[0, 0].set_xlabel("Round")
        axes[0, 0].set_ylabel("Cumulative Regret")
        axes[0, 0].set_title("Cumulative Regret Comparison")
        axes[0, 0].legend()

        axes[0, 1].set_xlabel("Round")
        axes[0, 1].set_ylabel("Average Reward")
        axes[0, 1].set_title("Average Reward Over Time")
        axes[0, 1].legend()

        axes[1, 0].set_xlabel("Round")
        axes[1, 0].set_ylabel("Arm Index")
        axes[1, 0].set_title("Arm Selection Pattern")

        # Final statistics
        comparison_df = self.compare_algorithms(algorithms)
        axes[1, 1].axis("tight")
        axes[1, 1].axis("off")
        table = axes[1, 1].table(
            cellText=comparison_df.round(3).values,
            colLabels=comparison_df.columns,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)

        plt.tight_layout()
        plt.show()


class DynamicBandit:
    """Bandit with time-varying reward probabilities."""

    def __init__(
        self, n_arms: int, change_points: list[int], prob_schedules: list[list[float]]
    ):
        """Initialize dynamic bandit.

        Args:
            n_arms: Number of arms
            change_points: Rounds where probabilities change
            prob_schedules: Probability values for each period
        """
        self.n_arms = n_arms
        self.change_points = [0] + change_points
        self.prob_schedules = prob_schedules
        self.current_period = 0

    def get_current_probs(self, round: int) -> list[float]:
        """Get current reward probabilities based on round."""
        # Find current period
        for i, cp in enumerate(self.change_points[1:]):
            if round < cp:
                return self.prob_schedules[i]
        return self.prob_schedules[-1]

    def simulate_dynamic(
        self, algorithm: BanditAlgorithm, n_rounds: int
    ) -> dict[str, Any]:
        """Simulate algorithm on dynamic bandit.

        Args:
            algorithm: Bandit algorithm
            n_rounds: Number of rounds

        Returns:
            Simulation results
        """
        results = []

        for round in range(n_rounds):
            # Get current probabilities
            current_probs = self.get_current_probs(round)

            # Select arm
            arm = algorithm.select_arm()

            # Generate reward
            reward = np.random.binomial(1, current_probs[arm])

            # Update algorithm
            algorithm.update(arm, reward)

            results.append(
                {
                    "round": round,
                    "arm": arm,
                    "reward": reward,
                    "optimal_arm": np.argmax(current_probs),
                }
            )

        return pd.DataFrame(results)


if __name__ == "__main__":
    # Example usage
    print("Multi-Armed Bandit Algorithms")
    print("=" * 60)

    # Set up simulator
    true_probs = [0.1, 0.2, 0.3, 0.25, 0.15]  # True reward probabilities
    simulator = BanditSimulator(true_probs, n_rounds=500)

    # Test algorithms
    algorithms = [
        EpsilonGreedy(len(true_probs), epsilon=0.1),
        ThompsonSampling(len(true_probs)),
        UCB(len(true_probs), confidence=2.0),
    ]

    # Compare algorithms
    print("\nAlgorithm Comparison:")
    comparison = simulator.compare_algorithms(algorithms)
    print(comparison)

    # Test contextual bandit
    print("\nContextual Bandit (LinUCB):")
    linucb = LinUCB(n_arms=3, n_features=5)
    context = np.random.randn(5)
    selected_arm = linucb.select_arm(context)
    print(f"Selected arm: {selected_arm} for context: {context}")
