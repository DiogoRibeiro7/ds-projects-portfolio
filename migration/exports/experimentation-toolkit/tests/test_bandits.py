import numpy as np

from experimentation_toolkit import BanditState, epsilon_greedy_arm, thompson_beta_arm, ucb1_arm


def test_epsilon_greedy_can_exploit_best_arm() -> None:
    arm = epsilon_greedy_arm([0.1, 0.4, 0.2], epsilon=0.0, rng=np.random.default_rng(7))

    assert arm == 1


def test_ucb1_selects_unobserved_arm_first() -> None:
    state = BanditState.from_sequences([10, 0, 4], [2, 0, 2])

    assert ucb1_arm(state) == 1


def test_thompson_beta_returns_valid_arm() -> None:
    arm = thompson_beta_arm([10, 30], [90, 70], rng=np.random.default_rng(7))

    assert arm in {0, 1}
