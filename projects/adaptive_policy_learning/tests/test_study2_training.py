from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from adaptive_policy_learning.study2_training import (
    STUDY2_MAX_ITER,
    _fit_study2_reward_model,
)


def test_study2_newton_cholesky_fit_passes_on_well_conditioned_binary_data() -> None:
    rng = np.random.default_rng(20260831)
    x = rng.normal(size=(1000, 6))
    linear = 0.6 * x[:, 0] - 0.4 * x[:, 1] + 0.2 * x[:, 2]
    probability = 1.0 / (1.0 + np.exp(-linear))
    y = rng.binomial(1, probability).astype(int)

    model, captured = _fit_study2_reward_model(x, y)

    assert model.solver == "newton-cholesky"
    assert model.max_iter == STUDY2_MAX_ITER
    assert int(model.n_iter_[0]) < STUDY2_MAX_ITER
    assert model.classes_.tolist() == [0, 1]
    assert np.all(np.isfinite(model.coef_))
    assert np.all(np.isfinite(model.intercept_))
    assert captured == []


def test_study2_training_gate_rejects_convergence_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    class WarningModel:
        solver = "newton-cholesky"
        max_iter = STUDY2_MAX_ITER
        classes_ = np.array([0, 1])
        n_iter_ = np.array([1])
        coef_ = np.zeros((1, 2))
        intercept_ = np.zeros(1)

        def fit(self, x: np.ndarray, y: np.ndarray) -> WarningModel:
            del x, y
            warnings.warn("synthetic convergence failure", ConvergenceWarning, stacklevel=2)
            return self

    monkeypatch.setattr(
        "adaptive_policy_learning.study2_training.LogisticRegression",
        lambda **kwargs: WarningModel(),
    )

    with pytest.raises(RuntimeError, match="ConvergenceWarning"):
        _fit_study2_reward_model(np.zeros((4, 2)), np.array([0, 1, 0, 1]))
