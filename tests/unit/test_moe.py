"""Unit tests for Gaussian Mixture-of-Experts."""

from __future__ import annotations

import numpy as np
import pytest

from src.statistics.moe import GaussianMixtureOfExperts

pytestmark = pytest.mark.unit


def _synth_data(seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, size=2000)
    y = np.where(
        x < 0,
        rng.normal(loc=-2 + 0.5 * x, scale=0.5, size=2000),
        rng.normal(loc=2 + 0.5 * x, scale=0.4, size=2000),
    )
    return x[:, None], y


def test_gating_probabilities_sum_to_one():
    x, y = _synth_data()
    model = GaussianMixtureOfExperts(n_experts=2, max_iters=10)
    model.fit(x, y)
    probs, _, _ = model.mixture_components(x[:10])
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_pdf_integrates_to_one():
    x, y = _synth_data()
    model = GaussianMixtureOfExperts(n_experts=2, max_iters=20)
    model.fit(x, y)
    x0 = np.array([[0.2]])
    grid = np.linspace(-6, 6, 4000)
    pdf_vals = model.pdf(np.repeat(x0, grid.size, axis=0), grid)
    integral = np.trapezoid(pdf_vals, grid)
    assert integral == pytest.approx(1.0, abs=5e-2)


def test_moe_improves_over_single_expert():
    x, y = _synth_data()
    moe = GaussianMixtureOfExperts(n_experts=2, max_iters=50, lr=0.2)
    moe.fit(x, y)
    single = GaussianMixtureOfExperts(n_experts=1, max_iters=50)
    single.fit(x, y)
    assert moe.negative_log_likelihood(x, y) < single.negative_log_likelihood(x, y)
