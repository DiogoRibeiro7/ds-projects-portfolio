"""
Tests for monotone quantile head and training utilities.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.statistics.quantile_models import (
    MonotoneQuantileHead,
    fit_constant_monotone_quantiles,
    pinball_loss,
)

pytestmark = pytest.mark.unit


def test_monotone_head_outputs_sorted():
    rng = np.random.default_rng(0)
    head = MonotoneQuantileHead([0.1, 0.5, 0.9])
    raw = rng.normal(size=(128, head.n_quantiles + 1))
    preds = head(raw)
    assert preds.shape == (128, head.n_quantiles)
    assert np.all(np.diff(preds, axis=1) >= 0.0)


def test_constant_fit_matches_empirical_quantiles():
    rng = np.random.default_rng(1)
    data = rng.lognormal(mean=0.3, sigma=0.6, size=5000)
    levels = [0.1, 0.5, 0.9]
    fit = fit_constant_monotone_quantiles(data, levels, epochs=4000, lr=0.05)
    learned = fit["quantiles"]
    empirical = np.quantile(data, levels)
    assert np.allclose(learned, empirical, atol=0.03)


def test_pinball_loss_compares_against_empirical_quantiles():
    rng = np.random.default_rng(42)
    data = rng.gamma(shape=2.0, scale=1.5, size=4000)
    levels = [0.05, 0.5, 0.95]
    fit = fit_constant_monotone_quantiles(data, levels, epochs=4000, lr=0.05)
    mono_preds = np.tile(fit["quantiles"], (data.shape[0], 1))
    empirical_quantiles = np.quantile(data, levels)
    cdf_preds = np.tile(empirical_quantiles, (data.shape[0], 1))
    mono_loss = pinball_loss(data, mono_preds, levels)
    cdf_loss = pinball_loss(data, cdf_preds, levels)
    assert mono_loss <= cdf_loss + 1e-4
