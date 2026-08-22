"""Tests for the conjugate Bayesian building blocks."""

from __future__ import annotations

import numpy as np
import pytest

from pt_gdp_bayes.bayes import (
    fit_bayesian_linear_regression,
    normal_normal_update,
    summarize_samples,
)


class TestNormalNormalUpdate:
    def test_matches_closed_form_precision_weighting(self) -> None:
        posterior = normal_normal_update(
            prior_mean=0.0, prior_sd=2.0, observation_mean=10.0, observation_sd=1.0
        )
        # Precisions 0.25 and 1.0 -> posterior mean 10*1/1.25 = 8, var = 1/1.25.
        assert posterior.mean == pytest.approx(8.0)
        assert posterior.sd == pytest.approx(np.sqrt(1 / 1.25))

    def test_posterior_is_sharper_than_either_input(self) -> None:
        posterior = normal_normal_update(
            prior_mean=1.0, prior_sd=0.5, observation_mean=2.0, observation_sd=0.5
        )
        assert posterior.sd < 0.5
        assert posterior.mean == pytest.approx(1.5)

    def test_a_far_sharper_observation_dominates(self) -> None:
        posterior = normal_normal_update(
            prior_mean=0.0, prior_sd=1.0, observation_mean=5.0, observation_sd=0.001
        )
        assert posterior.mean == pytest.approx(5.0, abs=1e-4)

    @pytest.mark.parametrize("prior_sd,obs_sd", [(0.0, 1.0), (1.0, 0.0), (-1.0, 1.0)])
    def test_rejects_non_positive_standard_deviations(self, prior_sd: float, obs_sd: float) -> None:
        with pytest.raises(ValueError, match="Standard deviations"):
            normal_normal_update(
                prior_mean=0.0, prior_sd=prior_sd, observation_mean=0.0, observation_sd=obs_sd
            )


class TestBayesianLinearRegression:
    def test_recovers_known_coefficients_from_clean_data(self, rng) -> None:
        X = np.column_stack([np.ones(200), rng.normal(size=200), rng.normal(size=200)])
        true_beta = np.array([1.5, -2.0, 0.75])
        y = X @ true_beta + rng.normal(scale=0.05, size=200)

        posterior = fit_bayesian_linear_regression(X, y, prior_variance=100.0)

        assert posterior.beta_mean == pytest.approx(true_beta, abs=0.05)

    def test_tighter_prior_shrinks_coefficients_toward_zero(self, rng) -> None:
        X = np.column_stack([np.ones(50), rng.normal(size=50)])
        y = X @ np.array([0.0, 5.0]) + rng.normal(scale=0.5, size=50)

        loose = fit_bayesian_linear_regression(X, y, prior_variance=1000.0)
        tight = fit_bayesian_linear_regression(X, y, prior_variance=0.001)

        assert abs(tight.beta_mean[1]) < abs(loose.beta_mean[1])

    def test_predictive_is_wider_with_observation_noise(self, rng) -> None:
        X = np.column_stack([np.ones(80), rng.normal(size=80)])
        y = X @ np.array([1.0, 2.0]) + rng.normal(scale=0.3, size=80)
        posterior = fit_bayesian_linear_regression(X, y, prior_variance=100.0)

        x_new = np.array([1.0, 0.5])
        with_noise = posterior.sample_predictive(x_new, 20_000, rng=rng)
        without = posterior.sample_predictive(
            x_new, 20_000, include_observation_noise=False, rng=rng
        )

        assert with_noise.std() > without.std()
        assert with_noise.mean() == pytest.approx(without.mean(), abs=0.05)

    def test_sample_parameters_respects_shapes_and_positivity(self, rng) -> None:
        X = np.column_stack([np.ones(40), rng.normal(size=40)])
        y = rng.normal(size=40)
        posterior = fit_bayesian_linear_regression(X, y)

        beta, sigma = posterior.sample_parameters(n_samples=500, rng=rng)

        assert beta.shape == (500, 2)
        assert sigma.shape == (500,)
        assert (sigma > 0).all()

    def test_sampling_is_reproducible_for_a_fixed_seed(self, rng) -> None:
        X = np.column_stack([np.ones(30), np.linspace(0, 1, 30)])
        y = np.linspace(0, 2, 30)
        posterior = fit_bayesian_linear_regression(X, y)

        first = posterior.sample_predictive(
            np.array([1.0, 0.5]), 100, rng=np.random.default_rng(5)
        )
        second = posterior.sample_predictive(
            np.array([1.0, 0.5]), 100, rng=np.random.default_rng(5)
        )

        np.testing.assert_allclose(first, second)

    def test_rejects_mismatched_predictor_length(self, rng) -> None:
        X = np.column_stack([np.ones(20), rng.normal(size=20)])
        posterior = fit_bayesian_linear_regression(X, rng.normal(size=20))

        with pytest.raises(ValueError, match="expects"):
            posterior.sample_predictive(np.array([1.0, 2.0, 3.0]), 10, rng=rng)

    def test_rejects_non_finite_inputs(self) -> None:
        X = np.array([[1.0, np.nan], [1.0, 2.0]])
        with pytest.raises(ValueError, match="non-finite"):
            fit_bayesian_linear_regression(X, np.array([1.0, 2.0]))

    def test_rejects_row_count_mismatch(self) -> None:
        with pytest.raises(ValueError, match="same number of rows"):
            fit_bayesian_linear_regression(np.ones((5, 2)), np.ones(4))


class TestSummarizeSamples:
    def test_reports_expected_quantile_keys_and_values(self) -> None:
        samples = np.linspace(0.0, 100.0, 100_001)
        stats = summarize_samples(samples, credible_mass=0.90)

        assert set(stats) == {"mean", "median", "sd", "q5", "q95"}
        assert stats["median"] == pytest.approx(50.0, abs=0.1)
        assert stats["q5"] == pytest.approx(5.0, abs=0.1)
        assert stats["q95"] == pytest.approx(95.0, abs=0.1)

    def test_credible_mass_changes_the_reported_keys(self) -> None:
        stats = summarize_samples(np.arange(1000.0), credible_mass=0.50)
        assert {"q25", "q75"}.issubset(stats)

    def test_rejects_empty_samples(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            summarize_samples(np.array([]))

    @pytest.mark.parametrize("mass", [0.0, 1.0, 1.5])
    def test_rejects_out_of_range_credible_mass(self, mass: float) -> None:
        with pytest.raises(ValueError, match="credible_mass"):
            summarize_samples(np.arange(10.0), credible_mass=mass)
