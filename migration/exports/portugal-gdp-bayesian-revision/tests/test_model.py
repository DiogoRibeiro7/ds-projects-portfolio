"""Tests for the population, GDP and labour-channel model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pt_gdp_bayes.model import (
    _beta_draw,
    _lognormal_draw,
    backtest_gdp_model,
    capture_sensitivity_curve,
    check_extrapolation,
    estimate_gdp_posterior_samples,
    estimate_population_posterior_samples,
    estimate_pps_index_samples,
    pps_index_samples,
    score_against_outturn,
    simulate_unmeasured_output_samples,
)


class TestCheckExtrapolation:
    def test_returns_none_inside_the_fitted_range(self) -> None:
        assert check_extrapolation("x", 0.5, np.array([0.0, 1.0])) is None

    def test_flags_a_value_far_outside_the_fitted_range(self) -> None:
        message = check_extrapolation("population growth", 0.066, np.linspace(-0.006, 0.014, 50))
        assert message is not None
        assert "population growth" in message
        assert "outside the fitted range" in message

    def test_margin_tolerates_a_small_overshoot(self) -> None:
        training = np.array([0.0, 1.0])
        assert check_extrapolation("x", 1.2, training, margin=0.25) is None
        assert check_extrapolation("x", 1.5, training, margin=0.25) is not None

    def test_handles_degenerate_training_data(self) -> None:
        assert check_extrapolation("x", 5.0, np.array([])) is None
        assert check_extrapolation("x", 5.0, np.array([1.0, 1.0])) is None

    def test_catches_the_cross_vintage_population_revision(self) -> None:
        """The regression test for the bug this guard exists to prevent.

        Portuguese population growth has always been within roughly +/-1.5% a
        year. A cross-vintage level revision entering as one year of growth
        shows up as 6.6%, and must not pass silently.
        """
        historical_growth = np.random.default_rng(0).normal(0.003, 0.004, 60)
        assert check_extrapolation("population growth", 0.066, historical_growth) is not None


class TestDrawHelpers:
    def test_beta_draw_preserves_the_requested_mean(self, rng) -> None:
        draws = _beta_draw(0.87, 0.12, 200_000, rng)
        assert draws.mean() == pytest.approx(0.87, abs=0.005)

    def test_beta_draw_stays_inside_the_unit_interval(self, rng) -> None:
        draws = _beta_draw(0.95, 0.30, 100_000, rng)
        assert draws.min() >= 0.0
        assert draws.max() <= 1.0

    def test_beta_draw_beats_a_clipped_normal_on_bias(self, rng) -> None:
        """A truncated normal shifts the mean where the clip bites; Beta does not."""
        mean, spread, n = 0.92, 0.15, 200_000
        clipped = np.clip(rng.normal(mean, mean * spread, n), 0.0, 1.0)
        beta = _beta_draw(mean, spread, n, rng)

        assert abs(beta.mean() - mean) < abs(clipped.mean() - mean)

    def test_beta_draw_is_degenerate_at_the_boundaries(self, rng) -> None:
        assert (_beta_draw(0.0, 0.2, 100, rng) == 0.0).all()
        assert (_beta_draw(1.0, 0.2, 100, rng) == 1.0).all()
        assert (_beta_draw(0.5, 0.0, 100, rng) == 0.5).all()

    def test_beta_draw_caps_an_impossible_spread(self, rng) -> None:
        draws = _beta_draw(0.5, 5.0, 10_000, rng)
        assert np.isfinite(draws).all()
        assert draws.min() >= 0.0 and draws.max() <= 1.0

    @pytest.mark.parametrize("mean", [-0.1, 1.1])
    def test_beta_draw_rejects_means_outside_the_unit_interval(self, mean: float, rng) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            _beta_draw(mean, 0.1, 10, rng)

    def test_lognormal_draw_is_positive_and_centred_on_its_median(self, rng) -> None:
        draws = _lognormal_draw(0.8, 0.12, 200_000, rng)
        assert (draws > 0).all()
        assert np.median(draws) == pytest.approx(0.8, rel=0.01)

    def test_lognormal_draw_may_exceed_one(self, rng) -> None:
        """Relative productivity is not bounded above, unlike a rate."""
        assert _lognormal_draw(1.0, 0.2, 10_000, rng).max() > 1.0


class TestUnmeasuredOutput:
    BASE = {
        "extra_residents": 600_000.0,
        "baseline_output_per_employed": 56_000.0,
        "working_age_share": 0.85,
        "participation_rate": 0.87,
        "employment_rate": 0.88,
        "productivity_relative": 0.80,
        "n_samples": 20_000,
    }

    def test_full_capture_produces_no_extra_output(self, rng) -> None:
        """The double-count guard: workers already in the accounts add nothing."""
        draws = simulate_unmeasured_output_samples(**self.BASE, statistical_capture=1.0, rng=rng)
        assert (draws == 0.0).all()

    def test_zero_capture_reproduces_the_gross_output(self, rng) -> None:
        draws = simulate_unmeasured_output_samples(**self.BASE, statistical_capture=0.0, rng=rng)
        expected = (
            600_000.0 * 0.85 * 0.87 * 0.88 * 56_000.0 * 0.80
        )
        assert draws.mean() == pytest.approx(expected, rel=0.02)

    def test_output_falls_monotonically_as_capture_rises(self, rng) -> None:
        means = [
            simulate_unmeasured_output_samples(
                **self.BASE, statistical_capture=c, rng=np.random.default_rng(3)
            ).mean()
            for c in [0.0, 0.25, 0.5, 0.75, 1.0]
        ]
        assert all(earlier > later for earlier, later in zip(means, means[1:]))

    def test_no_extra_residents_means_no_extra_output(self, rng) -> None:
        params = {**self.BASE, "extra_residents": 0.0}
        draws = simulate_unmeasured_output_samples(**params, statistical_capture=0.0, rng=rng)
        assert (draws == 0.0).all()

    def test_draws_are_non_negative(self, rng) -> None:
        draws = simulate_unmeasured_output_samples(**self.BASE, statistical_capture=0.5, rng=rng)
        assert (draws >= 0.0).all()

    @pytest.mark.parametrize("capture", [-0.1, 1.1])
    def test_rejects_capture_outside_the_unit_interval(self, capture: float, rng) -> None:
        with pytest.raises(ValueError, match="statistical_capture"):
            simulate_unmeasured_output_samples(**self.BASE, statistical_capture=capture, rng=rng)

    def test_rejects_negative_extra_residents(self, rng) -> None:
        params = {**self.BASE, "extra_residents": -1.0}
        with pytest.raises(ValueError, match="extra_residents"):
            simulate_unmeasured_output_samples(**params, statistical_capture=0.5, rng=rng)


class TestPpsIndexSamplesFromComponents:
    def test_reproduces_the_published_portugal_index(self) -> None:
        """Built from the definition, the 2025 index on the accounts population
        must reproduce Eurostat's published 81.4."""
        n = 1_000
        index = pps_index_samples(
            gdp_samples=np.full(n, 306_749.6e6),
            population_samples=np.full(n, 10_804_160.0),
            purchasing_power_parity=28_390.0 / 33_825.0,
            reference_gdp_pc_pps=41_565.7,
        )
        assert np.median(index) == pytest.approx(81.4, abs=0.1)

    def test_on_the_revised_population_it_gives_the_corrected_index(self) -> None:
        n = 1_000
        index = pps_index_samples(
            gdp_samples=np.full(n, 306_749.6e6),
            population_samples=np.full(n, 11_405_627.0),
            purchasing_power_parity=28_390.0 / 33_825.0,
            reference_gdp_pc_pps=41_565.7,
        )
        assert np.median(index) == pytest.approx(77.1, abs=0.1)

    def test_a_larger_population_lowers_the_index(self) -> None:
        n = 500
        kwargs = {
            "gdp_samples": np.full(n, 300e9),
            "purchasing_power_parity": 0.84,
            "reference_gdp_pc_pps": 41_000.0,
        }
        small = pps_index_samples(population_samples=np.full(n, 1.0e7), **kwargs)
        large = pps_index_samples(population_samples=np.full(n, 1.1e7), **kwargs)
        assert np.median(large) < np.median(small)

    def test_rejects_mismatched_sample_lengths(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            pps_index_samples(
                gdp_samples=np.full(10, 300e9), population_samples=np.full(5, 1e7),
                purchasing_power_parity=0.84, reference_gdp_pc_pps=41_000.0,
            )

    @pytest.mark.parametrize("ppp,ref", [(0.0, 41_000.0), (0.84, 0.0)])
    def test_rejects_non_positive_scaling(self, ppp: float, ref: float) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            pps_index_samples(
                gdp_samples=np.full(10, 300e9), population_samples=np.full(10, 1e7),
                purchasing_power_parity=ppp, reference_gdp_pc_pps=ref,
            )


class TestCaptureSensitivityCurve:
    BASE = {
        "extra_residents": 601_467.0,
        "baseline_output_per_employed": 56_837.0,
        "working_age_share": 0.85,
        "participation_rate": 0.87,
        "employment_rate": 0.88,
        "productivity_relative": 0.80,
        "purchasing_power_parity": 28_390.0 / 33_825.0,
        "reference_gdp_pc_pps": 41_565.7,
        "n_samples": 5_000,
    }

    def _curve(self, capture_values):
        n = self.BASE["n_samples"]
        # The population carries posterior spread, as it does in the pipeline, so
        # the index is a distribution at every capture value rather than a point.
        population = np.random.default_rng(5).normal(11_405_627.0, 42_000.0, n)
        return capture_sensitivity_curve(
            capture_values=capture_values,
            measured_gdp_samples=np.full(n, 306_749.6e6),
            population_samples=population,
            **self.BASE,
        )

    def test_index_decreases_as_capture_rises(self) -> None:
        curve = self._curve(np.linspace(0.0, 1.0, 5))

        assert curve["pps_median"].is_monotonic_decreasing
        assert curve["extra_gdp_bn_eur"].is_monotonic_decreasing
        assert (curve["pps_q5"] < curve["pps_q95"]).all()

    def test_full_capture_is_the_pure_denominator_correction(self) -> None:
        curve = self._curve(np.array([1.0]))

        assert curve.loc[0, "extra_gdp_bn_eur"] == pytest.approx(0.0)
        assert curve.loc[0, "pps_median"] == pytest.approx(77.1, abs=0.15)

    def test_zero_capture_roughly_offsets_the_denominator(self) -> None:
        curve = self._curve(np.array([0.0]))

        assert curve.loc[0, "extra_gdp_bn_eur"] == pytest.approx(18.0, abs=1.5)
        assert curve.loc[0, "pps_median"] == pytest.approx(81.6, abs=0.4)


class TestScoreAgainstOutturn:
    def test_scores_a_centred_posterior_as_accurate_and_covered(self) -> None:
        samples = np.random.default_rng(2).normal(100.0, 5.0, 50_000)
        score = score_against_outturn(samples, 100.0)

        assert score.pct_error == pytest.approx(0.0, abs=0.5)
        assert score.in_interval is True
        assert score.pit == pytest.approx(0.5, abs=0.02)

    def test_flags_a_posterior_that_sits_entirely_below_the_truth(self) -> None:
        samples = np.random.default_rng(2).normal(100.0, 1.0, 20_000)
        score = score_against_outturn(samples, 130.0)

        assert score.in_interval is False
        assert score.pit == pytest.approx(1.0)
        assert score.pct_error < 0

    def test_pit_is_near_zero_when_the_posterior_sits_above_the_truth(self) -> None:
        samples = np.random.default_rng(4).normal(100.0, 1.0, 20_000)
        assert score_against_outturn(samples, 70.0).pit == pytest.approx(0.0)

    def test_wider_credible_mass_covers_more(self) -> None:
        samples = np.random.default_rng(6).normal(100.0, 5.0, 50_000)
        narrow = score_against_outturn(samples, 108.0, credible_mass=0.50)
        wide = score_against_outturn(samples, 108.0, credible_mass=0.99)

        assert narrow.in_interval is False
        assert wide.in_interval is True

    def test_summary_mentions_the_interval_verdict(self) -> None:
        samples = np.random.default_rng(7).normal(100.0, 5.0, 10_000)
        assert "inside" in score_against_outturn(samples, 100.0).summary()
        assert "OUTSIDE" in score_against_outturn(samples, 500.0).summary()

    def test_rejects_empty_samples(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            score_against_outturn(np.array([]), 1.0)


class TestPopulationPosterior:
    def test_posterior_lies_between_the_prior_and_the_observation(self, synthetic_panel) -> None:
        """The defining property of the update, and the one the original model lost.

        Anchoring hard to the observation regardless of the prior, or ignoring
        the observation entirely, both indicate the two inputs are not describing
        the same quantity. A posterior strictly between them is the signature of
        a genuine reconciliation.
        """
        observed = float(synthetic_panel["population"].iloc[-1]) * 1.01
        result = estimate_population_posterior_samples(
            synthetic_panel,
            target_year=2025,
            observed_population=observed,
            n_samples=5_000,
        )

        low, high = sorted([result.prior_mean, observed])
        assert low <= result.posterior_mean <= high
        assert len(result.samples) == 5_000
        assert len(result.demographic_growth_samples) == 5_000

    def test_posterior_is_sharper_than_the_prior(self, synthetic_panel) -> None:
        observed = float(synthetic_panel["population"].iloc[-1]) * 1.01
        result = estimate_population_posterior_samples(
            synthetic_panel, target_year=2025, observed_population=observed, n_samples=5_000
        )

        assert result.posterior_sd < result.prior_sd

    def test_demographic_growth_draws_are_plausible(self, synthetic_panel) -> None:
        result = estimate_population_posterior_samples(
            synthetic_panel,
            target_year=2025,
            observed_population=float(synthetic_panel["population"].iloc[-1]) * 1.005,
            n_samples=5_000,
        )
        # The synthetic series grows at 0.4-0.6% a year.
        assert np.median(result.demographic_growth_samples) == pytest.approx(0.005, abs=0.004)

    def test_diagnostics_are_empty_for_a_consistent_vintage(self, synthetic_panel) -> None:
        result = estimate_population_posterior_samples(
            synthetic_panel,
            target_year=2025,
            observed_population=float(synthetic_panel["population"].iloc[-1]) * 1.005,
            n_samples=2_000,
        )
        assert result.diagnostics == ()

    def test_diagnostics_flag_a_cross_vintage_observation(self, synthetic_panel) -> None:
        """A 6% jump cannot be one year of demographic growth."""
        result = estimate_population_posterior_samples(
            synthetic_panel,
            target_year=2025,
            observed_population=float(synthetic_panel["population"].iloc[-1]) * 1.06,
            n_samples=2_000,
        )
        assert result.diagnostics
        assert "vintage mismatch" in result.diagnostics[0]

    def test_a_tighter_observation_sd_pulls_the_posterior_to_the_observation(
        self, synthetic_panel
    ) -> None:
        observed = float(synthetic_panel["population"].iloc[-1]) * 1.03
        tight = estimate_population_posterior_samples(
            synthetic_panel, target_year=2025, observed_population=observed,
            observation_relative_sd=0.0005, n_samples=3_000,
        )
        loose = estimate_population_posterior_samples(
            synthetic_panel, target_year=2025, observed_population=observed,
            observation_relative_sd=0.05, n_samples=3_000,
        )

        assert abs(tight.posterior_mean - observed) < abs(loose.posterior_mean - observed)

    @pytest.mark.parametrize("observed,sd", [(-1.0, 0.001), (1e7, 0.0)])
    def test_rejects_invalid_arguments(self, synthetic_panel, observed: float, sd: float) -> None:
        with pytest.raises(ValueError):
            estimate_population_posterior_samples(
                synthetic_panel, target_year=2025,
                observed_population=observed, observation_relative_sd=sd,
            )

    def test_requires_enough_history(self) -> None:
        short = pd.DataFrame({"year": range(2020, 2025), "population": [1e7] * 5})
        with pytest.raises(ValueError, match="At least 10"):
            estimate_population_posterior_samples(
                short, target_year=2025, observed_population=1e7
            )


class TestGdpPosterior:
    def test_recovers_the_nominal_growth_identity_on_clean_data(self, synthetic_panel) -> None:
        n = 4_000
        population = np.full(n, float(synthetic_panel["population"].iloc[-1]) * 1.005)
        result = estimate_gdp_posterior_samples(
            synthetic_panel,
            target_year=2025,
            population_samples=population,
            known_real_gdp_growth=0.02,
            estimation_start_year=None,
            n_samples=n,
        )

        previous_gdp = float(synthetic_panel["gdp_current_eur"].iloc[-1])
        implied = np.median(result.gdp_samples) / previous_gdp - 1.0
        # ~2% real plus a ~2.5% deflator.
        assert implied == pytest.approx(0.045, abs=0.02)

    def test_a_known_real_growth_signal_sharpens_the_real_growth_component(
        self, synthetic_panel
    ) -> None:
        """The signal informs real growth, so that is where the sharpening shows up.

        It cannot sharpen *nominal* growth below the deflator uncertainty, because
        the deflator is unknown in both cases. Asserting on the nominal spread
        instead would pass only for a model that wrongly cancels the shared
        deflator term — the bug this test now guards against.
        """
        n = 8_000
        population = np.full(n, float(synthetic_panel["population"].iloc[-1]) * 1.005)
        kwargs = {
            "target_year": 2025,
            "population_samples": population,
            "estimation_start_year": None,
            "n_samples": n,
        }
        without = estimate_gdp_posterior_samples(synthetic_panel, **kwargs)
        with_signal = estimate_gdp_posterior_samples(
            synthetic_panel, known_real_gdp_growth=0.02, **kwargs
        )

        def real_growth_sd(result) -> float:
            return float(
                (result.gdp_growth_samples - result.deflator_growth_samples).std()
            )

        assert real_growth_sd(with_signal) < real_growth_sd(without) / 2.0

    def test_diagnostics_flag_an_extrapolated_population_regressor(self, synthetic_panel) -> None:
        n = 2_000
        population = np.full(n, float(synthetic_panel["population"].iloc[-1]) * 1.07)
        result = estimate_gdp_posterior_samples(
            synthetic_panel, target_year=2025, population_samples=population,
            estimation_start_year=None, n_samples=n,
        )

        assert result.diagnostics
        assert "extrapolating on population growth" in result.diagnostics[0]

    def test_estimation_window_changes_the_posterior(self) -> None:
        """A structural break in the estimation window must actually matter."""
        years = np.arange(1970, 2025)
        # High inflation until 1998, low thereafter.
        deflator = np.where(years < 1999, 12.0, 2.5)
        nominal = np.log1p(deflator / 100.0) + np.log1p(0.02)
        panel = pd.DataFrame(
            {
                "year": years,
                "population": 1e7 * np.exp(np.cumsum(np.full(len(years), 0.004))),
                "gdp_current_eur": 50e9 * np.exp(np.cumsum(nominal)),
                "gdp_deflator_pct": deflator,
            }
        )

        n = 4_000
        population = np.full(n, float(panel["population"].iloc[-1]) * 1.004)
        full = estimate_gdp_posterior_samples(
            panel, target_year=2025, population_samples=population,
            estimation_start_year=None, n_samples=n,
        )
        euro_era = estimate_gdp_posterior_samples(
            panel, target_year=2025, population_samples=population,
            estimation_start_year=1999, n_samples=n,
        )

        previous = float(panel["gdp_current_eur"].iloc[-1])
        assert np.median(full.gdp_samples) / previous > np.median(euro_era.gdp_samples) / previous

    def test_rejects_mismatched_sample_lengths(self, synthetic_panel) -> None:
        with pytest.raises(ValueError, match="population_samples length"):
            estimate_gdp_posterior_samples(
                synthetic_panel, target_year=2025,
                population_samples=np.full(10, 1e7), n_samples=1_000,
            )


class TestBacktest:
    def test_produces_one_scored_row_per_holdout_year(self, synthetic_panel) -> None:
        result = backtest_gdp_model(
            synthetic_panel, holdout_years=[2020, 2021, 2022],
            estimation_start_year=None, n_samples=1_500,
        )

        assert result["year"].tolist() == [2020, 2021, 2022]
        assert {"actual", "predicted_median", "q_low", "q_high", "pct_error", "in_interval"} <= set(
            result.columns
        )
        assert (result["q_low"] < result["q_high"]).all()

    def test_is_accurate_on_a_smooth_series(self, synthetic_panel) -> None:
        result = backtest_gdp_model(
            synthetic_panel, holdout_years=list(range(2015, 2024)),
            estimation_start_year=None, n_samples=2_000,
        )
        assert result["pct_error"].abs().mean() < 5.0

    def test_skips_years_absent_from_the_panel(self, synthetic_panel) -> None:
        result = backtest_gdp_model(
            synthetic_panel, holdout_years=[2020, 2100],
            estimation_start_year=None, n_samples=1_000,
        )
        assert result["year"].tolist() == [2020]

    def test_does_not_leak_the_target_year(self, synthetic_panel) -> None:
        """Corrupting a target year's GDP must not change its own prediction."""
        baseline = backtest_gdp_model(
            synthetic_panel, holdout_years=[2022], estimation_start_year=None, n_samples=2_000,
        )

        corrupted = synthetic_panel.copy()
        mask = corrupted["year"] == 2022
        corrupted.loc[mask, "gdp_current_eur"] *= 10.0
        after = backtest_gdp_model(
            corrupted, holdout_years=[2022], estimation_start_year=None, n_samples=2_000,
        )

        assert after["predicted_median"].iloc[0] == pytest.approx(
            baseline["predicted_median"].iloc[0], rel=1e-9
        )


class TestPpsIndexSamples:
    def test_centres_on_the_mechanical_correction(self) -> None:
        samples = estimate_pps_index_samples(
            preliminary_index=81.3, population_correction_factor=1.056, n_samples=20_000
        )
        assert np.median(samples) == pytest.approx(81.3 / 1.056, rel=0.01)

    def test_a_unit_correction_factor_leaves_the_index_alone(self) -> None:
        samples = estimate_pps_index_samples(
            preliminary_index=81.3, population_correction_factor=1.0, n_samples=20_000
        )
        assert np.median(samples) == pytest.approx(81.3, rel=0.01)

    @pytest.mark.parametrize("index,factor", [(0.0, 1.05), (81.3, 0.0), (-1.0, 1.05)])
    def test_rejects_non_positive_inputs(self, index: float, factor: float) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            estimate_pps_index_samples(
                preliminary_index=index, population_correction_factor=factor
            )

    def test_rejects_mismatched_revision_sample_lengths(self) -> None:
        with pytest.raises(ValueError, match="length must equal"):
            estimate_pps_index_samples(
                preliminary_index=81.3, population_correction_factor=1.056,
                gdp_revision_factor_samples=np.ones(10), n_samples=1_000,
            )


class TestSharedDeflatorHandling:
    """The regression and the direct signal share the same deflator draws.

    Combining them as independent estimates of *nominal* growth cancels that
    shared term and yields a posterior narrower than the deflator uncertainty
    alone — which is impossible, since neither input pins the deflator down.
    """

    def test_posterior_is_never_narrower_than_the_deflator_it_contains(
        self, synthetic_panel
    ) -> None:
        n = 8_000
        population = np.full(n, float(synthetic_panel["population"].iloc[-1]) * 1.005)
        result = estimate_gdp_posterior_samples(
            synthetic_panel, target_year=2025, population_samples=population,
            known_real_gdp_growth=0.02, estimation_start_year=None, n_samples=n,
        )

        assert result.gdp_growth_samples.std() >= result.deflator_growth_samples.std() * 0.99

    def test_a_sharper_real_growth_signal_does_not_shrink_below_the_deflator(
        self, synthetic_panel
    ) -> None:
        n = 8_000
        population = np.full(n, float(synthetic_panel["population"].iloc[-1]) * 1.005)
        result = estimate_gdp_posterior_samples(
            synthetic_panel, target_year=2025, population_samples=population,
            known_real_gdp_growth=0.02, real_growth_observation_sd=1e-6,
            estimation_start_year=None, n_samples=n,
        )

        # Even with real growth known exactly, deflator uncertainty must survive.
        assert result.gdp_growth_samples.std() >= result.deflator_growth_samples.std() * 0.99


class TestYearGapDiagnostic:
    def test_flags_a_lagging_input_series(self, synthetic_panel) -> None:
        lagging = synthetic_panel[synthetic_panel["year"] != 2024].copy()
        result = estimate_gdp_posterior_samples(
            lagging, target_year=2025, population_samples=np.full(2_000, 1.1e7),
            estimation_start_year=None, n_samples=2_000,
        )

        assert any("2-year gap" in message for message in result.diagnostics)

    def test_silent_when_the_panel_reaches_the_previous_year(self, synthetic_panel) -> None:
        result = estimate_gdp_posterior_samples(
            synthetic_panel, target_year=2025,
            population_samples=np.full(2_000, float(synthetic_panel["population"].iloc[-1])),
            estimation_start_year=None, n_samples=2_000,
        )

        assert not any("gap" in message for message in result.diagnostics)


class TestNonFiniteGuards:
    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_pps_index_rejects_non_finite_scaling(self, bad: float) -> None:
        with pytest.raises(ValueError, match="finite"):
            pps_index_samples(
                gdp_samples=np.full(10, 300e9), population_samples=np.full(10, 1e7),
                purchasing_power_parity=bad, reference_gdp_pc_pps=41_000.0,
            )

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_estimate_pps_index_rejects_non_finite_inputs(self, bad: float) -> None:
        with pytest.raises(ValueError, match="finite"):
            estimate_pps_index_samples(
                preliminary_index=bad, population_correction_factor=1.056, n_samples=100
            )
