"""
Comprehensive unit tests for the statistics.core module.

This module provides extensive testing for all statistical functions
with edge cases, error conditions, and proper coverage.
"""

import math
import warnings
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.statistics.core import (
    DEFAULT_ALPHA,
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_POWER,
    ExperimentAnalyzer,
    apply_multiple_testing_correction,
    bootstrap_ci_diff,
    calculate_power,
    calculate_sample_size,
    sequential_testing_boundary,
    two_prop_ztest,
)

pytestmark = pytest.mark.unit


class TestTwoPropZTest:
    """Test suite for two-proportion z-test function."""

    def test_equal_proportions(self):
        """Test when both groups have the same proportion."""
        z_stat, p_value = two_prop_ztest(50, 100, 50, 100)
        assert abs(z_stat) < 0.01
        assert p_value > 0.95

    def test_different_proportions(self):
        """Test when groups have significantly different proportions."""
        z_stat, p_value = two_prop_ztest(20, 100, 40, 100)
        assert z_stat < -2.0  # Negative because group 1 < group 2
        assert p_value < 0.05

    def test_continuity_correction(self):
        """Test continuity correction effect."""
        # Without continuity correction
        z_no_correction, _ = two_prop_ztest(10, 50, 20, 50, continuity_correction=False)

        # With continuity correction
        z_with_correction, _ = two_prop_ztest(
            10, 50, 20, 50, continuity_correction=True
        )

        # Continuity correction should reduce the absolute z-statistic
        assert abs(z_with_correction) < abs(z_no_correction)

    def test_alternative_hypotheses(self):
        """Test different alternative hypotheses."""
        x1, n1, x2, n2 = 30, 100, 40, 100

        # Two-sided test
        _, p_two = two_prop_ztest(x1, n1, x2, n2, alternative="two-sided")

        # One-sided tests
        _, p_larger = two_prop_ztest(x1, n1, x2, n2, alternative="larger")
        _, p_smaller = two_prop_ztest(x1, n1, x2, n2, alternative="smaller")

        # Two-sided p-value should be approximately twice the smaller one-sided p-value
        assert p_two > p_larger
        assert p_two > p_smaller

    def test_edge_cases_zero_successes(self):
        """Test when both groups have zero successes."""
        z_stat, p_value = two_prop_ztest(0, 100, 0, 100)
        assert z_stat == 0.0
        assert p_value == 1.0

    def test_edge_cases_all_successes(self):
        """Test when both groups have 100% success rate."""
        z_stat, p_value = two_prop_ztest(100, 100, 100, 100)
        assert z_stat == 0.0
        assert p_value == 1.0

    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        # Negative sample size
        with pytest.raises(ValueError, match="Sample sizes must be positive"):
            two_prop_ztest(10, -1, 20, 50)

        # Negative success count
        with pytest.raises(ValueError, match="Success counts cannot be negative"):
            two_prop_ztest(-10, 100, 20, 100)

        # Success count > sample size
        with pytest.raises(
            ValueError, match="Success counts cannot exceed sample sizes"
        ):
            two_prop_ztest(150, 100, 50, 100)

    def test_extreme_difference(self):
        """Test extreme differences between proportions."""
        z_stat, p_value = two_prop_ztest(0, 100, 100, 100)
        assert abs(z_stat) > 10  # Should be a very large z-statistic
        assert p_value < 0.001


class TestBootstrapCI:
    """Test suite for bootstrap confidence interval function."""

    def test_basic_ci(self):
        """Test basic confidence interval calculation."""
        ci_lower, ci_upper = bootstrap_ci_diff(
            0.5, 0.6, 100, 100, B=1000, random_state=42
        )

        # CI should contain the true difference (0.1)
        assert ci_lower < 0.1 < ci_upper
        # CI should be reasonable width
        assert 0.0 < ci_upper - ci_lower < 0.5

    def test_percentile_method(self):
        """Test percentile bootstrap method."""
        ci_lower, ci_upper = bootstrap_ci_diff(
            0.3, 0.4, 200, 200, B=2000, method="percentile", random_state=42
        )

        assert ci_lower < 0.1 < ci_upper
        assert ci_lower > -0.1  # Should be close to true difference

    def test_bca_method(self):
        """Test bias-corrected accelerated bootstrap method."""
        ci_lower, ci_upper = bootstrap_ci_diff(
            0.3, 0.4, 200, 200, B=2000, method="bca", random_state=42
        )

        assert ci_lower < 0.1 < ci_upper
        # BCA should give tighter intervals than percentile
        assert ci_upper - ci_lower < 0.5

    def test_edge_case_zero_proportion(self):
        """Test when one proportion is zero."""
        ci_lower, ci_upper = bootstrap_ci_diff(
            0.0, 0.1, 100, 100, B=1000, random_state=42
        )

        assert ci_lower < 0.1
        assert ci_upper > 0.0

    def test_edge_case_one_proportion(self):
        """Test when one proportion is one."""
        ci_lower, ci_upper = bootstrap_ci_diff(
            0.9, 1.0, 100, 100, B=1000, random_state=42
        )

        assert ci_lower < 0.1
        assert ci_upper > 0.0

    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        # Invalid proportions
        with pytest.raises(ValueError, match="Proportions must be between 0 and 1"):
            bootstrap_ci_diff(1.5, 0.5, 100, 100)

        # Negative sample size
        with pytest.raises(ValueError, match="Sample sizes must be positive"):
            bootstrap_ci_diff(0.5, 0.5, -100, 100)

        # Invalid bootstrap samples
        with pytest.raises(
            ValueError, match="Number of bootstrap samples must be positive"
        ):
            bootstrap_ci_diff(0.5, 0.5, 100, 100, B=0)

    def test_different_alpha_levels(self):
        """Test different confidence levels."""
        # 95% CI
        ci95_lower, ci95_upper = bootstrap_ci_diff(
            0.5, 0.6, 100, 100, alpha=0.05, random_state=42
        )

        # 99% CI
        ci99_lower, ci99_upper = bootstrap_ci_diff(
            0.5, 0.6, 100, 100, alpha=0.01, random_state=42
        )

        # 99% CI should be wider than 95% CI
        assert ci99_lower < ci95_lower
        assert ci99_upper > ci95_upper


class TestSampleSizeCalculation:
    """Test suite for sample size calculation."""

    def test_basic_calculation(self):
        """Test basic sample size calculation."""
        n = calculate_sample_size(baseline_rate=0.1, mde=0.02, alpha=0.05, power=0.8)

        assert isinstance(n, int)
        assert n > 0
        assert n < 100000  # Reasonable upper bound

    def test_larger_effect_smaller_sample(self):
        """Test that larger effects require smaller samples."""
        n_small_effect = calculate_sample_size(0.1, 0.01)
        n_large_effect = calculate_sample_size(0.1, 0.05)

        assert n_large_effect < n_small_effect

    def test_higher_power_larger_sample(self):
        """Test that higher power requires larger samples."""
        n_low_power = calculate_sample_size(0.1, 0.02, power=0.7)
        n_high_power = calculate_sample_size(0.1, 0.02, power=0.9)

        assert n_high_power > n_low_power

    def test_one_sided_vs_two_sided(self):
        """Test one-sided vs two-sided tests."""
        n_two_sided = calculate_sample_size(0.1, 0.02, alternative="two-sided")
        n_one_sided = calculate_sample_size(0.1, 0.02, alternative="one-sided")

        # One-sided tests require smaller samples
        assert n_one_sided < n_two_sided

    def test_unequal_allocation(self):
        """Test unequal group allocation."""
        n_equal = calculate_sample_size(0.1, 0.02, ratio=1.0)
        n_unequal = calculate_sample_size(0.1, 0.02, ratio=2.0)

        # Unequal allocation requires more samples in control group
        assert n_unequal != n_equal

    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        # Invalid baseline rate
        with pytest.raises(ValueError, match="Baseline rate must be between 0 and 1"):
            calculate_sample_size(-0.1, 0.02)

        # Invalid MDE
        with pytest.raises(
            ValueError, match="Minimum detectable effect must be positive"
        ):
            calculate_sample_size(0.1, -0.02)

        # Baseline + MDE > 1
        with pytest.raises(ValueError, match="Baseline rate \\+ MDE cannot exceed 1"):
            calculate_sample_size(0.9, 0.2)

        # Invalid alpha
        with pytest.raises(ValueError, match="Alpha must be between 0 and 1"):
            calculate_sample_size(0.1, 0.02, alpha=1.5)

        # Invalid power
        with pytest.raises(ValueError, match="Power must be between 0 and 1"):
            calculate_sample_size(0.1, 0.02, power=1.5)


class TestPowerCalculation:
    """Test suite for power calculation."""

    def test_basic_power(self):
        """Test basic power calculation."""
        power = calculate_power(
            n_control=1000, n_treatment=1000, baseline_rate=0.1, effect_size=0.02
        )

        assert 0 <= power <= 1
        assert power > 0.5  # Should have reasonable power

    def test_larger_sample_higher_power(self):
        """Test that larger samples give higher power."""
        power_small = calculate_power(100, 100, 0.1, 0.05)
        power_large = calculate_power(1000, 1000, 0.1, 0.05)

        assert power_large > power_small

    def test_larger_effect_higher_power(self):
        """Test that larger effects give higher power."""
        power_small_effect = calculate_power(500, 500, 0.1, 0.01)
        power_large_effect = calculate_power(500, 500, 0.1, 0.05)

        assert power_large_effect > power_small_effect

    def test_zero_effect_low_power(self):
        """Test that zero effect gives power approximately equal to alpha."""
        power = calculate_power(1000, 1000, 0.1, 0.0, alpha=0.05)

        # Power should be close to alpha for null effect
        assert abs(power - 0.05) < 0.1

    def test_one_sided_vs_two_sided_power(self):
        """Test power for one-sided vs two-sided tests."""
        power_two = calculate_power(500, 500, 0.1, 0.02, alternative="two-sided")
        power_one = calculate_power(500, 500, 0.1, 0.02, alternative="one-sided")

        # One-sided tests have higher power
        assert power_one > power_two


class TestExperimentAnalyzer:
    """Test suite for ExperimentAnalyzer class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample experiment data."""
        np.random.seed(42)
        n_control = 1000
        n_treatment = 1000

        data = pd.DataFrame(
            {
                "user_id": range(n_control + n_treatment),
                "group": ["control"] * n_control + ["treatment"] * n_treatment,
                "converted": np.concatenate(
                    [
                        np.random.binomial(1, 0.1, n_control),
                        np.random.binomial(1, 0.12, n_treatment),
                    ]
                ),
                "revenue": np.concatenate(
                    [
                        np.random.exponential(10, n_control),
                        np.random.exponential(11, n_treatment),
                    ]
                ),
            }
        )
        return data

    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = ExperimentAnalyzer(alpha=0.05, power=0.8)

        assert analyzer.alpha == 0.05
        assert analyzer.power == 0.8
        assert isinstance(analyzer.results_cache, dict)

    def test_srm_check_balanced(self, sample_data):
        """Test SRM check for balanced groups."""
        analyzer = ExperimentAnalyzer()
        results = analyzer.check_srm(sample_data)

        assert "chi2_statistic" in results
        assert "p_value" in results
        assert "is_srm" in results
        assert not results["is_srm"]  # Should not detect SRM in balanced data

    def test_srm_check_imbalanced(self):
        """Test SRM check for imbalanced groups."""
        # Create imbalanced data
        data = pd.DataFrame({"group": ["control"] * 800 + ["treatment"] * 200})

        analyzer = ExperimentAnalyzer()
        results = analyzer.check_srm(data)

        assert results["is_srm"]  # Should detect SRM

    def test_srm_check_custom_ratio(self):
        """Test SRM check with custom expected ratio."""
        # Create 2:1 ratio data
        data = pd.DataFrame({"group": ["control"] * 667 + ["treatment"] * 333})

        analyzer = ExperimentAnalyzer()
        results = analyzer.check_srm(
            data, expected_ratio={"control": 2, "treatment": 1}
        )

        assert not results["is_srm"]  # Should not detect SRM with correct ratio

    def test_analyze_conversion(self, sample_data):
        """Test conversion analysis."""
        analyzer = ExperimentAnalyzer()
        results = analyzer.analyze_conversion(sample_data, "converted")

        assert "conversion_rates" in results
        assert "sample_sizes" in results
        assert "p_value" in results
        assert "significant" in results
        assert "absolute_lift" in results
        assert "relative_lift" in results
        assert "confidence_interval" in results
        assert "statistical_power" in results

    def test_analyze_conversion_multigroup(self):
        """Test conversion analysis with multiple groups."""
        data = pd.DataFrame(
            {
                "group": ["A"] * 100 + ["B"] * 100 + ["C"] * 100,
                "converted": np.random.binomial(1, 0.1, 300),
            }
        )

        analyzer = ExperimentAnalyzer()
        results = analyzer.analyze_conversion(data, "converted")

        assert "chi2_statistic" in results
        assert "degrees_of_freedom" in results
        assert results["test_type"] == "chi_square"

    def test_analyze_conversion_missing_column(self, sample_data):
        """Test error handling for missing columns."""
        analyzer = ExperimentAnalyzer()

        with pytest.raises(ValueError, match="Conversion column"):
            analyzer.analyze_conversion(sample_data, "nonexistent")

    def test_comprehensive_analysis(self, sample_data):
        """Test comprehensive multi-metric analysis."""
        analyzer = ExperimentAnalyzer()

        results = analyzer.run_comprehensive_analysis(
            sample_data, metrics=["converted", "revenue"], group_col="group"
        )

        assert "data_quality" in results
        assert "metrics_analysis" in results
        assert "recommendations" in results

        # Check data quality section
        assert "srm_check" in results["data_quality"]
        assert "sample_sizes" in results["data_quality"]
        assert "missing_data" in results["data_quality"]

        # Check metrics analysis
        assert "converted" in results["metrics_analysis"]
        assert "revenue" in results["metrics_analysis"]

    def test_single_group_error(self):
        """Test error handling for single group."""
        data = pd.DataFrame(
            {"group": ["control"] * 100, "converted": np.random.binomial(1, 0.1, 100)}
        )

        analyzer = ExperimentAnalyzer()

        with pytest.raises(ValueError, match="Need at least 2 groups"):
            analyzer.check_srm(data)


class TestMultipleTestingCorrection:
    """Test suite for multiple testing correction."""

    def test_bonferroni_correction(self):
        """Test Bonferroni correction."""
        p_values = [0.01, 0.02, 0.03, 0.04, 0.05]

        corrected, rejected = apply_multiple_testing_correction(
            p_values, method="bonferroni"
        )

        # Bonferroni multiplies p-values by number of tests
        assert corrected[0] == pytest.approx(0.05, rel=1e-6)
        assert corrected[1] == pytest.approx(0.10, rel=1e-6)

        # Only first test should be rejected at 0.05 level
        assert rejected[0]
        assert not rejected[1]

    def test_holm_correction(self):
        """Test Holm-Bonferroni correction."""
        p_values = [0.01, 0.02, 0.03, 0.04, 0.05]

        corrected, rejected = apply_multiple_testing_correction(p_values, method="holm")

        # Holm is step-down procedure
        assert len(corrected) == len(p_values)
        assert len(rejected) == len(p_values)

        # Should reject some hypotheses
        assert any(rejected)

    def test_fdr_bh_correction(self):
        """Test Benjamini-Hochberg FDR correction."""
        p_values = [0.001, 0.008, 0.039, 0.041, 0.042]

        corrected, rejected = apply_multiple_testing_correction(
            p_values, method="fdr_bh"
        )

        assert len(corrected) == len(p_values)
        assert len(rejected) == len(p_values)

        # FDR should be less conservative than Bonferroni
        # Should reject at least the smallest p-values
        assert rejected[0]

    def test_invalid_method(self):
        """Test error handling for invalid method."""
        p_values = [0.01, 0.02]

        with pytest.raises(ValueError, match="Unknown correction method"):
            apply_multiple_testing_correction(p_values, method="invalid")

    def test_empty_pvalues(self):
        """Test handling of empty p-value list."""
        corrected, rejected = apply_multiple_testing_correction([], method="holm")

        assert corrected == []
        assert rejected == []


class TestSequentialTesting:
    """Test suite for sequential testing boundaries."""

    def test_obrien_fleming_boundary(self):
        """Test O'Brien-Fleming boundary."""
        boundary = sequential_testing_boundary(
            n=1000, alpha=0.05, method="obrien_fleming"
        )

        assert isinstance(boundary, float)
        assert boundary > 0
        assert boundary < 10  # Reasonable range

    def test_pocock_boundary(self):
        """Test Pocock boundary."""
        boundary = sequential_testing_boundary(n=1000, alpha=0.05, method="pocock")

        assert isinstance(boundary, float)
        assert boundary > 0
        assert boundary < 10  # Reasonable range

    def test_different_alpha_levels(self):
        """Test boundaries with different alpha levels."""
        boundary_05 = sequential_testing_boundary(1000, alpha=0.05)
        boundary_01 = sequential_testing_boundary(1000, alpha=0.01)

        # Smaller alpha should give larger boundary
        assert boundary_01 > boundary_05

    def test_invalid_method(self):
        """Test error handling for invalid method."""
        with pytest.raises(ValueError, match="Unknown boundary method"):
            sequential_testing_boundary(1000, method="invalid")


# Integration tests
class TestIntegration:
    """Integration tests for combined functionality."""

    def test_full_experiment_workflow(self):
        """Test complete experiment analysis workflow."""
        # 1. Calculate required sample size
        n_required = calculate_sample_size(
            baseline_rate=0.1, mde=0.02, alpha=0.05, power=0.8
        )

        # 2. Generate experiment data
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "user_id": range(n_required * 2),
                "group": ["control"] * n_required + ["treatment"] * n_required,
                "converted": np.concatenate(
                    [
                        np.random.binomial(1, 0.10, n_required),
                        np.random.binomial(1, 0.12, n_required),
                    ]
                ),
            }
        )

        # 3. Analyze results
        analyzer = ExperimentAnalyzer(alpha=0.05, power=0.8)

        # Check SRM
        srm_results = analyzer.check_srm(data)
        assert not srm_results["is_srm"]

        # Analyze conversion
        conversion_results = analyzer.analyze_conversion(data, "converted")

        # 4. Calculate actual power
        actual_power = calculate_power(
            n_required,
            n_required,
            conversion_results["conversion_rates"]["control"],
            conversion_results["absolute_lift"],
        )

        assert 0 <= actual_power <= 1

    def test_multiple_metrics_with_correction(self):
        """Test analyzing multiple metrics with correction."""
        # Create data with multiple metrics
        np.random.seed(42)
        n = 1000
        data = pd.DataFrame(
            {
                "group": ["control"] * n + ["treatment"] * n,
                "metric1": np.random.binomial(1, 0.1, 2 * n),
                "metric2": np.random.binomial(1, 0.15, 2 * n),
                "metric3": np.random.binomial(1, 0.2, 2 * n),
            }
        )

        # Analyze each metric
        analyzer = ExperimentAnalyzer()
        p_values = []

        for metric in ["metric1", "metric2", "metric3"]:
            results = analyzer.analyze_conversion(data, metric)
            if "p_value" in results:
                p_values.append(results["p_value"])

        # Apply multiple testing correction
        if p_values:
            corrected, rejected = apply_multiple_testing_correction(
                p_values, method="holm"
            )

            assert len(corrected) == len(p_values)
            assert len(rejected) == len(p_values)


# Performance tests
class TestPerformance:
    """Test performance with larger datasets."""

    @pytest.mark.slow
    def test_large_dataset_bootstrap(self):
        """Test bootstrap with large dataset."""
        # This test is marked as slow and can be skipped with -m "not slow"
        ci_lower, ci_upper = bootstrap_ci_diff(
            0.1, 0.11, 10000, 10000, B=1000, random_state=42
        )

        assert ci_lower < 0.01 < ci_upper

    @pytest.mark.slow
    def test_large_dataset_analyzer(self):
        """Test analyzer with large dataset."""
        # Create large dataset
        n = 100000
        data = pd.DataFrame(
            {
                "group": ["control"] * n + ["treatment"] * n,
                "converted": np.random.binomial(1, 0.1, 2 * n),
            }
        )

        analyzer = ExperimentAnalyzer()
        results = analyzer.analyze_conversion(data, "converted")

        assert "p_value" in results
