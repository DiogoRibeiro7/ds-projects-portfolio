"""Test suite for AB testing module."""

import numpy as np
import pandas as pd
import pytest
from ab_testing import (
    ABTestRunner,
    BayesianABTest,
    MultiArmedBandit,
    PowerAnalysis,
    SampleSizeCalculator,
    SequentialTest,
)
from ab_testing.erfcinv_fix import erfcinv

pytestmark = pytest.mark.integration


class TestABTestRunner:
    """Test suite for AB test runner."""

    @pytest.fixture
    def ab_test(self):
        """Create AB test instance."""
        return ABTestRunner(
            control_name="Control", treatment_name="Treatment", metric_type="conversion"
        )

    @pytest.fixture
    def sample_data(self):
        """Create sample AB test data."""
        np.random.seed(42)
        n_control = 1000
        n_treatment = 1000

        return pd.DataFrame(
            {
                "user_id": range(n_control + n_treatment),
                "group": ["control"] * n_control + ["treatment"] * n_treatment,
                "converted": np.concatenate(
                    [
                        np.random.binomial(1, 0.10, n_control),
                        np.random.binomial(1, 0.12, n_treatment),
                    ]
                ),
                "revenue": np.concatenate(
                    [
                        np.random.exponential(50, n_control),
                        np.random.exponential(55, n_treatment),
                    ]
                ),
            }
        )

    def test_initialization(self, ab_test):
        """Test AB test initialization."""
        assert ab_test.control_name == "Control"
        assert ab_test.treatment_name == "Treatment"
        assert ab_test.metric_type == "conversion"

    def test_run_test_conversion(self, ab_test, sample_data):
        """Test running conversion rate test."""
        results = ab_test.run_test(
            data=sample_data, metric_column="converted", group_column="group"
        )

        assert "p_value" in results
        assert "confidence_interval" in results
        assert "effect_size" in results
        assert "statistical_power" in results
        assert "is_significant" in results

    def test_run_test_continuous(self, sample_data):
        """Test running continuous metric test."""
        ab_test = ABTestRunner(metric_type="continuous")

        results = ab_test.run_test(
            data=sample_data, metric_column="revenue", group_column="group"
        )

        assert "mean_control" in results
        assert "mean_treatment" in results
        assert "t_statistic" in results
        assert "p_value" in results

    def test_minimum_detectable_effect(self, ab_test):
        """Test MDE calculation."""
        mde = ab_test.calculate_mde(
            baseline_rate=0.10, sample_size=1000, alpha=0.05, power=0.80
        )

        assert mde > 0
        assert mde < 0.1  # Should be reasonable

    def test_confidence_intervals(self, ab_test, sample_data):
        """Test confidence interval calculation."""
        results = ab_test.run_test(sample_data, "converted", "group")

        ci = results["confidence_interval"]
        assert len(ci) == 2
        assert ci[0] < ci[1]
        assert ci[0] < results["effect_size"] < ci[1]


class TestBayesianABTest:
    """Test suite for Bayesian AB testing."""

    @pytest.fixture
    def bayesian_test(self):
        """Create Bayesian AB test instance."""
        return BayesianABTest(prior_alpha=1, prior_beta=1)

    def test_beta_binomial_model(self, bayesian_test):
        """Test Beta-Binomial model for conversion rates."""
        results = bayesian_test.analyze_conversion(
            successes_a=120, trials_a=1000, successes_b=150, trials_b=1000
        )

        assert "probability_b_better" in results
        assert "expected_loss_a" in results
        assert "expected_loss_b" in results
        assert "credible_interval_a" in results
        assert "credible_interval_b" in results

        # B should be better with high probability
        assert results["probability_b_better"] > 0.9

    def test_normal_model(self, bayesian_test):
        """Test Normal model for continuous metrics."""
        data_a = np.random.normal(100, 20, 1000)
        data_b = np.random.normal(105, 20, 1000)

        results = bayesian_test.analyze_continuous(data_a, data_b)

        assert "probability_b_better" in results
        assert "difference_credible_interval" in results
        assert results["probability_b_better"] > 0.9

    def test_early_stopping(self, bayesian_test):
        """Test early stopping based on posterior probabilities."""
        # Strong evidence for B
        should_stop = bayesian_test.check_early_stopping(
            successes_a=10, trials_a=100, successes_b=25, trials_b=100, threshold=0.95
        )

        assert should_stop["stop"]
        assert should_stop["winner"] == "B"

    def test_sample_size_recommendation(self, bayesian_test):
        """Test Bayesian sample size recommendation."""
        n_recommended = bayesian_test.recommend_sample_size(
            baseline_rate=0.10, minimum_effect=0.02, confidence_threshold=0.95
        )

        assert n_recommended > 0
        assert n_recommended < 100000


class TestMultiArmedBandit:
    """Test suite for Multi-Armed Bandit algorithms."""

    @pytest.fixture
    def bandit(self):
        """Create MAB instance."""
        return MultiArmedBandit(n_arms=3, algorithm="epsilon_greedy", epsilon=0.1)

    def test_epsilon_greedy(self, bandit):
        """Test epsilon-greedy algorithm."""
        # Simulate rewards
        for _ in range(100):
            arm = bandit.select_arm()
            reward = np.random.binomial(1, 0.1 + 0.1 * arm)
            bandit.update(arm, reward)

        # Best arm should be selected more often
        selections = bandit.get_arm_statistics()
        assert selections[2]["n_selections"] > selections[0]["n_selections"]

    def test_thompson_sampling(self):
        """Test Thompson Sampling algorithm."""
        bandit = MultiArmedBandit(n_arms=3, algorithm="thompson")

        # Simulate with known probabilities
        true_probs = [0.1, 0.3, 0.2]

        for _ in range(1000):
            arm = bandit.select_arm()
            reward = np.random.binomial(1, true_probs[arm])
            bandit.update(arm, reward)

        stats = bandit.get_arm_statistics()
        best_arm = max(stats, key=lambda x: stats[x]["mean_reward"])

        assert best_arm == 1  # Arm with 0.3 probability

    def test_ucb_algorithm(self):
        """Test Upper Confidence Bound algorithm."""
        bandit = MultiArmedBandit(n_arms=3, algorithm="ucb")

        for _ in range(500):
            arm = bandit.select_arm()
            reward = np.random.normal(arm * 0.5, 1)
            bandit.update(arm, reward)

        # Check exploration vs exploitation balance
        stats = bandit.get_arm_statistics()
        assert all(stats[i]["n_selections"] > 10 for i in range(3))

    def test_contextual_bandit(self):
        """Test contextual bandit functionality."""
        from ab_testing import ContextualBandit

        bandit = ContextualBandit(n_arms=3, n_features=5)

        # Train with contexts
        for _ in range(100):
            context = np.random.randn(5)
            arm = bandit.select_arm(context)
            reward = np.dot(context, np.random.randn(5)) + arm * 0.1
            bandit.update(arm, reward, context)

        # Test prediction with new context
        new_context = np.random.randn(5)
        selected_arm = bandit.select_arm(new_context)
        assert 0 <= selected_arm < 3


class TestSequentialTesting:
    """Test suite for sequential testing methods."""

    @pytest.fixture
    def sequential_test(self):
        """Create sequential test instance."""
        return SequentialTest(alpha=0.05, beta=0.20, effect_size=0.1)

    def test_sprt(self, sequential_test):
        """Test Sequential Probability Ratio Test."""
        # Simulate data collection
        for i in range(100):
            # Simulate conversions
            control = np.random.binomial(1, 0.10)
            treatment = np.random.binomial(1, 0.12)

            decision = sequential_test.update_sprt(control, treatment)

            if decision != "continue":
                assert decision in ["reject_null", "accept_null"]
                break

    def test_group_sequential(self, sequential_test):
        """Test group sequential testing."""
        # Define interim analysis points
        analysis_points = [0.25, 0.5, 0.75, 1.0]

        for point in analysis_points:
            n_current = int(point * 1000)

            # Simulate data up to this point
            control_conversions = np.random.binomial(n_current, 0.10)
            treatment_conversions = np.random.binomial(n_current, 0.12)

            boundary = sequential_test.get_obrien_fleming_boundary(
                information_fraction=point, alpha=0.05
            )

            z_score = sequential_test.calculate_z_score(
                control_conversions, n_current, treatment_conversions, n_current
            )

            if abs(z_score) > boundary:
                assert point < 1.0  # Should stop early
                break

    def test_adaptive_sample_size(self, sequential_test):
        """Test adaptive sample size re-estimation."""
        # Initial data
        initial_effect = 0.08  # Smaller than expected

        updated_sample_size = sequential_test.reestimate_sample_size(
            observed_effect=initial_effect, current_n=500, target_power=0.8
        )

        # Should increase sample size
        assert updated_sample_size > 1000


class TestPowerAnalysis:
    """Test suite for power analysis tools."""

    @pytest.fixture
    def power_analyzer(self):
        """Create power analyzer instance."""
        return PowerAnalysis()

    def test_calculate_power(self, power_analyzer):
        """Test statistical power calculation."""
        power = power_analyzer.calculate_power(
            effect_size=0.2, sample_size=1000, alpha=0.05
        )

        assert 0 <= power <= 1
        assert power > 0.7  # Should have decent power

    def test_effect_size_calculation(self, power_analyzer):
        """Test effect size calculations."""
        # Cohen's d
        cohens_d = power_analyzer.calculate_cohens_d(
            mean1=100, std1=15, mean2=105, std2=15
        )
        assert abs(cohens_d - 0.333) < 0.01

        # Cohen's h for proportions
        cohens_h = power_analyzer.calculate_cohens_h(0.10, 0.15)
        assert cohens_h > 0

    def test_power_curve(self, power_analyzer):
        """Test power curve generation."""
        sample_sizes = range(100, 1000, 100)

        power_curve = power_analyzer.generate_power_curve(
            effect_size=0.2, sample_sizes=sample_sizes, alpha=0.05
        )

        # Power should increase with sample size
        assert power_curve[0] < power_curve[-1]
        assert all(0 <= p <= 1 for p in power_curve)

    def test_minimum_sample_size(self, power_analyzer):
        """Test minimum sample size for desired power."""
        n_min = power_analyzer.find_minimum_sample_size(
            effect_size=0.3, desired_power=0.8, alpha=0.05
        )

        # Verify the sample size achieves desired power
        actual_power = power_analyzer.calculate_power(
            effect_size=0.3, sample_size=n_min, alpha=0.05
        )

        assert actual_power >= 0.79  # Allow small tolerance


class TestSampleSizeCalculator:
    """Test suite for sample size calculations."""

    @pytest.fixture
    def calculator(self):
        """Create sample size calculator."""
        return SampleSizeCalculator()

    def test_two_proportion_sample_size(self, calculator):
        """Test sample size for two-proportion test."""
        n = calculator.two_proportion_sample_size(
            p1=0.10, p2=0.12, alpha=0.05, power=0.80
        )

        assert n > 0
        assert n < 10000

    def test_continuous_sample_size(self, calculator):
        """Test sample size for continuous outcomes."""
        n = calculator.continuous_sample_size(
            mean_diff=5, std_dev=20, alpha=0.05, power=0.80
        )

        assert n > 0

    def test_unequal_allocation(self, calculator):
        """Test sample size with unequal allocation."""
        n_control, n_treatment = calculator.unequal_allocation_size(
            p1=0.10,
            p2=0.12,
            ratio=2.0,  # 2:1 allocation
            alpha=0.05,
            power=0.80,
        )

        assert n_treatment * 2 == pytest.approx(n_control, rel=0.01)

    def test_multiple_comparisons_adjustment(self, calculator):
        """Test sample size adjustment for multiple comparisons."""
        # Single comparison
        n_single = calculator.two_proportion_sample_size(p1=0.10, p2=0.12, alpha=0.05)

        # Multiple comparisons (Bonferroni)
        n_multiple = calculator.adjust_for_multiple_comparisons(
            base_sample_size=n_single, n_comparisons=3, method="bonferroni"
        )

        assert n_multiple > n_single


class TestErfcinvFix:
    """Test suite for erfcinv implementation."""

    def test_erfcinv_values(self):
        """Test erfcinv returns correct values."""
        # Test known values
        assert abs(erfcinv(0.0) - 1.0) < 1e-10
        assert abs(erfcinv(2.0) + 1.0) < 1e-10

        # Test intermediate values
        x = 0.5
        result = erfcinv(x)
        # Verify by computing erfc(result)
        from scipy.special import erfc

        assert abs(erfc(result) - x) < 1e-10

    def test_erfcinv_edge_cases(self):
        """Test erfcinv edge cases."""
        # Near boundaries
        assert erfcinv(1e-10) > 0
        assert erfcinv(2 - 1e-10) < 0

        # Invalid inputs should raise errors
        with pytest.raises(ValueError):
            erfcinv(-0.1)
        with pytest.raises(ValueError):
            erfcinv(2.1)
