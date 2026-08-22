import pandas as pd
import pytest

from experimentation_toolkit import (
    apply_cuped,
    bootstrap_ci_diff,
    cohens_h,
    power_two_proportions,
    sample_ratio_mismatch,
    sample_size_two_proportions,
    summarize_groups,
    two_proportion_z_test,
    welch_t_test,
)


def test_two_proportion_z_test_detects_treatment_lift() -> None:
    result = two_proportion_z_test(100, 1000, 150, 1000, continuity_correction=False)

    assert result.effect == pytest.approx(0.05)
    assert result.p_value < 0.01
    assert result.confidence_interval is not None


def test_bootstrap_ci_diff_is_seeded_and_contains_effect() -> None:
    interval = bootstrap_ci_diff(0.1, 0.13, 1000, 1000, n_bootstrap=1000, seed=42)

    assert interval.lower < interval.observed_effect < interval.upper


def test_power_and_sample_size_are_monotonic() -> None:
    smaller = sample_size_two_proportions(0.1, 0.03)
    larger = sample_size_two_proportions(0.1, 0.02)

    assert larger > smaller
    assert power_two_proportions(5000, 5000, 0.1, 0.02) > 0.7
    assert cohens_h(0.1, 0.12) > 0


def test_sample_ratio_mismatch_flags_large_imbalance() -> None:
    result = sample_ratio_mismatch({"control": 1000, "treatment": 700})

    assert result.is_mismatch
    assert result.p_value < 0.001


def test_cuped_and_group_summary() -> None:
    data = pd.DataFrame(
        {
            "group": ["control", "control", "treatment", "treatment"],
            "metric": [10.0, 12.0, 13.0, 15.0],
            "pre_metric": [8.0, 10.0, 11.0, 13.0],
        }
    )

    adjusted = apply_cuped(data, "metric", "pre_metric")
    summary = summarize_groups(adjusted, "group", "metric_cuped")

    assert "metric_cuped" in adjusted.columns
    assert set(summary["group"]) == {"control", "treatment"}


def test_welch_t_test_returns_treatment_minus_control_effect() -> None:
    result = welch_t_test([1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0])

    assert result.effect == 1.0
    assert result.confidence_interval is not None
