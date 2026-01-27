"""Tests for robust experiment analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.statistics.core import ExperimentAnalyzer

pytestmark = pytest.mark.unit


def make_outlier_data(seed: int = 0):
    rng = np.random.default_rng(seed)
    base = pd.DataFrame({
        "group": np.repeat(["control", "treatment"], 500),
        "converted": rng.binomial(1, 0.2, size=1000),
    })
    outliers = pd.DataFrame({
        "group": ["treatment"] * 10,
        "converted": np.ones(10),
    })
    return pd.concat([base, outliers], ignore_index=True)


def test_trimmed_analysis_less_sensitive_to_outliers():
    df = make_outlier_data()
    analyzer = ExperimentAnalyzer(alpha=0.05)
    vanilla = analyzer.analyze_conversion(df)
    robust = analyzer.analyze_conversion(df, robust=True, trim_fraction=0.05)
    assert vanilla["absolute_lift"] > robust["absolute_lift"]
    assert not vanilla["significant"]


def test_huber_continuous_metric_reduces_outlier_influence():
    rng = np.random.default_rng(123)
    base = pd.DataFrame(
        {
            "group": np.repeat(["control", "treatment"], 200),
            "metric": rng.normal(0, 1, size=400),
        }
    )
    outliers = pd.DataFrame(
        {
            "group": ["treatment"] * 5,
            "metric": np.full(5, 50.0),
        }
    )
    df = pd.concat([base, outliers], ignore_index=True)
    analyzer = ExperimentAnalyzer()
    summary_plain = analyzer.run_comprehensive_analysis(
        df, metrics=["metric"], group_col="group"
    )
    summary_robust = analyzer.run_comprehensive_analysis(
        df,
        metrics=["metric"],
        group_col="group",
        robust=True,
        trim_fraction=0.05,
        huber_delta=1.5,
    )
    plain_diff = summary_plain["metrics_analysis"]["metric"]["mean_diff"]
    robust_diff = summary_robust["metrics_analysis"]["metric"]["mean_diff"]
    assert abs(robust_diff) < abs(plain_diff)
