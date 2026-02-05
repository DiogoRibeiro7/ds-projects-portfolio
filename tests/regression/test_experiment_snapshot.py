"""Snapshot tests for conversion analysis outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.statistics.core import ExperimentAnalyzer
from tests.regression.baseline_utils import build_conversion_dataset

pytestmark = pytest.mark.regression

BASELINE_PATH = (
    Path(__file__).parent / "baselines" / "experiment_conversion_report.json"
)


def _load_baseline():
    return json.loads(BASELINE_PATH.read_text())


def test_conversion_snapshot_matches_baseline():
    baseline = _load_baseline()
    seed = baseline["metadata"]["seed"]
    df = build_conversion_dataset(seed=seed)

    analyzer = ExperimentAnalyzer()
    report = analyzer.analyze_conversion(df, conversion_col="converted")

    snap = baseline["snapshot"]

    for group, expected_rate in snap["conversion_rates"].items():
        assert report["conversion_rates"][group] == pytest.approx(
            expected_rate, rel=1e-6
        )

    for key in (
        "absolute_lift",
        "relative_lift",
        "z_statistic",
        "p_value",
        "statistical_power",
    ):
        assert report[key] == pytest.approx(snap[key], rel=1e-6)

    assert report["sample_sizes"] == snap["sample_sizes"]
    assert report["confidence_interval"][0] == pytest.approx(
        snap["confidence_interval"][0], rel=1e-6
    )
    assert report["confidence_interval"][1] == pytest.approx(
        snap["confidence_interval"][1], rel=1e-6
    )
