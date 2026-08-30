"""Tests for shared mobility headline-evaluation rules."""

from __future__ import annotations

import pandas as pd
import pytest

from mobility_optimization.evaluation import (
    filter_headline_policy_results,
    headline_mask,
    headline_timestamp_eligible,
)


def test_headline_mask_excludes_dst_transition_rows() -> None:
    """Prospectively flagged DST rows must be excluded from headline metrics."""
    frame = pd.DataFrame({"is_dst_transition_day": [False, True, False]})
    assert headline_mask(frame).tolist() == [True, False, True]


def test_timestamp_eligibility_requires_consistent_zone_flags() -> None:
    """All zones in one hour must share the same prospective eligibility flag."""
    frame = pd.DataFrame({"is_dst_transition_day": [False, True]})
    with pytest.raises(ValueError, match="share headline eligibility"):
        headline_timestamp_eligible(frame)


def test_policy_filter_preserves_non_dst_hours_only() -> None:
    """Policy trajectories may contain DST hours while headline summaries exclude them."""
    panel = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-03-07 12:00:00",
                    "2026-03-07 12:00:00",
                    "2026-03-08 12:00:00",
                    "2026-03-08 12:00:00",
                ]
            ),
            "zone_id": [1, 2, 1, 2],
            "is_dst_transition_day": [False, False, True, True],
        }
    )
    results = pd.DataFrame(
        {
            "timestamp": ["2026-03-07 12:00:00", "2026-03-08 12:00:00"],
            "policy": ["poisson_mean", "poisson_mean"],
            "total_cost": [10.0, 999.0],
        }
    )
    filtered = filter_headline_policy_results(results, panel)
    assert filtered["total_cost"].tolist() == [10.0]
