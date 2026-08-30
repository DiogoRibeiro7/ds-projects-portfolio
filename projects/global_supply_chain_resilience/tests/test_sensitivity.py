"""Tests for structural-dependency ranking sensitivity utilities."""

from __future__ import annotations

import pandas as pd
import pytest

from supply_chain_resilience.mapping import ICIOBlocks
from supply_chain_resilience.sensitivity import (
    dependency_metrics_after_threshold,
    foreign_intermediate_total,
    ranking_stability,
    threshold_intermediate_use_by_input_share,
    top_k_jaccard,
)


def _blocks() -> ICIOBlocks:
    labels = ["AAA_X", "BBB_Y", "CCC_Z"]
    z = pd.DataFrame(
        [
            [40.0, 5.0, 0.2],
            [20.0, 30.0, 8.0],
            [10.0, 5.0, 20.0],
        ],
        index=labels,
        columns=labels,
    )
    return ICIOBlocks(
        intermediate_use=z,
        gross_output=pd.Series([100.0, 100.0, 100.0], index=labels),
        final_demand=pd.DataFrame(index=labels),
        value_added=pd.Series([0.0, 0.0, 0.0], index=labels),
        taxes_less_subsidies=pd.Series([0.0, 0.0, 0.0], index=labels),
    )


def test_threshold_intermediate_use_uses_strict_input_share_cutoff() -> None:
    blocks = _blocks()
    thresholded = threshold_intermediate_use_by_input_share(
        blocks,
        minimum_input_share=0.05,
    )

    assert thresholded.loc["AAA_X", "BBB_Y"] == 0.0
    assert thresholded.loc["BBB_Y", "CCC_Z"] == pytest.approx(8.0)
    assert thresholded.loc["AAA_X", "CCC_Z"] == 0.0


def test_dependency_metrics_after_threshold_recomputes_foreign_dependence() -> None:
    blocks = _blocks()
    metrics = dependency_metrics_after_threshold(blocks, minimum_input_share=0.05)

    assert metrics.loc["AAA_X", "foreign_input"] == pytest.approx(30.0)
    assert metrics.loc["CCC_Z", "foreign_input"] == pytest.approx(8.0)


def test_foreign_intermediate_total_counts_only_cross_border_flows() -> None:
    assert foreign_intermediate_total(_blocks().intermediate_use) == pytest.approx(48.2)


def test_top_k_jaccard_detects_rank_membership_change() -> None:
    reference = pd.Series([3.0, 2.0, 1.0], index=["a", "b", "c"])
    candidate = pd.Series([3.0, 0.5, 2.0], index=["a", "b", "c"])

    assert top_k_jaccard(reference, candidate, k=2) == pytest.approx(1.0 / 3.0)


def test_ranking_stability_uses_fixed_eligible_universe() -> None:
    reference = pd.Series([4.0, 3.0, 2.0, 1.0], index=["a", "b", "c", "d"])
    candidate = pd.Series([4.0, 2.5, float("nan"), 1.0], index=reference.index)
    result = ranking_stability(
        reference,
        candidate,
        eligible_nodes=pd.Index(["a", "b", "c"]),
        top_k_values=(2,),
    )

    assert result["reference_eligible_nodes"] == 3
    assert result["common_finite_nodes"] == 2
    assert result["candidate_missing_nodes"] == 1
