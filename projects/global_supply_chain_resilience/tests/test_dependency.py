"""Tests for accounting-based supply-chain dependency metrics."""

from __future__ import annotations

import pandas as pd
import pytest

from supply_chain_resilience.dependency import (
    cross_border_intermediate_share,
    split_country_activity,
    structural_dependency_metrics,
)
from supply_chain_resilience.mapping import ICIOBlocks


def _blocks() -> ICIOBlocks:
    labels = ["AAA_X", "AAA_Y", "BBB_X"]
    z = pd.DataFrame(
        [
            [20.0, 10.0, 30.0],
            [10.0, 20.0, 10.0],
            [30.0, 10.0, 20.0],
        ],
        index=labels,
        columns=labels,
    )
    output = pd.Series([100.0, 100.0, 100.0], index=labels)
    final_demand = pd.DataFrame({"AAA_HFCE": [40.0, 60.0, 40.0]}, index=labels)
    value_added = pd.Series([40.0, 60.0, 40.0], index=labels)
    taxes = pd.Series([0.0, 0.0, 0.0], index=labels)
    return ICIOBlocks(z, output, final_demand, value_added, taxes)


def test_split_country_activity_preserves_activity_underscores() -> None:
    assert split_country_activity("PRT_C17_18") == ("PRT", "C17_18")


def test_structural_dependency_aggregates_supplier_activities_by_country() -> None:
    metrics = structural_dependency_metrics(_blocks())

    aaa_x = metrics.loc["AAA_X"]
    assert aaa_x["intermediate_input"] == pytest.approx(60.0)
    assert aaa_x["domestic_input"] == pytest.approx(30.0)
    assert aaa_x["foreign_input"] == pytest.approx(30.0)
    assert aaa_x["foreign_input_dependence"] == pytest.approx(0.5)
    assert aaa_x["supplier_country_hhi"] == pytest.approx(0.5)
    assert aaa_x["effective_supplier_countries"] == pytest.approx(2.0)
    assert aaa_x["largest_supplier_country_share"] == pytest.approx(0.5)
    assert aaa_x["foreign_supplier_country_hhi"] == pytest.approx(1.0)


def test_cross_border_intermediate_share_uses_flow_values() -> None:
    blocks = _blocks()
    # Foreign cells: BBB->AAA_X=30, BBB->AAA_Y=10, AAA_X->BBB=30, AAA_Y->BBB=10.
    assert cross_border_intermediate_share(blocks) == pytest.approx(80.0 / 160.0)
