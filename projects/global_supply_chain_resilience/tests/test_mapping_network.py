"""Tests for OECD 2022 mapping and production-network construction."""

from __future__ import annotations

import pandas as pd
import pytest

from supply_chain_resilience.mapping import (
    active_production_blocks,
    extract_2022_blocks,
    validate_2022_accounting,
)
from supply_chain_resilience.network import build_production_graph


def _balanced_frame() -> pd.DataFrame:
    """Return a minimal ICIO-like table satisfying row and column identities."""
    columns = [
        "AAA_A",
        "BBB_B",
        "AAA_HFCE",
        "AAA_NPISH",
        "AAA_GGFC",
        "AAA_GFCF",
        "AAA_INVNT",
        "AAA_DPABR",
        "OUT",
    ]
    return pd.DataFrame(
        [
            [20.0, 10.0, 50.0, 0.0, 0.0, 20.0, 0.0, 0.0, 100.0],
            [5.0, 30.0, 40.0, 0.0, 0.0, 25.0, 0.0, 0.0, 100.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [75.0, 60.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [100.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 200.0],
        ],
        index=["AAA_A", "BBB_B", "TLS", "VA", "OUT"],
        columns=columns,
    )


def _balanced_frame_with_inactive_industry() -> pd.DataFrame:
    """Extend the balanced fixture with one economically inactive industry."""
    frame = _balanced_frame()
    frame.insert(2, "CCC_C", 0.0)
    frame.loc["CCC_C"] = 0.0
    frame = frame.reindex(["AAA_A", "BBB_B", "CCC_C", "TLS", "VA", "OUT"])
    return frame


def test_extract_2022_blocks_is_label_driven() -> None:
    """Industry and final-demand blocks must be selected from labels, not positions."""
    blocks = extract_2022_blocks(_balanced_frame())

    assert list(blocks.intermediate_use.index) == ["AAA_A", "BBB_B"]
    assert list(blocks.intermediate_use.columns) == ["AAA_A", "BBB_B"]
    assert list(blocks.final_demand.columns) == [
        "AAA_HFCE",
        "AAA_NPISH",
        "AAA_GGFC",
        "AAA_GFCF",
        "AAA_INVNT",
        "AAA_DPABR",
    ]


def test_validate_2022_accounting_accepts_balanced_table() -> None:
    """A table satisfying both accounting identities should pass validation."""
    validate_2022_accounting(extract_2022_blocks(_balanced_frame()))


def test_zero_output_industry_is_retained_then_excluded_from_active_network() -> None:
    """Published inactive labels stay in accounting but not technical coefficients."""
    blocks = extract_2022_blocks(_balanced_frame_with_inactive_industry())
    validate_2022_accounting(blocks)

    active, inactive = active_production_blocks(blocks)

    assert inactive == ("CCC_C",)
    assert list(active.gross_output.index) == ["AAA_A", "BBB_B"]
    assert list(active.intermediate_use.columns) == ["AAA_A", "BBB_B"]


def test_zero_output_industry_with_intermediate_flow_is_rejected() -> None:
    """An inactive industry cannot be silently removed when it carries flows."""
    frame = _balanced_frame_with_inactive_industry()
    frame.loc["AAA_A", "CCC_C"] = 1.0
    blocks = extract_2022_blocks(frame)

    with pytest.raises(ValueError, match="material intermediate-use flows"):
        active_production_blocks(blocks)


def test_validate_2022_accounting_rejects_output_use_mismatch() -> None:
    """Output-use inconsistencies should block graph construction."""
    frame = _balanced_frame()
    frame.loc["AAA_A", "AAA_HFCE"] += 1.0
    blocks = extract_2022_blocks(frame)

    with pytest.raises(ValueError, match="Output-use"):
        validate_2022_accounting(blocks)


def test_build_production_graph_preserves_orientation_and_weights() -> None:
    """Edges must point from supplying industry to using industry with A as weight."""
    blocks = extract_2022_blocks(_balanced_frame())
    validate_2022_accounting(blocks)

    graph = build_production_graph(blocks)

    assert graph.number_of_nodes() == 2
    assert graph["AAA_A"]["BBB_B"]["weight"] == pytest.approx(0.10)
    assert graph["BBB_B"]["AAA_A"]["weight"] == pytest.approx(0.05)
