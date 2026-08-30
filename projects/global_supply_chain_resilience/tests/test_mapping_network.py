"""Tests for OECD 2022 mapping and production-network construction."""

from __future__ import annotations

import pandas as pd
import pytest

from supply_chain_resilience.mapping import (
    RELEASE_BALANCE_ATOL,
    RELEASE_BALANCE_RTOL,
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
    validate_2022_accounting(extract_2022_blocks(_balanced_frame()))


def test_release_gate_accepts_residual_inside_scale_aware_envelope() -> None:
    frame = _balanced_frame()
    allowance = RELEASE_BALANCE_ATOL + RELEASE_BALANCE_RTOL * 100.0
    frame.loc["AAA_A", "AAA_HFCE"] += 0.9 * allowance
    validate_2022_accounting(extract_2022_blocks(frame))


def test_release_gate_rejects_residual_outside_scale_aware_envelope() -> None:
    frame = _balanced_frame()
    allowance = RELEASE_BALANCE_ATOL + RELEASE_BALANCE_RTOL * 100.0
    frame.loc["AAA_A", "AAA_HFCE"] += 1.1 * allowance
    with pytest.raises(ValueError, match="release balance envelope"):
        validate_2022_accounting(extract_2022_blocks(frame))


def test_strict_override_remains_available_for_synthetic_checks() -> None:
    frame = _balanced_frame()
    frame.loc["AAA_A", "AAA_HFCE"] += 1e-4
    with pytest.raises(ValueError, match="Output-use"):
        validate_2022_accounting(extract_2022_blocks(frame), rtol=0.0, atol=1e-6)


def test_zero_output_industry_is_retained_then_excluded_from_active_network() -> None:
    blocks = extract_2022_blocks(_balanced_frame_with_inactive_industry())
    validate_2022_accounting(blocks)
    active, inactive = active_production_blocks(blocks)
    assert inactive == ("CCC_C",)
    assert list(active.gross_output.index) == ["AAA_A", "BBB_B"]
    assert list(active.intermediate_use.columns) == ["AAA_A", "BBB_B"]


def test_zero_output_industry_with_intermediate_flow_is_rejected() -> None:
    frame = _balanced_frame_with_inactive_industry()
    frame.loc["AAA_A", "CCC_C"] = 1.0
    blocks = extract_2022_blocks(frame)
    with pytest.raises(ValueError, match="material intermediate-use flows"):
        active_production_blocks(blocks)


def test_validate_2022_accounting_rejects_large_output_use_mismatch() -> None:
    frame = _balanced_frame()
    frame.loc["AAA_A", "AAA_HFCE"] += 1.0
    blocks = extract_2022_blocks(frame)
    with pytest.raises(ValueError, match="Output-use"):
        validate_2022_accounting(blocks)


def test_build_production_graph_preserves_orientation_and_semantics() -> None:
    """Graph edges expose both observed flow and input-share semantics."""
    blocks = extract_2022_blocks(_balanced_frame())
    validate_2022_accounting(blocks)
    graph = build_production_graph(blocks)

    assert graph.number_of_nodes() == 2
    assert graph.nodes["AAA_A"]["country"] == "AAA"
    assert graph.nodes["AAA_A"]["activity"] == "A"
    assert graph["AAA_A"]["BBB_B"]["flow_value"] == pytest.approx(10.0)
    assert graph["AAA_A"]["BBB_B"]["input_share"] == pytest.approx(0.10)
    assert graph["AAA_A"]["BBB_B"]["weight"] == pytest.approx(0.10)
    assert graph["BBB_B"]["AAA_A"]["flow_value"] == pytest.approx(5.0)
    assert graph["BBB_B"]["AAA_A"]["input_share"] == pytest.approx(0.05)
