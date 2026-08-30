"""Tests for ICIO accounting diagnostic helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from supply_chain_resilience.diagnostics import (
    accounting_residuals,
    output_vector_diagnostics,
    ranked_residuals,
    relative_residuals,
    threshold_counts,
)
from supply_chain_resilience.mapping import ICIOBlocks


def _blocks() -> ICIOBlocks:
    """Return a small accounting system with one deliberate row imbalance."""
    labels = ["AAA_A", "BBB_B"]
    intermediate = pd.DataFrame(
        [[20.0, 10.0], [5.0, 30.0]],
        index=labels,
        columns=labels,
    )
    return ICIOBlocks(
        intermediate_use=intermediate,
        gross_output=pd.Series([100.0, 100.0], index=labels),
        final_demand=pd.DataFrame(
            [[71.0], [65.0]],
            index=labels,
            columns=["AAA_HFCE"],
        ),
        value_added=pd.Series([75.0, 60.0], index=labels),
        taxes_less_subsidies=pd.Series([0.0, 0.0], index=labels),
    )


def test_accounting_residuals_keep_row_and_column_diagnostics_separate() -> None:
    """Row and column residuals should expose different accounting identities."""
    row_residual, column_residual = accounting_residuals(_blocks())

    assert row_residual.to_dict() == pytest.approx({"AAA_A": 1.0, "BBB_B": 0.0})
    assert column_residual.to_dict() == pytest.approx({"AAA_A": 0.0, "BBB_B": 0.0})


def test_relative_residuals_scale_by_output() -> None:
    """Relative residuals should use gross output as their natural scale."""
    residual = pd.Series([2.0, -5.0], index=["A", "B"])
    output = pd.Series([100.0, 50.0], index=["A", "B"])

    result = relative_residuals(residual, output)

    assert result.to_dict() == pytest.approx({"A": 0.02, "B": 0.10})


def test_ranked_residuals_return_largest_absolute_errors_first() -> None:
    """Diagnostics should prioritize the industries driving the imbalance."""
    residual = pd.Series([2.0, -5.0, 1.0], index=["A", "B", "C"])
    output = pd.Series([100.0, 50.0, 20.0], index=residual.index)

    ranked = ranked_residuals(residual, output, top_n=2)

    assert [item["label"] for item in ranked] == ["B", "A"]
    assert ranked[0]["relative_residual"] == pytest.approx(0.10)


def test_threshold_counts_are_cumulative() -> None:
    """Each threshold should count all industries at or below that relative error."""
    relative = pd.Series([1e-8, 5e-7, 2e-4])

    counts = threshold_counts(relative, thresholds=(1e-7, 1e-6, 1e-3))

    assert counts == {"1e-07": 1, "1e-06": 2, "1e-03": 3}


def test_output_vector_diagnostics_compare_out_row_and_column() -> None:
    """The duplicated published output vectors should be compared explicitly."""
    frame = pd.DataFrame(
        [
            [0.0, 0.0, 100.0],
            [0.0, 0.0, 200.0],
            [101.0, 198.0, 300.0],
        ],
        index=["AAA_A", "BBB_B", "OUT"],
        columns=["AAA_A", "BBB_B", "OUT"],
    )

    report = output_vector_diagnostics(frame, ["AAA_A", "BBB_B"])

    assert report["out_column_total"] == pytest.approx(300.0)
    assert report["out_row_total"] == pytest.approx(299.0)
    assert report["max_absolute_difference"] == pytest.approx(2.0)
    assert report["nonzero_difference_count"] == 2
