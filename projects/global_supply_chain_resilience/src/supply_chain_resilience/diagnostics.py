"""Accounting diagnostics for OECD ICIO mapping validation.

Diagnostics are deliberately separate from acceptance criteria. They quantify and
locate balance residuals, but they do not decide that an economically inconsistent
table is acceptable merely because the relative error is small.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from supply_chain_resilience.mapping import ICIOBlocks

DEFAULT_RELATIVE_THRESHOLDS: tuple[float, ...] = (
    1e-8,
    1e-7,
    1e-6,
    1e-5,
    1e-4,
    1e-3,
)


def accounting_residuals(blocks: ICIOBlocks) -> tuple[pd.Series, pd.Series]:
    """Return row-use and column-cost accounting residuals.

    Returns:
        A tuple ``(row_residual, column_residual)`` where zero denotes exact
        accounting balance under the current mapping.
    """
    z = blocks.intermediate_use
    x = blocks.gross_output
    row_residual = z.sum(axis=1) + blocks.final_demand.sum(axis=1) - x
    column_residual = z.sum(axis=0) + blocks.value_added + blocks.taxes_less_subsidies - x
    return row_residual.astype(float), column_residual.astype(float)


def relative_residuals(
    residual: pd.Series,
    gross_output: pd.Series,
    *,
    denominator_floor: float = 1.0,
) -> pd.Series:
    """Scale absolute residuals by gross output with an explicit denominator floor.

    The floor prevents zero or tiny published outputs from creating infinite or
    meaningless relative errors. It is a diagnostic convention only and is not an
    accounting-validation tolerance.
    """
    if denominator_floor <= 0.0 or not np.isfinite(denominator_floor):
        raise ValueError("denominator_floor must be a finite positive value.")
    if not residual.index.equals(gross_output.index):
        raise ValueError("residual and gross_output indexes must match exactly.")

    denominator = gross_output.abs().clip(lower=denominator_floor)
    return residual.abs().divide(denominator).astype(float)


def ranked_residuals(
    residual: pd.Series,
    gross_output: pd.Series,
    *,
    top_n: int = 20,
    denominator_floor: float = 1.0,
) -> list[dict[str, float | str]]:
    """Return the largest absolute accounting residuals with scale information."""
    if top_n <= 0:
        raise ValueError("top_n must be positive.")

    relative = relative_residuals(
        residual,
        gross_output,
        denominator_floor=denominator_floor,
    )
    ordered = residual.abs().sort_values(ascending=False).head(top_n)
    return [
        {
            "label": str(label),
            "residual": float(residual.loc[label]),
            "absolute_residual": float(abs(residual.loc[label])),
            "gross_output": float(gross_output.loc[label]),
            "relative_residual": float(relative.loc[label]),
        }
        for label in ordered.index
    ]


def threshold_counts(
    relative: pd.Series,
    *,
    thresholds: Sequence[float] = DEFAULT_RELATIVE_THRESHOLDS,
) -> dict[str, int]:
    """Count industries whose relative residual is no greater than each threshold."""
    result: dict[str, int] = {}
    for threshold in thresholds:
        value = float(threshold)
        if value < 0.0 or not np.isfinite(value):
            raise ValueError("thresholds must contain finite non-negative values.")
        result[f"{value:.0e}"] = int((relative <= value).sum())
    return result


def output_vector_diagnostics(
    frame: pd.DataFrame,
    labels: Sequence[str],
    *,
    top_n: int = 20,
    denominator_floor: float = 1.0,
) -> dict[str, object]:
    """Compare the published top-right and bottom-row gross-output vectors.

    The regular ICIO table exposes ``OUT`` both as a column and as a bottom row.
    Comparing them is diagnostic evidence about the table's valuation/accounting
    semantics; this function does not assume in advance that they must be identical.
    """
    if "OUT" not in frame.index or "OUT" not in frame.columns:
        raise ValueError("frame must contain both an OUT row and an OUT column.")
    if top_n <= 0:
        raise ValueError("top_n must be positive.")

    ordered_labels = [str(label) for label in labels]
    column_output = pd.to_numeric(frame.loc[ordered_labels, "OUT"], errors="raise").astype(float)
    row_output = pd.to_numeric(frame.loc["OUT", ordered_labels], errors="raise").astype(float)
    row_output.index = column_output.index

    difference = row_output - column_output
    relative = relative_residuals(
        difference,
        column_output,
        denominator_floor=denominator_floor,
    )
    ordered = difference.abs().sort_values(ascending=False).head(top_n)

    return {
        "out_column_total": float(column_output.sum()),
        "out_row_total": float(row_output.sum()),
        "total_difference": float(difference.sum()),
        "max_absolute_difference": float(difference.abs().max()),
        "max_relative_difference": float(relative.max()),
        "nonzero_difference_count": int((difference != 0.0).sum()),
        "top_differences": [
            {
                "label": str(label),
                "out_column": float(column_output.loc[label]),
                "out_row": float(row_output.loc[label]),
                "difference": float(difference.loc[label]),
                "absolute_difference": float(abs(difference.loc[label])),
                "relative_difference": float(relative.loc[label]),
            }
            for label in ordered.index
        ],
    }
