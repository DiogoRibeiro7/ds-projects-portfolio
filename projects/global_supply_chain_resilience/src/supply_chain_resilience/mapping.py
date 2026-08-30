"""Vintage-specific mapping for the OECD 2025 ICIO 2022 regular table."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FINAL_DEMAND_SUFFIXES = ("HFCE", "NPISH", "GGFC", "GFCF", "INVNT", "DPABR")
SPECIAL_ROWS = {"TLS", "VA", "OUT"}


@dataclass(frozen=True)
class ICIOBlocks:
    """Economically identified blocks extracted from an OECD ICIO table."""

    intermediate_use: pd.DataFrame
    gross_output: pd.Series
    final_demand: pd.DataFrame
    value_added: pd.Series
    taxes_less_subsidies: pd.Series


def _is_final_demand(label: str) -> bool:
    """Return whether a column label is one of the documented final-demand categories."""
    return any(label.endswith(f"_{suffix}") for suffix in FINAL_DEMAND_SUFFIXES)


def extract_2022_blocks(frame: pd.DataFrame) -> ICIOBlocks:
    """Extract Z, x, f, value added, and taxes from the OECD 2022 regular ICIO table.

    The mapping is label-driven and frozen from the observed 2022 schema artifact:
    country-industry identifiers appear in both rows and columns, final-demand columns
    use six explicit suffixes, and special accounting rows/columns are named.
    """
    if frame.empty:
        raise ValueError("ICIO frame must not be empty.")
    if frame.index.has_duplicates or frame.columns.has_duplicates:
        raise ValueError("ICIO labels must be unique.")
    for label in SPECIAL_ROWS:
        if label not in frame.index:
            raise ValueError(f"Missing required accounting row: {label}.")
    if "OUT" not in frame.columns:
        raise ValueError("Missing required gross-output column: OUT.")

    row_labels = [str(value) for value in frame.index]
    column_labels = [str(value) for value in frame.columns]
    industry_labels = [
        label
        for label in row_labels
        if label not in SPECIAL_ROWS and label in frame.columns
    ]
    if not industry_labels:
        raise ValueError("No overlapping country-industry labels were identified.")

    final_demand_columns = [label for label in column_labels if _is_final_demand(label)]
    if not final_demand_columns:
        raise ValueError("No final-demand columns were identified from documented suffixes.")

    numeric = frame.apply(pd.to_numeric, errors="raise").astype(float)
    z = numeric.loc[industry_labels, industry_labels]
    gross_output = numeric.loc[industry_labels, "OUT"]
    final_demand = numeric.loc[industry_labels, final_demand_columns]
    value_added = numeric.loc["VA", industry_labels]
    taxes_less_subsidies = numeric.loc["TLS", industry_labels]

    arrays = [
        z.to_numpy(),
        gross_output.to_numpy(),
        final_demand.to_numpy(),
        value_added.to_numpy(),
        taxes_less_subsidies.to_numpy(),
    ]
    if not all(np.all(np.isfinite(values)) for values in arrays):
        raise ValueError("Mapped ICIO blocks must contain only finite numeric values.")
    if (z < 0.0).any(axis=None):
        raise ValueError("Intermediate-use block must be non-negative.")
    if (gross_output <= 0.0).any():
        raise ValueError("Gross output must be strictly positive for all industries.")

    return ICIOBlocks(
        intermediate_use=z,
        gross_output=gross_output,
        final_demand=final_demand,
        value_added=value_added,
        taxes_less_subsidies=taxes_less_subsidies,
    )


def validate_2022_accounting(blocks: ICIOBlocks, *, rtol: float = 1e-7, atol: float = 1e-5) -> None:
    """Validate row and column accounting identities for the mapped 2022 ICIO table."""
    z = blocks.intermediate_use
    x = blocks.gross_output
    f = blocks.final_demand

    row_total = z.sum(axis=1) + f.sum(axis=1)
    if not np.allclose(row_total.to_numpy(), x.to_numpy(), rtol=rtol, atol=atol):
        max_error = float(np.max(np.abs(row_total.to_numpy() - x.to_numpy())))
        raise ValueError(f"Output-use accounting identity failed; max absolute error={max_error:.6g}.")

    input_total = z.sum(axis=0) + blocks.value_added + blocks.taxes_less_subsidies
    if not np.allclose(input_total.to_numpy(), x.to_numpy(), rtol=rtol, atol=atol):
        max_error = float(np.max(np.abs(input_total.to_numpy() - x.to_numpy())))
        raise ValueError(f"Input-cost accounting identity failed; max absolute error={max_error:.6g}.")
