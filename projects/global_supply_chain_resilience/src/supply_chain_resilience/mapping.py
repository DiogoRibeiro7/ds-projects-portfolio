"""Vintage-specific mapping for the OECD 2025 ICIO regular tables."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FINAL_DEMAND_SUFFIXES = ("HFCE", "NPISH", "GGFC", "GFCF", "INVNT", "DPABR")
SPECIAL_ROWS = {"TLS", "VA", "OUT"}

# Release-aware accounting envelope. OECD ICIO construction may retain small
# balancing residuals, and values below 0.1 million USD can be zeroed during the
# balancing process. The relative allowance was selected only after auditing every
# published year in the official 2016-2022 archive, not from the 2022 target alone.
RELEASE_BALANCE_ATOL = 0.1
RELEASE_BALANCE_RTOL = 2e-4


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
    """Extract Z, x, f, value added, and taxes from a regular ICIO table.

    The name is retained for API compatibility with the original 2022 mapping,
    but the label-driven extraction is also used by the 2016-2022 release audit.
    Zero-output country-industry labels remain in the published accounting system
    and are removed only when constructing the active production subsystem.
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
        label for label in row_labels if label not in SPECIAL_ROWS and label in frame.columns
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
    if (gross_output < 0.0).any():
        minimum = float(gross_output.min())
        raise ValueError(f"Gross output must be non-negative; minimum observed={minimum:.6g}.")

    return ICIOBlocks(
        intermediate_use=z,
        gross_output=gross_output,
        final_demand=final_demand,
        value_added=value_added,
        taxes_less_subsidies=taxes_less_subsidies,
    )


def accounting_residuals(blocks: ICIOBlocks) -> tuple[pd.Series, pd.Series]:
    """Return signed output-use and input-cost residuals."""
    z = blocks.intermediate_use
    x = blocks.gross_output
    row_residual = z.sum(axis=1) + blocks.final_demand.sum(axis=1) - x
    column_residual = z.sum(axis=0) + blocks.value_added + blocks.taxes_less_subsidies - x
    return row_residual.astype(float), column_residual.astype(float)


def _validate_residual_envelope(
    residual: pd.Series,
    gross_output: pd.Series,
    *,
    rtol: float,
    atol: float,
    identity_name: str,
) -> None:
    """Require every residual to lie within ``atol + rtol * |x|``."""
    if rtol < 0.0 or atol < 0.0 or not np.isfinite(rtol) or not np.isfinite(atol):
        raise ValueError("rtol and atol must be finite and non-negative.")
    if not residual.index.equals(gross_output.index):
        raise ValueError("residual and gross_output indexes must match exactly.")

    allowance = atol + rtol * gross_output.abs()
    violation = residual.abs() - allowance
    if (violation > 0.0).any():
        label = str(violation.idxmax())
        raise ValueError(
            f"{identity_name} accounting identity exceeded release balance envelope; "
            f"label={label}, residual={float(residual.loc[label]):.6g}, "
            f"allowance={float(allowance.loc[label]):.6g}."
        )


def validate_2022_accounting(
    blocks: ICIOBlocks,
    *,
    rtol: float = RELEASE_BALANCE_RTOL,
    atol: float = RELEASE_BALANCE_ATOL,
) -> None:
    """Validate ICIO identities against the release-aware balance envelope.

    The gate is deliberately per-industry rather than aggregate: every published
    country-industry residual must satisfy

    ``abs(residual_i) <= atol + rtol * abs(output_i)``.

    The defaults are release-level constants derived from the complete official
    2016-2022 audit. Callers may pass stricter values in tests or diagnostics.
    """
    row_residual, column_residual = accounting_residuals(blocks)
    _validate_residual_envelope(
        row_residual,
        blocks.gross_output,
        rtol=rtol,
        atol=atol,
        identity_name="Output-use",
    )
    _validate_residual_envelope(
        column_residual,
        blocks.gross_output,
        rtol=rtol,
        atol=atol,
        identity_name="Input-cost",
    )


def active_production_blocks(
    blocks: ICIOBlocks,
    *,
    output_atol: float = 1e-12,
    flow_atol: float = 1e-12,
) -> tuple[ICIOBlocks, tuple[str, ...]]:
    """Return the positive-output production subsystem and excluded inactive labels.

    Technical coefficients are undefined for zero-output using industries. A zero-
    output label is excluded only if its intermediate-use row and column are both
    zero up to ``flow_atol``. The published system must be validated before calling
    this function; the returned active subsystem is revalidated under the same
    release-aware envelope.
    """
    if output_atol < 0.0 or flow_atol < 0.0:
        raise ValueError("output_atol and flow_atol must be non-negative.")

    x = blocks.gross_output
    active_mask = x > output_atol
    inactive_labels = tuple(str(label) for label in x.index[~active_mask])
    if not inactive_labels:
        validate_2022_accounting(blocks)
        return blocks, inactive_labels

    inactive_rows = blocks.intermediate_use.loc[list(inactive_labels), :]
    inactive_columns = blocks.intermediate_use.loc[:, list(inactive_labels)]
    max_row_flow = float(np.max(inactive_rows.to_numpy(), initial=0.0))
    max_column_flow = float(np.max(inactive_columns.to_numpy(), initial=0.0))
    max_inactive_flow = max(max_row_flow, max_column_flow)
    if max_inactive_flow > flow_atol:
        raise ValueError(
            "Zero-output industries have material intermediate-use flows; "
            f"maximum flow={max_inactive_flow:.6g}."
        )

    active_labels = [str(label) for label in x.index[active_mask]]
    active = ICIOBlocks(
        intermediate_use=blocks.intermediate_use.loc[active_labels, active_labels],
        gross_output=blocks.gross_output.loc[active_labels],
        final_demand=blocks.final_demand.loc[active_labels, :],
        value_added=blocks.value_added.loc[active_labels],
        taxes_less_subsidies=blocks.taxes_less_subsidies.loc[active_labels],
    )
    validate_2022_accounting(active)
    return active, inactive_labels
