"""Build and validate the OECD 2022 country-industry production graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from supply_chain_resilience.diagnostics import (
    accounting_residuals,
    output_vector_diagnostics,
    ranked_residuals,
    relative_residuals,
    threshold_counts,
)
from supply_chain_resilience.mapping import (
    active_production_blocks,
    extract_2022_blocks,
    validate_2022_accounting,
)
from supply_chain_resilience.network import build_production_graph


def load_2022_csv(archive_path: Path, member: str = "2022_SML.csv") -> pd.DataFrame:
    """Load the observed 2022 ICIO member from the validated OECD archive."""
    with ZipFile(archive_path) as archive:
        if member not in archive.namelist():
            raise RuntimeError(f"Expected archive member {member!r} was not found.")
        with archive.open(member) as handle:
            return pd.read_csv(handle, index_col=0, low_memory=False)


def _write_report(path: Path, report: dict[str, object]) -> None:
    """Write deterministic JSON evidence and echo it to the workflow log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True)
    path.write_text(payload, encoding="utf-8")
    print(payload)


def main() -> None:
    """Diagnose accounting, validate strictly, then build the active graph."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--diagnostics-output",
        type=Path,
        required=True,
        help="JSON diagnostic evidence written before strict accounting validation.",
    )
    args = parser.parse_args()

    frame = load_2022_csv(args.archive)
    published_blocks = extract_2022_blocks(frame)
    z = published_blocks.intermediate_use
    x = published_blocks.gross_output
    f = published_blocks.final_demand

    row_residual, column_residual = accounting_residuals(published_blocks)
    row_relative = relative_residuals(row_residual, x)
    column_relative = relative_residuals(column_residual, x)
    zero_output = x.index[x <= 1e-12]

    diagnostic_report: dict[str, object] = {
        "published_industry_labels": int(len(x)),
        "zero_output_labels": int(len(zero_output)),
        "zero_output_sample": [str(label) for label in zero_output[:20]],
        "max_abs_row_balance_error": float(row_residual.abs().max()),
        "max_relative_row_balance_error": float(row_relative.max()),
        "max_abs_column_balance_error": float(column_residual.abs().max()),
        "max_relative_column_balance_error": float(column_relative.max()),
        "row_relative_threshold_counts": threshold_counts(row_relative),
        "column_relative_threshold_counts": threshold_counts(column_relative),
        "top_row_balance_residuals": ranked_residuals(row_residual, x),
        "top_column_balance_residuals": ranked_residuals(column_residual, x),
        "out_row_vs_column": output_vector_diagnostics(frame, list(x.index)),
        "gross_output_total": float(x.sum()),
        "intermediate_use_total": float(z.to_numpy().sum()),
        "final_demand_total": float(f.to_numpy().sum()),
        "negative_final_demand_cells": int((f < 0.0).sum().sum()),
    }
    _write_report(args.diagnostics_output, diagnostic_report)

    # Diagnostics do not weaken the scientific gate. The workflow must remain red
    # until the currently assumed ICIO identities pass at their prospective tolerance.
    validate_2022_accounting(published_blocks)
    blocks, inactive_labels = active_production_blocks(published_blocks)
    graph = build_production_graph(blocks)

    network_report: dict[str, object] = {
        "published_industry_labels": int(len(x)),
        "active_nodes": graph.number_of_nodes(),
        "inactive_zero_output_labels": int(len(inactive_labels)),
        "inactive_zero_output_sample": list(inactive_labels[:20]),
        "edges": graph.number_of_edges(),
        "density": float(graph.number_of_edges() / (graph.number_of_nodes() ** 2)),
        "published_intermediate_use_shape": [int(z.shape[0]), int(z.shape[1])],
        "active_intermediate_use_shape": [
            int(blocks.intermediate_use.shape[0]),
            int(blocks.intermediate_use.shape[1]),
        ],
        "final_demand_columns": int(f.shape[1]),
        "gross_output_total": float(x.sum()),
        "intermediate_use_total": float(z.to_numpy().sum()),
        "final_demand_total": float(f.to_numpy().sum()),
        "max_abs_row_balance_error": float(row_residual.abs().max()),
        "max_abs_column_balance_error": float(column_residual.abs().max()),
        "negative_final_demand_cells": int((f < 0.0).sum().sum()),
    }
    _write_report(args.output, network_report)


if __name__ == "__main__":
    main()
