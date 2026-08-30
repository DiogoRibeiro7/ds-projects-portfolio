"""Build and validate the OECD 2022 country-industry production graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

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


def main() -> None:
    """Validate the real 2022 mapping, build the active graph, and write evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = load_2022_csv(args.archive)
    published_blocks = extract_2022_blocks(frame)
    validate_2022_accounting(published_blocks)
    blocks, inactive_labels = active_production_blocks(published_blocks)

    graph = build_production_graph(blocks)
    z = published_blocks.intermediate_use
    x = published_blocks.gross_output
    f = published_blocks.final_demand

    row_residual = z.sum(axis=1) + f.sum(axis=1) - x
    column_residual = (
        z.sum(axis=0)
        + published_blocks.value_added
        + published_blocks.taxes_less_subsidies
        - x
    )

    report = {
        "published_industry_labels": int(len(published_blocks.gross_output)),
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
        "max_abs_row_balance_error": float(np.max(np.abs(row_residual.to_numpy()))),
        "max_abs_column_balance_error": float(np.max(np.abs(column_residual.to_numpy()))),
        "negative_final_demand_cells": int((f < 0.0).sum().sum()),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
