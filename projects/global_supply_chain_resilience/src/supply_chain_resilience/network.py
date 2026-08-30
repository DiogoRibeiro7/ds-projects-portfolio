"""Production-network construction from validated ICIO blocks."""

from __future__ import annotations

import networkx as nx
import numpy as np

from supply_chain_resilience.icio import technical_coefficients
from supply_chain_resilience.mapping import ICIOBlocks


def build_production_graph(
    blocks: ICIOBlocks,
    *,
    minimum_coefficient: float = 0.0,
) -> nx.DiGraph:
    """Build a directed weighted country-industry production graph.

    An edge ``i -> j`` represents supplier ``i`` providing intermediate inputs to
    using industry ``j``. Edge weight is the technical coefficient ``A[i, j]``.

    Args:
        blocks: Accounting-validated ICIO blocks.
        minimum_coefficient: Optional strict lower bound for retained edges.

    Returns:
        Directed graph with gross-output node attributes and technical-coefficient
        edge weights.
    """
    if minimum_coefficient < 0.0:
        raise ValueError("minimum_coefficient must be non-negative.")

    coefficients = technical_coefficients(blocks.intermediate_use, blocks.gross_output)
    graph = nx.DiGraph()
    for node, output in blocks.gross_output.items():
        graph.add_node(str(node), gross_output=float(output))

    values = coefficients.to_numpy(dtype=float)
    row_indices, column_indices = np.nonzero(values > minimum_coefficient)
    row_labels = coefficients.index.to_numpy(dtype=str)
    column_labels = coefficients.columns.to_numpy(dtype=str)

    for row_index, column_index in zip(row_indices, column_indices, strict=True):
        graph.add_edge(
            row_labels[row_index],
            column_labels[column_index],
            weight=float(values[row_index, column_index]),
        )

    return graph
