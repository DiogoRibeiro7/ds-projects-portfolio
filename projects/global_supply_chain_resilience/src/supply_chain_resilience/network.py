"""Production-network construction from validated ICIO blocks."""

from __future__ import annotations

import networkx as nx
import numpy as np

from supply_chain_resilience.icio import technical_coefficients
from supply_chain_resilience.mapping import ICIOBlocks


def _split_node_label(label: str) -> tuple[str, str]:
    """Split ``COUNTRY_ACTIVITY`` once, preserving activity underscores."""
    if "_" not in label:
        raise ValueError(f"ICIO node label lacks country/activity separator: {label!r}.")
    country, activity = label.split("_", maxsplit=1)
    if not country or not activity:
        raise ValueError(f"Invalid ICIO country-industry label: {label!r}.")
    return country, activity


def build_production_graph(
    blocks: ICIOBlocks,
    *,
    minimum_coefficient: float = 0.0,
) -> nx.DiGraph:
    """Build a directed weighted country-industry production graph.

    An edge ``i -> j`` represents supplier ``i`` providing intermediate inputs to
    using industry ``j``. ``flow_value`` stores observed intermediate use ``Z[i,j]``
    and ``input_share`` stores the technical coefficient ``A[i,j]``. The generic
    NetworkX ``weight`` attribute remains an alias of ``input_share`` for backwards
    compatibility.
    """
    if minimum_coefficient < 0.0:
        raise ValueError("minimum_coefficient must be non-negative.")

    coefficients = technical_coefficients(blocks.intermediate_use, blocks.gross_output)
    graph = nx.DiGraph()
    for node, output in blocks.gross_output.items():
        label = str(node)
        country, activity = _split_node_label(label)
        graph.add_node(
            label,
            gross_output=float(output),
            country=country,
            activity=activity,
        )

    values = coefficients.to_numpy(dtype=float)
    flows = blocks.intermediate_use.to_numpy(dtype=float)
    row_indices, column_indices = np.nonzero(values > minimum_coefficient)
    row_labels = coefficients.index.to_numpy(dtype=str)
    column_labels = coefficients.columns.to_numpy(dtype=str)

    for row_index, column_index in zip(row_indices, column_indices, strict=True):
        input_share = float(values[row_index, column_index])
        graph.add_edge(
            row_labels[row_index],
            column_labels[column_index],
            flow_value=float(flows[row_index, column_index]),
            input_share=input_share,
            weight=input_share,
        )

    return graph
