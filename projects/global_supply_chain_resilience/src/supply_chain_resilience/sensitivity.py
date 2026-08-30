"""Sensitivity utilities for ICIO structural-dependency rankings."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from supply_chain_resilience.dependency import split_country_activity, structural_dependency_metrics
from supply_chain_resilience.icio import technical_coefficients
from supply_chain_resilience.mapping import ICIOBlocks


def threshold_intermediate_use_by_input_share(
    blocks: ICIOBlocks,
    *,
    minimum_input_share: float,
) -> pd.DataFrame:
    """Retain only links with ``A[i,j] > minimum_input_share``.

    The transformation is diagnostic only. It does not rebalance the ICIO table and
    must not be passed through the accounting-identity validator as though it were a
    new observed table.
    """
    if minimum_input_share < 0.0 or not np.isfinite(minimum_input_share):
        raise ValueError("minimum_input_share must be finite and non-negative.")

    coefficients = technical_coefficients(blocks.intermediate_use, blocks.gross_output)
    retained = coefficients > minimum_input_share
    return blocks.intermediate_use.where(retained, other=0.0)


def dependency_metrics_after_threshold(
    blocks: ICIOBlocks,
    *,
    minimum_input_share: float,
) -> pd.DataFrame:
    """Recompute direct dependency descriptors after a diagnostic link threshold."""
    thresholded = threshold_intermediate_use_by_input_share(
        blocks,
        minimum_input_share=minimum_input_share,
    )
    transformed = replace(blocks, intermediate_use=thresholded)
    return structural_dependency_metrics(transformed)


def foreign_intermediate_total(intermediate_use: pd.DataFrame) -> float:
    """Return total cross-border intermediate use for a square ICIO flow block."""
    if not intermediate_use.index.equals(intermediate_use.columns):
        raise ValueError("intermediate_use must have identical row and column labels.")
    supplier_country = np.array(
        [split_country_activity(str(label))[0] for label in intermediate_use.index]
    )
    user_country = np.array(
        [split_country_activity(str(label))[0] for label in intermediate_use.columns]
    )
    values = intermediate_use.to_numpy(dtype=float)
    foreign_mask = supplier_country[:, None] != user_country[None, :]
    return float(values[foreign_mask].sum())


def top_k_jaccard(
    reference: pd.Series,
    candidate: pd.Series,
    *,
    k: int,
) -> float:
    """Return Jaccard overlap of two descending top-k rankings."""
    if k <= 0:
        raise ValueError("k must be positive.")
    reference_set = set(reference.dropna().nlargest(k).index)
    candidate_set = set(candidate.dropna().nlargest(k).index)
    union = reference_set | candidate_set
    if not union:
        return float("nan")
    return float(len(reference_set & candidate_set) / len(union))


def ranking_stability(
    reference: pd.Series,
    candidate: pd.Series,
    *,
    eligible_nodes: pd.Index,
    top_k_values: tuple[int, ...] = (20, 50),
) -> dict[str, float | int]:
    """Compare candidate values with a fixed reference ranking universe."""
    reference_subset = reference.reindex(eligible_nodes)
    candidate_subset = candidate.reindex(eligible_nodes)
    common = reference_subset.notna() & candidate_subset.notna()
    common_reference = reference_subset.loc[common]
    common_candidate = candidate_subset.loc[common]

    if len(common_reference) >= 2:
        spearman = float(common_reference.corr(common_candidate, method="spearman"))
    else:
        spearman = float("nan")

    result: dict[str, float | int] = {
        "reference_eligible_nodes": int(len(eligible_nodes)),
        "common_finite_nodes": int(common.sum()),
        "candidate_missing_nodes": int(len(eligible_nodes) - common.sum()),
        "spearman": spearman,
    }
    for k in top_k_values:
        result[f"top_{k}_jaccard"] = top_k_jaccard(
            reference_subset,
            candidate_subset,
            k=k,
        )
    return result
