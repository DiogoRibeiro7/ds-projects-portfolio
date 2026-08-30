"""Deterministic candidate persistence utilities for supplier shock selection."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


def persistent_top_k(
    rankings: Mapping[float, pd.Series],
    *,
    k: int,
) -> list[str]:
    """Return nodes present in the descending top-k set at every threshold.

    Ordering follows the unthresholded ranking when threshold ``0.0`` is available;
    otherwise the smallest supplied threshold is used as the reference ordering.
    """
    if not rankings:
        raise ValueError("rankings must not be empty.")
    if k <= 0:
        raise ValueError("k must be positive.")

    reference_threshold = 0.0 if 0.0 in rankings else min(rankings)
    top_sets = {
        threshold: set(series.dropna().nlargest(k).index.astype(str))
        for threshold, series in rankings.items()
    }
    persistent = set.intersection(*top_sets.values())
    reference_order = [
        str(label)
        for label in rankings[reference_threshold].dropna().nlargest(k).index
        if str(label) in persistent
    ]
    return reference_order


def real_country_candidates(nodes: list[str]) -> list[str]:
    """Remove aggregate rest-of-world nodes from real-country shock candidates."""
    return [node for node in nodes if not node.startswith("ROW_")]
