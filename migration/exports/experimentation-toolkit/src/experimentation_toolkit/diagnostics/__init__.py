"""Experiment diagnostic summaries."""

from __future__ import annotations

import pandas as pd


def summarize_groups(
    df: pd.DataFrame,
    group_col: str,
    metric_col: str,
) -> pd.DataFrame:
    """Return count, mean, standard deviation, and standard error by group."""

    missing = [col for col in (group_col, metric_col) if col not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    summary = (
        df.dropna(subset=[group_col, metric_col])
        .groupby(group_col, observed=True)[metric_col]
        .agg(["count", "mean", "std"])
        .rename(columns={"count": "n"})
    )
    summary["standard_error"] = summary["std"] / summary["n"].pow(0.5)
    return summary.reset_index()
