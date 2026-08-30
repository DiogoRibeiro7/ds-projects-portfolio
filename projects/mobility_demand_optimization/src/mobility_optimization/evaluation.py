"""Shared evaluation-contract helpers for mobility experiments."""

from __future__ import annotations

import pandas as pd


def headline_mask(frame: pd.DataFrame) -> pd.Series:
    """Return rows eligible for prospectively declared headline metrics."""
    if "is_dst_transition_day" not in frame.columns:
        return pd.Series(True, index=frame.index, dtype=bool)
    return ~frame["is_dst_transition_day"].astype(bool)


def headline_timestamp_eligible(frame: pd.DataFrame) -> bool:
    """Return whether one timestamp group is eligible for headline metrics."""
    if frame.empty:
        raise ValueError("Timestamp group must not be empty.")
    mask = headline_mask(frame)
    if mask.nunique() != 1:
        raise ValueError("All zones in one timestamp group must share headline eligibility.")
    return bool(mask.iloc[0])


def filter_headline_policy_results(
    results: pd.DataFrame,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Filter policy-hour results using the panel's prospective DST marker.

    Stateful simulations should still propagate every hourly state. This helper is
    therefore applied only at summary time, after the full trajectory has run.
    """
    required_results = {"timestamp", "policy"}
    missing_results = required_results.difference(results.columns)
    if missing_results:
        raise ValueError(f"Policy results are missing columns: {sorted(missing_results)}")
    if "timestamp" not in panel.columns:
        raise ValueError("Panel is missing timestamp.")

    output = results.copy()
    output["timestamp"] = pd.to_datetime(output["timestamp"], errors="coerce")
    if output["timestamp"].isna().any():
        raise ValueError("Policy results contain invalid timestamps.")

    if "is_dst_transition_day" not in panel.columns:
        return output

    eligibility = panel[["timestamp", "is_dst_transition_day"]].copy()
    eligibility["timestamp"] = pd.to_datetime(eligibility["timestamp"], errors="coerce")
    eligibility = eligibility.drop_duplicates()
    conflicting = eligibility.groupby("timestamp")["is_dst_transition_day"].nunique()
    if (conflicting > 1).any():
        raise ValueError("Panel contains conflicting DST eligibility within a timestamp.")
    eligibility = eligibility.drop_duplicates("timestamp")

    joined = output.merge(eligibility, on="timestamp", how="left", validate="many_to_one")
    if joined["is_dst_transition_day"].isna().any():
        raise ValueError("Policy results contain timestamps absent from the panel.")
    return joined.loc[~joined["is_dst_transition_day"].astype(bool)].drop(
        columns=["is_dst_transition_day"]
    )
