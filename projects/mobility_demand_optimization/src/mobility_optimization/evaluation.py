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
