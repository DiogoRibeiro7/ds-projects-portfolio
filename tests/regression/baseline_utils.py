"""Shared helpers for regression baselines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_conversion_dataset(seed: int = 202401) -> pd.DataFrame:
    """Create a deterministic two-arm experiment DataFrame.

    Args:
        seed: Random seed for reproducibility.

    Returns:
        pandas.DataFrame with ``user_id``, ``group``, and ``converted`` columns.
    """
    rng = np.random.default_rng(seed)
    n_control = 800
    n_treatment = 820

    control = pd.DataFrame(
        {
            "user_id": range(n_control),
            "group": "control",
            "converted": rng.binomial(1, 0.12, size=n_control),
        }
    )
    treatment = pd.DataFrame(
        {
            "user_id": range(n_control, n_control + n_treatment),
            "group": "treatment",
            "converted": rng.binomial(1, 0.15, size=n_treatment),
        }
    )
    df = pd.concat([control, treatment], ignore_index=True)
    return df
