from __future__ import annotations

import numpy as np
import pytest

from adaptive_policy_learning.empirical import moving_block_bootstrap_mean


def test_moving_block_bootstrap_is_deterministic() -> None:
    hour_ns = 3_600_000_000_000
    timestamps = np.arange(72, dtype=np.int64) * hour_ns
    values = np.linspace(-0.5, 0.75, 72)

    left = moving_block_bootstrap_mean(
        timestamps,
        values,
        block_length_hours=24,
        replications=41,
        seed=17,
        confidence_level=0.95,
    )
    right = moving_block_bootstrap_mean(
        timestamps,
        values,
        block_length_hours=24,
        replications=41,
        seed=17,
        confidence_level=0.95,
    )

    assert left == right
    assert left["replications"] == 41
    assert float(left["lower"]) <= float(left["upper"])
    assert float(left["point_estimate"]) == pytest.approx(float(np.mean(values)))


def test_moving_block_bootstrap_rejects_unsorted_timestamps() -> None:
    timestamps = np.array([0, 2, 1], dtype=np.int64)
    values = np.array([0.1, 0.2, 0.3])

    with pytest.raises(ValueError, match="non-decreasing"):
        moving_block_bootstrap_mean(
            timestamps,
            values,
            block_length_hours=1,
            replications=5,
            seed=1,
            confidence_level=0.95,
        )
