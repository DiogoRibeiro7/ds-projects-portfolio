from __future__ import annotations

import pandas as pd
import pytest

from adaptive_policy_learning.study2_empirical import (
    EXPECTED_QUALIFIED_COEFFICIENT_SHA256,
    EXPECTED_QUALIFIED_FEATURES,
    EXPECTED_QUALIFIED_N_ITER,
    _assert_qualification_reproduced,
)
from adaptive_policy_learning.study2_evaluation import _timestamps_to_ns


def _qualification() -> dict[str, object]:
    return {
        "training": {
            "coefficient_sha256": EXPECTED_QUALIFIED_COEFFICIENT_SHA256,
            "n_features": EXPECTED_QUALIFIED_FEATURES,
            "n_iter": EXPECTED_QUALIFIED_N_ITER,
            "warnings": [],
        },
        "runtime": {
            "python": "3.11.16",
            "numpy": "2.4.6",
            "pandas": "3.0.5",
            "scipy": "1.17.1",
            "scikit_learn": "1.9.0",
        },
    }


def test_study2_evaluation_authorization_rejects_coefficient_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "adaptive_policy_learning.study2_empirical._runtime_versions",
        lambda: {
            "python": "3.11.16",
            "numpy": "2.4.6",
            "pandas": "3.0.5",
            "scipy": "1.17.1",
            "scikit_learn": "1.9.0",
        },
    )

    with pytest.raises(RuntimeError, match="coefficient SHA mismatch"):
        _assert_qualification_reproduced(
            _qualification(),
            coefficient_sha256="0" * 64,
            n_features=EXPECTED_QUALIFIED_FEATURES,
            n_iter=EXPECTED_QUALIFIED_N_ITER,
            warnings_captured=[],
        )


def test_study2_timestamp_conversion_is_explicitly_nanoseconds() -> None:
    timestamp = "2019-11-28 16:55:17.867529+00:00"
    values = pd.Series([timestamp])

    converted = _timestamps_to_ns(values)

    assert converted.tolist() == [pd.Timestamp(timestamp).value]
