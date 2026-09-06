from __future__ import annotations

import pandas as pd
import pytest

from adaptive_policy_learning.study3_empirical import (
    EXPECTED_QUALIFIED_COEFFICIENT_SHA256,
    EXPECTED_QUALIFIED_FEATURES,
    EXPECTED_QUALIFIED_N_ITER,
    _assert_qualification_reproduced,
)
from adaptive_policy_learning.study3_empirical_erratum import _parse_mixed_utc


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


def _patch_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "adaptive_policy_learning.study3_empirical._runtime_versions",
        lambda: {
            "python": "3.11.16",
            "numpy": "2.4.6",
            "pandas": "3.0.5",
            "scipy": "1.17.1",
            "scikit_learn": "1.9.0",
        },
    )


def test_study3_authorization_rejects_coefficient_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_runtime(monkeypatch)
    with pytest.raises(RuntimeError, match="coefficient SHA mismatch"):
        _assert_qualification_reproduced(
            _qualification(),
            coefficient_sha256="0" * 64,
            n_features=EXPECTED_QUALIFIED_FEATURES,
            n_iter=EXPECTED_QUALIFIED_N_ITER,
            warnings_captured=[],
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("n_features", EXPECTED_QUALIFIED_FEATURES + 1, "feature-count mismatch"),
        ("n_iter", EXPECTED_QUALIFIED_N_ITER + 1, "optimizer-iteration mismatch"),
    ],
)
def test_study3_authorization_rejects_training_state_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: int,
    message: str,
) -> None:
    _patch_runtime(monkeypatch)
    kwargs = {
        "coefficient_sha256": EXPECTED_QUALIFIED_COEFFICIENT_SHA256,
        "n_features": EXPECTED_QUALIFIED_FEATURES,
        "n_iter": EXPECTED_QUALIFIED_N_ITER,
        "warnings_captured": [],
    }
    kwargs[field] = value
    with pytest.raises(RuntimeError, match=message):
        _assert_qualification_reproduced(_qualification(), **kwargs)


def test_study3_authorization_rejects_warning_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_runtime(monkeypatch)
    with pytest.raises(RuntimeError, match="optimizer warnings"):
        _assert_qualification_reproduced(
            _qualification(),
            coefficient_sha256=EXPECTED_QUALIFIED_COEFFICIENT_SHA256,
            n_features=EXPECTED_QUALIFIED_FEATURES,
            n_iter=EXPECTED_QUALIFIED_N_ITER,
            warnings_captured=["warning"],
        )


def test_study3_authorization_rejects_runtime_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "adaptive_policy_learning.study3_empirical._runtime_versions",
        lambda: {
            "python": "3.11.99",
            "numpy": "2.4.6",
            "pandas": "3.0.5",
            "scipy": "1.17.1",
            "scikit_learn": "1.9.0",
        },
    )
    with pytest.raises(RuntimeError, match="runtime mismatch for python"):
        _assert_qualification_reproduced(
            _qualification(),
            coefficient_sha256=EXPECTED_QUALIFIED_COEFFICIENT_SHA256,
            n_features=EXPECTED_QUALIFIED_FEATURES,
            n_iter=EXPECTED_QUALIFIED_N_ITER,
            warnings_captured=[],
        )


def test_study3_mixed_iso_timestamp_erratum_preserves_instants() -> None:
    values = pd.Series(
        [
            "2019-11-28 15:23:48.989271+00:00",
            "2019-11-30 09:20:56+00:00",
        ]
    )

    parsed = _parse_mixed_utc(values)

    assert parsed.iloc[0] == pd.Timestamp("2019-11-28 15:23:48.989271+00:00")
    assert parsed.iloc[1] == pd.Timestamp("2019-11-30 09:20:56+00:00")
