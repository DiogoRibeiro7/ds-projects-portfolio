"""Tests for frozen v2 empirical inputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mobility_optimization.frozen_inputs import (
    FROZEN_RELOCATION_CANONICAL_SHA256,
    FROZEN_ZONES,
    load_frozen_relocation_matrix,
    relocation_matrix_canonical_sha256,
)


def test_canonical_hash_ignores_equivalent_decimal_formatting(tmp_path: Path) -> None:
    """Equivalent decimal tokens must canonicalize to the same checksum."""
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    header = "," + ",".join(str(zone) for zone in FROZEN_ZONES)
    rows_first = [header]
    rows_second = [header]
    for row_zone in FROZEN_ZONES:
        values_first = ["0.250000000000" for _ in FROZEN_ZONES]
        values_second = ["0.25" for _ in FROZEN_ZONES]
        rows_first.append(f"{row_zone}," + ",".join(values_first))
        rows_second.append(f"{row_zone}," + ",".join(values_second))
    first.write_text("\n".join(rows_first) + "\n", encoding="utf-8")
    second.write_text("\n".join(rows_second) + "\n", encoding="utf-8")
    assert relocation_matrix_canonical_sha256(first) == relocation_matrix_canonical_sha256(second)


def test_committed_frozen_matrix_matches_canonical_lock() -> None:
    """The repository copy must retain the frozen ordered v1.1 matrix."""
    path = Path("evidence/v2_relocation_cost_matrix.csv")
    zones, matrix, repository_sha = load_frozen_relocation_matrix(path)
    assert zones == FROZEN_ZONES
    assert matrix.shape == (30, 30)
    assert np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-15)
    assert np.array_equal(np.diag(matrix), np.zeros(30))
    assert len(repository_sha) == 64
    assert relocation_matrix_canonical_sha256(path) == FROZEN_RELOCATION_CANONICAL_SHA256


def test_canonical_lock_detects_numerical_change(tmp_path: Path) -> None:
    """A materially changed matrix value must fail the frozen-input gate."""
    source = pd.read_csv("evidence/v2_relocation_cost_matrix.csv", index_col=0)
    source.iloc[0, 1] += 1e-6
    path = tmp_path / "changed.csv"
    source.to_csv(path)
    try:
        load_frozen_relocation_matrix(path)
    except ValueError as exc:
        assert "canonical checksum mismatch" in str(exc) or "symmetric" in str(exc)
    else:
        raise AssertionError("Changed relocation matrix unexpectedly passed the frozen-input gate.")
