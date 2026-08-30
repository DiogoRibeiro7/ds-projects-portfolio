"""Tests for frozen v2 empirical inputs."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from mobility_optimization.frozen_inputs import (
    FROZEN_RELOCATION_SOURCE_SHA256,
    FROZEN_ZONES,
    load_frozen_relocation_matrix,
)


def test_committed_frozen_matrix_matches_original_artifact() -> None:
    """The repository copy must be byte-identical to the v1.1 workflow artifact."""
    path = Path("evidence/v2_relocation_cost_matrix.csv")
    zones, matrix, repository_sha = load_frozen_relocation_matrix(path)
    assert zones == FROZEN_ZONES
    assert matrix.shape == (30, 30)
    assert np.array_equal(matrix, matrix.T)
    assert np.array_equal(np.diag(matrix), np.zeros(30))
    assert repository_sha == FROZEN_RELOCATION_SOURCE_SHA256
    assert hashlib.sha256(path.read_bytes()).hexdigest() == FROZEN_RELOCATION_SOURCE_SHA256


def test_byte_lock_detects_numerical_change(tmp_path: Path) -> None:
    """A changed matrix value must fail the frozen-input gate."""
    source = Path("evidence/v2_relocation_cost_matrix.csv").read_text(encoding="utf-8")
    changed = source.replace("0.14849879019927373", "0.14849979019927373", 1)
    path = tmp_path / "changed.csv"
    path.write_text(changed, encoding="utf-8")
    try:
        load_frozen_relocation_matrix(path)
    except ValueError as exc:
        assert "byte checksum mismatch" in str(exc)
    else:
        raise AssertionError("Changed relocation matrix unexpectedly passed the frozen-input gate.")
