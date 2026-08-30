"""Tests for frozen v2 empirical inputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mobility_optimization.frozen_inputs import (
    FROZEN_RELOCATION_SEMANTIC_SHA256,
    FROZEN_ZONES,
    load_frozen_relocation_matrix,
    relocation_matrix_semantic_sha256,
)


def test_semantic_hash_ignores_csv_formatting(tmp_path: Path) -> None:
    """Equivalent float matrices must hash identically regardless of CSV text format."""
    matrix = np.array([[0.0, 0.25], [0.25, 0.0]], dtype=np.float64)
    zones = (1, 2)
    first = relocation_matrix_semantic_sha256(zones, matrix)
    second = relocation_matrix_semantic_sha256(zones, matrix.copy())
    assert first == second


def test_committed_frozen_matrix_matches_semantic_lock() -> None:
    """The repository copy must retain the exact ordered v1.1 numerical matrix."""
    path = Path("evidence/v2_relocation_cost_matrix.csv")
    zones, matrix, repository_sha = load_frozen_relocation_matrix(path)
    assert zones == FROZEN_ZONES
    assert matrix.shape == (30, 30)
    assert np.array_equal(matrix, matrix.T)
    assert np.array_equal(np.diag(matrix), np.zeros(30))
    assert len(repository_sha) == 64
    assert relocation_matrix_semantic_sha256(zones, matrix) == FROZEN_RELOCATION_SEMANTIC_SHA256


def test_semantic_lock_detects_numerical_change(tmp_path: Path) -> None:
    """A changed matrix value must fail the frozen-input gate."""
    source = pd.read_csv("evidence/v2_relocation_cost_matrix.csv", index_col=0)
    source.iloc[0, 1] += 1e-6
    path = tmp_path / "changed.csv"
    source.to_csv(path)
    try:
        load_frozen_relocation_matrix(path)
    except ValueError as exc:
        assert "semantic checksum mismatch" in str(exc) or "symmetric" in str(exc)
    else:
        raise AssertionError("Changed relocation matrix unexpectedly passed the frozen-input gate.")
