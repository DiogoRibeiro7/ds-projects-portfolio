"""Frozen empirical inputs reused by the preregistered v2 mobility study."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

FROZEN_RELOCATION_SOURCE_SHA256 = (
    "bf3ebdf7eaa8391c4a5c4554fbb39d0a098f5d4fc31af429cd39f7b4b17bb8b4"
)
FROZEN_RELOCATION_SEMANTIC_SHA256 = (
    "6e6436a44be238f240f150c1e07e63400b44672b521bcddaa5c234d6e27ba8a2"
)
FROZEN_RELOCATION_CANONICAL_DECIMALS = 12
FROZEN_ZONES: tuple[int, ...] = (
    48,
    68,
    79,
    90,
    100,
    107,
    113,
    114,
    132,
    138,
    140,
    141,
    142,
    161,
    162,
    163,
    164,
    170,
    186,
    229,
    230,
    231,
    234,
    236,
    237,
    238,
    239,
    246,
    249,
    263,
)


def relocation_matrix_semantic_sha256(
    zones: tuple[int, ...],
    matrix: np.ndarray,
) -> str:
    """Hash ordered zones and canonically rounded matrix values.

    Rounding to 12 decimal places removes parser/CSV round-trip noise while being
    far tighter than any operationally meaningful cost difference in this study.
    """
    zone_bytes = np.asarray(zones, dtype="<i8").tobytes(order="C")
    canonical = np.round(
        np.asarray(matrix, dtype=np.float64),
        decimals=FROZEN_RELOCATION_CANONICAL_DECIMALS,
    )
    matrix_bytes = np.asarray(canonical, dtype="<f8").tobytes(order="C")
    digest = hashlib.sha256()
    digest.update(zone_bytes)
    digest.update(matrix_bytes)
    return digest.hexdigest()


def load_frozen_relocation_matrix(path: Path) -> tuple[tuple[int, ...], np.ndarray, str]:
    """Load and verify the exact v1.1 relocation matrix reused by v2.

    The original workflow-artifact byte checksum is retained as provenance. The
    executable invariant hashes ordered zone IDs plus matrix values rounded to 12
    decimal places, so harmless CSV round-trip noise cannot block the study while
    any scientifically meaningful numerical or ordering drift still fails.
    """
    frame = pd.read_csv(path, index_col=0)
    frame.index = frame.index.astype(int)
    frame.columns = frame.columns.astype(int)
    rows = tuple(int(value) for value in frame.index)
    columns = tuple(int(value) for value in frame.columns)
    if rows != FROZEN_ZONES or columns != FROZEN_ZONES:
        raise ValueError("Frozen v2 relocation matrix zone order differs from v1.1.")

    matrix = frame.to_numpy(dtype=np.float64)
    if matrix.shape != (30, 30):
        raise ValueError("Frozen v2 relocation matrix must be 30x30.")
    if not np.isfinite(matrix).all() or (matrix < 0.0).any():
        raise ValueError("Frozen v2 relocation matrix must be finite and non-negative.")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-15):
        raise ValueError("Frozen v2 relocation matrix must be symmetric to numerical precision.")
    if not np.allclose(np.diag(matrix), 0.0, rtol=0.0, atol=0.0):
        raise ValueError("Frozen v2 relocation matrix diagonal must be exactly zero.")

    mask = ~np.eye(matrix.shape[0], dtype=bool)
    if not np.isclose(np.median(matrix[mask]), 0.25, rtol=0.0, atol=1e-15):
        raise ValueError("Frozen v2 relocation matrix median off-diagonal cost must be 0.25.")

    semantic = relocation_matrix_semantic_sha256(FROZEN_ZONES, matrix)
    if semantic != FROZEN_RELOCATION_SEMANTIC_SHA256:
        raise ValueError(f"Frozen v2 relocation matrix semantic checksum mismatch: {semantic}")
    repository_bytes_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    return FROZEN_ZONES, matrix, repository_bytes_sha256
