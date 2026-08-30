"""Frozen empirical inputs reused by the preregistered v2 mobility study."""

from __future__ import annotations

import csv
import hashlib
import io
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path

import numpy as np
import pandas as pd

FROZEN_RELOCATION_SOURCE_SHA256 = (
    "bf3ebdf7eaa8391c4a5c4554fbb39d0a098f5d4fc31af429cd39f7b4b17bb8b4"
)
FROZEN_RELOCATION_CANONICAL_SHA256 = (
    "453c50bd86326f784cd1ce6c2158c96454891302aca74a003b816775d55c126d"
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


def relocation_matrix_canonical_sha256(path: Path) -> str:
    """Hash decimal CSV content in a parser-independent canonical form.

    The ordered zone labels are preserved exactly. Matrix values are parsed with
    :class:`decimal.Decimal`, rounded to 12 decimal places with half-even rounding,
    and emitted with exactly 12 decimal places before hashing. This avoids any
    dependence on pandas or NumPy float parsing and rounding implementations.
    """
    text = path.read_text(encoding="utf-8-sig")
    rows = list(csv.reader(io.StringIO(text)))
    if len(rows) != len(FROZEN_ZONES) + 1:
        raise ValueError("Frozen v2 relocation matrix CSV must contain one header and 30 rows.")
    if len(rows[0]) != len(FROZEN_ZONES) + 1:
        raise ValueError("Frozen v2 relocation matrix CSV header must contain 30 zones.")

    header_zones = tuple(int(value.strip()) for value in rows[0][1:])
    if header_zones != FROZEN_ZONES:
        raise ValueError("Frozen v2 relocation matrix column order differs from v1.1.")

    quantum = Decimal(1).scaleb(-FROZEN_RELOCATION_CANONICAL_DECIMALS)
    canonical_lines = [",".join(str(zone) for zone in FROZEN_ZONES)]
    for expected_zone, row in zip(FROZEN_ZONES, rows[1:], strict=True):
        if len(row) != len(FROZEN_ZONES) + 1:
            raise ValueError("Frozen v2 relocation matrix rows must each contain 30 costs.")
        row_zone = int(row[0].strip())
        if row_zone != expected_zone:
            raise ValueError("Frozen v2 relocation matrix row order differs from v1.1.")
        values: list[str] = []
        for token in row[1:]:
            value = Decimal(token.strip()).quantize(quantum, rounding=ROUND_HALF_EVEN)
            values.append(format(value, f".{FROZEN_RELOCATION_CANONICAL_DECIMALS}f"))
        canonical_lines.append(f"{row_zone}," + ",".join(values))

    canonical = ("\n".join(canonical_lines) + "\n").encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def load_frozen_relocation_matrix(path: Path) -> tuple[tuple[int, ...], np.ndarray, str]:
    """Load and verify the exact v1.1 relocation matrix reused by v2.

    The original workflow-artifact byte checksum is retained as provenance. The
    executable invariant is a parser-independent canonical decimal checksum derived
    from that artifact. Structural numerical checks are applied after parsing.
    """
    canonical = relocation_matrix_canonical_sha256(path)
    if canonical != FROZEN_RELOCATION_CANONICAL_SHA256:
        raise ValueError(f"Frozen v2 relocation matrix canonical checksum mismatch: {canonical}")

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

    repository_bytes_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    return FROZEN_ZONES, matrix, repository_bytes_sha256
