"""OECD ICIO ingestion and validation utilities.

The raw OECD tables are treated as accounting matrices. This module deliberately
separates ingestion/validation from graph construction and stress testing so that
later analyses cannot silently reinterpret malformed or incomplete tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ICIOTable:
    """Validated raw ICIO table plus immutable source metadata."""

    frame: pd.DataFrame
    source_name: str
    source_sha256: str


def _validate_sha256(value: str) -> None:
    """Validate a hexadecimal SHA-256 digest."""
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("source_sha256 must be a lowercase hexadecimal SHA-256 digest.")


def _validate_frame(frame: pd.DataFrame) -> None:
    """Reject structurally invalid raw ICIO tables before downstream analysis."""
    if frame.empty:
        raise ValueError("ICIO table must not be empty.")
    if frame.index.has_duplicates:
        raise ValueError("ICIO row labels must be unique.")
    if frame.columns.has_duplicates:
        raise ValueError("ICIO column labels must be unique.")
    if frame.shape[1] < 2:
        raise ValueError("ICIO table must contain at least two numeric columns.")

    numeric = frame.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().all(axis=None):
        raise ValueError("ICIO table contains no numeric accounting values.")
    values = numeric.to_numpy(dtype=np.float64, na_value=np.nan)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("ICIO table contains no finite accounting values.")


def _parse_csv_bytes(raw: bytes, *, source_name: str, digest: str) -> ICIOTable:
    """Parse a CSV payload while preserving the first column as row identifiers."""
    _validate_sha256(digest)
    if not raw:
        raise ValueError("ICIO source is empty.")

    try:
        frame = pd.read_csv(BytesIO(raw), index_col=0, low_memory=False)
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not parse ICIO CSV {source_name!r}.") from exc

    frame.index = frame.index.map(str)
    frame.columns = frame.columns.map(str)
    _validate_frame(frame)
    return ICIOTable(frame=frame, source_name=source_name, source_sha256=digest)


def load_icio_csv(path: str | Path) -> ICIOTable:
    """Load and validate one OECD ICIO CSV file."""
    source = Path(path)
    raw = source.read_bytes()
    digest = sha256(raw).hexdigest()
    return _parse_csv_bytes(raw, source_name=source.name, digest=digest)


def load_icio_zip(path: str | Path, *, member: str | None = None) -> ICIOTable:
    """Load one CSV member from an OECD ICIO ZIP archive."""
    source = Path(path)
    raw_archive = source.read_bytes()
    digest = sha256(raw_archive).hexdigest()

    try:
        with ZipFile(BytesIO(raw_archive)) as archive:
            csv_members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if member is None:
                if len(csv_members) != 1:
                    raise ValueError(
                        "ICIO archive must contain exactly one CSV when member is not specified."
                    )
                selected = csv_members[0]
            else:
                if member not in csv_members:
                    raise ValueError(f"CSV member {member!r} was not found in the ICIO archive.")
                selected = member
            raw_csv = archive.read(selected)
    except BadZipFile as exc:
        raise ValueError("ICIO source is not a valid ZIP archive.") from exc

    return _parse_csv_bytes(raw_csv, source_name=selected, digest=digest)


def technical_coefficients(
    intermediate_use: pd.DataFrame,
    gross_output: pd.Series,
) -> pd.DataFrame:
    """Compute the technical-coefficient matrix ``A`` from intermediate use ``Z``.

    For supplier row ``i`` and using industry column ``j``:

    ``A[i, j] = Z[i, j] / x[j]``.

    Gross output must be strictly positive for every using industry. Coefficient
    construction does not require each column sum to be at most one: that stronger
    condition is neither part of the definition of ``A`` nor necessary for graph
    analysis. Productivity/invertibility checks belong to any later Leontief layer.
    """
    if intermediate_use.empty:
        raise ValueError("intermediate_use must not be empty.")
    if intermediate_use.columns.has_duplicates or intermediate_use.index.has_duplicates:
        raise ValueError("intermediate_use labels must be unique.")
    if not intermediate_use.columns.equals(gross_output.index):
        raise ValueError("gross_output index must exactly match intermediate_use columns.")

    z = intermediate_use.apply(pd.to_numeric, errors="raise").astype(float)
    x = pd.to_numeric(gross_output, errors="raise").astype(float)
    if not np.all(np.isfinite(z.to_numpy())) or not np.all(np.isfinite(x.to_numpy())):
        raise ValueError("intermediate use and gross output must contain only finite values.")
    if (z < 0.0).any(axis=None):
        raise ValueError("intermediate use must be non-negative.")
    if (x <= 0.0).any():
        raise ValueError("gross output must be strictly positive.")

    coefficients = z.divide(x, axis="columns")
    if not np.all(np.isfinite(coefficients.to_numpy())):
        raise ValueError("technical coefficients must contain only finite values.")
    if (coefficients < 0.0).any(axis=None):
        raise ValueError("technical coefficients must be non-negative.")
    return coefficients
