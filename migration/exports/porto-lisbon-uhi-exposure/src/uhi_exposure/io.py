"""Input/output helpers for the UHI exposure workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

DEFAULT_ENCODING: Final[str] = "utf-8"


def load_exposure_cells(path: str | Path) -> pd.DataFrame:
    """Load a prepared cell-level exposure table from CSV or Parquet.

    Parameters
    ----------
    path:
        Path to a CSV or Parquet file.

    Returns
    -------
    pandas.DataFrame
        Loaded data.

    Raises
    ------
    ValueError
        If the extension is unsupported.
    """
    input_path = Path(path)
    suffix = input_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(input_path, encoding=DEFAULT_ENCODING)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(input_path)

    raise ValueError(f"Unsupported input format: {suffix}. Use CSV or Parquet.")


def write_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    """Write a DataFrame to CSV or Parquet based on the file extension."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    if suffix == ".csv":
        df.to_csv(output_path, index=False, encoding=DEFAULT_ENCODING)
        return
    if suffix in {".parquet", ".pq"}:
        df.to_parquet(output_path, index=False)
        return

    raise ValueError(f"Unsupported output format: {suffix}. Use CSV or Parquet.")
