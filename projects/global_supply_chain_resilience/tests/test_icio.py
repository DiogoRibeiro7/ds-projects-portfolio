"""Tests for OECD ICIO ingestion and accounting validation."""

from __future__ import annotations

from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import pytest

from supply_chain_resilience.icio import load_icio_csv, load_icio_zip, technical_coefficients


def test_load_icio_csv_preserves_labels_and_hash(tmp_path) -> None:
    """Raw row/column identifiers and exact-file provenance must be preserved."""
    path = tmp_path / "icio.csv"
    path.write_text(
        "node,USA_A,PRT_B,HFCE\nUSA_A,10,2,4\nPRT_B,1,8,3\n",
        encoding="utf-8",
    )

    table = load_icio_csv(path)

    assert table.source_name == "icio.csv"
    assert len(table.source_sha256) == 64
    assert list(table.frame.index) == ["USA_A", "PRT_B"]
    assert list(table.frame.columns) == ["USA_A", "PRT_B", "HFCE"]


def test_load_icio_zip_requires_unambiguous_member(tmp_path) -> None:
    """Archives with multiple CSV files require an explicit member selection."""
    archive_path = tmp_path / "icio.zip"
    buffer = BytesIO()
    with ZipFile(buffer, mode="w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("a.csv", "node,A,B\nA,1,0\nB,0,1\n")
        archive.writestr("b.csv", "node,A,B\nA,2,0\nB,0,2\n")
    archive_path.write_bytes(buffer.getvalue())

    with pytest.raises(ValueError, match="exactly one CSV"):
        load_icio_zip(archive_path)

    table = load_icio_zip(archive_path, member="b.csv")
    assert table.source_name == "b.csv"
    assert table.frame.loc["A", "A"] == 2


def test_raw_table_allows_negative_inventory_like_values(tmp_path) -> None:
    """Raw final-demand entries may be negative before economic components are mapped."""
    path = tmp_path / "raw.csv"
    path.write_text("node,A,INVNT\nA,1,-0.5\n", encoding="utf-8")

    table = load_icio_csv(path)

    assert table.frame.loc["A", "INVNT"] == pytest.approx(-0.5)


def test_technical_coefficients_match_definition() -> None:
    """Each using-industry column is divided by that industry's gross output."""
    intermediate = pd.DataFrame(
        [[20.0, 10.0], [5.0, 30.0]],
        index=["supplier_a", "supplier_b"],
        columns=["industry_a", "industry_b"],
    )
    output = pd.Series([100.0, 100.0], index=intermediate.columns)

    coefficients = technical_coefficients(intermediate, output)

    assert coefficients.loc["supplier_a", "industry_a"] == pytest.approx(0.2)
    assert coefficients.loc["supplier_b", "industry_b"] == pytest.approx(0.3)
    assert coefficients.sum(axis=0).to_dict() == pytest.approx(
        {"industry_a": 0.25, "industry_b": 0.40}
    )


def test_technical_coefficients_reject_negative_intermediate_use() -> None:
    """Negative values remain invalid once the intermediate-use block is identified."""
    intermediate = pd.DataFrame(
        [[20.0], [-1.0]],
        index=["supplier_a", "supplier_b"],
        columns=["industry_a"],
    )
    output = pd.Series([100.0], index=intermediate.columns)

    with pytest.raises(ValueError, match="non-negative"):
        technical_coefficients(intermediate, output)


def test_technical_coefficients_reject_impossible_column_sum() -> None:
    """Intermediate-input coefficients above one fail the accounting gate."""
    intermediate = pd.DataFrame(
        [[70.0], [50.0]],
        index=["supplier_a", "supplier_b"],
        columns=["industry_a"],
    )
    output = pd.Series([100.0], index=intermediate.columns)

    with pytest.raises(ValueError, match="exceed one"):
        technical_coefficients(intermediate, output)
