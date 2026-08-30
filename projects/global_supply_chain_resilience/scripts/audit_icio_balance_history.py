"""Audit and gate OECD ICIO accounting residuals across the official release archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from supply_chain_resilience.diagnostics import accounting_residuals, relative_residuals
from supply_chain_resilience.mapping import (
    RELEASE_BALANCE_ATOL,
    RELEASE_BALANCE_RTOL,
    extract_2022_blocks,
    validate_2022_accounting,
)

YEARS = tuple(range(2016, 2023))


def _load_member(archive: ZipFile, member: str) -> pd.DataFrame:
    """Load one ICIO CSV member with its first column as the row index."""
    with archive.open(member) as handle:
        return pd.read_csv(handle, index_col=0, low_memory=False)


def _member_for_year(archive: ZipFile, year: int) -> str:
    """Return the unique CSV member whose stem contains ``year``."""
    members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
    matches = [name for name in members if str(year) in Path(name).stem]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one CSV for {year}, found {matches!r}.")
    return matches[0]


def _activity(label: str) -> str:
    """Return the activity code in a COUNTRY_ACTIVITY ICIO label."""
    if "_" not in label:
        return label
    return label.split("_", maxsplit=1)[1]


def _year_report(frame: pd.DataFrame, *, year: int, member: str) -> dict[str, object]:
    """Build accounting diagnostics and evaluate the frozen release envelope."""
    blocks = extract_2022_blocks(frame)
    row_residual, column_residual = accounting_residuals(blocks)
    row_relative = relative_residuals(row_residual, blocks.gross_output)
    column_relative = relative_residuals(column_residual, blocks.gross_output)

    allowance = RELEASE_BALANCE_ATOL + RELEASE_BALANCE_RTOL * blocks.gross_output.abs()
    row_excess = row_residual.abs() - allowance
    column_excess = column_residual.abs() - allowance

    activity_rows = pd.DataFrame(
        {
            "activity": [_activity(str(label)) for label in row_residual.index],
            "absolute_residual": row_residual.abs().to_numpy(dtype=float),
        },
        index=row_residual.index,
    )
    by_activity = (
        activity_rows.groupby("activity", sort=True)["absolute_residual"]
        .sum()
        .sort_values(ascending=False)
    )

    top_row = row_residual.abs().sort_values(ascending=False).head(10)
    top_column = column_residual.abs().sort_values(ascending=False).head(10)

    gate_status = "PASS"
    gate_error: str | None = None
    try:
        validate_2022_accounting(blocks)
    except ValueError as exc:
        gate_status = "FAIL"
        gate_error = str(exc)

    return {
        "year": year,
        "archive_member": member,
        "release_balance_gate": gate_status,
        "release_balance_gate_error": gate_error,
        "release_balance_atol": RELEASE_BALANCE_ATOL,
        "release_balance_rtol": RELEASE_BALANCE_RTOL,
        "max_row_envelope_excess": float(max(0.0, row_excess.max())),
        "max_column_envelope_excess": float(max(0.0, column_excess.max())),
        "row_envelope_violations": int((row_excess > 0.0).sum()),
        "column_envelope_violations": int((column_excess > 0.0).sum()),
        "published_industry_labels": int(len(blocks.gross_output)),
        "zero_output_labels": int((blocks.gross_output == 0.0).sum()),
        "max_abs_row_balance_error": float(row_residual.abs().max()),
        "max_abs_column_balance_error": float(column_residual.abs().max()),
        "max_relative_row_balance_error": float(row_relative.max()),
        "max_relative_column_balance_error": float(column_relative.max()),
        "global_row_residual_sum": float(row_residual.sum()),
        "global_column_residual_sum": float(column_residual.sum()),
        "gross_output_total": float(blocks.gross_output.sum()),
        "top_row_residuals": [
            {
                "label": str(label),
                "activity": _activity(str(label)),
                "residual": float(row_residual.loc[label]),
                "relative_residual": float(row_relative.loc[label]),
            }
            for label in top_row.index
        ],
        "top_column_residuals": [
            {
                "label": str(label),
                "activity": _activity(str(label)),
                "residual": float(column_residual.loc[label]),
                "relative_residual": float(column_relative.loc[label]),
            }
            for label in top_column.index
        ],
        "top_activities_by_row_absolute_residual": [
            {"activity": str(activity), "absolute_residual_sum": float(value)}
            for activity, value in by_activity.head(10).items()
        ],
        "row_share_within_1e_4": float(np.mean(row_relative.to_numpy() <= 1e-4)),
        "column_share_within_1e_4": float(np.mean(column_relative.to_numpy() <= 1e-4)),
        "row_share_within_1e_3": float(np.mean(row_relative.to_numpy() <= 1e-3)),
        "column_share_within_1e_3": float(np.mean(column_relative.to_numpy() <= 1e-3)),
    }


def main() -> None:
    """Gate every 2016-2022 table and write the complete evidence artifact."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reports: list[dict[str, object]] = []
    with ZipFile(args.archive) as archive:
        for year in YEARS:
            member = _member_for_year(archive, year)
            reports.append(_year_report(_load_member(archive, member), year=year, member=member))

    failed_years = [int(report["year"]) for report in reports if report["release_balance_gate"] != "PASS"]
    summary = {
        "years": list(YEARS),
        "release_balance_atol": RELEASE_BALANCE_ATOL,
        "release_balance_rtol": RELEASE_BALANCE_RTOL,
        "release_gate_status": "PASS" if not failed_years else "FAIL",
        "failed_years": failed_years,
        "reports": reports,
        "note": (
            "Release gate uses abs(residual_i) <= 0.1 + 2e-4 * abs(output_i) for every "
            "published country-industry row and column in every official 2016-2022 table."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))

    if failed_years:
        raise RuntimeError(f"ICIO release balance gate failed for years: {failed_years!r}.")


if __name__ == "__main__":
    main()
