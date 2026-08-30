"""Build the preregistered v2 panel through the June 2026 holdout."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from mobility_optimization.data import (
    YELLOW_PICKUP_COLUMNS,
    aggregate_pickups_frame,
    build_dense_demand_panel,
    combine_hourly_counts,
    download_yellow_month,
    download_zone_lookup,
    load_service_zone_ids,
    month_range,
)

RAW_DIR = Path("data/raw/tlc")
OUTPUT = Path("data/v2/processed/mobility_demand_hourly.parquet")
MATRIX_PATH = Path("evidence/v2_relocation_cost_matrix.csv")
MATRIX_SHA256 = "bf3ebdf7eaa8391c4a5c4554fbb39d0a098f5d4fc31af429cd39f7b4b17bb8b4"
START = pd.Timestamp("2025-01-01")
END = pd.Timestamp("2026-07-01")


def _frozen_zones() -> tuple[int, ...]:
    """Load the exact v1.1 zone order from the frozen relocation matrix."""
    digest = hashlib.sha256(MATRIX_PATH.read_bytes()).hexdigest()
    if digest != MATRIX_SHA256:
        raise ValueError(f"Frozen v2 relocation matrix checksum mismatch: {digest}")
    matrix = pd.read_csv(MATRIX_PATH, index_col=0)
    columns = tuple(int(value) for value in matrix.columns)
    rows = tuple(int(value) for value in matrix.index)
    if rows != columns or len(columns) != 30:
        raise ValueError("Frozen v2 relocation matrix must contain the same 30 zones on both axes.")
    return columns


def _nominal_month_counts(
    path: Path,
    *,
    year: int,
    month: int,
    valid_zone_ids: tuple[int, ...],
) -> tuple[pd.DataFrame, dict[str, int]]:
    frame = pd.read_parquet(path, columns=list(YELLOW_PICKUP_COLUMNS))
    pickup = pd.to_datetime(frame["tpep_pickup_datetime"], errors="coerce")
    month_start = pd.Timestamp(year=year, month=month, day=1)
    month_end = month_start + pd.offsets.MonthBegin(1)
    missing = pickup.isna()
    in_month = pickup.ge(month_start) & pickup.lt(month_end)
    filtered = frame.loc[in_month].copy()
    counts, report = aggregate_pickups_frame(
        filtered,
        valid_zone_ids=valid_zone_ids,
        start=START,
        end=END,
    )
    return counts, {
        "total_rows": int(len(frame)),
        "valid_rows": report.valid_rows,
        "missing_pickup_time": int(missing.sum()),
        "missing_pickup_zone": report.missing_pickup_zone,
        "invalid_zone": report.invalid_zone,
        "outside_study_window": report.outside_study_window,
        "outside_source_month": int((~missing & ~in_month).sum()),
        "rejected_rows": int(len(frame) - report.valid_rows),
    }


def main() -> None:
    """Download official TLC inputs and build the fixed-zone v2 panel."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zone_lookup = RAW_DIR / "taxi_zone_lookup.csv"
    download_zone_lookup(destination=zone_lookup)
    valid_zone_ids = load_service_zone_ids(zone_lookup)
    zones = _frozen_zones()
    if not set(zones).issubset(valid_zone_ids):
        raise ValueError("Frozen v1.1 zones are not all present in the official TLC lookup.")

    quality: dict[str, dict[str, int]] = {}
    monthly: list[pd.DataFrame] = []
    specs = month_range(start_year=2025, start_month=1, end_year=2026, end_month=6)
    for spec in specs:
        path = download_yellow_month(spec, destination_dir=RAW_DIR)
        counts, report = _nominal_month_counts(
            path,
            year=spec.year,
            month=spec.month,
            valid_zone_ids=valid_zone_ids,
        )
        monthly.append(counts)
        quality[spec.filename] = report

    counts = combine_hourly_counts(monthly)
    panel = build_dense_demand_panel(counts, zone_ids=zones, start=START, end=END)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUTPUT, index=False)
    manifest = {
        "design": "v2.0-preregistered",
        "panel_path": str(OUTPUT),
        "rows": int(len(panel)),
        "zones": list(zones),
        "zone_source": "evidence/v2_relocation_cost_matrix.csv",
        "relocation_matrix_sha256": MATRIX_SHA256,
        "start": str(START),
        "end_exclusive": str(END),
        "source_month_rule": "each TLC parquet restricted to its nominal month",
        "quality_by_file": quality,
    }
    OUTPUT.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
