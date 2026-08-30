"""Download TLC Yellow Taxi data and build the frozen hourly demand panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mobility_optimization.backtest import FROZEN_BACKTEST
from mobility_optimization.data import (
    TripQualityReport,
    YELLOW_PICKUP_COLUMNS,
    aggregate_pickups_frame,
    build_dense_demand_panel,
    combine_hourly_counts,
    download_yellow_month,
    download_zone_lookup,
    load_service_zone_ids,
    month_range,
    select_top_zones,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/tlc"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/mobility_demand_hourly.parquet"),
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--overwrite-downloads", action="store_true")
    return parser.parse_args()


def report_to_dict(report: TripQualityReport) -> dict[str, int]:
    """Convert an immutable QA report into a JSON-serialisable dictionary."""
    return {
        "total_rows": report.total_rows,
        "valid_rows": report.valid_rows,
        "missing_pickup_time": report.missing_pickup_time,
        "missing_pickup_zone": report.missing_pickup_zone,
        "invalid_zone": report.invalid_zone,
        "outside_study_window": report.outside_study_window,
        "rejected_rows": report.rejected_rows,
    }


def _aggregate_nominal_month(
    path: Path,
    *,
    year: int,
    month: int,
    valid_zone_ids: tuple[int, ...],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Aggregate one source file after enforcing its nominal calendar month."""
    frame = pd.read_parquet(path, columns=list(YELLOW_PICKUP_COLUMNS))
    pickup_time = pd.to_datetime(frame["tpep_pickup_datetime"], errors="coerce")
    month_start = pd.Timestamp(year=year, month=month, day=1)
    month_end = month_start + pd.offsets.MonthBegin(1)
    missing_time = pickup_time.isna()
    in_source_month = pickup_time.ge(month_start) & pickup_time.lt(month_end)
    outside_source_month = (~missing_time & ~in_source_month).sum()

    filtered = frame.loc[in_source_month].copy()
    counts, report = aggregate_pickups_frame(
        filtered,
        valid_zone_ids=valid_zone_ids,
        start=FROZEN_BACKTEST.train_start,
        end=FROZEN_BACKTEST.test_end,
    )
    quality = report_to_dict(report)
    quality["total_rows"] = int(len(frame))
    quality["missing_pickup_time"] = int(missing_time.sum())
    quality["outside_source_month"] = int(outside_source_month)
    quality["rejected_rows"] = int(len(frame) - report.valid_rows)
    return counts, quality


def main() -> None:
    """Execute the frozen TLC panel build."""
    args = parse_args()
    raw_dir: Path = args.data_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    zone_lookup_path = raw_dir / "taxi_zone_lookup.csv"
    specs = month_range(start_year=2025, start_month=1, end_year=2026, end_month=5)

    if args.download:
        download_zone_lookup(
            destination=zone_lookup_path,
            overwrite=args.overwrite_downloads,
        )
        for spec in specs:
            download_yellow_month(
                spec,
                destination_dir=raw_dir,
                overwrite=args.overwrite_downloads,
            )

    if not zone_lookup_path.exists():
        raise FileNotFoundError(
            f"{zone_lookup_path} is missing. Re-run with --download or provide the official lookup."
        )

    valid_zone_ids = load_service_zone_ids(zone_lookup_path)
    monthly_counts: list[pd.DataFrame] = []
    quality: dict[str, dict[str, int]] = {}

    for spec in specs:
        path = raw_dir / spec.filename
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing. Re-run with --download or provide the official parquet."
            )
        counts, source_quality = _aggregate_nominal_month(
            path,
            year=spec.year,
            month=spec.month,
            valid_zone_ids=valid_zone_ids,
        )
        monthly_counts.append(counts)
        quality[spec.filename] = source_quality

    counts = combine_hourly_counts(monthly_counts)
    selected_zones = select_top_zones(
        counts,
        training_start=FROZEN_BACKTEST.train_start,
        training_end=FROZEN_BACKTEST.train_end,
        top_k=FROZEN_BACKTEST.top_k_zones,
    )
    panel = build_dense_demand_panel(
        counts,
        zone_ids=selected_zones,
        start=FROZEN_BACKTEST.train_start,
        end=FROZEN_BACKTEST.test_end,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(args.output, index=False)

    manifest = {
        "panel_path": str(args.output),
        "rows": int(len(panel)),
        "zones": list(selected_zones),
        "start": str(FROZEN_BACKTEST.train_start),
        "end_exclusive": str(FROZEN_BACKTEST.test_end),
        "test_origins": int(len(FROZEN_BACKTEST.test_origins())),
        "source_month_rule": "each monthly parquet is restricted to its nominal calendar month before aggregation",
        "quality_by_file": quality,
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
