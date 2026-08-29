"""Fetch verified Lisbon Inside Airbnb snapshots and build annual summaries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from housing_tourism.inside_airbnb import (  # noqa: E402
    annualise_listing_snapshots,
    fetch_snapshot,
    load_snapshot_manifest,
    summarise_listing_snapshot,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data" / "manifests" / "inside_airbnb_lisbon.csv",
    )
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Fetch manifest snapshots and save dated and annual listing summaries."""
    args = parse_args()
    specs = load_snapshot_manifest(args.manifest)
    if not specs:
        raise SystemExit(
            "Snapshot manifest is empty. Add only URLs that have been independently "
            "verified as dated Lisbon Inside Airbnb snapshots."
        )

    summaries: list[pd.DataFrame] = []
    provenance: list[dict[str, object]] = []
    cache_dir = ROOT / "data" / "raw" / "inside_airbnb" / "lisbon"
    for spec in specs:
        fetched = fetch_snapshot(spec, cache_dir, refresh=args.refresh)
        frame = pd.read_csv(fetched.path, low_memory=False)
        summaries.append(summarise_listing_snapshot(frame, snapshot_date=fetched.snapshot_date))
        provenance.append(
            {
                "snapshot_date": fetched.snapshot_date,
                "source_url": spec.url,
                "sha256": fetched.sha256,
                "rows": fetched.rows,
            }
        )

    snapshot_summary = pd.concat(summaries, ignore_index=True)
    annual = annualise_listing_snapshots(snapshot_summary)
    output_dir = ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_summary.to_parquet(
        output_dir / "inside_airbnb_lisbon_snapshots.parquet", index=False
    )
    annual.to_parquet(output_dir / "inside_airbnb_lisbon_annual.parquet", index=False)
    pd.DataFrame(provenance).to_parquet(
        output_dir / "inside_airbnb_lisbon_provenance.parquet",
        index=False,
    )
    print(f"Verified snapshots: {len(snapshot_summary):,}")
    print(f"Observed years: {annual['year'].tolist()}")


if __name__ == "__main__":
    main()
