"""Fetch the current official RNAL snapshot and build a historical proxy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from housing_tourism.rnal import RNALClient, surviving_registration_panel  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-date", required=True, help="ISO retrieval date, e.g. 2026-08-29")
    parser.add_argument("--start-year", type=int, default=2017)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Fetch RNAL, cache the dated snapshot, and save the survivor proxy."""
    args = parse_args()
    client = RNALClient(ROOT / "data" / "raw" / "rnal")
    snapshot = client.fetch_current_snapshot(
        snapshot_date=args.snapshot_date,
        refresh=args.refresh,
    )
    proxy = surviving_registration_panel(
        snapshot,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    output = ROOT / "data" / "processed" / "rnal_surviving_registrations_annual.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    proxy.to_parquet(output, index=False)
    print(f"RNAL snapshot rows: {len(snapshot):,}")
    print(f"Historical proxy rows: {len(proxy):,}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
