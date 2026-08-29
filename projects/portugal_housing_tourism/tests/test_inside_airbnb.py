from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from housing_tourism.inside_airbnb import (
    SnapshotSpec,
    annualise_listing_snapshots,
    load_snapshot_manifest,
    summarise_listing_snapshot,
)


def test_snapshot_spec_requires_https() -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        SnapshotSpec("2020-01-01", "http://example.com/listings.csv.gz")


def test_manifest_rejects_duplicate_dates(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    path.write_text(
        "snapshot_date,url\n"
        "2020-01-01,https://example.com/a.csv.gz\n"
        "2020-01-01,https://example.com/b.csv.gz\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unique"):
        load_snapshot_manifest(path)


def test_snapshot_summary_deduplicates_listing_ids() -> None:
    frame = pd.DataFrame(
        {
            "id": [1, 1, 2, 3, 4],
            "room_type": [
                "Entire home/apt",
                "Entire home/apt",
                "Private room",
                "Shared room",
                "Hotel room",
            ],
        }
    )
    result = summarise_listing_snapshot(frame, snapshot_date="2020-06-30")
    row = result.iloc[0]
    assert row["listed_units"] == 4
    assert row["entire_home_units"] == 1
    assert row["private_room_units"] == 1
    assert row["shared_room_units"] == 1
    assert row["hotel_room_units"] == 1


def test_annualise_keeps_latest_snapshot_without_filling_years() -> None:
    summaries = pd.DataFrame(
        {
            "snapshot_date": ["2019-03-01", "2019-12-01", "2021-06-01"],
            "year": [2019, 2019, 2021],
            "listed_units": [10, 12, 20],
            "entire_home_units": [8, 9, 15],
            "private_room_units": [2, 3, 5],
            "shared_room_units": [0, 0, 0],
            "hotel_room_units": [0, 0, 0],
        }
    )
    result = annualise_listing_snapshots(summaries)
    assert result["year"].tolist() == [2019, 2021]
    assert result["listed_units"].tolist() == [12, 20]
