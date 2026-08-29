"""Dated Inside Airbnb snapshots for the Lisbon case study."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd
import requests

_REQUIRED_COLUMNS: Final = {"id", "room_type"}


@dataclass(frozen=True)
class SnapshotSpec:
    """One verified point-in-time Inside Airbnb snapshot."""

    snapshot_date: str
    url: str

    def __post_init__(self) -> None:
        parsed = pd.Timestamp(self.snapshot_date)
        if pd.isna(parsed):
            raise ValueError("snapshot_date must be parseable as a date.")
        if parsed.tzinfo is not None:
            raise ValueError("snapshot_date must be timezone-naive.")
        if not self.url.startswith("https://"):
            raise ValueError("url must use HTTPS.")

    @property
    def date(self) -> pd.Timestamp:
        """Return the normalised snapshot date."""
        return pd.Timestamp(self.snapshot_date).normalize()


@dataclass(frozen=True)
class SnapshotFetchResult:
    """Metadata for one cached snapshot download."""

    snapshot_date: pd.Timestamp
    path: Path
    sha256: str
    rows: int


def load_snapshot_manifest(path: Path) -> list[SnapshotSpec]:
    """Load a manifest without silently filling missing years."""
    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path.")
    frame = pd.read_csv(path, dtype=str)
    required = {"snapshot_date", "url"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Snapshot manifest is missing columns: {sorted(missing)}")
    if frame.empty:
        return []
    if frame[["snapshot_date", "url"]].isna().any().any():
        raise ValueError("Snapshot manifest contains missing dates or URLs.")
    specs = [
        SnapshotSpec(snapshot_date=row.snapshot_date, url=row.url)
        for row in frame[["snapshot_date", "url"]].itertuples(index=False)
    ]
    dates = [spec.date for spec in specs]
    if len(dates) != len(set(dates)):
        raise ValueError("Snapshot dates must be unique.")
    return sorted(specs, key=lambda spec: spec.date)


def fetch_snapshot(
    spec: SnapshotSpec,
    cache_dir: Path,
    *,
    timeout_seconds: float = 60.0,
    refresh: bool = False,
    session: requests.Session | None = None,
) -> SnapshotFetchResult:
    """Download one dated listings CSV and cache the exact response bytes."""
    if not isinstance(cache_dir, Path):
        raise TypeError("cache_dir must be a pathlib.Path.")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive.")
    output_dir = cache_dir / spec.date.strftime("%Y-%m-%d")
    output_path = output_dir / "listings.csv.gz"
    if not output_path.exists() or refresh:
        client = session or requests.Session()
        response = client.get(spec.url, timeout=timeout_seconds)
        response.raise_for_status()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(response.content)
    content = output_path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    frame = pd.read_csv(output_path, low_memory=False)
    _validate_listing_snapshot(frame)
    return SnapshotFetchResult(
        snapshot_date=spec.date,
        path=output_path,
        sha256=digest,
        rows=len(frame),
    )


def summarise_listing_snapshot(
    frame: pd.DataFrame,
    *,
    snapshot_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Summarise one point-in-time listing census without inferring occupancy."""
    _validate_listing_snapshot(frame)
    date = pd.Timestamp(snapshot_date).normalize()
    listing_ids = pd.to_numeric(frame["id"], errors="coerce")
    valid = frame.loc[listing_ids.notna()].copy()
    valid["id"] = listing_ids.loc[listing_ids.notna()].astype("int64")
    valid = valid.drop_duplicates(subset="id")
    room_type = valid["room_type"].astype("string")
    return pd.DataFrame(
        {
            "snapshot_date": [date],
            "year": [date.year],
            "listed_units": [valid["id"].nunique()],
            "entire_home_units": [(room_type == "Entire home/apt").sum()],
            "private_room_units": [(room_type == "Private room").sum()],
            "shared_room_units": [(room_type == "Shared room").sum()],
            "hotel_room_units": [(room_type == "Hotel room").sum()],
        }
    )


def annualise_listing_snapshots(summaries: pd.DataFrame) -> pd.DataFrame:
    """Keep the latest observed snapshot in each year; never interpolate gaps."""
    required = {
        "snapshot_date",
        "year",
        "listed_units",
        "entire_home_units",
        "private_room_units",
        "shared_room_units",
        "hotel_room_units",
    }
    missing = required.difference(summaries.columns)
    if missing:
        raise KeyError(f"Snapshot summaries are missing columns: {sorted(missing)}")
    if summaries.empty:
        return summaries.copy()
    result = summaries.copy()
    result["snapshot_date"] = pd.to_datetime(result["snapshot_date"], errors="raise")
    result["year"] = pd.to_numeric(result["year"], errors="raise").astype(int)
    result = result.sort_values(["year", "snapshot_date"])
    result = result.groupby("year", as_index=False, observed=True).tail(1)
    return result.sort_values("year").reset_index(drop=True)


def _validate_listing_snapshot(frame: pd.DataFrame) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")
    missing = _REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise KeyError(f"Listing snapshot is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Listing snapshot is empty.")
