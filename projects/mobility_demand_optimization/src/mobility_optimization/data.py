"""NYC TLC ingestion and demand-panel construction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from shutil import copyfileobj
from urllib.request import urlopen

import numpy as np
import pandas as pd

TLC_TRIP_BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
TLC_ZONE_LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
YELLOW_PICKUP_COLUMNS = ("tpep_pickup_datetime", "PULocationID")


@dataclass(frozen=True, slots=True)
class MonthSpec:
    """One monthly Yellow Taxi source file."""

    year: int
    month: int

    def __post_init__(self) -> None:
        """Validate the calendar month."""
        if self.year < 2009:
            raise ValueError("Yellow Taxi monthly data are only supported from 2009 onward.")
        if not 1 <= self.month <= 12:
            raise ValueError("month must be between 1 and 12.")

    @property
    def filename(self) -> str:
        """Return the official TLC parquet filename."""
        return f"yellow_tripdata_{self.year:04d}-{self.month:02d}.parquet"

    @property
    def url(self) -> str:
        """Return the official TLC parquet URL."""
        return f"{TLC_TRIP_BASE_URL}/{self.filename}"


@dataclass(frozen=True, slots=True)
class TripQualityReport:
    """Row-level QA counts produced before monthly aggregation."""

    total_rows: int
    valid_rows: int
    missing_pickup_time: int
    missing_pickup_zone: int
    invalid_zone: int
    outside_study_window: int

    @property
    def rejected_rows(self) -> int:
        """Return the number of rows excluded from the study panel."""
        return self.total_rows - self.valid_rows


def month_range(
    *,
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
) -> tuple[MonthSpec, ...]:
    """Return an inclusive sequence of monthly TLC file specifications."""
    start = pd.Period(year=start_year, month=start_month, freq="M")
    end = pd.Period(year=end_year, month=end_month, freq="M")
    if end < start:
        raise ValueError("end month must not precede start month.")
    return tuple(MonthSpec(period.year, period.month) for period in pd.period_range(start, end))


def download_yellow_month(
    spec: MonthSpec,
    *,
    destination_dir: Path,
    overwrite: bool = False,
) -> Path:
    """Download one official Yellow Taxi parquet file atomically."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / spec.filename
    if destination.exists() and not overwrite:
        return destination

    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urlopen(spec.url, timeout=120) as response, temporary.open("wb") as output:  # noqa: S310
            copyfileobj(response, output, length=1024 * 1024)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def download_zone_lookup(*, destination: Path, overwrite: bool = False) -> Path:
    """Download the official TLC Taxi Zone lookup table atomically."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        return destination

    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urlopen(TLC_ZONE_LOOKUP_URL, timeout=60) as response, temporary.open("wb") as output:  # noqa: S310
            copyfileobj(response, output, length=1024 * 1024)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_service_zone_ids(path: Path) -> tuple[int, ...]:
    """Load geographic TLC Taxi Zone identifiers from the official lookup CSV."""
    frame = pd.read_csv(path)
    required = {"LocationID", "Borough", "Zone"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Taxi zone lookup is missing columns: {sorted(missing)}")

    ids = pd.to_numeric(frame["LocationID"], errors="coerce")
    if ids.isna().any():
        raise ValueError("Taxi zone lookup contains non-numeric LocationID values.")
    if ids.duplicated().any():
        raise ValueError("Taxi zone lookup contains duplicate LocationID values.")

    zone = frame["Zone"].astype(str).str.strip()
    borough = frame["Borough"].astype(str).str.strip()
    geographic = ~zone.isin({"Unknown", "Outside of NYC", "N/A"}) & ~borough.isin(
        {"Unknown", "N/A"}
    )
    selected = ids.loc[geographic].astype(int).sort_values().tolist()
    if not selected:
        raise ValueError("Taxi zone lookup contains no geographic service zones.")
    return tuple(int(value) for value in selected)


def aggregate_pickups_frame(
    frame: pd.DataFrame,
    *,
    valid_zone_ids: tuple[int, ...],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, TripQualityReport]:
    """Validate trip rows and aggregate valid pickups to civil clock hour and zone.

    TLC timestamps are published as local wall-clock timestamps without an explicit
    UTC offset. Aggregation therefore remains on the supplied civil clock. DST
    transition dates are flagged later in the dense panel and can be excluded from
    headline evaluation.
    """
    if end <= start:
        raise ValueError("end must be later than start.")
    missing_columns = set(YELLOW_PICKUP_COLUMNS).difference(frame.columns)
    if missing_columns:
        raise ValueError(f"Trip data are missing columns: {sorted(missing_columns)}")

    pickup_time = pd.to_datetime(frame["tpep_pickup_datetime"], errors="coerce")
    pickup_zone = pd.to_numeric(frame["PULocationID"], errors="coerce")

    missing_time = pickup_time.isna()
    missing_zone = pickup_zone.isna()
    valid_zone = pickup_zone.isin(valid_zone_ids)
    in_window = pickup_time.ge(start) & pickup_time.lt(end)

    keep = ~missing_time & ~missing_zone & valid_zone & in_window
    valid = pd.DataFrame(
        {
            "timestamp": pickup_time.loc[keep].dt.floor("h"),
            "zone_id": pickup_zone.loc[keep].astype(np.int64),
        }
    )
    counts = (
        valid.groupby(["timestamp", "zone_id"], as_index=False, sort=True)
        .size()
        .rename(columns={"size": "demand"})
    )
    counts["demand"] = counts["demand"].astype(np.int64)

    report = TripQualityReport(
        total_rows=int(len(frame)),
        valid_rows=int(keep.sum()),
        missing_pickup_time=int(missing_time.sum()),
        missing_pickup_zone=int((~missing_time & missing_zone).sum()),
        invalid_zone=int((~missing_time & ~missing_zone & ~valid_zone).sum()),
        outside_study_window=int((~missing_time & ~missing_zone & valid_zone & ~in_window).sum()),
    )
    return counts, report


def aggregate_yellow_file(
    path: Path,
    *,
    valid_zone_ids: tuple[int, ...],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, TripQualityReport]:
    """Read the minimal Yellow Taxi columns and aggregate one parquet file."""
    frame = pd.read_parquet(path, columns=list(YELLOW_PICKUP_COLUMNS))
    return aggregate_pickups_frame(
        frame,
        valid_zone_ids=valid_zone_ids,
        start=start,
        end=end,
    )


def combine_hourly_counts(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine monthly hourly counts into one unique timestamp-zone table."""
    if not frames:
        raise ValueError("At least one hourly-count frame is required.")
    combined = pd.concat(frames, ignore_index=True)
    required = {"timestamp", "zone_id", "demand"}
    if not required.issubset(combined.columns):
        raise ValueError("Hourly-count frames must contain timestamp, zone_id, and demand.")
    result = (
        combined.groupby(["timestamp", "zone_id"], as_index=False, sort=True)["demand"]
        .sum()
        .sort_values(["timestamp", "zone_id"], ignore_index=True)
    )
    result["demand"] = result["demand"].astype(np.int64)
    return result


def select_top_zones(
    counts: pd.DataFrame,
    *,
    training_start: pd.Timestamp,
    training_end: pd.Timestamp,
    top_k: int,
) -> tuple[int, ...]:
    """Select service zones by pickup volume using training data only."""
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    mask = counts["timestamp"].ge(training_start) & counts["timestamp"].lt(training_end)
    training = counts.loc[mask]
    if training.empty:
        raise ValueError("No demand observations fall inside the training window.")

    totals = (
        training.groupby("zone_id", as_index=False)["demand"]
        .sum()
        .sort_values(["demand", "zone_id"], ascending=[False, True], ignore_index=True)
    )
    if len(totals) < top_k:
        raise ValueError("top_k exceeds the number of zones observed during training.")
    return tuple(int(value) for value in totals.head(top_k)["zone_id"])


def _dst_transition_dates(
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    timezone: str,
) -> set[object]:
    """Return local dates whose UTC offset changes before the next midnight."""
    days = pd.date_range(start=start.normalize(), end=end.normalize(), freq="D", tz=timezone)
    transitions: set[object] = set()
    for current, following in zip(days[:-1], days[1:], strict=False):
        if current.utcoffset() != following.utcoffset():
            transitions.add(current.date())
    return transitions


def build_dense_demand_panel(
    counts: pd.DataFrame,
    *,
    zone_ids: tuple[int, ...],
    start: pd.Timestamp,
    end: pd.Timestamp,
    timezone: str = "America/New_York",
) -> pd.DataFrame:
    """Build a complete hourly civil-time demand panel for selected zones."""
    if end <= start:
        raise ValueError("end must be later than start.")
    if not zone_ids or len(set(zone_ids)) != len(zone_ids):
        raise ValueError("zone_ids must be a non-empty sequence of unique identifiers.")

    hours = pd.date_range(start=start, end=end - pd.Timedelta(hours=1), freq="h")
    index = pd.MultiIndex.from_product(
        [hours, sorted(zone_ids)],
        names=["timestamp", "zone_id"],
    )
    selected = counts.loc[counts["zone_id"].isin(zone_ids), ["timestamp", "zone_id", "demand"]]
    series = selected.set_index(["timestamp", "zone_id"])["demand"]
    if series.index.has_duplicates:
        series = series.groupby(level=["timestamp", "zone_id"]).sum()

    panel = series.reindex(index, fill_value=0).rename("demand").reset_index()
    panel["demand"] = panel["demand"].astype(np.int64)
    transitions = _dst_transition_dates(start=start, end=end, timezone=timezone)
    panel["is_dst_transition_day"] = panel["timestamp"].dt.date.isin(transitions)
    return panel
