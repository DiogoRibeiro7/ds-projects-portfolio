"""Official RNAL acquisition and survivorship-aware historical proxy."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

RNAL_LAYER_URL = (
    "https://geo.turismodeportugal.pt/server/rest/services/TDP/OpenData_AL/MapServer/6/query"
)


@dataclass(frozen=True)
class RNALFetchConfig:
    """Network configuration for the official RNAL ArcGIS layer."""

    page_size: int = 2_000
    timeout_seconds: float = 60.0
    max_retries: int = 3

    def __post_init__(self) -> None:
        if self.page_size <= 0:
            raise ValueError("page_size must be positive.")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative.")


class RNALClient:
    """Fetch and cache a dated snapshot of the current official RNAL register."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        config: RNALFetchConfig | None = None,
        session: requests.Session | None = None,
    ) -> None:
        if not isinstance(cache_dir, Path):
            raise TypeError("cache_dir must be a pathlib.Path.")
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or RNALFetchConfig()
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": "portugal-housing-tourism/0.2"})

    def fetch_current_snapshot(
        self,
        *,
        snapshot_date: str,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch the current RNAL layer while preserving the retrieval date."""
        parsed_snapshot = pd.Timestamp(snapshot_date).normalize()
        output_dir = self.cache_dir / parsed_snapshot.strftime("%Y-%m-%d")
        output_path = output_dir / "rnal.parquet"
        if output_path.exists() and not refresh:
            return pd.read_parquet(output_path)

        records: list[dict[str, Any]] = []
        offset = 0
        while True:
            params: dict[str, str | int] = {
                "where": "1=1",
                "outFields": "*",
                "returnGeometry": "false",
                "f": "json",
                "resultOffset": offset,
                "resultRecordCount": self.config.page_size,
                "orderByFields": "OBJECTID",
            }
            payload: dict[str, Any] | None = None
            for attempt in range(self.config.max_retries + 1):
                try:
                    response = self.session.get(
                        RNAL_LAYER_URL,
                        params=params,
                        timeout=self.config.timeout_seconds,
                    )
                    response.raise_for_status()
                    payload = response.json()
                    break
                except (requests.RequestException, ValueError):
                    if attempt >= self.config.max_retries:
                        raise
                    time.sleep(2**attempt)
            if payload is None:
                raise RuntimeError("RNAL request produced no payload.")
            if "error" in payload:
                raise RuntimeError(f"RNAL ArcGIS error: {payload['error']}")
            features = payload.get("features", [])
            if not isinstance(features, list):
                raise ValueError("Unexpected RNAL payload: features must be a list.")
            for feature in features:
                attributes = feature.get("attributes", {})
                if isinstance(attributes, dict):
                    records.append(attributes)
            if not payload.get("exceededTransferLimit", False):
                break
            if not features:
                raise RuntimeError("RNAL pagination stalled with an empty page.")
            offset += len(features)

        frame = pd.DataFrame.from_records(records)
        if frame.empty:
            raise ValueError("RNAL returned no registrations.")
        for column in ("DataRegisto", "DataAberturaPublico"):
            if column in frame.columns:
                frame[column] = pd.to_datetime(
                    frame[column],
                    unit="ms",
                    errors="coerce",
                )
        if "NrRNAL" in frame.columns:
            frame["NrRNAL"] = pd.to_numeric(
                frame["NrRNAL"],
                errors="coerce",
            ).astype("Int64")
        if "Concelho" in frame.columns:
            frame["Concelho"] = frame["Concelho"].astype("string").str.strip()
        frame["snapshot_date"] = parsed_snapshot
        output_dir.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(output_path, index=False)
        return frame


def surviving_registration_panel(
    current_snapshot: pd.DataFrame,
    *,
    start_year: int = 2017,
    end_year: int | None = None,
    municipality_col: str = "Concelho",
    registration_col: str = "DataRegisto",
) -> pd.DataFrame:
    """Build a survivorship-biased lower-bound proxy from a current snapshot.

    This deliberately does not claim to reconstruct historical active stock:
    registrations cancelled before the current snapshot are absent.
    """
    if not isinstance(current_snapshot, pd.DataFrame):
        raise TypeError("current_snapshot must be a pandas DataFrame.")
    if not isinstance(start_year, int) or isinstance(start_year, bool):
        raise TypeError("start_year must be an integer.")
    if end_year is None:
        if "snapshot_date" not in current_snapshot.columns:
            raise KeyError("snapshot_date is required when end_year is omitted.")
        end_year = int(pd.to_datetime(current_snapshot["snapshot_date"]).max().year)
    if end_year < start_year:
        raise ValueError("end_year must be greater than or equal to start_year.")
    required = {"NrRNAL", municipality_col, registration_col}
    missing = required.difference(current_snapshot.columns)
    if missing:
        raise KeyError(f"RNAL frame is missing columns: {sorted(missing)}")

    data = current_snapshot[["NrRNAL", municipality_col, registration_col]].copy()
    data[registration_col] = pd.to_datetime(data[registration_col], errors="coerce")
    data = data.dropna(subset=[municipality_col, registration_col, "NrRNAL"])
    data["registration_year"] = data[registration_col].dt.year.astype(int)
    data = data.loc[data["registration_year"] <= end_year].copy()

    annual_new = (
        data.groupby([municipality_col, "registration_year"], observed=True)["NrRNAL"]
        .nunique()
        .rename("new_surviving_registrations")
        .reset_index()
    )
    municipalities = data[municipality_col].drop_duplicates().sort_values().tolist()
    grid = pd.MultiIndex.from_product(
        [municipalities, range(start_year, end_year + 1)],
        names=[municipality_col, "year"],
    ).to_frame(index=False)
    result = grid.merge(
        annual_new,
        left_on=[municipality_col, "year"],
        right_on=[municipality_col, "registration_year"],
        how="left",
    )
    result["new_surviving_registrations"] = (
        result["new_surviving_registrations"].fillna(0).astype(int)
    )
    pre_start = (
        data.loc[data["registration_year"] < start_year]
        .groupby(municipality_col, observed=True)["NrRNAL"]
        .nunique()
        .rename("pre_start")
    )
    result = result.join(pre_start, on=municipality_col)
    result["pre_start"] = result["pre_start"].fillna(0).astype(int)
    result["al_surviving_registrations"] = (
        result.groupby(municipality_col, observed=True)["new_surviving_registrations"].cumsum()
        + result["pre_start"]
    )
    return result.drop(columns=["registration_year", "pre_start"])
