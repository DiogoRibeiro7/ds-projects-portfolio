"""Download and inspect the official OECD 2025 ICIO archive for 2016-2022.

This script deliberately produces a schema/provenance report before any economic
mapping is hard-coded. It is the empirical bridge between the generic ingestion
layer and the vintage-specific extraction of Z, x, and final demand.
"""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

import pandas as pd

OFFICIAL_URL = "https://webfs-sti.oecd.org/files/STI-PIE/ICIO/2025/2016-2022_SML.zip"


def download(url: str, destination: Path) -> str:
    """Download ``url`` to ``destination`` and return its SHA-256 digest."""
    request = Request(url, headers={"User-Agent": "ds-projects-portfolio/1.0"})
    with urlopen(request, timeout=180) as response:  # noqa: S310 - fixed HTTPS OECD source
        raw = response.read()
    if not raw:
        raise RuntimeError("OECD ICIO download returned an empty response.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(raw)
    return sha256(raw).hexdigest()


def choose_2022_member(archive_path: Path) -> str:
    """Select the unique CSV member corresponding to 2022."""
    with ZipFile(archive_path) as archive:
        csv_members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
    matches = [name for name in csv_members if "2022" in Path(name).stem]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one 2022 CSV member, found {len(matches)}: {matches!r}."
        )
    return matches[0]


def build_report(archive_path: Path, *, digest: str, member: str) -> dict[str, object]:
    """Return a compact schema report for the real 2022 OECD table."""
    with ZipFile(archive_path) as archive:
        with archive.open(member) as handle:
            frame = pd.read_csv(handle, index_col=0, low_memory=False)

    row_labels = [str(value) for value in frame.index]
    column_labels = [str(value) for value in frame.columns]
    common_labels = sorted(set(row_labels).intersection(column_labels))
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    negative_count = int((numeric < 0.0).sum().sum())

    return {
        "source_url": OFFICIAL_URL,
        "source_sha256": digest,
        "archive_member": member,
        "shape": [int(frame.shape[0]), int(frame.shape[1])],
        "row_count": len(row_labels),
        "column_count": len(column_labels),
        "common_row_column_labels": len(common_labels),
        "negative_numeric_cells": negative_count,
        "first_20_rows": row_labels[:20],
        "last_20_rows": row_labels[-20:],
        "first_20_columns": column_labels[:20],
        "last_20_columns": column_labels[-20:],
        "common_label_sample": common_labels[:20],
    }


def main() -> None:
    """Download the official archive and write the 2022 schema report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/icio_probe"))
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    archive_path = output_dir / "2016-2022_SML.zip"
    digest = download(OFFICIAL_URL, archive_path)
    member = choose_2022_member(archive_path)
    report = build_report(archive_path, digest=digest, member=member)

    report_path = output_dir / "icio_2022_schema.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
