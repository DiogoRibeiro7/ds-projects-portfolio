"""Probe official UN Comtrade coverage and schema for the frozen 2022 HS 8542 study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from supply_chain_resilience.comtrade import extract_data_rows, get_official_json, response_schema

DATA_AVAILABILITY_ENDPOINT = "https://comtradeapi.un.org/public/v1/getDA/C/A/HS"
PREVIEW_ENDPOINT = "https://comtradeapi.un.org/public/v1/preview/C/A/HS"
REFERENCE_YEAR = 2022
HS_HEADING = "8542"
PROBE_REPORTER = "842"  # United States; schema probe only, not an analytical selection.
MAX_PREVIEW_RECORDS = 500


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    availability = get_official_json(
        DATA_AVAILABILITY_ENDPOINT,
        {"period": REFERENCE_YEAR},
    )
    availability_rows = extract_data_rows(availability.payload)
    if not availability_rows:
        raise RuntimeError("UN Comtrade data-availability endpoint returned no 2022 rows.")

    preview = get_official_json(
        PREVIEW_ENDPOINT,
        {
            "period": REFERENCE_YEAR,
            "reporterCode": PROBE_REPORTER,
            "cmdCode": HS_HEADING,
            "flowCode": "M",
            "partnerCode": None,
            "partner2Code": "0",
            "customsCode": "C00",
            "motCode": "0",
            "maxRecords": MAX_PREVIEW_RECORDS,
            "aggregateBy": None,
            "breakdownMode": "classic",
            "includeDesc": "true",
        },
    )
    preview_rows = extract_data_rows(preview.payload)
    if not preview_rows:
        raise RuntimeError("UN Comtrade preview returned no 2022 HS 8542 import rows.")
    if len(preview_rows) >= MAX_PREVIEW_RECORDS:
        raise RuntimeError(
            "UN Comtrade preview reached the 500-row public cap; schema sample may be truncated."
        )

    report = {
        "reference_year": REFERENCE_YEAR,
        "commodity_heading": HS_HEADING,
        "probe_reporter_code": PROBE_REPORTER,
        "probe_purpose": "schema/provenance validation only; reporter is not selected for substantive analysis",
        "data_availability": {
            "endpoint": availability.endpoint,
            "query": availability.query,
            "retrieved_at_utc": availability.retrieved_at_utc,
            "canonical_sha256": availability.canonical_sha256,
            "schema": response_schema(availability_rows),
        },
        "bilateral_preview": {
            "endpoint": preview.endpoint,
            "query": preview.query,
            "retrieved_at_utc": preview.retrieved_at_utc,
            "canonical_sha256": preview.canonical_sha256,
            "schema": response_schema(preview_rows),
            "public_preview_cap": MAX_PREVIEW_RECORDS,
            "cap_reached": False,
        },
        "scientific_boundary": (
            "This artifact audits official UN Comtrade availability, provenance and returned schema only. "
            "It contains no concentration ranking or substantive semiconductor result."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
