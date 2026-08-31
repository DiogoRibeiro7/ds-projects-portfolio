"""Freeze the positive-import reporter universe for the 2022 HS 8542 case study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from supply_chain_resilience.comtrade import extract_data_rows, get_official_json, response_schema
from supply_chain_resilience.semiconductor import freeze_positive_reporter_universe

DATA_AVAILABILITY_ENDPOINT = "https://comtradeapi.un.org/public/v1/getDA/C/A/HS"
PREVIEW_ENDPOINT = "https://comtradeapi.un.org/public/v1/preview/C/A/HS"
REFERENCE_YEAR = 2022
HS_HEADING = "8542"
MAX_PREVIEW_RECORDS = 500


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    availability = get_official_json(DATA_AVAILABILITY_ENDPOINT, {"period": REFERENCE_YEAR})
    availability_rows = extract_data_rows(availability.payload)
    if not availability_rows:
        raise RuntimeError("UN Comtrade returned no 2022 annual HS data-availability rows.")

    world_imports = get_official_json(
        PREVIEW_ENDPOINT,
        {
            "period": REFERENCE_YEAR,
            "reporterCode": "0",
            "cmdCode": HS_HEADING,
            "flowCode": "M",
            "partnerCode": "0",
            "partner2Code": "0",
            "customsCode": "C00",
            "motCode": "0",
            "maxRecords": MAX_PREVIEW_RECORDS,
            "breakdownMode": "classic",
            "includeDesc": "true",
        },
    )
    world_rows = extract_data_rows(world_imports.payload)
    if not world_rows:
        raise RuntimeError("UN Comtrade returned no World-import HS 8542 rows.")
    if len(world_rows) >= MAX_PREVIEW_RECORDS:
        raise RuntimeError(
            "UN Comtrade all-reporter World-import preview reached the 500-row cap; "
            "the reporter universe cannot be frozen from a potentially truncated response."
        )

    reporters = freeze_positive_reporter_universe(
        availability_rows,
        world_rows,
        reference_year=REFERENCE_YEAR,
        commodity_heading=HS_HEADING,
    )
    if not reporters:
        raise RuntimeError("No positive 2022 HS 8542 import reporters were identified.")

    report = {
        "reference_year": REFERENCE_YEAR,
        "commodity_heading": HS_HEADING,
        "data_availability": {
            "endpoint": availability.endpoint,
            "query": availability.query,
            "retrieved_at_utc": availability.retrieved_at_utc,
            "canonical_sha256": availability.canonical_sha256,
            "schema": response_schema(availability_rows),
        },
        "world_import_gate": {
            "endpoint": world_imports.endpoint,
            "query": world_imports.query,
            "retrieved_at_utc": world_imports.retrieved_at_utc,
            "canonical_sha256": world_imports.canonical_sha256,
            "schema": response_schema(world_rows),
            "public_preview_cap": MAX_PREVIEW_RECORDS,
            "cap_reached": False,
        },
        "available_2022_hs_reporter_datasets": len(availability_rows),
        "positive_hs8542_import_reporters": len(reporters),
        "reporters": reporters,
        "scientific_boundary": (
            "This artifact freezes the exact positive-import reporter universe and Comtrade "
            "dataset-release metadata. It intentionally omits trade values, concentration metrics, "
            "and rankings; those remain unseen until the subsequent bilateral-analysis gate."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
