"""Freeze the positive-import reporter universe for the 2022 HS 8542 case study."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from supply_chain_resilience.comtrade import extract_data_rows, get_official_json, response_schema
from supply_chain_resilience.semiconductor import freeze_positive_reporter_universe

DATA_AVAILABILITY_ENDPOINT = "https://comtradeapi.un.org/public/v1/getDA/C/A/HS"
PREVIEW_ENDPOINT = "https://comtradeapi.un.org/public/v1/preview/C/A/HS"
REFERENCE_YEAR = 2022
HS_HEADING = "8542"
MAX_WORLD_ROWS_PER_REPORTER = 2
PUBLIC_PREVIEW_MIN_INTERVAL_SECONDS = 1.05
AGGREGATE_REPORTER_GROUPS = {
    97: "European Union",
    975: "ASEAN",
}


def _fetch_reporter_world_imports(
    availability_rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Fetch one reporter-scoped HS 8542 World-import aggregate at a time.

    UN Comtrade documents the public/free API at one request per second. The
    loop therefore enforces a small margin above that interval rather than using
    429 responses as implicit pacing. This affects transport only.
    """
    reporter_codes = sorted({int(row["reporterCode"]) for row in availability_rows})
    if len(reporter_codes) != len(availability_rows):
        raise RuntimeError("2022 data availability contains duplicate reporter codes.")

    world_rows: list[dict[str, object]] = []
    query_evidence: list[dict[str, object]] = []
    previous_started: float | None = None
    for reporter_code in reporter_codes:
        if previous_started is not None:
            elapsed = time.monotonic() - previous_started
            if elapsed < PUBLIC_PREVIEW_MIN_INTERVAL_SECONDS:
                time.sleep(PUBLIC_PREVIEW_MIN_INTERVAL_SECONDS - elapsed)
        previous_started = time.monotonic()

        response = get_official_json(
            PREVIEW_ENDPOINT,
            {
                "period": REFERENCE_YEAR,
                "reportercode": reporter_code,
                "cmdCode": HS_HEADING,
                "flowCode": "M",
                "partnerCode": "0",
                "partner2Code": "0",
                "customsCode": "C00",
                "motCode": "0",
                "maxRecords": MAX_WORLD_ROWS_PER_REPORTER,
                "breakdownMode": "classic",
                "includeDesc": "true",
            },
        )
        rows = extract_data_rows(response.payload)
        if len(rows) >= MAX_WORLD_ROWS_PER_REPORTER:
            raise RuntimeError(
                f"UN Comtrade reporter {reporter_code} returned at least "
                f"{MAX_WORLD_ROWS_PER_REPORTER} World-import rows; expected at most one."
            )
        world_rows.extend(rows)
        query_evidence.append(
            {
                "reporter_code": reporter_code,
                "endpoint": response.endpoint,
                "query": response.query,
                "retrieved_at_utc": response.retrieved_at_utc,
                "canonical_sha256": response.canonical_sha256,
                "row_count": len(rows),
            }
        )

    return world_rows, query_evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    availability = get_official_json(DATA_AVAILABILITY_ENDPOINT, {"period": REFERENCE_YEAR})
    availability_rows = extract_data_rows(availability.payload)
    if not availability_rows:
        raise RuntimeError("UN Comtrade returned no 2022 annual HS data-availability rows.")

    world_rows, query_evidence = _fetch_reporter_world_imports(availability_rows)
    if not world_rows:
        raise RuntimeError("UN Comtrade returned no reporter-scoped World-import HS 8542 rows.")

    all_positive_reporters = freeze_positive_reporter_universe(
        availability_rows,
        world_rows,
        reference_year=REFERENCE_YEAR,
        commodity_heading=HS_HEADING,
    )
    observed_positive_codes = {int(row["reporter_code"]) for row in all_positive_reporters}
    missing_groups = set(AGGREGATE_REPORTER_GROUPS).difference(observed_positive_codes)
    if missing_groups:
        raise RuntimeError(
            "Prospectively identified aggregate reporter groups were absent from the positive "
            f"reporter gate: {sorted(missing_groups)}"
        )

    reporters = [
        row
        for row in all_positive_reporters
        if int(row["reporter_code"]) not in AGGREGATE_REPORTER_GROUPS
    ]
    if not reporters:
        raise RuntimeError("No positive 2022 HS 8542 individual reporters were identified.")

    excluded_groups = [
        {"reporter_code": code, "reporter_desc": AGGREGATE_REPORTER_GROUPS[code]}
        for code in sorted(AGGREGATE_REPORTER_GROUPS)
    ]

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
            "strategy": "one paced official public-preview World-import query per available 2022 reporter",
            "documented_public_rate_limit": "1 request per second",
            "minimum_request_interval_seconds": PUBLIC_PREVIEW_MIN_INTERVAL_SECONDS,
            "reporter_queries": len(query_evidence),
            "reporters_with_returned_world_row": len(world_rows),
            "max_rows_per_reporter_query": MAX_WORLD_ROWS_PER_REPORTER,
            "schema": response_schema(world_rows),
            "query_evidence": query_evidence,
        },
        "available_2022_hs_reporter_datasets": len(availability_rows),
        "positive_hs8542_import_reporters_before_group_exclusion": len(all_positive_reporters),
        "excluded_aggregate_reporter_groups": excluded_groups,
        "primary_positive_hs8542_import_reporters": len(reporters),
        "reporters": reporters,
        "scientific_boundary": (
            "This artifact freezes the exact positive-import individual country/area reporter "
            "universe and Comtrade dataset-release metadata. European Union (97) and ASEAN "
            "(975) are excluded prospectively because they overlap their member reporters. "
            "Residual/special individual reporting areas are retained. The artifact intentionally "
            "omits trade values, concentration metrics, and rankings."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
