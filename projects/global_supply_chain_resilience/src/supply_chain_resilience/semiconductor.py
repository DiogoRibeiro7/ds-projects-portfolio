"""Prospective semiconductor-case-study data-selection utilities."""

from __future__ import annotations

from math import isfinite
from typing import Any


def _require_fields(row: dict[str, Any], fields: set[str], *, context: str) -> None:
    missing = fields.difference(row)
    if missing:
        raise ValueError(f"{context} row is missing required fields: {sorted(missing)}")


def freeze_positive_reporter_universe(
    availability_rows: list[dict[str, Any]],
    world_import_rows: list[dict[str, Any]],
    *,
    reference_year: int,
    commodity_heading: str,
) -> list[dict[str, object]]:
    """Freeze reporters with positive World-import value for the specified HS heading.

    Trade values are used only as a mechanical positive-value inclusion gate and
    are deliberately omitted from the returned metadata.  Substantive trade
    magnitudes remain unseen until the later concentration-analysis gate.
    """
    required_availability = {
        "period",
        "reporterCode",
        "reporterDesc",
        "reporterISO",
        "classificationCode",
        "classificationSearchCode",
        "datasetCode",
        "datasetChecksum",
        "firstReleased",
        "lastReleased",
        "totalRecords",
        "isOriginalClassification",
    }
    availability_by_code: dict[int, dict[str, Any]] = {}
    for row in availability_rows:
        _require_fields(row, required_availability, context="data-availability")
        if int(row["period"]) != reference_year:
            raise ValueError("data-availability response contains an unexpected period.")
        code = int(row["reporterCode"])
        if code in availability_by_code:
            raise ValueError(f"duplicate data-availability reporterCode={code}.")
        availability_by_code[code] = row

    if not availability_by_code:
        raise ValueError("data-availability response contains no reporters.")

    required_world = {
        "reporterCode",
        "reporterDesc",
        "reporterISO",
        "partnerCode",
        "flowCode",
        "cmdCode",
        "primaryValue",
        "classificationCode",
        "classificationSearchCode",
    }
    positive_world_by_code: dict[int, dict[str, Any]] = {}
    seen_world_codes: set[int] = set()
    for row in world_import_rows:
        _require_fields(row, required_world, context="World-import")
        code = int(row["reporterCode"])
        if code in seen_world_codes:
            raise ValueError(f"duplicate World-import reporterCode={code}.")
        seen_world_codes.add(code)
        if code not in availability_by_code:
            raise ValueError(f"World-import reporterCode={code} is absent from data availability.")
        if int(row["partnerCode"]) != 0:
            raise ValueError("World-import response contains a non-World partner row.")
        if str(row["flowCode"]) != "M":
            raise ValueError("World-import response contains a non-import flow.")
        if str(row["cmdCode"]) != commodity_heading:
            raise ValueError("World-import response contains an unexpected commodity code.")

        value = float(row["primaryValue"])
        if not isfinite(value) or value < 0.0:
            raise ValueError("World-import primaryValue must be finite and non-negative.")
        if value > 0.0:
            positive_world_by_code[code] = row

    reporters: list[dict[str, object]] = []
    for code in sorted(positive_world_by_code):
        availability = availability_by_code[code]
        trade_row = positive_world_by_code[code]
        reporters.append(
            {
                "reporter_code": code,
                "reporter_desc": str(availability["reporterDesc"]),
                "reporter_iso": str(availability["reporterISO"]),
                "dataset_code": int(availability["datasetCode"]),
                "dataset_checksum": int(availability["datasetChecksum"]),
                "first_released": str(availability["firstReleased"]),
                "last_released": str(availability["lastReleased"]),
                "dataset_total_records": int(availability["totalRecords"]),
                "availability_classification_code": str(availability["classificationCode"]),
                "availability_classification_search_code": str(
                    availability["classificationSearchCode"]
                ),
                "is_original_classification": bool(availability["isOriginalClassification"]),
                "trade_classification_code": str(trade_row["classificationCode"]),
                "trade_classification_search_code": str(trade_row["classificationSearchCode"]),
            }
        )

    return reporters
