"""Prospective semiconductor-case-study data-selection and concentration utilities."""

from __future__ import annotations

from math import isfinite
import re
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
    are deliberately omitted from the returned metadata. Substantive trade
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


def is_named_country_partner(partner_iso: object) -> bool:
    """Return whether a Comtrade partner has an ordinary three-letter ISO code."""
    return bool(re.fullmatch(r"[A-Z]{3}", str(partner_iso)))


def importer_concentration_metrics(
    rows: list[dict[str, Any]],
    *,
    reporter_code: int,
    commodity_heading: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Compute the frozen 2022 HS 8542 importer concentration metrics.

    World rows are excluded from the bilateral denominator. Residual/special
    partner categories remain in the all-reported HHI and are reported separately;
    named-country HHI is renormalized over ordinary three-letter ISO partners.
    """
    required = {
        "reporterCode",
        "partnerCode",
        "partnerDesc",
        "partnerISO",
        "flowCode",
        "cmdCode",
        "primaryValue",
    }
    bilateral: list[dict[str, object]] = []
    world_value: float | None = None
    seen_partner_codes: set[int] = set()

    for row in rows:
        _require_fields(row, required, context="bilateral import")
        if int(row["reporterCode"]) != reporter_code:
            raise ValueError("bilateral response contains an unexpected reporter.")
        if str(row["flowCode"]) != "M":
            raise ValueError("bilateral response contains a non-import flow.")
        if str(row["cmdCode"]) != commodity_heading:
            raise ValueError("bilateral response contains an unexpected commodity code.")
        partner_code = int(row["partnerCode"])
        if partner_code in seen_partner_codes:
            raise ValueError(f"duplicate partnerCode={partner_code} for reporter {reporter_code}.")
        seen_partner_codes.add(partner_code)
        value = float(row["primaryValue"])
        if not isfinite(value) or value < 0.0:
            raise ValueError("bilateral primaryValue must be finite and non-negative.")
        if partner_code == 0:
            world_value = value
            continue
        if value == 0.0:
            continue
        bilateral.append(
            {
                "partner_code": partner_code,
                "partner_desc": str(row["partnerDesc"]),
                "partner_iso": str(row["partnerISO"]),
                "trade_value": value,
                "is_named_country": is_named_country_partner(row["partnerISO"]),
            }
        )

    total = sum(float(row["trade_value"]) for row in bilateral)
    if total <= 0.0:
        raise ValueError(f"reporter {reporter_code} has no positive bilateral imports.")

    for row in bilateral:
        row["partner_share"] = float(row["trade_value"]) / total

    shares = [float(row["partner_share"]) for row in bilateral]
    hhi = sum(share * share for share in shares)
    ordered = sorted(shares, reverse=True)
    named = [row for row in bilateral if bool(row["is_named_country"])]
    named_total = sum(float(row["trade_value"]) for row in named)
    named_hhi = float("nan")
    if named_total > 0.0:
        named_hhi = sum((float(row["trade_value"]) / named_total) ** 2 for row in named)
    residual_share = 1.0 - named_total / total
    reconciliation = float("nan")
    if world_value is not None and world_value > 0.0:
        reconciliation = total / world_value

    metrics: dict[str, object] = {
        "reporter_code": reporter_code,
        "total_import_value": total,
        "partner_hhi_all_reported": hhi,
        "effective_partner_count": 1.0 / hhi,
        "largest_partner_share": ordered[0],
        "top3_partner_share": sum(ordered[:3]),
        "named_country_hhi": named_hhi,
        "residual_partner_share": residual_share,
        "world_reconciliation_ratio": reconciliation,
        "positive_partner_count": len(bilateral),
        "named_country_partner_count": len(named),
    }
    return metrics, bilateral
