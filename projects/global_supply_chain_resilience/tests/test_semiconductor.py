from __future__ import annotations

import math

import pytest

from supply_chain_resilience.semiconductor import (
    freeze_positive_reporter_universe,
    importer_concentration_metrics,
    is_named_country_partner,
)


def _availability(code: int, iso: str, name: str) -> dict[str, object]:
    return {
        "period": 2022,
        "reporterCode": code,
        "reporterDesc": name,
        "reporterISO": iso,
        "classificationCode": "H6",
        "classificationSearchCode": "HS",
        "datasetCode": 1000 + code,
        "datasetChecksum": 2000 + code,
        "firstReleased": "2023-01-01T00:00:00",
        "lastReleased": "2026-01-01T00:00:00",
        "totalRecords": 123,
        "isOriginalClassification": True,
    }


def _world_import(code: int, iso: str, name: str, value: float) -> dict[str, object]:
    return {
        "reporterCode": code,
        "reporterDesc": name,
        "reporterISO": iso,
        "partnerCode": 0,
        "flowCode": "M",
        "cmdCode": "8542",
        "primaryValue": value,
        "classificationCode": "H6",
        "classificationSearchCode": "HS",
    }


def _partner(
    partner_code: int,
    partner_iso: str,
    value: float,
    *,
    reporter_code: int = 1,
) -> dict[str, object]:
    return {
        "reporterCode": reporter_code,
        "partnerCode": partner_code,
        "partnerDesc": f"Partner {partner_code}",
        "partnerISO": partner_iso,
        "flowCode": "M",
        "cmdCode": "8542",
        "primaryValue": value,
    }


def test_freeze_positive_reporter_universe_excludes_zero_and_omits_trade_values() -> None:
    availability = [_availability(1, "AAA", "Alpha"), _availability(2, "BBB", "Beta")]
    world = [_world_import(1, "AAA", "Alpha", 10.0), _world_import(2, "BBB", "Beta", 0.0)]

    reporters = freeze_positive_reporter_universe(
        availability, world, reference_year=2022, commodity_heading="8542"
    )

    assert [row["reporter_code"] for row in reporters] == [1]
    assert reporters[0]["dataset_checksum"] == 2001
    assert "primaryValue" not in reporters[0]


def test_freeze_positive_reporter_universe_rejects_duplicate_world_rows() -> None:
    availability = [_availability(1, "AAA", "Alpha")]
    world = [
        _world_import(1, "AAA", "Alpha", 10.0),
        _world_import(1, "AAA", "Alpha", 11.0),
    ]

    with pytest.raises(ValueError, match="duplicate World-import"):
        freeze_positive_reporter_universe(
            availability, world, reference_year=2022, commodity_heading="8542"
        )


def test_freeze_positive_reporter_universe_requires_world_partner() -> None:
    availability = [_availability(1, "AAA", "Alpha")]
    row = _world_import(1, "AAA", "Alpha", 10.0)
    row["partnerCode"] = 2

    with pytest.raises(ValueError, match="non-World"):
        freeze_positive_reporter_universe(
            availability, [row], reference_year=2022, commodity_heading="8542"
        )


def test_named_country_partner_requires_three_letter_iso() -> None:
    assert is_named_country_partner("CHN")
    assert not is_named_country_partner("R4")
    assert not is_named_country_partner("S19")
    assert not is_named_country_partner("")


def test_importer_concentration_metrics_follow_frozen_denominators() -> None:
    rows = [
        _partner(0, "W00", 120.0),
        _partner(10, "AAA", 60.0),
        _partner(20, "BBB", 30.0),
        _partner(30, "R4", 10.0),
    ]

    metrics, bilateral = importer_concentration_metrics(
        rows, reporter_code=1, commodity_heading="8542"
    )

    assert metrics["total_import_value"] == pytest.approx(100.0)
    assert metrics["largest_partner_share"] == pytest.approx(0.6)
    assert metrics["top3_partner_share"] == pytest.approx(1.0)
    assert metrics["partner_hhi_all_reported"] == pytest.approx(0.46)
    assert metrics["effective_partner_count"] == pytest.approx(1.0 / 0.46)
    assert metrics["named_country_hhi"] == pytest.approx((2 / 3) ** 2 + (1 / 3) ** 2)
    assert metrics["residual_partner_share"] == pytest.approx(0.1)
    assert metrics["world_reconciliation_ratio"] == pytest.approx(100.0 / 120.0)
    assert len(bilateral) == 3


def test_importer_concentration_metrics_reject_duplicate_partner() -> None:
    rows = [_partner(10, "AAA", 60.0), _partner(10, "AAA", 30.0)]
    with pytest.raises(ValueError, match="duplicate partnerCode"):
        importer_concentration_metrics(rows, reporter_code=1, commodity_heading="8542")


def test_importer_concentration_metrics_allows_no_world_row() -> None:
    rows = [_partner(10, "AAA", 60.0), _partner(20, "BBB", 40.0)]
    metrics, _ = importer_concentration_metrics(rows, reporter_code=1, commodity_heading="8542")
    assert math.isnan(float(metrics["world_reconciliation_ratio"]))
