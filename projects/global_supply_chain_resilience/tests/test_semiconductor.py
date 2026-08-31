from __future__ import annotations

import pytest

from supply_chain_resilience.semiconductor import freeze_positive_reporter_universe


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
