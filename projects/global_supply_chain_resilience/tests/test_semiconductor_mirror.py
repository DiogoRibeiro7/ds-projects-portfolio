from __future__ import annotations

import math

import pytest

from supply_chain_resilience.semiconductor_mirror import exporter_mirror_value, mirror_diagnostics


def _row(exporter: int, importer: int, value: float) -> dict[str, object]:
    return {
        "reporterCode": exporter,
        "partnerCode": importer,
        "flowCode": "X",
        "cmdCode": "8542",
        "primaryValue": value,
    }


def test_exporter_mirror_value_extracts_exact_pair() -> None:
    assert exporter_mirror_value([_row(410, 156, 80.0)], exporter_code=410, importer_code=156, commodity_heading="8542") == 80.0


def test_exporter_mirror_value_preserves_missing() -> None:
    assert exporter_mirror_value([], exporter_code=410, importer_code=156, commodity_heading="8542") is None


def test_exporter_mirror_value_rejects_wrong_partner() -> None:
    with pytest.raises(ValueError, match="unexpected importer partner"):
        exporter_mirror_value([_row(410, 344, 80.0)], exporter_code=410, importer_code=156, commodity_heading="8542")


def test_mirror_diagnostics_follow_frozen_definitions() -> None:
    metrics = mirror_diagnostics(100.0, 80.0)
    assert metrics["mirror_observed"] is True
    assert metrics["absolute_difference"] == pytest.approx(20.0)
    assert metrics["relative_difference_max_denominator"] == pytest.approx(0.2)
    assert metrics["signed_log_ratio_export_over_import"] == pytest.approx(math.log(0.8))


def test_mirror_diagnostics_missing_mirror_stays_missing() -> None:
    metrics = mirror_diagnostics(100.0, None)
    assert metrics["mirror_observed"] is False
    assert metrics["exporter_reported_value"] is None
    assert metrics["absolute_difference"] is None
