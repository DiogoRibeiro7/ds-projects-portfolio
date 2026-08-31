from __future__ import annotations

import pandas as pd
import pytest

from supply_chain_resilience.semiconductor_exporter import (
    supplier_dependency_diagnostics,
    world_export_value,
)


def test_world_export_value_requires_one_world_export_row() -> None:
    rows = [{
        "reporterCode": 1,
        "flowCode": "X",
        "cmdCode": "8542",
        "partnerCode": 0,
        "primaryValue": 12.0,
    }]
    assert world_export_value(rows, reporter_code=1, commodity_heading="8542") == 12.0


def test_world_export_value_zero_is_not_positive_exporter() -> None:
    rows = [{
        "reporterCode": 1,
        "flowCode": "X",
        "cmdCode": "8542",
        "partnerCode": 0,
        "primaryValue": 0.0,
    }]
    assert world_export_value(rows, reporter_code=1, commodity_heading="8542") is None


def test_supplier_dependency_diagnostics_uses_frozen_material_denominator() -> None:
    metrics = pd.DataFrame({"reporter_code": [1, 2], "material_importer": [True, True]})
    links = pd.DataFrame([
        {"reporter_code": 1, "partner_code": 10, "partner_desc": "A", "partner_iso": "AAA", "trade_value": 60.0, "partner_share": 0.6, "is_named_country": True, "material_importer": True},
        {"reporter_code": 1, "partner_code": 20, "partner_desc": "B", "partner_iso": "BBB", "trade_value": 40.0, "partner_share": 0.4, "is_named_country": True, "material_importer": True},
        {"reporter_code": 2, "partner_code": 10, "partner_desc": "A", "partner_iso": "AAA", "trade_value": 20.0, "partner_share": 0.2, "is_named_country": True, "material_importer": True},
        {"reporter_code": 2, "partner_code": 490, "partner_desc": "Other Asia, nes", "partner_iso": "S19", "trade_value": 80.0, "partner_share": 0.8, "is_named_country": False, "material_importer": True},
    ])
    suppliers, residual = supplier_dependency_diagnostics(metrics, links, expected_material_importers=2)
    a = suppliers.loc[suppliers["partner_code"] == 10].iloc[0]
    assert a["total_importer_reported_value"] == pytest.approx(80.0)
    assert a["largest_named_supplier_count"] == 2
    assert a["material_importer_count_ge_10pct"] == 2
    assert a["material_importer_count_ge_50pct"] == 1
    assert residual["material_importer_count_ge_50pct"] == 1


def test_supplier_dependency_diagnostics_rejects_changed_material_membership() -> None:
    metrics = pd.DataFrame({"reporter_code": [1], "material_importer": [True]})
    links = pd.DataFrame(columns=[
        "reporter_code", "partner_code", "partner_desc", "partner_iso", "trade_value",
        "partner_share", "is_named_country", "material_importer",
    ])
    with pytest.raises(ValueError, match="expected exactly 84 material importers"):
        supplier_dependency_diagnostics(metrics, links)
