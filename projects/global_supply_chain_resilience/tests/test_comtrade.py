from __future__ import annotations

from urllib.parse import parse_qs, urlsplit

import pytest

from supply_chain_resilience.comtrade import (
    canonical_json_sha256,
    extract_data_rows,
    redact_subscription_key,
    response_schema,
)


def test_canonical_json_sha256_is_key_order_invariant() -> None:
    left = {"b": [2, 1], "a": {"y": 2, "x": 1}}
    right = {"a": {"x": 1, "y": 2}, "b": [2, 1]}
    assert canonical_json_sha256(left) == canonical_json_sha256(right)


def test_extract_data_rows_accepts_standard_payload() -> None:
    rows = extract_data_rows({"data": [{"reporterCode": 842, "primaryValue": 10.0}]})
    assert rows == [{"reporterCode": 842, "primaryValue": 10.0}]


def test_extract_data_rows_rejects_non_row_payload() -> None:
    with pytest.raises(ValueError, match="data row list"):
        extract_data_rows({"data": {"reporterCode": 842}})


def test_response_schema_records_union_of_fields_and_types() -> None:
    schema = response_schema(
        [
            {"reporterCode": 842, "primaryValue": 1.0},
            {"reporterCode": 276, "primaryValue": None, "partnerDesc": "World"},
        ]
    )
    assert schema["row_count"] == 2
    assert schema["fields"] == ["partnerDesc", "primaryValue", "reporterCode"]
    assert schema["field_types"]["reporterCode"] == ["int"]
    assert schema["field_types"]["primaryValue"] == ["NoneType", "float"]


def test_redact_subscription_key_removes_secret_only() -> None:
    redacted = redact_subscription_key(
        "https://comtradeapi.un.org/public/v1/preview/C/A/HS?period=2022&subscription-key=secret&cmdCode=8542"
    )
    query = parse_qs(urlsplit(redacted).query)
    assert "subscription-key" not in query
    assert query == {"period": ["2022"], "cmdCode": ["8542"]}
