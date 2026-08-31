from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_2022_semiconductor_hs6_gate.py"
spec = importlib.util.spec_from_file_location("hs6_gate", SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
reference_result = module.reference_result


def _payload(classification: str, codes: list[str]) -> dict[str, object]:
    return {
        "classCode": classification,
        "results": [
            {"id": code, "aggrlevel": 6, "parent": "8542", "text": code}
            for code in codes
        ],
    }


def test_reference_result_detects_incompatible_h2() -> None:
    result = reference_result(_payload("H2", ["854290"]), "H2")
    assert not result["all_frozen_codes_present"]
    assert result["present_codes"] == ["854290"]
    assert result["missing_codes"] == ["854231", "854232", "854233", "854239"]


def test_reference_result_accepts_complete_revision() -> None:
    result = reference_result(_payload("H6", list(module.FROZEN_CODES)), "H6")
    assert result["all_frozen_codes_present"]
    assert result["missing_codes"] == []


def test_reference_result_rejects_wrong_parent() -> None:
    payload = _payload("H6", list(module.FROZEN_CODES))
    payload["results"][0]["parent"] = "8541"
    with pytest.raises(ValueError, match="six-digit child of 8542"):
        reference_result(payload, "H6")
