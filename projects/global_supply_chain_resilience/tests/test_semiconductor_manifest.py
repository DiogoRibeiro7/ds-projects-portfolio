from __future__ import annotations

import json
from pathlib import Path


def test_frozen_semiconductor_reporter_manifest_is_self_consistent() -> None:
    manifest_path = Path(__file__).parents[1] / "protocol" / "semiconductor_reporter_universe_2022.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reporters = manifest["reporters"]
    declared = int(manifest["positive_hs8542_import_reporters"])

    assert declared == 169
    assert len(reporters) == declared
    codes = [int(row["reporter_code"]) for row in reporters]
    assert len(set(codes)) == declared
    assert sum(bool(row["is_special_reporter"]) for row in reporters) == 2
    assert {row["reporter_iso"] for row in reporters if row["is_special_reporter"]} == {"S19", "R4"}
