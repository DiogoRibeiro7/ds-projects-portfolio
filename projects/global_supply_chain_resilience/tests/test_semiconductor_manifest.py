from __future__ import annotations

import json
from pathlib import Path


def test_authoritative_semiconductor_reporter_lock_is_self_consistent() -> None:
    lock_path = Path(__file__).parents[1] / "protocol" / "semiconductor_reporter_lock_2022.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    reporters = lock["source_reporters"]

    assert int(lock["source_positive_reporter_count"]) == 169
    assert int(lock["primary_reporter_count"]) == 167
    assert lock["excluded_aggregate_reporter_codes"] == [97, 975]
    assert len(reporters) == 169

    codes = [int(row["reporter_code"]) for row in reporters]
    assert len(set(codes)) == 169
    assert {97, 975}.issubset(codes)
    assert 490 in codes
    assert all(isinstance(row["dataset_checksum"], int) for row in reporters)
    assert all(str(row["classification_code"]).startswith("H") for row in reporters)
    assert lock["source_workflow_run"] == 33386592868
    assert lock["source_artifact_id"] == 9756068440
