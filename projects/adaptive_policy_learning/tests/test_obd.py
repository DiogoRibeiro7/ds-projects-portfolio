from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from adaptive_policy_learning.obd import audit_archive


def _write_csv(archive: ZipFile, member: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("test fixture requires rows")
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    archive.writestr(member, buffer.getvalue())


def _logged_rows(*, pscore: float = 0.5) -> list[dict[str, object]]:
    return [
        {
            "timestamp": "2019-11-25 00:00:00",
            "item_id": 0,
            "position": 1,
            "click": 0,
            "propensity_score": pscore,
            "user_feature_0": "A",
        },
        {
            "timestamp": "2019-11-25 00:01:00",
            "item_id": 1,
            "position": 2,
            "click": 1,
            "propensity_score": pscore,
            "user_feature_0": "B",
        },
    ]


def _item_rows() -> list[dict[str, object]]:
    return [
        {"item_id": 0, "item_feature_0": 0.1},
        {"item_id": 1, "item_feature_0": 0.2},
    ]


def _build_archive(path: Path, *, pscore: float = 0.5) -> None:
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for policy in ("bts", "random"):
            prefix = f"open_bandit_dataset/{policy}/all"
            _write_csv(archive, f"{prefix}/all.csv", _logged_rows(pscore=pscore))
            _write_csv(archive, f"{prefix}/item_context.csv", _item_rows())


def test_audit_archive_reports_schema_without_outcome_aggregates(tmp_path: Path) -> None:
    path = tmp_path / "open_bandit_dataset.zip"
    _build_archive(path)

    report = audit_archive(path, enforce_official_contract=False)

    assert report["campaign"] == "all"
    assert report["catalog_action_count"] == 81
    assert report["leftmost_raw_position"] == "1"
    assert report["normalized_leftmost_position"] == 0
    assert report["logged_files"]["bts"]["propensity_field"] == "propensity_score"
    assert report["logged_files"]["bts"]["row_count"] == 2
    assert report["logged_action_support"]["bts"]["observed_action_ids"] == [0, 1]
    assert "ctr" not in report
    assert "reward_mean" not in report


def test_audit_archive_strict_contract_rejects_nonofficial_catalog(tmp_path: Path) -> None:
    path = tmp_path / "open_bandit_dataset.zip"
    _build_archive(path)

    with pytest.raises(ValueError, match="item context universe"):
        audit_archive(path)


def test_audit_archive_rejects_invalid_propensity(tmp_path: Path) -> None:
    path = tmp_path / "open_bandit_dataset.zip"
    _build_archive(path, pscore=0.0)

    with pytest.raises(ValueError, match="propensity"):
        audit_archive(path, enforce_official_contract=False)
