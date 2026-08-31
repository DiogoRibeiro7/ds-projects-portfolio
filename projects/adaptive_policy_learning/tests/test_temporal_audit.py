from __future__ import annotations

import csv
import json
import subprocess
import sys
from io import StringIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def _write_csv(archive: ZipFile, member: str, rows: list[dict[str, object]]) -> None:
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    archive.writestr(member, buffer.getvalue())


def _row(minute: int, item_id: int, *, position: int = 1) -> dict[str, object]:
    return {
        "timestamp": f"2019-11-25 00:{minute:02d}:00+00:00",
        "item_id": item_id,
        "position": position,
        "click": 0,
        "propensity_score": 0.5,
    }


def test_temporal_audit_freezes_stable_70_30_split(tmp_path: Path) -> None:
    archive_path = tmp_path / "obd.zip"
    output_path = tmp_path / "temporal.json"
    bts_rows = [_row(i, i % 2) for i in range(10)]
    random_rows = [_row(i, i % 2) for i in range(10)]
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        _write_csv(archive, "open_bandit_dataset/bts/all/all.csv", bts_rows)
        _write_csv(archive, "open_bandit_dataset/random/all/all.csv", random_rows)

    subprocess.run(
        [
            sys.executable,
            "scripts/audit_obd_temporal_split.py",
            "--archive",
            str(archive_path),
            "--output",
            str(output_path),
        ],
        check=True,
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
    )
    report = json.loads(output_path.read_text(encoding="utf-8"))

    assert report["bts_position_1"]["row_count"] == 10
    assert report["bts_position_1"]["training_row_count"] == 7
    assert report["bts_position_1"]["evaluation_row_count"] == 3
    assert report["bts_position_1"]["training_last_timestamp"] == "2019-11-25 00:06:00+00:00"
    assert report["bts_position_1"]["evaluation_first_timestamp"] == "2019-11-25 00:07:00+00:00"
    assert report["random_reference"]["row_count"] == 3
    assert report["random_reference"]["observed_action_ids"] == [0, 1]
