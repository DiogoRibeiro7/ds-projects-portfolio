"""Freeze a value-free temporal split for one official OBD campaign."""

from __future__ import annotations

import argparse
import csv
import json
import re
from io import TextIOWrapper
from pathlib import Path
from zipfile import ZipFile, ZipInfo

RAW_POSITION = "1"
TRAIN_NUMERATOR = 7
TRAIN_DENOMINATOR = 10
_CAMPAIGN_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def _validate_campaign(campaign: str) -> str:
    value = campaign.strip()
    if not value or _CAMPAIGN_PATTERN.fullmatch(value) is None:
        raise ValueError("campaign must contain only letters, numbers, underscores, or hyphens")
    return value


def _find_unique_member(archive: ZipFile, suffix: str) -> ZipInfo:
    normalized = suffix.lstrip("/")
    matches = [info for info in archive.infolist() if info.filename.endswith(normalized)]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one member ending in {suffix!r}; found {len(matches)}")
    return matches[0]


def _iter_position_rows(archive: ZipFile, info: ZipInfo):
    """Yield timestamp/action metadata only; click values are never accessed or converted."""
    with archive.open(info) as raw:
        reader = csv.DictReader(TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
        for row in reader:
            if str(row["position"]).strip() == RAW_POSITION:
                yield str(row["timestamp"]).strip(), int(row["item_id"])


def audit_temporal_split(path: Path, *, campaign: str = "all") -> dict[str, object]:
    """Instantiate the frozen 70/30 position-1 split without inspecting outcomes."""
    if not path.is_file():
        raise FileNotFoundError(path)
    campaign = _validate_campaign(campaign)

    with ZipFile(path) as archive:
        bts_info = _find_unique_member(archive, f"/bts/{campaign}/{campaign}.csv")
        random_info = _find_unique_member(archive, f"/random/{campaign}/{campaign}.csv")

        bts_n = 0
        previous_timestamp: str | None = None
        for timestamp, _ in _iter_position_rows(archive, bts_info):
            if not timestamp:
                raise ValueError("BTS position-1 row has empty timestamp.")
            if previous_timestamp is not None and timestamp < previous_timestamp:
                raise ValueError("BTS position-1 rows are not chronological in source order.")
            previous_timestamp = timestamp
            bts_n += 1
        if bts_n < 2:
            raise ValueError("BTS position-1 sample is too small for a 70/30 split.")

        train_n = bts_n * TRAIN_NUMERATOR // TRAIN_DENOMINATOR
        eval_n = bts_n - train_n
        if train_n <= 0 or eval_n <= 0:
            raise ValueError("BTS position-1 split produced an empty partition.")

        train_last_timestamp: str | None = None
        eval_first_timestamp: str | None = None
        eval_last_timestamp: str | None = None
        eval_actions: set[int] = set()
        for index, (timestamp, action) in enumerate(_iter_position_rows(archive, bts_info)):
            if index == train_n - 1:
                train_last_timestamp = timestamp
            if index == train_n:
                eval_first_timestamp = timestamp
            if index >= train_n:
                eval_last_timestamp = timestamp
                eval_actions.add(action)

        if train_last_timestamp is None or eval_first_timestamp is None or eval_last_timestamp is None:
            raise ValueError("failed to resolve BTS split timestamps.")
        if train_last_timestamp == eval_first_timestamp:
            raise ValueError("BTS split boundary timestamp is tied across training and evaluation.")

        random_reference_n = 0
        random_reference_actions: set[int] = set()
        previous_random_timestamp: str | None = None
        for timestamp, action in _iter_position_rows(archive, random_info):
            if previous_random_timestamp is not None and timestamp < previous_random_timestamp:
                raise ValueError("Random position-1 rows are not chronological in source order.")
            previous_random_timestamp = timestamp
            if eval_first_timestamp <= timestamp <= eval_last_timestamp:
                random_reference_n += 1
                random_reference_actions.add(action)
        if random_reference_n == 0:
            raise ValueError("Random has no position-1 observations in the BTS evaluation interval.")

    return {
        "campaign": campaign,
        "raw_position": 1,
        "split_fraction": {"training": 0.70, "evaluation": 0.30},
        "split_rule": "Stable source order after raw-position-1 filtering; train_n=floor(0.70*n).",
        "boundary_tie_rule": "Hard fail if the last training and first evaluation timestamps are equal.",
        "bts_position_1": {
            "row_count": bts_n,
            "training_row_count": train_n,
            "evaluation_row_count": eval_n,
            "training_last_timestamp": train_last_timestamp,
            "evaluation_first_timestamp": eval_first_timestamp,
            "evaluation_last_timestamp": eval_last_timestamp,
            "evaluation_action_count": len(eval_actions),
            "evaluation_action_ids": sorted(eval_actions),
        },
        "random_reference": {
            "window_start": eval_first_timestamp,
            "window_end": eval_last_timestamp,
            "row_count": random_reference_n,
            "observed_action_count": len(random_reference_actions),
            "observed_action_ids": sorted(random_reference_actions),
        },
        "outcome_values_parsed": False,
        "scientific_boundary": (
            "This temporal audit uses timestamps, positions, item IDs, and row counts only. "
            "The click field is never accessed or converted, and the audit produces no CTR, "
            "reward, OPE, challenger, ranking, bootstrap, or promotion result."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--campaign", default="all")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_temporal_split(args.archive, campaign=args.campaign)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
