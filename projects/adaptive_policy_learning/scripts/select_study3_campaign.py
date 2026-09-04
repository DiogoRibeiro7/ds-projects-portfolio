"""Select the prospective Study 3 campaign from value-free source-audit reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CANDIDATES = ("men", "women")


def _load(path: Path) -> dict[str, Any]:
    """Load one JSON object from disk."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return data


def _source_lock_archive_sha256(source_lock: dict[str, Any]) -> str:
    """Extract the frozen OBD archive digest from the source lock."""
    archive = source_lock.get("archive")
    if not isinstance(archive, dict):
        raise TypeError("source lock must contain an archive object")
    digest = archive.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("source lock archive.sha256 must be a 64-character string")
    return digest


def _positive_row_count(value: object) -> bool:
    """Return whether a JSON value is a strictly positive integer row count."""
    return type(value) is int and value > 0


def _integer_id_set(value: object, *, require_non_empty: bool) -> set[int] | None:
    """Validate a JSON action-ID list without coercing floats, strings, or booleans."""
    if not isinstance(value, list):
        return None
    if require_non_empty and not value:
        return None
    if any(type(item_id) is not int for item_id in value):
        return None
    return set(value)


def _qualifies(
    report: dict[str, Any],
    *,
    campaign: str,
    expected_archive_sha256: str,
) -> bool:
    """Validate one report against the frozen value-free source rule."""
    if report.get("campaign") != campaign:
        return False
    if report.get("click_value_validation_performed") is not False:
        return False

    archive = report.get("archive")
    if not isinstance(archive, dict) or archive.get("sha256") != expected_archive_sha256:
        return False

    catalog = _integer_id_set(
        report.get("archive_catalog_action_ids"),
        require_non_empty=True,
    )
    if catalog is None:
        return False

    logged = report.get("logged_files")
    support = report.get("logged_action_support")
    if not isinstance(logged, dict) or not isinstance(support, dict):
        return False

    timestamp_mins: list[str] = []
    timestamp_maxs: list[str] = []
    for policy in ("bts", "random"):
        policy_logged = logged.get(policy)
        policy_support = support.get(policy)
        if not isinstance(policy_logged, dict) or not isinstance(policy_support, dict):
            return False
        if policy_logged.get("propensity_field") != "propensity_score":
            return False
        if set(policy_logged.get("raw_positions", [])) != {"1", "2", "3"}:
            return False
        if not _positive_row_count(policy_logged.get("row_count")):
            return False

        observed = _integer_id_set(
            policy_support.get("observed_action_ids"),
            require_non_empty=True,
        )
        if observed is None or not observed.issubset(catalog):
            return False

        timestamp_min = policy_logged.get("timestamp_min")
        timestamp_max = policy_logged.get("timestamp_max")
        if not isinstance(timestamp_min, str) or not isinstance(timestamp_max, str):
            return False
        if timestamp_min > timestamp_max:
            return False
        timestamp_mins.append(timestamp_min)
        timestamp_maxs.append(timestamp_max)

    return max(timestamp_mins) <= min(timestamp_maxs)


def select_campaign(
    reports: dict[str, dict[str, Any]],
    *,
    code_sha: str,
    expected_archive_sha256: str,
) -> dict[str, Any]:
    """Apply the frozen Study 3 value-free campaign-selection rule."""
    if not code_sha:
        raise ValueError("code_sha must be non-empty")

    qualified: list[tuple[int, str]] = []
    for campaign in CANDIDATES:
        report = reports.get(campaign)
        if report is None or not _qualifies(
            report,
            campaign=campaign,
            expected_archive_sha256=expected_archive_sha256,
        ):
            continue
        logged = report["logged_files"]
        random_rows = int(logged["random"]["row_count"])
        qualified.append((random_rows, campaign))

    base = {
        "protocol_version": "2.1-study3-source-selection-execution-erratum",
        "code_sha": code_sha,
        "archive_sha256": expected_archive_sha256,
        "selection_basis": (
            "largest Random-policy row_count among qualifying campaigns; lexical tie-break"
        ),
        "outcome_values_parsed": False,
    }
    if not qualified:
        return {
            **base,
            "status": "no_qualifying_campaign",
            "selected_campaign": None,
            "qualified_campaigns": [],
        }

    qualified.sort(key=lambda pair: (-pair[0], pair[1]))
    random_rows, selected = qualified[0]
    return {
        **base,
        "status": "selected",
        "selected_campaign": selected,
        "selected_random_row_count": random_rows,
        "qualified_campaigns": [campaign for _, campaign in qualified],
    }


def main() -> None:
    """Load source audits, apply the frozen rule, and write the selection record."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--men", type=Path, required=True)
    parser.add_argument("--women", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_lock = _load(args.source_lock)
    expected_archive_sha256 = _source_lock_archive_sha256(source_lock)
    result = select_campaign(
        {"men": _load(args.men), "women": _load(args.women)},
        code_sha=args.code_sha,
        expected_archive_sha256=expected_archive_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))

    if result["status"] != "selected":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
