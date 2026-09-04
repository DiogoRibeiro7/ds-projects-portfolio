"""Select the prospective Study 3 campaign from value-free source-audit reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CANDIDATES = ("men", "women")


def _load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return data


def _qualifies(report: dict[str, Any]) -> bool:
    logged = report.get("logged_files")
    support = report.get("logged_action_support")
    if not isinstance(logged, dict) or not isinstance(support, dict):
        return False
    if report.get("click_value_validation_performed") is not False:
        return False
    for policy in ("bts", "random"):
        policy_logged = logged.get(policy)
        policy_support = support.get(policy)
        if not isinstance(policy_logged, dict) or not isinstance(policy_support, dict):
            return False
        if policy_logged.get("propensity_field") != "propensity_score":
            return False
        if set(policy_logged.get("raw_positions", [])) != {"1", "2", "3"}:
            return False
        if policy_support.get("missing_catalog_action_ids"):
            return False
    return True


def select_campaign(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Apply the frozen Study 3 value-free campaign-selection rule."""
    qualified: list[tuple[int, str]] = []
    for campaign in CANDIDATES:
        report = reports.get(campaign)
        if report is None or not _qualifies(report):
            continue
        logged = report["logged_files"]
        random_rows = int(logged["random"]["row_count"])
        qualified.append((random_rows, campaign))

    if not qualified:
        return {
            "status": "no_qualifying_campaign",
            "selected_campaign": None,
            "selection_basis": "largest Random-policy row_count among qualifying campaigns; lexical tie-break",
        }

    qualified.sort(key=lambda pair: (-pair[0], pair[1]))
    random_rows, selected = qualified[0]
    return {
        "status": "selected",
        "selected_campaign": selected,
        "selected_random_row_count": random_rows,
        "qualified_campaigns": [campaign for _, campaign in qualified],
        "selection_basis": "largest Random-policy row_count among qualifying campaigns; lexical tie-break",
        "outcome_values_parsed": False,
    }


def main() -> None:
    """Load source audits, apply the frozen rule, and write the selection record."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--men", type=Path, required=True)
    parser.add_argument("--women", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = select_campaign({"men": _load(args.men), "women": _load(args.women)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))

    if result["status"] != "selected":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
