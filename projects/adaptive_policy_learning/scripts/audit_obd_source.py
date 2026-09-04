"""Audit one official Open Bandit Dataset campaign without outcome summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from adaptive_policy_learning.obd import audit_archive


def main() -> None:
    """Run a value-free source audit and write the JSON report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--campaign", default="all")
    parser.add_argument(
        "--skip-click-validation",
        action="store_true",
        help="Do not parse click values; intended for prospective source-selection audits.",
    )
    args = parser.parse_args()

    report = audit_archive(
        args.archive,
        campaign=args.campaign,
        validate_click_values=not args.skip_click_validation,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
