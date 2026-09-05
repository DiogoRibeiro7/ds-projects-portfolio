"""Run the prospective Study 3 training-only qualification gate."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from adaptive_policy_learning.study3_training import (
    ARCHIVE_SHA256,
    CAMPAIGN,
    protocol_hash,
    run_study3_training_qualification,
)


def _protocol_hash_or_none(protocol_dir: Path) -> str | None:
    """Return frozen protocol identity when available, without hiding the original failure."""
    try:
        return protocol_hash(protocol_dir)
    except OSError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--protocol-dir", type=Path, default=Path("protocol"))
    args = parser.parse_args()

    try:
        result = run_study3_training_qualification(
            args.archive,
            args.output,
            code_sha=args.code_sha,
            protocol_dir=args.protocol_dir,
        )
    except Exception as exc:
        failure = {
            "status": "training_gate_failure",
            "study": "Adaptive Policy Learning Study 3",
            "protocol_version": "2.5-study3-training-only-qualification",
            "code_sha": args.code_sha,
            "protocol_hash": _protocol_hash_or_none(args.protocol_dir),
            "source_archive_sha256": ARCHIVE_SHA256,
            "campaign": CAMPAIGN,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "evaluation_outcomes_loaded": False,
            "random_reference_outcomes_loaded": False,
            "ope_estimates_computed": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(failure, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(failure, indent=2, sort_keys=True))
        raise

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
