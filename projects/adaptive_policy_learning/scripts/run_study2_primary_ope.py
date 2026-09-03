"""Run the authorized Adaptive Policy Learning Study 2 primary OPE."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from adaptive_policy_learning.study2_empirical import run_study2_primary_ope


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--protocol-dir", type=Path, default=Path("protocol"))
    parser.add_argument("--chunk-size", type=int, default=100_000)
    args = parser.parse_args()

    try:
        result = run_study2_primary_ope(
            args.archive,
            args.output,
            code_sha=args.code_sha,
            protocol_dir=args.protocol_dir,
            chunk_size=args.chunk_size,
        )
    except Exception as exc:
        failure = {
            "status": "failure",
            "study": "Adaptive Policy Learning Study 2",
            "protocol_version": "1.1-qualified-study2-primary-ope",
            "code_sha": args.code_sha,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(failure, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(failure, indent=2, sort_keys=True))
        raise

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
