"""Run the v0.8 optimizer-budget primary OPE study."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from adaptive_policy_learning.optimizer_v08 import run_primary_ope_v08


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--protocol-dir", type=Path, default=Path("protocol"))
    parser.add_argument("--chunk-size", type=int, default=100_000)
    args = parser.parse_args()

    try:
        result = run_primary_ope_v08(
            args.archive,
            args.output,
            code_sha=args.code_sha,
            protocol_dir=args.protocol_dir,
            chunk_size=args.chunk_size,
        )
    except Exception as exc:
        failure = {
            "status": "failure",
            "protocol_version": "0.8-optimizer-budget-amendment",
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
