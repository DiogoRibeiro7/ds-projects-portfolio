from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pt_salary_gamma_distribution.pipeline import (
    build_pipeline_paths,
    build_pipeline_steps,
    run_pipeline,
    steps_to_frame,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the full analysis pipeline."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the full Portugal salary distribution analysis pipeline: "
            "sync the paired notebook, execute it, and refresh the summary outputs."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Override the project root. Defaults to the repository project directory.",
    )
    parser.add_argument(
        "--python",
        dest="python_executable",
        default=None,
        help="Python executable to use for the notebook and summary steps.",
    )
    parser.add_argument(
        "--notebook-timeout-seconds",
        type=int,
        default=1800,
        help="Execution timeout for the notebook run.",
    )
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="Skip the compact markdown and JSON summary step.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned pipeline steps as JSON and exit without running them.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full notebook-based analysis pipeline."""
    args = parse_args()
    planned_steps = build_pipeline_steps(
        paths=build_pipeline_paths(args.project_root),
        python_executable=args.python_executable,
        notebook_timeout_seconds=args.notebook_timeout_seconds,
        include_summary=not args.skip_summary,
    )
    if args.dry_run:
        print(json.dumps(steps_to_frame(planned_steps), indent=2))
        return
    steps = run_pipeline(
        project_root=args.project_root,
        python_executable=args.python_executable,
        notebook_timeout_seconds=args.notebook_timeout_seconds,
        include_summary=not args.skip_summary,
    )
    print(f"Completed {len(steps)} pipeline step(s).")


if __name__ == "__main__":
    main()
