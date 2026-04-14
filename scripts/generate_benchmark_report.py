#!/usr/bin/env python3
"""Generate a benchmark report from saved benchmark JSON results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from performance_optimization.benchmarking import PerformanceBenchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate benchmark reports from stored benchmark JSON files."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default="benchmark_results",
        help=(
            "Path to a benchmark JSON file or directory containing benchmark_data_*.json. "
            "If a directory is provided, the latest benchmark JSON file is used."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_reports",
        help="Directory where the generated report, plots, and summary files will be saved.",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="When input_path is a directory, choose the latest benchmark JSON file.",
    )
    return parser.parse_args()


def find_latest_benchmark_file(directory: Path) -> Path | None:
    candidates = sorted(
        directory.glob("benchmark_data_*.json"),
        key=lambda path: path.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def main() -> int:
    args = parse_args()
    source = Path(args.input_path)

    if source.is_dir():
        benchmark_file = find_latest_benchmark_file(source)
        if benchmark_file is None:
            print(f"No benchmark JSON files found in {source}")
            return 1
        print(f"Using latest benchmark file: {benchmark_file}")
    elif source.is_file():
        benchmark_file = source
    else:
        print(f"Benchmark file or directory not found: {source}")
        return 1

    benchmark = PerformanceBenchmark(output_dir=args.output_dir)
    benchmark.load_results(benchmark_file)
    report_path = benchmark.generate_report()

    if report_path is None:
        print("Benchmark report generation failed.")
        return 1

    print(f"Benchmark report written to: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
