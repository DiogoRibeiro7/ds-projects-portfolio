#!/usr/bin/env python
"""
Verify docstring coverage for public modules, classes, and functions.

The script scans key source packages, counts the number of documented symbols,
and fails when coverage falls below the configured threshold or when a missing
docstring is detected. Files listed in ``KNOWN_PARSE_ISSUES`` are skipped until
their syntax errors are fixed (see docs/DOCS_STYLE.md for the rationale).
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TARGETS: tuple[str, ...] = (
    "src",
    "statistical_methods",
    "dashboard_enhanced",
    "ab_testing",
    "time-series",
    "tests",
    "modern-bank-churn",
)

# File-specific ignores are documented in docs/DOCS_STYLE.md.
KNOWN_PARSE_ISSUES = {
    Path("modern-bank-churn/enhanced_feature_engineering.py"),
}

COVERAGE_THRESHOLD = 1.0

IGNORED_DIR_PARTS = {".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache"}


@dataclass
class CoverageStats:
    documented: int = 0
    total: int = 0

    @property
    def ratio(self) -> float:
        if self.total == 0:
            return 1.0
        return self.documented / self.total


def is_public(name: str) -> bool:
    """Return True when a symbol should be documented per project policy."""
    if name.startswith("__") and name.endswith("__"):
        return False
    return not name.startswith("_")


def iter_python_files() -> Iterable[Path]:
    """Yield Python files under the configured targets."""
    for entry in TARGETS:
        base = Path(entry)
        if not base.exists():
            continue
        paths: Iterable[Path]
        if base.is_file() and base.suffix == ".py":
            paths = [base]
        else:
            paths = base.rglob("*.py")
        for path in paths:
            if KNOWN_PARSE_ISSUES and path in KNOWN_PARSE_ISSUES:
                continue
            if any(part in IGNORED_DIR_PARTS for part in path.parts):
                continue
            yield path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail when docstring coverage falls below the threshold."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=COVERAGE_THRESHOLD,
        help="Required coverage ratio (default: %(default)s)",
    )
    args = parser.parse_args()

    stats = CoverageStats()
    missing: list[str] = []
    parse_failures: list[str] = []

    for path in iter_python_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            parse_failures.append(f"{path}: {exc}")
            continue

        stats.total += 1
        if ast.get_docstring(tree):
            stats.documented += 1
        else:
            missing.append(f"{path}:<module>")

        for node in tree.body:
            if isinstance(node, ast.ClassDef) and is_public(node.name):
                stats.total += 1
                if ast.get_docstring(node):
                    stats.documented += 1
                else:
                    missing.append(f"{path}:{node.name} (class)")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and is_public(
                node.name
            ):
                stats.total += 1
                if ast.get_docstring(node):
                    stats.documented += 1
                else:
                    missing.append(f"{path}:{node.name}()")

    if parse_failures:
        print("Docstring coverage check failed due to parse errors:")
        for entry in parse_failures:
            print(f"  - {entry}")
        return 1

    if missing:
        print("Missing docstrings detected:")
        for entry in missing:
            print(f"  - {entry}")

    coverage_pct = stats.ratio * 100
    print(
        f"Docstring coverage: {stats.documented}/{stats.total} "
        f"({coverage_pct:.2f}% minimum required {args.threshold * 100:.2f}%)"
    )

    if stats.ratio < args.threshold or missing:
        return 1

    if KNOWN_PARSE_ISSUES:
        print(
            "Skipped files with known parser issues:\n  - "
            + "\n  - ".join(str(p) for p in KNOWN_PARSE_ISSUES)
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
