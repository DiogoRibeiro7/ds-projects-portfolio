from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_runner() -> object:
    path = Path("scripts/run_2022_semiconductor_concentration.py")
    spec = importlib.util.spec_from_file_location("semiconductor_concentration_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load semiconductor concentration runner.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest() -> dict[str, object]:
    reporters = [
        {"reporter_code": code}
        for code in list(range(1, 168)) + [97, 975]
    ]
    # Ensure reporter codes are unique despite 97 appearing in range(1, 168).
    reporters = [{"reporter_code": code} for code in sorted(set(r["reporter_code"] for r in reporters))]
    # Add enough neutral reporters to make the source universe exactly 169.
    next_code = 1000
    while len(reporters) < 169:
        reporters.append({"reporter_code": next_code})
        next_code += 1
    return {"reporters": reporters}


def test_primary_reporters_exclude_only_overlap_groups() -> None:
    runner = _load_runner()
    manifest = _manifest()
    reporters = runner.primary_reporters_from_manifest(manifest)
    codes = {int(row["reporter_code"]) for row in reporters}

    assert len(reporters) == 167
    assert 97 not in codes
    assert 975 not in codes


def test_primary_reporters_reject_missing_source_reporter() -> None:
    runner = _load_runner()
    manifest = _manifest()
    manifest["reporters"] = [
        row for row in manifest["reporters"] if int(row["reporter_code"]) != 975
    ]

    with pytest.raises(ValueError, match="expected exactly 169 unique source reporters"):
        runner.primary_reporters_from_manifest(manifest)
