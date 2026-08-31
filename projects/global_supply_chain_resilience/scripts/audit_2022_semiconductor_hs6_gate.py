"""Audit whether the frozen global HS 8542 reporter universe supports the preregistered HS6 split."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

FROZEN_CODES = ("854231", "854232", "854233", "854239", "854290")
AGGREGATE_REPORTERS = {97, 975}


def reference_result(payload: dict[str, object], classification: str) -> dict[str, object]:
    if str(payload.get("classCode")) != classification:
        raise ValueError(f"unexpected classCode for {classification}")
    rows = payload.get("results")
    if not isinstance(rows, list):
        raise ValueError("commodity reference must contain results")
    by_code = {str(row.get("id")): row for row in rows if isinstance(row, dict)}
    present = []
    missing = []
    for code in FROZEN_CODES:
        row = by_code.get(code)
        if row is None:
            missing.append(code)
            continue
        if int(row.get("aggrlevel", -1)) != 6 or str(row.get("parent")) != "8542":
            raise ValueError(f"{classification} code {code} is not a six-digit child of 8542")
        present.append(code)
    return {
        "classification": classification,
        "present_codes": present,
        "missing_codes": missing,
        "all_frozen_codes_present": not missing,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reporter-lock", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    lock = json.loads(args.reporter_lock.read_text(encoding="utf-8"))
    rows = lock["source_reporters"]
    primary = [row for row in rows if int(row["reporter_code"]) not in AGGREGATE_REPORTERS]
    if len(rows) != 169 or len(primary) != 167:
        raise ValueError("unexpected frozen reporter cardinality")
    counts = Counter(str(row["classification_code"]) for row in primary)

    coverage: dict[str, dict[str, object]] = {}
    for classification in sorted(counts):
        payload = json.loads((args.reference_dir / f"{classification}.json").read_text(encoding="utf-8"))
        coverage[classification] = reference_result(payload, classification)

    incompatible = [
        classification
        for classification in sorted(counts)
        if not coverage[classification]["all_frozen_codes_present"]
    ]
    affected = sum(counts[classification] for classification in incompatible)
    result = {
        "reference_year": 2022,
        "frozen_codes": list(FROZEN_CODES),
        "primary_reporter_count": 167,
        "classification_reporter_counts": dict(sorted(counts.items())),
        "reference_coverage": coverage,
        "global_gate_pass": not incompatible,
        "incompatible_classifications": incompatible,
        "affected_reporter_count": affected,
        "decision": (
            "Global 167-reporter HS6 decomposition is permitted."
            if not incompatible
            else "Global 167-reporter HS6 decomposition is not permitted; stop before HS6 trade-value analysis."
        ),
        "scientific_boundary": "This gate audits classification/code compatibility only. It contains no six-digit trade values, concentrations, rankings, or post hoc reporter exclusions.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
