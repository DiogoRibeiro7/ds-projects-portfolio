#!/usr/bin/env python
"""
Generate regression baseline artifacts.

Usage:
    python scripts/generate_regression_baselines.py

Outputs land in tests/regression/baselines and include metadata such as
sha256 checksum, environment info, and RNG seed.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.statistics.core import ExperimentAnalyzer  # noqa: E402
from tests.regression.baseline_utils import build_conversion_dataset  # noqa: E402

BASELINE_DIR = ROOT / "tests" / "regression" / "baselines"
BASELINE_DIR.mkdir(parents=True, exist_ok=True)


def write_conversion_report(seed: int = 202401) -> Path:
    df = build_conversion_dataset(seed=seed)
    analyzer = ExperimentAnalyzer()
    report = analyzer.analyze_conversion(df, conversion_col="converted")

    snapshot = {
        "conversion_rates": report["conversion_rates"],
        "sample_sizes": report["sample_sizes"],
        "absolute_lift": report["absolute_lift"],
        "relative_lift": report["relative_lift"],
        "z_statistic": report["z_statistic"],
        "p_value": report["p_value"],
        "confidence_interval": report["confidence_interval"],
        "statistical_power": report["statistical_power"],
    }

    payload = {
        "metadata": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "seed": seed,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "np_version": np.__version__,
        },
        "snapshot": snapshot,
    }

    output_path = BASELINE_DIR / "experiment_conversion_report.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    checksum = hashlib.sha256(output_path.read_bytes()).hexdigest()
    print(f"Wrote {output_path} (sha256={checksum[:12]}...)")
    return output_path


def main():
    write_conversion_report()


if __name__ == "__main__":
    main()
