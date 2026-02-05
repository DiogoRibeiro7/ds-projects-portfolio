"""Minimal demo that exercises the data cleaning + analysis stack.

Usage:
    python examples/run_demo.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_processing.cleaning import clean_ab_data
from src.statistics.core import ExperimentAnalyzer


def build_dataset(seed: int = 202401) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    control = pd.DataFrame(
        {
            "user_id": range(800),
            "group": "control",
            "converted": rng.binomial(1, 0.12, size=800),
        }
    )
    treatment = pd.DataFrame(
        {
            "user_id": range(800, 1620),
            "group": "treatment",
            "converted": rng.binomial(1, 0.14, size=820),
        }
    )
    df = pd.concat([control, treatment], ignore_index=True)
    df.loc[5, "converted"] = np.nan  # introduce small data issue for cleaning demo
    return df


def main() -> None:
    raw = build_dataset()
    cleaned = clean_ab_data(raw, user_col="user_id", group_col="group")

    analyzer = ExperimentAnalyzer()
    report = analyzer.analyze_conversion(cleaned, conversion_col="converted")

    control_rate = report["conversion_rates"]["control"]
    treatment_rate = report["conversion_rates"]["treatment"]

    print("=== Data Science Portfolio Demo ===")
    print(f"Rows before cleaning: {len(raw):,} -> after: {len(cleaned):,}")
    print("control vs treatment conversion")
    print(f"  control   : {control_rate:.5f}")
    print(f"  treatment : {treatment_rate:.5f}")
    print(f"absolute lift : {report['absolute_lift']:.5f}")
    print(f"p-value        : {report['p_value']:.4f}")
    print(f"statistical power (observed): {report['statistical_power']:.4f}")


if __name__ == "__main__":
    main()
