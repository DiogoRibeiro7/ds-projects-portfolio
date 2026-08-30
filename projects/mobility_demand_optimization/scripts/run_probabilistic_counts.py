"""Calibrate count uncertainty on validation data and evaluate once on test data."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mobility_optimization.evaluation import headline_mask
from mobility_optimization.probabilistic import probabilistic_summary, select_dispersion_alpha

BASELINE_DIR = Path("data/results/baselines")
OUTPUT_DIR = Path("data/results/probabilistic_counts")


def main() -> None:
    """Select Negative-Binomial dispersion on validation and evaluate the test window."""
    validation_path = BASELINE_DIR / "validation_poisson_hour_of_week.parquet"
    test_path = BASELINE_DIR / "test_poisson_hour_of_week.parquet"
    if not validation_path.exists() or not test_path.exists():
        raise FileNotFoundError("Baseline Poisson forecast artifacts are required before calibration.")

    validation_all = pd.read_parquet(validation_path)
    test_all = pd.read_parquet(test_path)
    validation = validation_all.loc[headline_mask(validation_all)].copy()
    test = test_all.loc[headline_mask(test_all)].copy()
    if validation.empty or test.empty:
        raise ValueError("No rows remain after the prospective headline exclusions.")

    selected_alpha, selection_table = select_dispersion_alpha(validation)

    poisson_validation = probabilistic_summary(validation, alpha=0.0)
    selected_validation = probabilistic_summary(validation, alpha=selected_alpha)
    poisson_test = probabilistic_summary(test, alpha=0.0)
    selected_test = probabilistic_summary(test, alpha=selected_alpha)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selection_table.to_csv(OUTPUT_DIR / "validation_alpha_grid.csv", index=False)
    summary = {
        "selection_rule": "minimum mean pinball loss across q={0.1,0.5,0.9} on validation only",
        "headline_exclusion": "DST transition target rows excluded prospectively",
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "selected_alpha": selected_alpha,
        "validation": {
            "poisson": poisson_validation,
            "selected_negative_binomial": selected_validation,
        },
        "test": {
            "poisson": poisson_test,
            "selected_negative_binomial": selected_test,
        },
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
