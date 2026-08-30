"""Run descriptive municipality TWFE associations for the 2022-2024 panel."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_municipality_panel_support import _build_panel  # noqa: E402

from housing_tourism.twfe import fit_twfe_bundle, prepare_twfe_sample  # noqa: E402

YEARS = (2022, 2023, 2024)
PRIMARY_SAMPLE_PATH = ROOT / "results" / "processed" / "municipality_twfe_sample_unbalanced.csv"
BALANCED_SAMPLE_PATH = ROOT / "results" / "processed" / "municipality_twfe_sample_balanced.csv"
RESULTS_PATH = ROOT / "results" / "processed" / "municipality_twfe_results.json"


def main() -> None:
    """Build the municipality panel and estimate primary and balanced-sample TWFE models."""
    panel = _build_panel()
    primary = prepare_twfe_sample(panel, years=YEARS, balanced=False)
    balanced = prepare_twfe_sample(panel, years=YEARS, balanced=True)

    results = {
        "design": {
            "years": list(YEARS),
            "estimand": "descriptive within-municipality association",
            "exposure": "log tourism intensity: overnight stays per resident",
            "fixed_effects": ["municipality", "year"],
            "standard_errors": "clustered by municipality",
            "primary_sample": "complete municipality-years, municipalities with at least two years",
            "sensitivity_sample": "municipalities complete in all three years",
            "causal_claim": False,
        },
        "primary_unbalanced": fit_twfe_bundle(primary),
        "balanced_sensitivity": fit_twfe_bundle(balanced),
    }

    PRIMARY_SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    primary.to_csv(PRIMARY_SAMPLE_PATH, index=False, float_format="%.8f")
    balanced.to_csv(BALANCED_SAMPLE_PATH, index=False, float_format="%.8f")
    RESULTS_PATH.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("\nMunicipality TWFE association results:")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
