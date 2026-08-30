"""Evaluate 2022 dependency-ranking stability under prospective A-link thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from supply_chain_resilience.dependency import structural_dependency_metrics
from supply_chain_resilience.mapping import active_production_blocks, extract_2022_blocks, validate_2022_accounting
from supply_chain_resilience.sensitivity import (
    dependency_metrics_after_threshold,
    foreign_intermediate_total,
    ranking_stability,
    threshold_intermediate_use_by_input_share,
)

THRESHOLDS = (0.0, 0.001, 0.005, 0.01)


def load_2022_csv(archive_path: Path, member: str = "2022_SML.csv") -> pd.DataFrame:
    """Load the observed 2022 ICIO member from the validated OECD archive."""
    with ZipFile(archive_path) as archive:
        with archive.open(member) as handle:
            return pd.read_csv(handle, index_col=0, low_memory=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    published = extract_2022_blocks(load_2022_csv(args.archive))
    validate_2022_accounting(published)
    blocks, _ = active_production_blocks(published)
    baseline = structural_dependency_metrics(blocks)

    positive_intermediate = baseline.loc[baseline["intermediate_input"] > 0.0, "intermediate_input"]
    material_threshold = float(positive_intermediate.median())
    material_nodes = baseline.index[baseline["intermediate_input"] >= material_threshold]

    positive_foreign = baseline.loc[baseline["foreign_input"] > 0.0, "foreign_input"]
    material_foreign_threshold = float(positive_foreign.median())
    material_importers = baseline.index[baseline["foreign_input"] >= material_foreign_threshold]

    baseline_total = float(blocks.intermediate_use.to_numpy(dtype=float).sum())
    baseline_foreign = foreign_intermediate_total(blocks.intermediate_use)

    reports: list[dict[str, object]] = []
    for threshold in THRESHOLDS:
        thresholded = threshold_intermediate_use_by_input_share(
            blocks,
            minimum_input_share=threshold,
        )
        metrics = dependency_metrics_after_threshold(
            blocks,
            minimum_input_share=threshold,
        )
        retained_total = float(thresholded.to_numpy(dtype=float).sum())
        retained_foreign = foreign_intermediate_total(thresholded)
        retained_links = int((thresholded.to_numpy(dtype=float) > 0.0).sum())

        reports.append(
            {
                "minimum_input_share": threshold,
                "retained_links": retained_links,
                "retained_total_intermediate_share": retained_total / baseline_total,
                "retained_foreign_intermediate_share": retained_foreign / baseline_foreign,
                "foreign_input_value_ranking": ranking_stability(
                    baseline["foreign_input"],
                    metrics["foreign_input"],
                    eligible_nodes=baseline.index,
                ),
                "foreign_dependence_material_ranking": ranking_stability(
                    baseline["foreign_input_dependence"],
                    metrics["foreign_input_dependence"],
                    eligible_nodes=material_nodes,
                ),
                "foreign_concentration_material_ranking": ranking_stability(
                    baseline["foreign_supplier_country_hhi"],
                    metrics["foreign_supplier_country_hhi"],
                    eligible_nodes=material_importers,
                ),
            }
        )

    summary = {
        "year": 2022,
        "thresholds": list(THRESHOLDS),
        "threshold_definition": "retain links only when A[i,j] > threshold; diagnostic only, without rebalancing Z",
        "material_intermediate_input_threshold": material_threshold,
        "material_foreign_input_threshold": material_foreign_threshold,
        "material_universe_definition": "fixed from the unthresholded 2022 baseline for all comparisons",
        "reports": reports,
        "interpretation_boundary": (
            "This sensitivity analysis tests ranking dependence on individually small direct input links. "
            "It does not validate a thresholded table as a new accounting system and does not measure shock propagation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
