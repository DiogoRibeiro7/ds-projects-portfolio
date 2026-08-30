"""Compute supplier-side importance and threshold stability for OECD ICIO 2022."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from supply_chain_resilience.mapping import active_production_blocks, extract_2022_blocks, validate_2022_accounting
from supply_chain_resilience.sensitivity import ranking_stability, threshold_intermediate_use_by_input_share
from supply_chain_resilience.supplier import supplier_importance_metrics

THRESHOLDS = (0.0, 0.001, 0.005, 0.01)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()

    raw = args.archive.read_bytes()
    with ZipFile(args.archive) as archive:
        with archive.open("2022_SML.csv") as handle:
            frame = pd.read_csv(handle, index_col=0, low_memory=False)

    published = extract_2022_blocks(frame)
    validate_2022_accounting(published)
    blocks, inactive = active_production_blocks(published)
    baseline = supplier_importance_metrics(blocks)

    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    baseline.to_csv(args.csv_output)

    positive_foreign = baseline.loc[baseline["foreign_intermediate_sales"] > 0.0]
    foreign_scale = float(positive_foreign["foreign_intermediate_sales"].median())
    material = baseline.index[baseline["foreign_intermediate_sales"] >= foreign_scale]

    reports = []
    original_value = float(blocks.intermediate_use.to_numpy().sum())
    original_links = int((blocks.intermediate_use.to_numpy() > 0.0).sum())
    for threshold in THRESHOLDS:
        z_t = threshold_intermediate_use_by_input_share(blocks, minimum_input_share=threshold)
        candidate = supplier_importance_metrics(replace(blocks, intermediate_use=z_t))
        reports.append(
            {
                "minimum_input_share": threshold,
                "retained_links": int((z_t.to_numpy() > 0.0).sum()),
                "retained_link_share": float((z_t.to_numpy() > 0.0).sum() / original_links),
                "retained_intermediate_value_share": float(z_t.to_numpy().sum() / original_value),
                "foreign_sales_value_ranking": ranking_stability(
                    baseline["foreign_intermediate_sales"],
                    candidate["foreign_intermediate_sales"],
                    eligible_nodes=baseline.index,
                ),
                "foreign_downstream_mass_material_ranking": ranking_stability(
                    baseline["foreign_downstream_input_share_mass"],
                    candidate["foreign_downstream_input_share_mass"],
                    eligible_nodes=material,
                ),
            }
        )

    summary = {
        "year": 2022,
        "source_sha256": sha256(raw).hexdigest(),
        "active_nodes": int(len(baseline)),
        "inactive_zero_output_labels": int(len(inactive)),
        "material_foreign_sales_threshold": foreign_scale,
        "material_supplier_universe": int(len(material)),
        "top_foreign_intermediate_sales": baseline.nlargest(20, "foreign_intermediate_sales").reset_index().to_dict("records"),
        "top_foreign_downstream_mass_material": baseline.loc[material].nlargest(20, "foreign_downstream_input_share_mass").reset_index().to_dict("records"),
        "thresholds": list(THRESHOLDS),
        "stability_reports": reports,
        "selection_boundary": (
            "Shock candidates must be supplier nodes. Direct user-side foreign dependence is not a supplier-importance score. "
            "Candidate promotion requires stability under the prospective input-share threshold audit."
        ),
    }
    args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
