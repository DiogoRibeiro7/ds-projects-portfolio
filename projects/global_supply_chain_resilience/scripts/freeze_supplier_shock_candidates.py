"""Freeze supplier shock candidates from threshold-persistent 2022 rankings."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from supply_chain_resilience.candidates import persistent_top_k, real_country_candidates
from supply_chain_resilience.mapping import active_production_blocks, extract_2022_blocks, validate_2022_accounting
from supply_chain_resilience.sensitivity import threshold_intermediate_use_by_input_share
from supply_chain_resilience.supplier import supplier_importance_metrics

THRESHOLDS = (0.0, 0.001, 0.005, 0.01)
TOP_K = 20


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw = args.archive.read_bytes()
    with ZipFile(args.archive) as archive:
        with archive.open("2022_SML.csv") as handle:
            frame = pd.read_csv(handle, index_col=0, low_memory=False)

    published = extract_2022_blocks(frame)
    validate_2022_accounting(published)
    blocks, inactive = active_production_blocks(published)
    baseline = supplier_importance_metrics(blocks)

    positive_foreign = baseline.loc[baseline["foreign_intermediate_sales"] > 0.0]
    foreign_scale = float(positive_foreign["foreign_intermediate_sales"].median())
    material = baseline.index[baseline["foreign_intermediate_sales"] >= foreign_scale]

    sales_rankings: dict[float, pd.Series] = {}
    downstream_rankings: dict[float, pd.Series] = {}
    for threshold in THRESHOLDS:
        z_t = threshold_intermediate_use_by_input_share(blocks, minimum_input_share=threshold)
        metrics = supplier_importance_metrics(replace(blocks, intermediate_use=z_t))
        sales_rankings[threshold] = metrics["foreign_intermediate_sales"]
        downstream_rankings[threshold] = metrics.loc[material, "foreign_downstream_input_share_mass"]

    persistent_sales = persistent_top_k(sales_rankings, k=TOP_K)
    persistent_downstream = persistent_top_k(downstream_rankings, k=TOP_K)
    persistent_both = [node for node in persistent_downstream if node in set(persistent_sales)]
    real_country_both = real_country_candidates(persistent_both)

    details = baseline.loc[real_country_both].reset_index().to_dict("records") if real_country_both else []
    summary = {
        "year": 2022,
        "source_sha256": sha256(raw).hexdigest(),
        "thresholds": list(THRESHOLDS),
        "top_k": TOP_K,
        "inactive_zero_output_labels": int(len(inactive)),
        "material_foreign_sales_threshold": foreign_scale,
        "material_supplier_universe": int(len(material)),
        "persistent_top20_foreign_sales": persistent_sales,
        "persistent_top20_foreign_downstream_mass": persistent_downstream,
        "persistent_top20_both": persistent_both,
        "persistent_real_country_candidates": real_country_both,
        "persistent_real_country_candidate_details": details,
        "selection_rule": (
            "Candidate supplier nodes must appear in the top 20 of both foreign intermediate sales "
            "and foreign downstream input-share mass at every prospective threshold A[i,j] > "
            "0, 0.001, 0.005, and 0.01. Aggregate ROW nodes are excluded from real-country shocks."
        ),
        "scientific_boundary": (
            "This artifact freezes eligible supplier nodes only. Shock magnitudes, propagation equations, "
            "and outcome metrics remain unspecified and must be frozen prospectively in a later gate."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
