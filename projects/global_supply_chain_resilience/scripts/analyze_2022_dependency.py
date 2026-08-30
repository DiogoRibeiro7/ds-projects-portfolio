"""Compute the first structural-dependency baseline from the validated 2022 ICIO table."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from supply_chain_resilience.dependency import (
    cross_border_intermediate_share,
    structural_dependency_metrics,
)
from supply_chain_resilience.mapping import (
    active_production_blocks,
    extract_2022_blocks,
    validate_2022_accounting,
)


def _load_2022(archive_path: Path) -> pd.DataFrame:
    with ZipFile(archive_path) as archive:
        with archive.open("2022_SML.csv") as handle:
            return pd.read_csv(handle, index_col=0, low_memory=False)


def _records(frame: pd.DataFrame, columns: list[str], n: int = 20) -> list[dict[str, object]]:
    selected = frame.head(n).reset_index()
    return [
        {
            key: (None if pd.isna(value) else float(value) if isinstance(value, (float, np.floating)) else value)
            for key, value in row.items()
            if key in {"node", "country", "activity", *columns}
        }
        for row in selected.to_dict(orient="records")
    ]


def _quantiles(series: pd.Series) -> dict[str, float]:
    clean = series.dropna().astype(float)
    if clean.empty:
        return {}
    return {
        f"p{int(q * 100):02d}": float(clean.quantile(q))
        for q in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()

    raw_archive = args.archive.read_bytes()
    frame = _load_2022(args.archive)
    published = extract_2022_blocks(frame)
    validate_2022_accounting(published)
    active, inactive = active_production_blocks(published)
    metrics = structural_dependency_metrics(active)

    positive_intermediate = metrics.loc[metrics["intermediate_input"] > 0.0, "intermediate_input"]
    material_threshold = float(positive_intermediate.median())
    positive_foreign = metrics.loc[metrics["foreign_input"] > 0.0, "foreign_input"]
    foreign_scale_threshold = float(positive_foreign.median())

    top_foreign_value = metrics.sort_values(
        ["foreign_input", "foreign_input_dependence"], ascending=[False, False]
    )
    top_material_dependence = metrics.loc[
        metrics["intermediate_input"] >= material_threshold
    ].sort_values(["foreign_input_dependence", "foreign_input"], ascending=[False, False])
    top_material_foreign_hhi = metrics.loc[
        metrics["foreign_input"] >= foreign_scale_threshold
    ].sort_values(["foreign_supplier_country_hhi", "foreign_input"], ascending=[False, False])

    summary = {
        "year": 2022,
        "source_sha256": sha256(raw_archive).hexdigest(),
        "active_nodes": int(len(metrics)),
        "inactive_zero_output_labels": int(len(inactive)),
        "cross_border_intermediate_share": cross_border_intermediate_share(active),
        "positive_intermediate_input_nodes": int((metrics["intermediate_input"] > 0.0).sum()),
        "positive_foreign_input_nodes": int((metrics["foreign_input"] > 0.0).sum()),
        "material_intermediate_input_threshold": material_threshold,
        "material_foreign_input_threshold": foreign_scale_threshold,
        "material_threshold_definition": "median positive intermediate input across active 2022 nodes",
        "foreign_scale_threshold_definition": "median positive foreign input across active 2022 nodes",
        "foreign_input_dependence_quantiles": _quantiles(metrics["foreign_input_dependence"]),
        "supplier_country_hhi_quantiles": _quantiles(metrics["supplier_country_hhi"]),
        "foreign_supplier_country_hhi_quantiles": _quantiles(metrics["foreign_supplier_country_hhi"]),
        "effective_supplier_countries_quantiles": _quantiles(metrics["effective_supplier_countries"]),
        "top_foreign_input_by_value": _records(
            top_foreign_value,
            ["gross_output", "intermediate_input", "foreign_input", "foreign_input_dependence"],
        ),
        "top_foreign_dependence_material_nodes": _records(
            top_material_dependence,
            ["gross_output", "intermediate_input", "foreign_input", "foreign_input_dependence", "supplier_country_hhi", "effective_supplier_countries"],
        ),
        "top_foreign_concentration_material_importers": _records(
            top_material_foreign_hhi,
            ["gross_output", "foreign_input", "foreign_input_dependence", "foreign_supplier_country_hhi", "foreign_effective_supplier_countries", "largest_foreign_supplier_country_share"],
        ),
        "interpretation_boundary": (
            "These are direct observed 2022 input-dependency descriptors. They are not causal, "
            "do not measure propagation, and are not yet a composite systemic-vulnerability score."
        ),
    }

    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.csv_output, index=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
