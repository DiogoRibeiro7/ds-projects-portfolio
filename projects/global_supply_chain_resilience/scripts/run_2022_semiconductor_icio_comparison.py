"""Run the frozen descriptive HS 8542 versus OECD ICIO C26 comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from supply_chain_resilience.semiconductor_icio import (
    c26_supplier_frame,
    compare_ranked_measures,
    trade_downstream_share_mass,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exporters", type=Path, required=True)
    parser.add_argument("--importer-links", type=Path, required=True)
    parser.add_argument("--icio-suppliers", type=Path, required=True)
    parser.add_argument("--export-table-output", type=Path, required=True)
    parser.add_argument("--downstream-table-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()

    exporters = pd.read_csv(args.exporters)
    links = pd.read_csv(args.importer_links)
    icio = c26_supplier_frame(pd.read_csv(args.icio_suppliers))

    if "S19" in set(exporters["reporter_iso"].astype(str)) and "TWN" not in set(
        exporters["reporter_iso"].astype(str)
    ):
        pass

    export_table, export_summary = compare_ranked_measures(
        exporters,
        icio,
        trade_code_column="reporter_iso",
        trade_label_column="reporter_desc",
        trade_value_column="world_export_value",
        icio_value_column="foreign_intermediate_sales",
    )

    downstream = trade_downstream_share_mass(links)
    if "S19" in set(downstream["partner_iso"].astype(str)) and "TWN" in set(
        downstream.loc[downstream["partner_iso"].eq("S19"), "partner_iso"].astype(str)
    ):
        raise AssertionError("Other Asia, nes must not be remapped to TWN.")
    downstream_table, downstream_summary = compare_ranked_measures(
        downstream,
        icio,
        trade_code_column="partner_iso",
        trade_label_column="partner_desc",
        trade_value_column="trade_downstream_share_mass",
        icio_value_column="foreign_downstream_input_share_mass",
    )

    summary = {
        "reference_year": 2022,
        "trade_scope": "UN Comtrade HS 8542",
        "icio_scope": "OECD ICIO C26",
        "export_scale": export_summary,
        "downstream_importance": downstream_summary,
        "trade_exporter_count": int(len(exporters)),
        "trade_named_supplier_count": int(len(downstream)),
        "icio_c26_country_count": int(len(icio)),
        "excluded_export_trade_codes": sorted(
            set(exporters["reporter_iso"].astype(str)).difference(set(export_table["country"]))
        ),
        "excluded_downstream_trade_codes": sorted(
            set(downstream["partner_iso"].astype(str)).difference(set(downstream_table["country"]))
        ),
        "excluded_icio_export_codes": sorted(
            set(icio["country"].astype(str)).difference(set(export_table["country"]))
        ),
        "excluded_icio_downstream_codes": sorted(
            set(icio["country"].astype(str)).difference(set(downstream_table["country"]))
        ),
        "evidence": {
            "importer_workflow_run": 33417837534,
            "importer_artifact_id": 9767995550,
            "exporter_workflow_run": 33421394224,
            "exporter_artifact_id": 9769297706,
            "mirror_workflow_run": 33423306450,
            "mirror_artifact_id": 9769862014,
            "icio_workflow_run": 33332046522,
            "icio_artifact_id": 9737938679,
        },
        "scientific_boundary": (
            "This is a descriptive cross-system rank comparison. HS 8542 is narrower than ICIO C26; "
            "agreement or disagreement is not validation, causal evidence, fabrication-origin evidence, "
            "or a technological-dependence measure. Other Asia, nes is not mapped to TWN, and mirror "
            "values are not used to correct trade evidence."
        ),
    }

    for path in (args.export_table_output, args.downstream_table_output, args.summary_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    export_table.to_csv(args.export_table_output, index=False)
    downstream_table.to_csv(args.downstream_table_output, index=False)
    args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
