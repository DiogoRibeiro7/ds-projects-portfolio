"""Run frozen 2022 HS 8542 exporter and supplier-dependency diagnostics."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from supply_chain_resilience.comtrade import extract_data_rows, get_official_json
from supply_chain_resilience.semiconductor_exporter import (
    supplier_dependency_diagnostics,
    world_export_value,
)

PREVIEW_ENDPOINT = "https://comtradeapi.un.org/public/v1/preview/C/A/HS"
REFERENCE_YEAR = 2022
HS_HEADING = "8542"
PUBLIC_PREVIEW_MIN_INTERVAL_SECONDS = 1.05
EXPECTED_PRIMARY_REPORTERS = 167


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reporter-lock", type=Path, required=True)
    parser.add_argument("--importer-metrics", type=Path, required=True)
    parser.add_argument("--importer-links", type=Path, required=True)
    parser.add_argument("--exporters-output", type=Path, required=True)
    parser.add_argument("--suppliers-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()

    lock = json.loads(args.reporter_lock.read_text(encoding="utf-8"))
    source = lock["source_reporters"]
    reporters = [r for r in source if int(r["reporter_code"]) not in {97, 975}]
    if len(reporters) != EXPECTED_PRIMARY_REPORTERS:
        raise ValueError(f"expected {EXPECTED_PRIMARY_REPORTERS} primary reporters; got {len(reporters)}")

    metrics = pd.read_csv(args.importer_metrics)
    links = pd.read_csv(args.importer_links)
    suppliers, residual = supplier_dependency_diagnostics(metrics, links)

    exporter_rows: list[dict[str, object]] = []
    evidence: list[dict[str, object]] = []
    previous_started: float | None = None
    for reporter in reporters:
        if previous_started is not None:
            elapsed = time.monotonic() - previous_started
            if elapsed < PUBLIC_PREVIEW_MIN_INTERVAL_SECONDS:
                time.sleep(PUBLIC_PREVIEW_MIN_INTERVAL_SECONDS - elapsed)
        previous_started = time.monotonic()
        code = int(reporter["reporter_code"])
        response = get_official_json(
            PREVIEW_ENDPOINT,
            {
                "period": REFERENCE_YEAR,
                "reportercode": code,
                "cmdCode": HS_HEADING,
                "flowCode": "X",
                "partnerCode": 0,
                "partner2Code": 0,
                "customsCode": "C00",
                "motCode": 0,
                "maxRecords": 500,
                "breakdownMode": "classic",
                "includeDesc": "true",
            },
        )
        rows = extract_data_rows(response.payload)
        value = world_export_value(rows, reporter_code=code, commodity_heading=HS_HEADING)
        evidence.append({
            "reporter_code": code,
            "retrieved_at_utc": response.retrieved_at_utc,
            "canonical_sha256": response.canonical_sha256,
            "row_count": len(rows),
        })
        if value is not None:
            exporter_rows.append({
                "reporter_code": code,
                "reporter_desc": rows[0].get("reporterDesc") if rows else None,
                "reporter_iso": rows[0].get("reporterISO") if rows else None,
                "world_export_value": value,
            })

    exporters = pd.DataFrame(exporter_rows)
    total_exports = float(exporters["world_export_value"].sum())
    exporters["share_of_positive_primary_reporter_world_exports"] = exporters["world_export_value"] / total_exports
    exporters = exporters.sort_values(["world_export_value", "reporter_code"], ascending=[False, True]).reset_index(drop=True)
    exporters["exporter_rank_by_world_exports"] = range(1, len(exporters) + 1)

    suppliers = suppliers.sort_values(
        ["total_importer_reported_value", "partner_code"], ascending=[False, True]
    ).reset_index(drop=True)

    summary = {
        "reference_year": REFERENCE_YEAR,
        "commodity_heading": HS_HEADING,
        "primary_reporter_count": len(reporters),
        "positive_exporter_count": len(exporters),
        "positive_primary_reporter_world_exports_total": total_exports,
        "material_importer_count": int(metrics["material_importer"].astype(bool).sum()),
        "named_supplier_count": len(suppliers),
        "other_asia_nes": residual,
        "importer_evidence_workflow_run": 33417837534,
        "importer_evidence_artifact_id": 9767995550,
        "importer_evidence_artifact_digest": "sha256:0445610ba5a9d4df2d617cac543d1a14d9c4b1e5c5d295665541a9eadc99f41c",
        "query_count": len(evidence),
        "query_evidence": evidence,
        "scientific_boundary": "Exporter-reported World exports and importer-reported bilateral imports remain distinct commercial observations; neither identifies fabrication origin or technological dependence.",
    }

    for path in (args.exporters_output, args.suppliers_output, args.summary_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    exporters.to_csv(args.exporters_output, index=False)
    suppliers.to_csv(args.suppliers_output, index=False)
    args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "query_evidence"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
