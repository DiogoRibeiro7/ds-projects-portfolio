"""Run the frozen top-50 HS 8542 bilateral mirror-data audit."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from supply_chain_resilience.comtrade import extract_data_rows, get_official_json
from supply_chain_resilience.semiconductor_mirror import exporter_mirror_value, mirror_diagnostics

PREVIEW_ENDPOINT = "https://comtradeapi.un.org/public/v1/preview/C/A/HS"
REFERENCE_YEAR = 2022
HS_HEADING = "8542"
TOP_N = 50
PUBLIC_PREVIEW_MIN_INTERVAL_SECONDS = 1.05


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--importer-links", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()

    links = pd.read_csv(args.importer_links)
    eligible = links.loc[links["material_importer"].astype(bool) & links["is_named_country"].astype(bool)].copy()
    frozen = eligible.sort_values(
        ["trade_value", "partner_share", "reporter_code", "partner_code"],
        ascending=[False, False, True, True],
    ).head(TOP_N).reset_index(drop=True)
    if len(frozen) != TOP_N:
        raise ValueError(f"expected exactly {TOP_N} frozen top-value named links; got {len(frozen)}")

    audited: list[dict[str, object]] = []
    evidence: list[dict[str, object]] = []
    previous_started: float | None = None
    for rank, row in frozen.iterrows():
        if previous_started is not None:
            elapsed = time.monotonic() - previous_started
            if elapsed < PUBLIC_PREVIEW_MIN_INTERVAL_SECONDS:
                time.sleep(PUBLIC_PREVIEW_MIN_INTERVAL_SECONDS - elapsed)
        previous_started = time.monotonic()
        importer = int(row["reporter_code"])
        exporter = int(row["partner_code"])
        response = get_official_json(
            PREVIEW_ENDPOINT,
            {
                "period": REFERENCE_YEAR,
                "reportercode": exporter,
                "cmdCode": HS_HEADING,
                "flowCode": "X",
                "partnerCode": importer,
                "partner2Code": 0,
                "customsCode": "C00",
                "motCode": 0,
                "maxRecords": 500,
                "breakdownMode": "classic",
                "includeDesc": "true",
            },
        )
        rows = extract_data_rows(response.payload)
        if len(rows) >= 500:
            raise RuntimeError(f"mirror query exporter={exporter}, importer={importer} reached preview cap")
        export_value = exporter_mirror_value(
            rows,
            exporter_code=exporter,
            importer_code=importer,
            commodity_heading=HS_HEADING,
        )
        diagnostics = mirror_diagnostics(float(row["trade_value"]), export_value)
        audited.append(
            {
                "frozen_value_rank": rank + 1,
                "importer_code": importer,
                "importer_desc": row["reporter_desc"],
                "importer_iso": row["reporter_iso"],
                "exporter_code": exporter,
                "exporter_desc": row["partner_desc"],
                "exporter_iso": row["partner_iso"],
                "importer_reported_value": float(row["trade_value"]),
                "importer_partner_share": float(row["partner_share"]),
                **diagnostics,
            }
        )
        evidence.append(
            {
                "frozen_value_rank": rank + 1,
                "importer_code": importer,
                "exporter_code": exporter,
                "retrieved_at_utc": response.retrieved_at_utc,
                "canonical_sha256": response.canonical_sha256,
                "row_count": len(rows),
            }
        )

    result = pd.DataFrame(audited)
    observed = result.loc[result["mirror_observed"].astype(bool)].copy()
    summary = {
        "reference_year": REFERENCE_YEAR,
        "commodity_heading": HS_HEADING,
        "frozen_link_count": len(result),
        "observed_mirror_count": len(observed),
        "missing_mirror_count": len(result) - len(observed),
        "median_relative_difference_observed": (
            float(observed["relative_difference_max_denominator"].median()) if len(observed) else None
        ),
        "p90_relative_difference_observed": (
            float(observed["relative_difference_max_denominator"].quantile(0.9)) if len(observed) else None
        ),
        "max_relative_difference_observed": (
            float(observed["relative_difference_max_denominator"].max()) if len(observed) else None
        ),
        "authoritative_importer_workflow_run": 33417837534,
        "authoritative_importer_artifact_id": 9767995550,
        "authoritative_importer_artifact_digest": "sha256:0445610ba5a9d4df2d617cac543d1a14d9c4b1e5c5d295665541a9eadc99f41c",
        "query_count": len(evidence),
        "query_evidence": evidence,
        "scientific_boundary": "Mirror asymmetry is a reporting, valuation and timing diagnostic. Importer-reported primary values are never overwritten, averaged or reconciled with exporter mirrors.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "query_evidence"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
