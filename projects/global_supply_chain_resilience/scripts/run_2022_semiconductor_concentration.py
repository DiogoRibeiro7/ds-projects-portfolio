"""Run the preregistered 2022 HS 8542 importer-concentration analysis."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from supply_chain_resilience.comtrade import extract_data_rows, get_official_json
from supply_chain_resilience.semiconductor import importer_concentration_metrics

DATA_AVAILABILITY_ENDPOINT = "https://comtradeapi.un.org/public/v1/getDA/C/A/HS"
PREVIEW_ENDPOINT = "https://comtradeapi.un.org/public/v1/preview/C/A/HS"
REFERENCE_YEAR = 2022
HS_HEADING = "8542"
MAX_PREVIEW_RECORDS = 500
PUBLIC_PREVIEW_MIN_INTERVAL_SECONDS = 1.05
AGGREGATE_REPORTER_CODES = frozenset({97, 975})
EXPECTED_PRIMARY_REPORTERS = 167


def primary_reporters_from_manifest(manifest: dict[str, object]) -> list[dict[str, object]]:
    """Return the prospectively corrected individual country/area reporter universe."""
    frozen = manifest["reporters"]
    if not isinstance(frozen, list):
        raise ValueError("reporter manifest reporters must be a list.")
    source_codes = {int(row["reporter_code"]) for row in frozen if isinstance(row, dict)}
    missing = AGGREGATE_REPORTER_CODES.difference(source_codes)
    if missing:
        raise ValueError(f"source manifest is missing expected aggregate reporters: {sorted(missing)}")
    reporters = [
        row
        for row in frozen
        if isinstance(row, dict) and int(row["reporter_code"]) not in AGGREGATE_REPORTER_CODES
    ]
    if len(reporters) != EXPECTED_PRIMARY_REPORTERS:
        raise ValueError(
            f"expected exactly {EXPECTED_PRIMARY_REPORTERS} primary reporters after aggregate "
            f"exclusion; got {len(reporters)}."
        )
    return reporters


def _verify_live_dataset_contract(reporters: list[dict[str, object]]) -> None:
    response = get_official_json(DATA_AVAILABILITY_ENDPOINT, {"period": REFERENCE_YEAR})
    rows = extract_data_rows(response.payload)
    live = {int(row["reporterCode"]): row for row in rows}
    for reporter in reporters:
        code = int(reporter["reporter_code"])
        if code not in live:
            raise RuntimeError(f"frozen reporter {code} is absent from current 2022 data availability.")
        row = live[code]
        if int(row["datasetChecksum"]) != int(reporter["dataset_checksum"]):
            raise RuntimeError(
                f"UN Comtrade revised reporter {code} since the reporter-universe freeze; "
                "rerun the prospective provenance gate before substantive analysis."
            )
        if str(row["classificationCode"]) != str(reporter["classification_code"]):
            raise RuntimeError(f"classification changed for frozen reporter {code}.")


def _fetch_reporter_panel(
    reporter_code: int, previous_started: float | None
) -> tuple[list[dict[str, object]], float, dict[str, object]]:
    if previous_started is not None:
        elapsed = time.monotonic() - previous_started
        if elapsed < PUBLIC_PREVIEW_MIN_INTERVAL_SECONDS:
            time.sleep(PUBLIC_PREVIEW_MIN_INTERVAL_SECONDS - elapsed)
    started = time.monotonic()
    response = get_official_json(
        PREVIEW_ENDPOINT,
        {
            "period": REFERENCE_YEAR,
            "reportercode": reporter_code,
            "cmdCode": HS_HEADING,
            "flowCode": "M",
            "partner2Code": "0",
            "customsCode": "C00",
            "motCode": "0",
            "maxRecords": MAX_PREVIEW_RECORDS,
            "breakdownMode": "classic",
            "includeDesc": "true",
        },
    )
    rows = extract_data_rows(response.payload)
    if len(rows) >= MAX_PREVIEW_RECORDS:
        raise RuntimeError(
            f"reporter {reporter_code} reached the 500-row public preview cap; "
            "bilateral panel completeness is not guaranteed."
        )
    evidence = {
        "reporter_code": reporter_code,
        "retrieved_at_utc": response.retrieved_at_utc,
        "canonical_sha256": response.canonical_sha256,
        "row_count": len(rows),
    }
    return rows, started, evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--links-output", type=Path, required=True)
    parser.add_argument("--rankings-output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if int(manifest["reference_year"]) != REFERENCE_YEAR:
        raise ValueError("reporter manifest year does not match executable lock.")
    if str(manifest["commodity_heading"]) != HS_HEADING:
        raise ValueError("reporter manifest commodity does not match executable lock.")
    source_reporters = manifest["reporters"]
    if not isinstance(source_reporters, list):
        raise ValueError("reporter manifest reporters must be a list.")
    reporters = primary_reporters_from_manifest(manifest)

    _verify_live_dataset_contract(reporters)

    metric_rows: list[dict[str, object]] = []
    link_rows: list[dict[str, object]] = []
    query_evidence: list[dict[str, object]] = []
    previous_started: float | None = None

    for reporter in reporters:
        code = int(reporter["reporter_code"])
        rows, previous_started, evidence = _fetch_reporter_panel(code, previous_started)
        metrics, bilateral = importer_concentration_metrics(
            rows, reporter_code=code, commodity_heading=HS_HEADING
        )
        metrics.update(
            {
                "reporter_desc": str(reporter["reporter_desc"]),
                "reporter_iso": str(reporter["reporter_iso"]),
                "is_special_reporter": bool(reporter["is_special_reporter"]),
            }
        )
        metric_rows.append(metrics)
        for row in bilateral:
            link_rows.append({"reporter_code": code, **row})
        query_evidence.append(evidence)

    metrics = pd.DataFrame(metric_rows)
    links = pd.DataFrame(link_rows)
    materiality_threshold = float(metrics["total_import_value"].median())
    metrics["material_importer"] = metrics["total_import_value"] >= materiality_threshold
    material = metrics.loc[metrics["material_importer"]].copy()

    ranked_import_value = material.sort_values(
        ["total_import_value", "reporter_code"], ascending=[False, True]
    )
    ranked_hhi = material.sort_values(
        ["partner_hhi_all_reported", "reporter_code"], ascending=[False, True]
    )

    links = links.merge(
        metrics[["reporter_code", "reporter_desc", "reporter_iso", "material_importer"]],
        on="reporter_code",
        how="left",
        validate="many_to_one",
    )
    material_links = links.loc[links["material_importer"]].copy()
    named_links = material_links.loc[material_links["is_named_country"]].copy()
    ranked_share = named_links.sort_values(
        ["partner_share", "trade_value", "reporter_code", "partner_code"],
        ascending=[False, False, True, True],
    )
    ranked_value = named_links.sort_values(
        ["trade_value", "partner_share", "reporter_code", "partner_code"],
        ascending=[False, False, True, True],
    )

    rankings = {
        "largest_material_importers": ranked_import_value.head(25).to_dict(orient="records"),
        "highest_material_importer_hhi": ranked_hhi.head(25).to_dict(orient="records"),
        "largest_named_bilateral_dependency_share": ranked_share.head(50).to_dict(orient="records"),
        "largest_named_bilateral_dependency_value": ranked_value.head(50).to_dict(orient="records"),
    }

    summary = {
        "reference_year": REFERENCE_YEAR,
        "commodity_heading": HS_HEADING,
        "source_reporter_manifest_count": len(source_reporters),
        "excluded_aggregate_reporter_codes": sorted(AGGREGATE_REPORTER_CODES),
        "frozen_reporter_count": len(reporters),
        "special_reporter_count": int(metrics["is_special_reporter"].sum()),
        "materiality_threshold": materiality_threshold,
        "material_importer_count": int(metrics["material_importer"].sum()),
        "bilateral_positive_link_count": int(len(links)),
        "named_country_positive_link_count": int(links["is_named_country"].sum()),
        "median_hhi_material": float(material["partner_hhi_all_reported"].median()),
        "median_largest_partner_share_material": float(material["largest_partner_share"].median()),
        "median_top3_partner_share_material": float(material["top3_partner_share"].median()),
        "max_world_reconciliation_abs_deviation": float(
            np.nanmax(np.abs(metrics["world_reconciliation_ratio"].to_numpy(dtype=float) - 1.0))
        ),
        "source_reporter_manifest_run": manifest["source_workflow_run"],
        "source_reporter_manifest_artifact_digest": manifest["source_artifact_digest"],
        "query_count": len(query_evidence),
        "query_evidence": query_evidence,
        "scientific_boundary": (
            "Importer-reported HS 8542 trade values measure commercial bilateral concentration, "
            "not technological dependence or fabrication origin. European Union (97) and ASEAN "
            "(975) are excluded from the primary reporter universe because they overlap member "
            "reporters; the exclusion was prospectively registered in PR #593 before any bilateral "
            "concentration result was produced."
        ),
    }

    for output in (args.summary_output, args.metrics_output, args.links_output, args.rankings_output):
        output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    metrics.to_csv(args.metrics_output, index=False)
    links.to_csv(args.links_output, index=False)
    args.rankings_output.write_text(json.dumps(rankings, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "query_evidence"},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
