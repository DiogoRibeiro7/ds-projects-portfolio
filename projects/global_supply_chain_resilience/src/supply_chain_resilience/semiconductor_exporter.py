"""Frozen exporter-scale and importer-side supplier diagnostics for HS 8542."""

from __future__ import annotations

from math import isfinite
from typing import Any

import pandas as pd

DEPENDENCY_THRESHOLDS = (0.10, 0.25, 0.50)


def world_export_value(
    rows: list[dict[str, Any]], *, reporter_code: int, commodity_heading: str
) -> float | None:
    """Return one reporter's exporter-reported World export value, if present and positive."""
    matches: list[float] = []
    for row in rows:
        if int(row.get("reporterCode", -1)) != reporter_code:
            raise ValueError("export response contains an unexpected reporter.")
        if str(row.get("flowCode")) != "X":
            raise ValueError("export response contains a non-export flow.")
        if str(row.get("cmdCode")) != commodity_heading:
            raise ValueError("export response contains an unexpected commodity code.")
        if int(row.get("partnerCode", -1)) != 0:
            raise ValueError("export response contains a non-World partner row.")
        value = float(row.get("primaryValue", float("nan")))
        if not isfinite(value) or value < 0.0:
            raise ValueError("World-export primaryValue must be finite and non-negative.")
        matches.append(value)
    if len(matches) > 1:
        raise ValueError(f"duplicate World-export rows for reporter {reporter_code}.")
    if not matches or matches[0] == 0.0:
        return None
    return matches[0]


def supplier_dependency_diagnostics(
    metrics: pd.DataFrame,
    links: pd.DataFrame,
    *,
    expected_material_importers: int = 84,
    residual_partner_code: int = 490,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Compute frozen importer-side supplier diagnostics over material importers."""
    required_metric = {"reporter_code", "material_importer"}
    required_link = {
        "reporter_code",
        "partner_code",
        "partner_desc",
        "partner_iso",
        "trade_value",
        "partner_share",
        "is_named_country",
        "material_importer",
    }
    if missing := required_metric.difference(metrics.columns):
        raise ValueError(f"importer metrics missing columns: {sorted(missing)}")
    if missing := required_link.difference(links.columns):
        raise ValueError(f"bilateral links missing columns: {sorted(missing)}")

    material_codes = set(
        metrics.loc[metrics["material_importer"].astype(bool), "reporter_code"].astype(int)
    )
    if len(material_codes) != expected_material_importers:
        raise ValueError(
            f"expected exactly {expected_material_importers} material importers; "
            f"got {len(material_codes)}."
        )

    material_links = links.loc[
        links["material_importer"].astype(bool)
        & links["reporter_code"].astype(int).isin(material_codes)
    ].copy()
    named = material_links.loc[material_links["is_named_country"].astype(bool)].copy()
    if named.empty:
        raise ValueError("authoritative importer artifact contains no named-country links.")

    named = named.sort_values(
        ["reporter_code", "partner_share", "trade_value", "partner_code"],
        ascending=[True, False, False, True],
    )
    largest = named.groupby("reporter_code", as_index=False).first()

    diagnostics = (
        named.groupby(["partner_code", "partner_desc", "partner_iso"], as_index=False)
        .agg(
            total_importer_reported_value=("trade_value", "sum"),
            material_importers_positive=("reporter_code", "nunique"),
        )
    )
    largest_counts = largest.groupby("partner_code").size().rename("largest_named_supplier_count")
    diagnostics = diagnostics.merge(largest_counts, on="partner_code", how="left")
    diagnostics["largest_named_supplier_count"] = (
        diagnostics["largest_named_supplier_count"].fillna(0).astype(int)
    )
    for threshold in DEPENDENCY_THRESHOLDS:
        column = f"material_importer_count_ge_{int(threshold * 100)}pct"
        counts = (
            named.loc[named["partner_share"] >= threshold]
            .groupby("partner_code")["reporter_code"]
            .nunique()
            .rename(column)
        )
        diagnostics = diagnostics.merge(counts, on="partner_code", how="left")
        diagnostics[column] = diagnostics[column].fillna(0).astype(int)

    residual = material_links.loc[
        material_links["partner_code"].astype(int) == residual_partner_code
    ].copy()
    residual_diag: dict[str, object] = {
        "partner_code": residual_partner_code,
        "partner_desc": None if residual.empty else str(residual.iloc[0]["partner_desc"]),
        "total_importer_reported_value": float(residual["trade_value"].sum()),
        "material_importers_positive": int(residual["reporter_code"].nunique()),
    }
    for threshold in DEPENDENCY_THRESHOLDS:
        residual_diag[f"material_importer_count_ge_{int(threshold * 100)}pct"] = int(
            residual.loc[residual["partner_share"] >= threshold, "reporter_code"].nunique()
        )

    return diagnostics, residual_diag
