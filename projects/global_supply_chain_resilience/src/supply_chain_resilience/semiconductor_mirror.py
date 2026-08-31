"""Mirror-data diagnostics for frozen HS 8542 bilateral links."""

from __future__ import annotations

from math import isfinite, log
from typing import Any


def exporter_mirror_value(
    rows: list[dict[str, Any]], *, exporter_code: int, importer_code: int, commodity_heading: str
) -> float | None:
    """Return exporter-reported value for one exact exporter-importer pair."""
    matches: list[float] = []
    for row in rows:
        if int(row.get("reporterCode", -1)) != exporter_code:
            raise ValueError("mirror response contains an unexpected exporter reporter.")
        if int(row.get("partnerCode", -1)) != importer_code:
            raise ValueError("mirror response contains an unexpected importer partner.")
        if str(row.get("flowCode")) != "X":
            raise ValueError("mirror response contains a non-export flow.")
        if str(row.get("cmdCode")) != commodity_heading:
            raise ValueError("mirror response contains an unexpected commodity code.")
        value = float(row.get("primaryValue", float("nan")))
        if not isfinite(value) or value < 0.0:
            raise ValueError("mirror export primaryValue must be finite and non-negative.")
        matches.append(value)
    if len(matches) > 1:
        raise ValueError("duplicate exporter mirror rows for one frozen bilateral link.")
    return matches[0] if matches else None


def mirror_diagnostics(import_value: float, export_value: float | None) -> dict[str, float | bool | None]:
    """Compute frozen mirror-asymmetry metrics without reconciling either observation."""
    if not isfinite(import_value) or import_value <= 0.0:
        raise ValueError("frozen importer value must be finite and strictly positive.")
    if export_value is None:
        return {
            "mirror_observed": False,
            "exporter_reported_value": None,
            "absolute_difference": None,
            "relative_difference_max_denominator": None,
            "signed_log_ratio_export_over_import": None,
        }
    if not isfinite(export_value) or export_value < 0.0:
        raise ValueError("exporter mirror value must be finite and non-negative.")
    absolute = abs(export_value - import_value)
    denom = max(import_value, export_value)
    relative = absolute / denom if denom > 0 else None
    signed_log = log(export_value / import_value) if export_value > 0.0 else None
    return {
        "mirror_observed": True,
        "exporter_reported_value": export_value,
        "absolute_difference": absolute,
        "relative_difference_max_denominator": relative,
        "signed_log_ratio_export_over_import": signed_log,
    }
