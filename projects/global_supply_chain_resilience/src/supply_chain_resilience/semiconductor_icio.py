"""Descriptive HS 8542 versus OECD ICIO C26 rank-comparison utilities."""

from __future__ import annotations

from math import isfinite

import pandas as pd
from scipy.stats import spearmanr


def _require_columns(frame: pd.DataFrame, columns: set[str], *, context: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{context} is missing required columns: {sorted(missing)}")


def _bool_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false"}).all():
        raise ValueError("boolean evidence column contains values other than true/false.")
    return normalized.eq("true")


def trade_downstream_share_mass(links: pd.DataFrame) -> pd.DataFrame:
    """Aggregate named-supplier partner shares over the frozen material-importer set."""
    _require_columns(
        links,
        {
            "partner_code",
            "partner_desc",
            "partner_iso",
            "partner_share",
            "is_named_country",
            "material_importer",
        },
        context="importer links",
    )
    eligible = links.loc[
        _bool_mask(links["material_importer"]) & _bool_mask(links["is_named_country"])
    ].copy()
    if eligible.empty:
        raise ValueError("no named links remain for material importers.")
    if not eligible["partner_share"].map(
        lambda value: isfinite(float(value)) and float(value) > 0.0
    ).all():
        raise ValueError("partner shares must be finite and strictly positive.")

    grouped = (
        eligible.groupby(["partner_iso", "partner_code", "partner_desc"], as_index=False)[
            "partner_share"
        ]
        .sum()
        .rename(columns={"partner_share": "trade_downstream_share_mass"})
    )
    if grouped["partner_iso"].duplicated().any():
        raise ValueError("one ISO alpha-3 code maps to multiple named supplier identities.")
    return grouped


def c26_supplier_frame(icio_suppliers: pd.DataFrame) -> pd.DataFrame:
    """Return the unique OECD ICIO C26 supplier row for each country."""
    _require_columns(
        icio_suppliers,
        {
            "country",
            "activity",
            "foreign_intermediate_sales",
            "foreign_downstream_input_share_mass",
        },
        context="ICIO supplier evidence",
    )
    c26 = icio_suppliers.loc[icio_suppliers["activity"].astype(str) == "C26"].copy()
    if c26.empty:
        raise ValueError("ICIO supplier evidence contains no C26 rows.")
    if c26["country"].duplicated().any():
        raise ValueError("ICIO C26 evidence contains duplicate country rows.")
    for column in ("foreign_intermediate_sales", "foreign_downstream_input_share_mass"):
        values = c26[column].astype(float)
        if not values.map(lambda value: isfinite(value) and value >= 0.0).all():
            raise ValueError(f"ICIO C26 {column} must be finite and non-negative.")
    return c26


def compare_ranked_measures(
    trade: pd.DataFrame,
    icio: pd.DataFrame,
    *,
    trade_code_column: str,
    trade_label_column: str,
    trade_value_column: str,
    icio_value_column: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Compare exact-matched country ranks under the preregistered descriptive contract."""
    _require_columns(
        trade,
        {trade_code_column, trade_label_column, trade_value_column},
        context="trade comparison input",
    )
    _require_columns(icio, {"country", icio_value_column}, context="ICIO comparison input")

    trade_part = trade[
        [trade_code_column, trade_label_column, trade_value_column]
    ].rename(
        columns={
            trade_code_column: "country",
            trade_label_column: "trade_label",
            trade_value_column: "trade_value",
        }
    )
    icio_part = icio[["country", icio_value_column]].rename(
        columns={icio_value_column: "icio_value"}
    )

    if trade_part["country"].duplicated().any():
        raise ValueError("trade comparison input contains duplicate country codes.")
    if icio_part["country"].duplicated().any():
        raise ValueError("ICIO comparison input contains duplicate country codes.")

    matched = trade_part.merge(icio_part, on="country", how="inner", validate="one_to_one")
    if len(matched) < 2:
        raise ValueError("rank comparison requires at least two exact-matched countries.")

    trade_values = matched["trade_value"].astype(float)
    icio_values = matched["icio_value"].astype(float)
    if not trade_values.map(lambda value: isfinite(value) and value > 0.0).all():
        raise ValueError("matched trade values must be finite and strictly positive.")
    if not icio_values.map(lambda value: isfinite(value) and value >= 0.0).all():
        raise ValueError("matched ICIO values must be finite and non-negative.")

    trade_order = matched.sort_values(
        ["trade_value", "country"], ascending=[False, True]
    ).reset_index(drop=True)
    trade_order["trade_rank"] = range(1, len(trade_order) + 1)
    icio_order = matched.sort_values(
        ["icio_value", "country"], ascending=[False, True]
    ).reset_index(drop=True)
    icio_order["icio_rank"] = range(1, len(icio_order) + 1)

    matched = matched.merge(
        trade_order[["country", "trade_rank"]], on="country", validate="one_to_one"
    ).merge(icio_order[["country", "icio_rank"]], on="country", validate="one_to_one")
    matched["rank_difference"] = matched["trade_rank"] - matched["icio_rank"]
    matched["absolute_rank_difference"] = matched["rank_difference"].abs()

    trade_sum = float(matched["trade_value"].sum())
    icio_sum = float(matched["icio_value"].sum())
    matched["trade_matched_share"] = matched["trade_value"] / trade_sum
    matched["icio_matched_share"] = (
        matched["icio_value"] / icio_sum if icio_sum > 0.0 else float("nan")
    )

    correlation = spearmanr(
        matched["trade_value"].to_numpy(dtype=float),
        matched["icio_value"].to_numpy(dtype=float),
    )
    top_n = 10 if len(matched) >= 10 else len(matched)
    trade_top = set(matched.nsmallest(top_n, "trade_rank")["country"])
    icio_top = set(matched.nsmallest(top_n, "icio_rank")["country"])
    union = trade_top | icio_top
    overlap = trade_top & icio_top

    outliers = matched.sort_values(
        ["absolute_rank_difference", "country"], ascending=[False, True]
    ).head(10)

    summary: dict[str, object] = {
        "matched_country_count": int(len(matched)),
        "spearman_rho": float(correlation.statistic),
        "spearman_p_value": float(correlation.pvalue),
        "top_n": top_n,
        "top_overlap_count": int(len(overlap)),
        "top_jaccard": float(len(overlap) / len(union)) if union else None,
        "top_overlap_countries": sorted(overlap),
        "largest_absolute_rank_difference": int(matched["absolute_rank_difference"].max()),
        "largest_rank_discrepancy_outliers": outliers[
            [
                "country",
                "trade_label",
                "trade_value",
                "icio_value",
                "trade_rank",
                "icio_rank",
                "rank_difference",
                "absolute_rank_difference",
            ]
        ].to_dict(orient="records"),
    }

    return matched.sort_values(["trade_rank", "country"]).reset_index(drop=True), summary
