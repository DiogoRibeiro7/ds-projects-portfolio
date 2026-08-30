"""Accounting-based structural dependency metrics for ICIO production systems."""

from __future__ import annotations

import numpy as np
import pandas as pd

from supply_chain_resilience.mapping import ICIOBlocks


def split_country_activity(label: str) -> tuple[str, str]:
    """Split a canonical ``COUNTRY_ACTIVITY`` ICIO label once."""
    if "_" not in label:
        raise ValueError(f"ICIO industry label lacks country/activity separator: {label!r}.")
    country, activity = label.split("_", maxsplit=1)
    if not country or not activity:
        raise ValueError(f"Malformed ICIO industry label: {label!r}.")
    return country, activity


def structural_dependency_metrics(blocks: ICIOBlocks) -> pd.DataFrame:
    """Compute direct supplier-country dependency metrics for every using industry.

    Metrics are based on observed intermediate-use flows ``Z`` rather than graph
    centrality. Supplier concentration is calculated at country level, so multiple
    supplying activities from the same economy are aggregated before shares/HHI.
    Ratios are left as NaN when the relevant denominator is zero.
    """
    z = blocks.intermediate_use
    x = blocks.gross_output
    if z.empty:
        raise ValueError("intermediate-use block must not be empty.")
    if not z.columns.equals(x.index):
        raise ValueError("gross-output index must match intermediate-use columns.")
    if not z.index.equals(z.columns):
        raise ValueError("structural dependency metrics require a square labelled production block.")

    supplier_countries = pd.Series(
        [split_country_activity(str(label))[0] for label in z.index],
        index=z.index,
        dtype="object",
    )
    grouped = z.groupby(supplier_countries, axis=0, sort=True).sum()

    records: list[dict[str, float | str]] = []
    for label in z.columns:
        label_str = str(label)
        user_country, activity = split_country_activity(label_str)
        by_country = grouped[label].astype(float)
        intermediate_input = float(by_country.sum())
        domestic_input = float(by_country.get(user_country, 0.0))
        foreign_input = intermediate_input - domestic_input

        if intermediate_input > 0.0:
            country_shares = by_country / intermediate_input
            supplier_country_hhi = float(np.square(country_shares).sum())
            effective_supplier_countries = 1.0 / supplier_country_hhi
            largest_supplier_country_share = float(country_shares.max())
            foreign_input_dependence = foreign_input / intermediate_input
            domestic_input_share = domestic_input / intermediate_input
        else:
            supplier_country_hhi = np.nan
            effective_supplier_countries = np.nan
            largest_supplier_country_share = np.nan
            foreign_input_dependence = np.nan
            domestic_input_share = np.nan

        if foreign_input > 0.0:
            foreign_by_country = by_country.drop(labels=[user_country], errors="ignore")
            foreign_shares = foreign_by_country / foreign_input
            foreign_supplier_country_hhi = float(np.square(foreign_shares).sum())
            foreign_effective_supplier_countries = 1.0 / foreign_supplier_country_hhi
            largest_foreign_supplier_country_share = float(foreign_shares.max())
        else:
            foreign_supplier_country_hhi = np.nan
            foreign_effective_supplier_countries = np.nan
            largest_foreign_supplier_country_share = np.nan

        records.append(
            {
                "node": label_str,
                "country": user_country,
                "activity": activity,
                "gross_output": float(x.loc[label]),
                "intermediate_input": intermediate_input,
                "domestic_input": domestic_input,
                "foreign_input": foreign_input,
                "foreign_input_dependence": foreign_input_dependence,
                "domestic_input_share": domestic_input_share,
                "supplier_country_hhi": supplier_country_hhi,
                "effective_supplier_countries": effective_supplier_countries,
                "largest_supplier_country_share": largest_supplier_country_share,
                "foreign_supplier_country_hhi": foreign_supplier_country_hhi,
                "foreign_effective_supplier_countries": foreign_effective_supplier_countries,
                "largest_foreign_supplier_country_share": largest_foreign_supplier_country_share,
            }
        )

    return pd.DataFrame.from_records(records).set_index("node")


def cross_border_intermediate_share(blocks: ICIOBlocks) -> float:
    """Return the share of observed intermediate use crossing country borders."""
    z = blocks.intermediate_use
    supplier_country = np.array([split_country_activity(str(v))[0] for v in z.index])
    user_country = np.array([split_country_activity(str(v))[0] for v in z.columns])
    values = z.to_numpy(dtype=float)
    total = float(values.sum())
    if total <= 0.0:
        return float("nan")
    foreign_mask = supplier_country[:, None] != user_country[None, :]
    return float(values[foreign_mask].sum() / total)
