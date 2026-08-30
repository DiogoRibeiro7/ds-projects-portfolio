"""Supplier-side importance metrics for ICIO production systems."""

from __future__ import annotations

import numpy as np
import pandas as pd

from supply_chain_resilience.dependency import split_country_activity
from supply_chain_resilience.icio import technical_coefficients
from supply_chain_resilience.mapping import ICIOBlocks


def supplier_importance_metrics(blocks: ICIOBlocks) -> pd.DataFrame:
    """Compute direct downstream importance metrics for each supplying industry."""
    z = blocks.intermediate_use
    x = blocks.gross_output
    if not z.index.equals(z.columns) or not z.columns.equals(x.index):
        raise ValueError("supplier metrics require a square labelled production block.")

    a = technical_coefficients(z, x)
    buyer_countries = np.array([split_country_activity(str(v))[0] for v in z.columns])
    records: list[dict[str, float | int | str]] = []

    for label in z.index:
        label_str = str(label)
        country, activity = split_country_activity(label_str)
        flows = z.loc[label].to_numpy(dtype=float)
        shares = a.loc[label].to_numpy(dtype=float)
        foreign = buyer_countries != country
        total_sales = float(flows.sum())
        foreign_sales = float(flows[foreign].sum())
        positive_foreign = shares[foreign & (shares > 0.0)]

        records.append(
            {
                "node": label_str,
                "country": country,
                "activity": activity,
                "gross_output": float(x.loc[label]),
                "intermediate_sales": total_sales,
                "foreign_intermediate_sales": foreign_sales,
                "foreign_sales_share": foreign_sales / total_sales if total_sales > 0 else np.nan,
                "downstream_input_share_mass": float(shares.sum()),
                "foreign_downstream_input_share_mass": float(shares[foreign].sum()),
                "max_foreign_buyer_input_share": (
                    float(positive_foreign.max()) if positive_foreign.size else np.nan
                ),
                "foreign_buyers_above_0_1pct": int((shares[foreign] > 0.001).sum()),
                "foreign_buyers_above_0_5pct": int((shares[foreign] > 0.005).sum()),
                "foreign_buyers_above_1pct": int((shares[foreign] > 0.01).sum()),
            }
        )

    return pd.DataFrame.from_records(records).set_index("node")
