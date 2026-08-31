"""Constrained sourcing-diversification optimization utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from supply_chain_resilience.dependency import split_country_activity


@dataclass(frozen=True)
class DiversificationResult:
    feasible: bool
    allocation: pd.Series | None
    reallocation_burden: float | None
    achieved_worst_case_direct_risk: float | None
    solver_status: int
    solver_message: str


def baseline_direct_risk(
    observed: pd.Series,
    shock_nodes: tuple[str, ...],
    *,
    shock_fraction: float,
) -> float:
    """Worst-case direct loss share across the frozen supplier shocks."""
    total = float(observed.sum())
    if total <= 0.0:
        return float("nan")
    return float(
        max(shock_fraction * float(observed.get(node, 0.0)) / total for node in shock_nodes)
    )


def rank_exposed_buyers(exposed: pd.DataFrame, *, limit: int) -> pd.DataFrame:
    """Apply the frozen buyer ranking without colliding with a ``node`` index name."""
    required = {"baseline_worst_case_direct_risk", "intermediate_input"}
    missing = required.difference(exposed.columns)
    if missing:
        raise ValueError(f"buyer ranking is missing required columns: {sorted(missing)}")
    if limit <= 0:
        raise ValueError("limit must be positive.")

    ranked = exposed.copy()
    ranked["_node_label"] = ranked.index.astype(str)
    ranked = ranked.sort_values(
        ["baseline_worst_case_direct_risk", "intermediate_input", "_node_label"],
        ascending=[False, False, True],
        kind="stable",
    ).drop(columns="_node_label")
    return ranked.head(limit)


def optimize_buyer_sourcing(
    observed: pd.Series,
    foreign_sales: pd.Series,
    shock_nodes: tuple[str, ...],
    *,
    buyer_node: str,
    shock_fraction: float,
    risk_reduction_target: float,
    headroom_fraction: float,
) -> DiversificationResult:
    """Minimize normalized sourcing turnover under the frozen diversification design."""
    if not observed.index.equals(foreign_sales.index):
        raise ValueError("observed and foreign_sales indexes must match exactly.")
    if not 0.0 < risk_reduction_target < 1.0:
        raise ValueError("risk_reduction_target must lie strictly between zero and one.")
    if headroom_fraction < 0.0 or not np.isfinite(headroom_fraction):
        raise ValueError("headroom_fraction must be finite and non-negative.")
    if shock_fraction <= 0.0 or not np.isfinite(shock_fraction):
        raise ValueError("shock_fraction must be finite and positive.")

    observed = observed.astype(float)
    foreign_sales = foreign_sales.astype(float)
    if not np.isfinite(observed.to_numpy()).all() or (observed < 0.0).any():
        raise ValueError("observed sourcing must be finite and non-negative.")
    if not np.isfinite(foreign_sales.to_numpy()).all() or (foreign_sales < 0.0).any():
        raise ValueError("foreign sales must be finite and non-negative.")

    labels = observed.index.astype(str)
    n = len(labels)
    total = float(observed.sum())
    if total <= 0.0:
        return DiversificationResult(False, None, None, None, 2, "Buyer has no intermediate input.")

    baseline_risk = baseline_direct_risk(observed, shock_nodes, shock_fraction=shock_fraction)
    if not np.isfinite(baseline_risk) or baseline_risk <= 0.0:
        return DiversificationResult(False, None, None, None, 2, "Buyer has no positive frozen-shock exposure.")
    risk_cap_share = (1.0 - risk_reduction_target) * baseline_risk
    max_shocked_flow = risk_cap_share * total / shock_fraction

    supplier_activity = pd.Series(
        [split_country_activity(label)[1] for label in labels], index=labels, dtype="object"
    )
    supplier_country = pd.Series(
        [split_country_activity(label)[0] for label in labels], index=labels, dtype="object"
    )

    # Variables: z'_i (n) followed by absolute deviations d_i (n).
    c = np.concatenate([np.zeros(n), np.full(n, 0.5 / total)])
    bounds: list[tuple[float, float | None]] = []
    for label in labels:
        base = float(observed.loc[label])
        if label == buyer_node:
            lower = upper = base
        elif base <= 0.0 and float(foreign_sales.loc[label]) <= 0.0:
            lower = upper = 0.0
        else:
            lower = 0.0
            upper = base + headroom_fraction * float(foreign_sales.loc[label])
        bounds.append((lower, upper))
    bounds.extend([(0.0, None)] * n)

    activities = sorted(set(supplier_activity))
    activity_array = supplier_activity.to_numpy()
    country_array = supplier_country.to_numpy()
    a_eq = lil_matrix((len(activities), 2 * n), dtype=float)
    b_eq = np.zeros(len(activities), dtype=float)
    for row, activity in enumerate(activities):
        idx = np.flatnonzero(activity_array == activity)
        a_eq[row, idx] = 1.0
        b_eq[row] = float(observed.iloc[idx].sum())

    concentration_pairs: list[tuple[str, str]] = []
    for activity in activities:
        idx = np.flatnonzero(activity_array == activity)
        if float(observed.iloc[idx].sum()) <= 0.0:
            continue
        concentration_pairs.extend((activity, country) for country in sorted(set(country_array[idx])))

    shock_set = set(shock_nodes)
    shock_positions = [i for i, label in enumerate(labels) if label in shock_set]
    row_count = 2 * n + len(shock_positions) + len(concentration_pairs)
    a_ub = lil_matrix((row_count, 2 * n), dtype=float)
    b_ub = np.zeros(row_count, dtype=float)
    row = 0

    # Absolute-deviation linearization.
    for i in range(n):
        a_ub[row, i] = 1.0
        a_ub[row, n + i] = -1.0
        b_ub[row] = float(observed.iloc[i])
        row += 1
        a_ub[row, i] = -1.0
        a_ub[row, n + i] = -1.0
        b_ub[row] = -float(observed.iloc[i])
        row += 1

    # Frozen-shock direct-risk cap.
    for i in shock_positions:
        a_ub[row, i] = 1.0
        b_ub[row] = max_shocked_flow
        row += 1

    # Country concentration safeguard within each supplying activity. The row is
    # added for every country with an active supplier in that activity, including
    # newly activated countries that had zero observed flow to this buyer.
    for activity, country in concentration_pairs:
        activity_idx = np.flatnonzero(activity_array == activity)
        activity_total = float(observed.iloc[activity_idx].sum())
        observed_by_country = observed.iloc[activity_idx].groupby(
            supplier_country.iloc[activity_idx], sort=False
        ).sum()
        largest_observed_share = float(observed_by_country.max() / activity_total)
        country_idx = np.flatnonzero((activity_array == activity) & (country_array == country))
        a_ub[row, country_idx] = 1.0
        b_ub[row] = largest_observed_share * activity_total
        row += 1

    result = linprog(
        c,
        A_ub=a_ub.tocsr(),
        b_ub=b_ub,
        A_eq=a_eq.tocsr(),
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        return DiversificationResult(False, None, None, None, int(result.status), str(result.message))

    allocation = pd.Series(result.x[:n], index=labels, dtype=float)
    achieved = baseline_direct_risk(allocation, shock_nodes, shock_fraction=shock_fraction)
    return DiversificationResult(
        True,
        allocation,
        float(result.fun),
        achieved,
        int(result.status),
        str(result.message),
    )
