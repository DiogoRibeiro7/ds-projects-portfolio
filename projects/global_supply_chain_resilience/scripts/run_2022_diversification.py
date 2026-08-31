"""Run the preregistered 2022 sourcing-diversification experiment."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from supply_chain_resilience.dependency import split_country_activity, structural_dependency_metrics
from supply_chain_resilience.diversification import (
    baseline_direct_risk,
    optimize_buyer_sourcing,
    rank_exposed_buyers,
)
from supply_chain_resilience.icio import technical_coefficients
from supply_chain_resilience.mapping import active_production_blocks, extract_2022_blocks, validate_2022_accounting
from supply_chain_resilience.propagation import one_node_shock, solve_downstream_exposure, spectral_radius_diagnostics
from supply_chain_resilience.supplier import supplier_importance_metrics

SHOCK_NODES = (
    "CHN_C26",
    "CHN_C20",
    "USA_G",
    "CHN_C27",
    "RUS_B06",
    "NOR_B06",
    "USA_B06",
)
SHOCK_FRACTION = 0.10
RISK_TARGETS = (0.25, 0.50, 0.75)
HEADROOMS = (0.01, 0.05, 0.10)
PRIMARY_TARGET = 0.50
PRIMARY_HEADROOM = 0.05
SELECTED_BUYERS = 10
CHANGE_TOLERANCE = 1e-8


def _scenario_exposure_metrics(
    coefficients: pd.DataFrame,
    gross_output: pd.Series,
    shock_node: str,
    *,
    buyer: str,
) -> dict[str, float]:
    shock = one_node_shock(coefficients.index, shock_node, SHOCK_FRACTION)
    q, _ = solve_downstream_exposure(coefficients, shock)
    contribution = gross_output * q
    shocked_country, _ = split_country_activity(shock_node)
    foreign_mask = pd.Series(
        [split_country_activity(str(label))[0] != shocked_country for label in q.index],
        index=q.index,
    )
    return {
        "global_output_equivalent_exposure": float(contribution.sum()),
        "buyer_exposure_index": float(q.loc[buyer]),
        "foreign_output_equivalent_exposure": float(contribution.loc[foreign_mask].sum()),
    }


def _select_buyers(blocks: object) -> tuple[pd.DataFrame, float]:
    dependency = structural_dependency_metrics(blocks)  # type: ignore[arg-type]
    positive = dependency.loc[dependency["intermediate_input"] > 0.0]
    material_threshold = float(positive["intermediate_input"].median())
    material = dependency.loc[dependency["intermediate_input"] >= material_threshold].copy()

    risks = []
    for buyer in material.index:
        observed = blocks.intermediate_use.loc[:, buyer]  # type: ignore[attr-defined]
        risks.append(
            baseline_direct_risk(observed, SHOCK_NODES, shock_fraction=SHOCK_FRACTION)
        )
    material["baseline_worst_case_direct_risk"] = risks
    exposed = material.loc[material["baseline_worst_case_direct_risk"] > 0.0].copy()
    return rank_exposed_buyers(exposed, limit=SELECTED_BUYERS), material_threshold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--frontier-output", type=Path, required=True)
    parser.add_argument("--allocation-output", type=Path, required=True)
    parser.add_argument("--system-output", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if tuple(protocol["frozen_shock_nodes"]) != SHOCK_NODES:
        raise ValueError("protocol shock nodes do not match executable lock.")
    if float(protocol["shock_fraction"]) != SHOCK_FRACTION:
        raise ValueError("protocol shock fraction does not match executable lock.")

    raw = args.archive.read_bytes()
    source_hash = sha256(raw).hexdigest()
    if source_hash != protocol["source_sha256"]:
        raise ValueError("official archive SHA-256 does not match diversification protocol.")

    with ZipFile(args.archive) as archive:
        with archive.open("2022_SML.csv") as handle:
            frame = pd.read_csv(handle, index_col=0, low_memory=False)

    published = extract_2022_blocks(frame)
    validate_2022_accounting(published)
    blocks, inactive = active_production_blocks(published)
    supplier_metrics = supplier_importance_metrics(blocks)
    foreign_sales = supplier_metrics["foreign_intermediate_sales"].reindex(blocks.intermediate_use.index)

    buyers, material_threshold = _select_buyers(blocks)
    selected_buyers = list(buyers.index.astype(str))
    if len(selected_buyers) != SELECTED_BUYERS:
        raise ValueError(f"expected {SELECTED_BUYERS} exposed material buyers, got {len(selected_buyers)}")

    frontier_rows: list[dict[str, object]] = []
    allocation_rows: list[dict[str, object]] = []
    primary_allocations: dict[str, pd.Series] = {}

    for buyer in selected_buyers:
        observed = blocks.intermediate_use.loc[:, buyer].astype(float)
        baseline_risk = baseline_direct_risk(observed, SHOCK_NODES, shock_fraction=SHOCK_FRACTION)
        for headroom in HEADROOMS:
            for target in RISK_TARGETS:
                result = optimize_buyer_sourcing(
                    observed,
                    foreign_sales,
                    SHOCK_NODES,
                    buyer_node=buyer,
                    shock_fraction=SHOCK_FRACTION,
                    risk_reduction_target=target,
                    headroom_fraction=headroom,
                )
                frontier_row: dict[str, object] = {
                    "buyer": buyer,
                    "risk_reduction_target": target,
                    "headroom_fraction": headroom,
                    "feasible": result.feasible,
                    "baseline_worst_case_direct_risk": baseline_risk,
                    "achieved_worst_case_direct_risk": result.achieved_worst_case_direct_risk,
                    "reallocation_burden": result.reallocation_burden,
                    "solver_status": result.solver_status,
                    "solver_message": result.solver_message,
                    "changed_supplier_nodes": np.nan,
                    "largest_supplier_increase": np.nan,
                    "largest_supplier_decrease": np.nan,
                }
                if result.feasible and result.allocation is not None:
                    change = result.allocation - observed
                    changed = change.abs() > CHANGE_TOLERANCE
                    frontier_row["changed_supplier_nodes"] = int(changed.sum())
                    frontier_row["largest_supplier_increase"] = float(change.max())
                    frontier_row["largest_supplier_decrease"] = float(change.min())
                    for supplier in result.allocation.index[changed]:
                        allocation_rows.append(
                            {
                                "buyer": buyer,
                                "risk_reduction_target": target,
                                "headroom_fraction": headroom,
                                "supplier": str(supplier),
                                "observed_flow": float(observed.loc[supplier]),
                                "counterfactual_flow": float(result.allocation.loc[supplier]),
                                "flow_change": float(change.loc[supplier]),
                            }
                        )
                    if target == PRIMARY_TARGET and headroom == PRIMARY_HEADROOM:
                        primary_allocations[buyer] = result.allocation
                frontier_rows.append(frontier_row)

    baseline_coefficients = technical_coefficients(blocks.intermediate_use, blocks.gross_output)
    baseline_diag = spectral_radius_diagnostics(baseline_coefficients)
    if not baseline_diag.admissible:
        raise ValueError("merged baseline propagation matrix is unexpectedly inadmissible.")

    baseline_by_buyer: dict[str, dict[str, dict[str, float]]] = {
        buyer: {
            node: _scenario_exposure_metrics(
                baseline_coefficients,
                blocks.gross_output,
                node,
                buyer=buyer,
            )
            for node in SHOCK_NODES
        }
        for buyer in selected_buyers
    }

    system_rows: list[dict[str, object]] = []
    for buyer in selected_buyers:
        if buyer not in primary_allocations:
            system_rows.append(
                {
                    "buyer": buyer,
                    "primary_policy_feasible": False,
                    "post_spectral_radius": np.nan,
                    "post_admissibility_status": "NOT_RUN_INFEASIBLE",
                }
            )
            continue

        z_cf = blocks.intermediate_use.copy()
        z_cf.loc[:, buyer] = primary_allocations[buyer]
        cf_blocks = replace(blocks, intermediate_use=z_cf)
        cf_coefficients = technical_coefficients(cf_blocks.intermediate_use, cf_blocks.gross_output)
        diag = spectral_radius_diagnostics(cf_coefficients)
        row: dict[str, object] = {
            "buyer": buyer,
            "primary_policy_feasible": True,
            "post_spectral_radius": diag.spectral_radius,
            "post_eigen_residual": diag.eigen_residual,
            "post_admissibility_status": "PASS" if diag.admissible else "FAIL",
        }
        if diag.admissible:
            cf_metrics = {
                node: _scenario_exposure_metrics(
                    cf_coefficients,
                    blocks.gross_output,
                    node,
                    buyer=buyer,
                )
                for node in SHOCK_NODES
            }
            baseline_worst = max(
                values["global_output_equivalent_exposure"]
                for values in baseline_by_buyer[buyer].values()
            )
            counterfactual_worst = max(
                values["global_output_equivalent_exposure"] for values in cf_metrics.values()
            )
            row["baseline_worst_global_exposure"] = baseline_worst
            row["counterfactual_worst_global_exposure"] = counterfactual_worst
            row["worst_global_exposure_change"] = counterfactual_worst - baseline_worst
            for node in SHOCK_NODES:
                before = baseline_by_buyer[buyer][node]
                after = cf_metrics[node]
                row[f"buyer_exposure_before__{node}"] = before["buyer_exposure_index"]
                row[f"buyer_exposure_after__{node}"] = after["buyer_exposure_index"]
                row[f"global_exposure_before__{node}"] = before["global_output_equivalent_exposure"]
                row[f"global_exposure_after__{node}"] = after["global_output_equivalent_exposure"]
                row[f"foreign_spillover_before__{node}"] = before["foreign_output_equivalent_exposure"]
                row[f"foreign_spillover_after__{node}"] = after["foreign_output_equivalent_exposure"]
        system_rows.append(row)

    frontier = pd.DataFrame(frontier_rows)
    allocations = pd.DataFrame(allocation_rows)
    system = pd.DataFrame(system_rows)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    frontier.to_csv(args.frontier_output, index=False)
    allocations.to_csv(args.allocation_output, index=False)
    system.to_csv(args.system_output, index=False)

    primary = frontier.loc[
        (frontier["risk_reduction_target"] == PRIMARY_TARGET)
        & (frontier["headroom_fraction"] == PRIMARY_HEADROOM)
    ]
    summary = {
        "year": 2022,
        "source_sha256": source_hash,
        "active_nodes": int(len(blocks.gross_output)),
        "inactive_zero_output_labels": int(len(inactive)),
        "material_intermediate_input_threshold": material_threshold,
        "selected_buyers": selected_buyers,
        "selected_buyer_baseline_risk": buyers["baseline_worst_case_direct_risk"].to_dict(),
        "frontier_rows": int(len(frontier)),
        "primary_policy": {
            "risk_reduction_target": PRIMARY_TARGET,
            "headroom_fraction": PRIMARY_HEADROOM,
            "feasible_buyers": int(primary["feasible"].sum()),
            "infeasible_buyers": int((~primary["feasible"]).sum()),
        },
        "baseline_spectral_radius": baseline_diag.spectral_radius,
        "system_evaluations_run": int(system["post_admissibility_status"].eq("PASS").sum()),
        "outcome_contract": {
            "frontier_includes_changed_supplier_count": True,
            "frontier_includes_largest_supplier_increase_decrease": True,
            "system_includes_buyer_exposure_by_shock": True,
            "system_includes_global_exposure_by_shock": True,
            "system_includes_foreign_spillover_by_shock": True,
        },
        "interpretation": (
            "Sourcing reallocation burden is normalized turnover relative to observed 2022 ICIO flows, "
            "not monetary procurement cost. Buyer-specific counterfactuals are evaluated separately."
        ),
    }
    args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
