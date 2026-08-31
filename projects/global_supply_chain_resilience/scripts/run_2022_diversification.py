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

from supply_chain_resilience.dependency import structural_dependency_metrics
from supply_chain_resilience.diversification import baseline_direct_risk, optimize_buyer_sourcing
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


def _global_exposure(coefficients: pd.DataFrame, gross_output: pd.Series, shock_node: str) -> float:
    shock = one_node_shock(coefficients.index, shock_node, SHOCK_FRACTION)
    q, _ = solve_downstream_exposure(coefficients, shock)
    return float((gross_output * q).sum())


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
    exposed["node"] = exposed.index.astype(str)
    exposed = exposed.sort_values(
        ["baseline_worst_case_direct_risk", "intermediate_input", "node"],
        ascending=[False, False, True],
        kind="stable",
    )
    return exposed.head(SELECTED_BUYERS), material_threshold


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
                frontier_rows.append(
                    {
                        "buyer": buyer,
                        "risk_reduction_target": target,
                        "headroom_fraction": headroom,
                        "feasible": result.feasible,
                        "baseline_worst_case_direct_risk": baseline_risk,
                        "achieved_worst_case_direct_risk": result.achieved_worst_case_direct_risk,
                        "reallocation_burden": result.reallocation_burden,
                        "solver_status": result.solver_status,
                        "solver_message": result.solver_message,
                    }
                )
                if result.feasible and result.allocation is not None:
                    changed = (result.allocation - observed).abs() > 1e-8
                    for supplier in result.allocation.index[changed]:
                        allocation_rows.append(
                            {
                                "buyer": buyer,
                                "risk_reduction_target": target,
                                "headroom_fraction": headroom,
                                "supplier": str(supplier),
                                "observed_flow": float(observed.loc[supplier]),
                                "counterfactual_flow": float(result.allocation.loc[supplier]),
                                "flow_change": float(result.allocation.loc[supplier] - observed.loc[supplier]),
                            }
                        )
                    if target == PRIMARY_TARGET and headroom == PRIMARY_HEADROOM:
                        primary_allocations[buyer] = result.allocation

    baseline_coefficients = technical_coefficients(blocks.intermediate_use, blocks.gross_output)
    baseline_diag = spectral_radius_diagnostics(baseline_coefficients)
    if not baseline_diag.admissible:
        raise ValueError("merged baseline propagation matrix is unexpectedly inadmissible.")
    baseline_global = {
        node: _global_exposure(baseline_coefficients, blocks.gross_output, node)
        for node in SHOCK_NODES
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
            cf_global = {
                node: _global_exposure(cf_coefficients, blocks.gross_output, node)
                for node in SHOCK_NODES
            }
            row["baseline_worst_global_exposure"] = max(baseline_global.values())
            row["counterfactual_worst_global_exposure"] = max(cf_global.values())
            row["worst_global_exposure_change"] = max(cf_global.values()) - max(baseline_global.values())
            for node in SHOCK_NODES:
                row[f"global_exposure_before__{node}"] = baseline_global[node]
                row[f"global_exposure_after__{node}"] = cf_global[node]
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
        "interpretation": (
            "Sourcing reallocation burden is normalized turnover relative to observed 2022 ICIO flows, "
            "not monetary procurement cost. Buyer-specific counterfactuals are evaluated separately."
        ),
    }
    args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
