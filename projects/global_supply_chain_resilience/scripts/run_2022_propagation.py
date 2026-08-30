"""Run the preregistered 2022 downstream fixed-coefficient exposure experiment."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from supply_chain_resilience.dependency import split_country_activity
from supply_chain_resilience.icio import technical_coefficients
from supply_chain_resilience.mapping import active_production_blocks, extract_2022_blocks, validate_2022_accounting
from supply_chain_resilience.propagation import one_node_shock, solve_downstream_exposure, spectral_radius_diagnostics
from supply_chain_resilience.sensitivity import threshold_intermediate_use_by_input_share

CANDIDATES = (
    "CHN_C26",
    "CHN_C20",
    "USA_G",
    "CHN_C27",
    "RUS_B06",
    "NOR_B06",
    "USA_B06",
)
PRIMARY_FRACTION = 0.10
SCALE_CHECKS = (0.05, 0.20)
THRESHOLDS = (0.001, 0.005, 0.01)


def _jaccard_top3(reference: pd.Series, candidate: pd.Series) -> float:
    left = set(reference.nlargest(3).index)
    right = set(candidate.nlargest(3).index)
    return float(len(left & right) / len(left | right))


def _scenario_metrics(
    coefficients: pd.DataFrame,
    gross_output: pd.Series,
    node: str,
    fraction: float,
) -> tuple[dict[str, float | str], pd.DataFrame, pd.Series]:
    shock = one_node_shock(coefficients.index, node, fraction)
    q, solver_residual = solve_downstream_exposure(coefficients, shock)
    first_round = coefficients.T.dot(shock)
    contribution = gross_output * q
    exogenous_burden = float(fraction * gross_output.loc[node])
    total = float(contribution.sum())
    propagated = total - exogenous_burden
    higher_order = q - shock - first_round
    shocked_country, _ = split_country_activity(node)
    foreign_mask = pd.Series(
        [split_country_activity(str(label))[0] != shocked_country for label in q.index],
        index=q.index,
    )
    foreign_total = float(contribution.loc[foreign_mask].sum())
    nonshocked = q.drop(index=node)

    rows = pd.DataFrame(
        {
            "node": q.index.astype(str),
            "exposure_index": q.to_numpy(),
            "gross_output": gross_output.to_numpy(dtype=float),
            "output_equivalent_exposure": contribution.to_numpy(dtype=float),
        }
    )
    rows[["country", "activity"]] = rows["node"].apply(
        lambda value: pd.Series(split_country_activity(str(value)))
    )
    rows["shocked_supplier"] = node
    rows["shock_fraction"] = fraction
    rows = rows.sort_values("output_equivalent_exposure", ascending=False, kind="stable")

    metrics: dict[str, float | str] = {
        "shocked_supplier": node,
        "shock_fraction": fraction,
        "exogenous_burden": exogenous_burden,
        "total_output_equivalent_exposure": total,
        "propagated_exposure": propagated,
        "amplification": total / exogenous_burden,
        "first_round_downstream_exposure": float((gross_output * first_round).sum()),
        "higher_order_exposure": float((gross_output * higher_order).sum()),
        "foreign_output_equivalent_exposure": foreign_total,
        "foreign_spillover_share": foreign_total / propagated if propagated > 0.0 else float("nan"),
        "max_nonshocked_exposure_index": float(nonshocked.max()),
        "solver_relative_residual": solver_residual,
    }
    return metrics, rows.head(20), q


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--scenario-output", type=Path, required=True)
    parser.add_argument("--top-affected-output", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if tuple(protocol["candidate_nodes"]) != CANDIDATES:
        raise ValueError("protocol candidate set does not match executable frozen set.")
    if float(protocol["shock_design"]["primary_fraction"]) != PRIMARY_FRACTION:
        raise ValueError("protocol primary shock fraction does not match executable lock.")

    raw = args.archive.read_bytes()
    source_hash = sha256(raw).hexdigest()
    if source_hash != protocol["source_sha256"]:
        raise ValueError("official archive SHA-256 does not match propagation protocol.")

    with ZipFile(args.archive) as archive:
        with archive.open("2022_SML.csv") as handle:
            frame = pd.read_csv(handle, index_col=0, low_memory=False)

    published = extract_2022_blocks(frame)
    validate_2022_accounting(published)
    blocks, inactive = active_production_blocks(published)
    coefficients = technical_coefficients(blocks.intermediate_use, blocks.gross_output)

    missing = [node for node in CANDIDATES if node not in coefficients.index]
    if missing:
        raise ValueError(f"frozen candidates missing from active matrix: {missing}")

    diagnostic = spectral_radius_diagnostics(coefficients)
    admissibility = {
        "spectral_radius": diagnostic.spectral_radius,
        "eigen_residual": diagnostic.eigen_residual,
        "spectral_radius_limit": 1.0 - 1e-8,
        "eigen_residual_limit": 1e-8,
        "status": "PASS" if diagnostic.admissible else "FAIL",
    }

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    if not diagnostic.admissible:
        summary = {
            "year": 2022,
            "source_sha256": source_hash,
            "active_nodes": int(len(coefficients)),
            "inactive_zero_output_labels": int(len(inactive)),
            "admissibility": admissibility,
            "propagation_status": "NOT_RUN_ADMISSIBILITY_FAILURE",
            "candidate_nodes": list(CANDIDATES),
        }
        args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
        raise SystemExit("Propagation stopped because the preregistered admissibility gate failed.")

    scenario_records: list[dict[str, float | str]] = []
    top_frames: list[pd.DataFrame] = []
    baseline_amplification: dict[str, float] = {}
    baseline_total: dict[str, float] = {}
    baseline_q: dict[str, pd.Series] = {}

    for node in CANDIDATES:
        metrics, top, q = _scenario_metrics(coefficients, blocks.gross_output, node, PRIMARY_FRACTION)
        scenario_records.append(metrics)
        top_frames.append(top)
        baseline_amplification[node] = float(metrics["amplification"])
        baseline_total[node] = float(metrics["total_output_equivalent_exposure"])
        baseline_q[node] = q

        for fraction in SCALE_CHECKS:
            scaled_metrics, _, scaled_q = _scenario_metrics(coefficients, blocks.gross_output, node, fraction)
            expected = fraction / PRIMARY_FRACTION
            scale_error = float(np.max(np.abs(scaled_q.to_numpy() - expected * q.to_numpy())))
            if scale_error > 1e-10:
                raise ValueError(f"linearity check failed for {node} at {fraction}: {scale_error}")
            scaled_metrics["linearity_max_abs_error_vs_10pct"] = scale_error
            scenario_records.append(scaled_metrics)

    primary = pd.DataFrame([row for row in scenario_records if row["shock_fraction"] == PRIMARY_FRACTION])
    primary = primary.set_index("shocked_supplier")
    reference_amp = primary["amplification"].astype(float)

    threshold_reports: list[dict[str, object]] = []
    for threshold in THRESHOLDS:
        z_t = threshold_intermediate_use_by_input_share(blocks, minimum_input_share=threshold)
        threshold_blocks = replace(blocks, intermediate_use=z_t)
        a_t = technical_coefficients(threshold_blocks.intermediate_use, threshold_blocks.gross_output)
        threshold_diag = spectral_radius_diagnostics(a_t)
        report: dict[str, object] = {
            "minimum_input_share": threshold,
            "spectral_radius": threshold_diag.spectral_radius,
            "eigen_residual": threshold_diag.eigen_residual,
            "admissibility_status": "PASS" if threshold_diag.admissible else "FAIL",
        }
        if threshold_diag.admissible:
            candidate_amp: dict[str, float] = {}
            for node in CANDIDATES:
                metrics, _, _ = _scenario_metrics(a_t, blocks.gross_output, node, PRIMARY_FRACTION)
                candidate_amp[node] = float(metrics["amplification"])
            candidate_series = pd.Series(candidate_amp, dtype=float)
            report["amplification_spearman"] = float(reference_amp.corr(candidate_series, method="spearman"))
            report["amplification_top3_jaccard"] = _jaccard_top3(reference_amp, candidate_series)
            report["candidate_amplification"] = candidate_amp
        threshold_reports.append(report)

    scenario_frame = pd.DataFrame(scenario_records)
    scenario_frame.to_csv(args.scenario_output, index=False)
    pd.concat(top_frames, ignore_index=True).to_csv(args.top_affected_output, index=False)

    summary = {
        "year": 2022,
        "source_sha256": source_hash,
        "active_nodes": int(len(coefficients)),
        "inactive_zero_output_labels": int(len(inactive)),
        "candidate_nodes": list(CANDIDATES),
        "admissibility": admissibility,
        "propagation_status": "PASS",
        "primary_shock_fraction": PRIMARY_FRACTION,
        "scale_check_fractions": list(SCALE_CHECKS),
        "primary_amplification_ranking": reference_amp.sort_values(ascending=False).to_dict(),
        "primary_total_exposure_ranking": primary["total_output_equivalent_exposure"].astype(float).sort_values(ascending=False).to_dict(),
        "threshold_sensitivity": threshold_reports,
        "interpretation": (
            "Fixed-coefficient downstream exposure under proportional input loss, no substitution, "
            "no inventories, no price response and linear retransmission. Not a causal forecast."
        ),
    }
    args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
