"""Authorized primary OPE execution for Adaptive Policy Learning Study 2."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import scipy
import sklearn

from adaptive_policy_learning import empirical as base
from adaptive_policy_learning.ope import overlap_diagnostics, promotion_decision
from adaptive_policy_learning.study2_evaluation import evaluate_bts_study2
from adaptive_policy_learning.study2_training import _fit_study2_reward_model

QUALIFICATION_LOCK_FILE = "study2_training_qualification_lock_v1_1.json"
EXPECTED_QUALIFIED_COEFFICIENT_SHA256 = (
    "7438db24286013c628dc7f74e2dd7f4913cdd05297c717fff78a214c9afb5684"
)
EXPECTED_QUALIFIED_FEATURES = 73
EXPECTED_QUALIFIED_N_ITER = 6


def run_study2_primary_ope(
    archive_path: Path,
    output_path: Path,
    *,
    code_sha: str,
    protocol_dir: Path,
    chunk_size: int = 100_000,
) -> dict[str, object]:
    """Reproduce the qualified model, then execute Study 2 OPE exactly once."""
    base._verify_archive(archive_path)
    qualification = _load_qualification_lock(protocol_dir)
    design_hash = protocol_hash_study2(protocol_dir)

    with ZipFile(archive_path) as archive:
        item_context = base._load_item_context(archive)
        training = base._extract_training_arrays(archive, chunk_size=chunk_size)

    scaling = base._training_numeric_scaling(training, item_context)
    x_train, layout = base._build_training_matrix(training, item_context, scaling)
    model, captured = _fit_study2_reward_model(x_train, training.reward)
    frozen_model = base._freeze_linear_model(model, training, item_context, scaling, layout)
    del x_train, training

    _assert_qualification_reproduced(
        qualification,
        coefficient_sha256=frozen_model.coefficient_sha256,
        n_features=layout.n_features,
        n_iter=frozen_model.n_iter,
        warnings_captured=captured,
    )

    incomplete = {
        "status": "incomplete",
        "study": "Adaptive Policy Learning Study 2",
        "protocol_version": "1.2-study2-primary-ope-execution-erratum",
        "code_sha": code_sha,
        "stage": "qualified_model_reproduced_before_evaluation",
        "qualification": {
            "coefficient_sha256": frozen_model.coefficient_sha256,
            "n_features": layout.n_features,
            "n_iter": frozen_model.n_iter,
            "warnings": captured,
        },
        "note": (
            "This record is overwritten by a terminal success or handled failure. "
            "If preserved, evaluation was interrupted or timed out after the exact "
            "qualified Study 2 model had been reproduced."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(incomplete, indent=2, sort_keys=True), encoding="utf-8")

    with ZipFile(archive_path) as archive:
        evaluation = evaluate_bts_study2(
            archive,
            frozen_model,
            item_context,
            chunk_size=chunk_size,
        )
        random_reference = base._random_reference(archive, chunk_size=chunk_size)

    uniform = base._policy_summary(
        evaluation,
        evaluation.uniform_weights,
        evaluation.uniform_dm,
    )
    challenger = base._policy_summary(
        evaluation,
        evaluation.challenger_weights,
        evaluation.challenger_dm,
    )
    uniform_sensitivity = base._clipping_sensitivity(
        evaluation,
        evaluation.uniform_weights,
        evaluation.uniform_dm,
    )
    challenger_sensitivity = base._clipping_sensitivity(
        evaluation,
        evaluation.challenger_weights,
        evaluation.challenger_dm,
    )

    bts_observed_value = float(np.mean(evaluation.reward))
    reference_value = float(random_reference["value"])  # type: ignore[arg-type]
    benchmark_errors = base._benchmark_errors(uniform, reference_value)

    challenger_dr_contribution = (
        evaluation.challenger_dm
        + evaluation.challenger_weights * (evaluation.reward - evaluation.q_logged)
    )
    paired_difference = challenger_dr_contribution - evaluation.reward
    bootstrap = base.moving_block_bootstrap_mean(
        evaluation.timestamps_ns,
        paired_difference,
        block_length_hours=24,
        replications=1999,
        seed=20260831,
        confidence_level=0.95,
    )
    challenger_overlap = overlap_diagnostics(
        evaluation.propensity,
        evaluation.challenger_weights,
    )
    decision = promotion_decision(
        float(bootstrap["lower"]),  # type: ignore[arg-type]
        challenger_overlap.ess_fraction,
        minimum_ess_fraction=0.10,
    )

    result: dict[str, object] = {
        "status": "success",
        "study": "Adaptive Policy Learning Study 2",
        "protocol_version": "1.2-study2-primary-ope-execution-erratum",
        "code_sha": code_sha,
        "design_hash": design_hash,
        "source_archive_sha256": base.ARCHIVE_SHA256,
        "qualification": {
            "qualification_code_sha": qualification["code_sha"],
            "workflow_run_id": qualification["workflow_run_id"],
            "artifact": qualification["artifact"],
            "coefficient_sha256": frozen_model.coefficient_sha256,
            "reproduced": True,
        },
        "runtime": _runtime_versions(),
        "frozen_sample": {
            "action_count": 80,
            "target_probability": base.UNIFORM_TARGET_PROBABILITY,
            "bts_training_rows": base.TRAIN_ROWS,
            "bts_evaluation_rows": base.EVALUATION_ROWS,
            "random_reference_rows": base.RANDOM_REFERENCE_ROWS,
            "evaluation_start": base.EVALUATION_START,
            "evaluation_end": base.EVALUATION_END,
        },
        "reward_model": {
            "family": "L2 logistic regression",
            "solver": "newton-cholesky",
            "regularization": "l2",
            "l1_ratio": 0.0,
            "C": 1.0,
            "tol": 0.0001,
            "max_iter": 100,
            "n_iter": frozen_model.n_iter,
            "coefficient_sha256": frozen_model.coefficient_sha256,
            "n_features": layout.n_features,
            "warnings": captured,
        },
        "random_reference": random_reference,
        "uniform_random_benchmark": {
            "estimators": uniform,
            "errors_vs_random_reference": benchmark_errors,
            "overlap": base._diagnostics_dict(
                overlap_diagnostics(evaluation.propensity, evaluation.uniform_weights)
            ),
            "clipping_sensitivity": uniform_sensitivity,
        },
        "challenger": {
            "estimators": challenger,
            "bts_observed_value": bts_observed_value,
            "dr_minus_bts_point_difference": float(np.mean(paired_difference)),
            "paired_moving_block_bootstrap": bootstrap,
            "overlap": base._diagnostics_dict(challenger_overlap),
            "clipping_sensitivity": challenger_sensitivity,
            "promotion_decision": decision,
        },
        "promotion_rule": {
            "lower_95_ci_must_exceed_zero": True,
            "minimum_ess_fraction": 0.10,
        },
    }
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _load_qualification_lock(protocol_dir: Path) -> dict[str, object]:
    path = protocol_dir / QUALIFICATION_LOCK_FILE
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Study 2 qualification lock must be a JSON object")
    if payload.get("status") != "training_gate_passed":
        raise RuntimeError("Study 2 evaluation is not authorized by the qualification lock")
    if payload.get("evaluation_outcomes_loaded") is not False:
        raise RuntimeError("qualification lock does not preserve unopened evaluation outcomes")
    if payload.get("random_reference_outcomes_loaded") is not False:
        raise RuntimeError("qualification lock does not preserve unopened Random outcomes")
    if payload.get("ope_estimates_computed") is not False:
        raise RuntimeError("qualification lock unexpectedly records prior OPE estimates")
    return payload


def _assert_qualification_reproduced(
    qualification: dict[str, object],
    *,
    coefficient_sha256: str,
    n_features: int,
    n_iter: int,
    warnings_captured: list[str],
) -> None:
    training = qualification.get("training")
    runtime = qualification.get("runtime")
    if not isinstance(training, dict) or not isinstance(runtime, dict):
        raise TypeError("Study 2 qualification lock is missing training/runtime metadata")

    expected_sha = training.get("coefficient_sha256")
    if expected_sha != EXPECTED_QUALIFIED_COEFFICIENT_SHA256:
        raise RuntimeError("Study 2 qualification lock coefficient SHA is not the frozen value")
    if coefficient_sha256 != expected_sha:
        raise RuntimeError("Study 2 coefficient SHA mismatch; evaluation is not authorized")
    if n_features != EXPECTED_QUALIFIED_FEATURES or training.get("n_features") != n_features:
        raise RuntimeError("Study 2 feature-count mismatch; evaluation is not authorized")
    if n_iter != EXPECTED_QUALIFIED_N_ITER or training.get("n_iter") != n_iter:
        raise RuntimeError("Study 2 optimizer-iteration mismatch; evaluation is not authorized")
    if warnings_captured or training.get("warnings") != []:
        raise RuntimeError("Study 2 optimizer warnings present; evaluation is not authorized")

    current_runtime = _runtime_versions()
    for key in ("python", "numpy", "pandas", "scipy", "scikit_learn"):
        if runtime.get(key) != current_runtime[key]:
            raise RuntimeError(
                f"Study 2 runtime mismatch for {key}; evaluation is not authorized"
            )


def protocol_hash_study2(protocol_dir: Path) -> str:
    """Hash the complete Study 2 prospective and qualification protocol chain."""
    names = (
        "design_lock.json",
        "source_contract_amendment_v0_2.json",
        "source_contract_amendment_v0_3.json",
        "source_contract_amendment_v0_4.json",
        "source_contract_amendment_v0_5.json",
        "temporal_split_lock_v0_6.json",
        "empirical_model_lock_v0_7.json",
        "optimizer_amendment_v0_8.json",
        "primary_empirical_terminal_status_v0_8.json",
        "obd_source_lock.json",
        "study2_design_lock_v1_0.json",
        QUALIFICATION_LOCK_FILE,
        "study2_primary_ope_execution_erratum_v1_2.json",
    )
    digest = hashlib.sha256()
    for name in names:
        path = protocol_dir / name
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }
