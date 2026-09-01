"""Pre-evaluation optimizer-budget amendment for the frozen primary OPE study."""

from __future__ import annotations

import hashlib
import json
import platform
import warnings
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from adaptive_policy_learning import empirical as base

MAX_ITER_V08 = 1000


def run_primary_ope_v08(
    archive_path: Path,
    output_path: Path,
    *,
    code_sha: str,
    protocol_dir: Path,
    chunk_size: int = 100_000,
) -> dict[str, object]:
    """Execute the frozen primary study with only the amended optimizer budget."""
    base._verify_archive(archive_path)
    design_hash = protocol_hash_v08(protocol_dir)

    with ZipFile(archive_path) as archive:
        item_context = base._load_item_context(archive)
        training = base._extract_training_arrays(archive, chunk_size=chunk_size)

    scaling = base._training_numeric_scaling(training, item_context)
    x_train, layout = base._build_training_matrix(training, item_context, scaling)
    model = _fit_reward_model_v08(x_train, training.reward)
    frozen_model = base._freeze_linear_model(model, training, item_context, scaling, layout)
    del x_train, training

    if frozen_model.n_iter >= MAX_ITER_V08:
        raise RuntimeError("reward model reached amended max_iter=1000; empirical OPE is not authorized")

    with ZipFile(archive_path) as archive:
        evaluation = base._evaluate_bts(
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
    challenger_overlap = base.overlap_diagnostics(
        evaluation.propensity,
        evaluation.challenger_weights,
    )
    decision = base.promotion_decision(
        float(bootstrap["lower"]),  # type: ignore[arg-type]
        challenger_overlap.ess_fraction,
        minimum_ess_fraction=0.10,
    )

    result: dict[str, object] = {
        "status": "success",
        "protocol_version": "0.8-optimizer-budget-amendment",
        "code_sha": code_sha,
        "design_hash": design_hash,
        "source_archive_sha256": base.ARCHIVE_SHA256,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
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
            "solver": "saga",
            "C": 1.0,
            "tol": 0.0001,
            "max_iter": MAX_ITER_V08,
            "n_iter": frozen_model.n_iter,
            "coefficient_sha256": frozen_model.coefficient_sha256,
            "n_features": layout.n_features,
        },
        "random_reference": random_reference,
        "uniform_random_benchmark": {
            "estimators": uniform,
            "errors_vs_random_reference": benchmark_errors,
            "overlap": base._diagnostics_dict(
                base.overlap_diagnostics(evaluation.propensity, evaluation.uniform_weights)
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def protocol_hash_v08(protocol_dir: Path) -> str:
    """Hash the complete frozen protocol chain including the optimizer amendment."""
    names = (
        "design_lock.json",
        "source_contract_amendment_v0_2.json",
        "source_contract_amendment_v0_3.json",
        "source_contract_amendment_v0_4.json",
        "source_contract_amendment_v0_5.json",
        "temporal_split_lock_v0_6.json",
        "empirical_model_lock_v0_7.json",
        "optimizer_amendment_v0_8.json",
        "obd_source_lock.json",
    )
    digest = hashlib.sha256()
    for name in names:
        path = protocol_dir / name
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _fit_reward_model_v08(x_train: Any, y_train: np.ndarray) -> Any:
    model = LogisticRegression(
        solver="saga",
        penalty="l2",
        C=1.0,
        fit_intercept=True,
        max_iter=MAX_ITER_V08,
        tol=0.0001,
        random_state=20260831,
        class_weight=None,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_train, y_train)
    if any(issubclass(warning.category, ConvergenceWarning) for warning in caught):
        raise RuntimeError("reward model failed to converge at amended max_iter=1000")
    if model.classes_.tolist() != [0, 1]:
        raise ValueError("reward model must fit binary click classes [0, 1]")
    return model
