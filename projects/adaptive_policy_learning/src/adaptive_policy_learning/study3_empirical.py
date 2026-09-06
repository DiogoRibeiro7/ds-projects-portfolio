"""Authorized primary OPE execution for Adaptive Policy Learning Study 3."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.special import expit

from adaptive_policy_learning import empirical as base
from adaptive_policy_learning.ope import importance_weights, overlap_diagnostics, promotion_decision
from adaptive_policy_learning.study3_training import (
    ACTION_COUNT,
    ACTION_IDS,
    AFFINITY_COLUMNS,
    ARCHIVE_SHA256,
    CAMPAIGN,
    ITEM_CATEGORICAL_COLUMNS,
    TRAIN_ROWS,
    USER_COLUMNS,
    _build_training_matrix,
    _extract_training_arrays,
    _fit_reward_model,
    _load_item_context,
    _training_numeric_scaling,
)

QUALIFIED_MODEL_LOCK_FILE = "study3_qualified_model_lock_v2_6.json"
AUTHORIZATION_FILE = "study3_primary_ope_authorization_v2_7.json"
EXPECTED_QUALIFIED_COEFFICIENT_SHA256 = (
    "8e8ba7827c80c256e1c980007053fdcbb22d2ac8793673df3fe6669fafd3802c"
)
EXPECTED_QUALIFIED_FEATURES = 65
EXPECTED_QUALIFIED_N_ITER = 6
EVALUATION_ROWS = 776_442
RANDOM_REFERENCE_ROWS = 85_990
EVALUATION_START = "2019-11-28 15:23:48.989271+00:00"
EVALUATION_END = "2019-11-30 23:59:59.862467+00:00"
UNIFORM_TARGET_PROBABILITY = 1.0 / ACTION_COUNT
CHALLENGER_EPSILON = 0.10


def run_study3_primary_ope(
    archive_path: Path,
    output_path: Path,
    *,
    code_sha: str,
    protocol_dir: Path,
    chunk_size: int = 100_000,
) -> dict[str, object]:
    """Reproduce the qualified Study 3 model, then execute frozen OPE exactly once."""
    _verify_archive(archive_path)
    qualification = _load_qualification_lock(protocol_dir)
    design_hash = protocol_hash_study3(protocol_dir)

    with ZipFile(archive_path) as archive:
        item_context = _load_item_context(archive)
        training = _extract_training_arrays(archive)

    scaling = _training_numeric_scaling(training, item_context)
    x_train, layout = _build_training_matrix(training, item_context, scaling)
    model, captured = _fit_reward_model(x_train, training.reward)
    frozen_model = _freeze_linear_model(model, training, item_context, scaling, layout)
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
        "study": "Adaptive Policy Learning Study 3",
        "protocol_version": "2.7-study3-primary-ope-authorization",
        "code_sha": code_sha,
        "stage": "qualified_model_reproduced_before_evaluation",
        "qualification": {
            "coefficient_sha256": frozen_model.coefficient_sha256,
            "n_features": layout.n_features,
            "n_iter": frozen_model.n_iter,
            "warnings": captured,
        },
        "note": (
            "This record is overwritten by terminal success or failure. If preserved, "
            "execution stopped after reproducing the qualified model and before a terminal "
            "Study 3 OPE result was written."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(incomplete, indent=2, sort_keys=True), encoding="utf-8")

    with ZipFile(archive_path) as archive:
        _validate_evaluation_window_without_outcomes(archive, chunk_size=chunk_size)
        evaluation = _evaluate_bts(
            archive,
            frozen_model,
            chunk_size=chunk_size,
        )
        random_reference = _random_reference(archive, chunk_size=chunk_size)

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
        "study": "Adaptive Policy Learning Study 3",
        "protocol_version": "2.7-study3-primary-ope-authorization",
        "code_sha": code_sha,
        "design_hash": design_hash,
        "source_archive_sha256": ARCHIVE_SHA256,
        "campaign": CAMPAIGN,
        "qualification": {
            "qualification_code_sha": qualification["code_sha"],
            "workflow_run_id": qualification["workflow_run_id"],
            "artifact": qualification["artifact"],
            "coefficient_sha256": frozen_model.coefficient_sha256,
            "reproduced": True,
        },
        "runtime": _runtime_versions(),
        "frozen_sample": {
            "action_count": ACTION_COUNT,
            "target_probability": UNIFORM_TARGET_PROBABILITY,
            "bts_training_rows": TRAIN_ROWS,
            "bts_evaluation_rows": EVALUATION_ROWS,
            "random_reference_rows": RANDOM_REFERENCE_ROWS,
            "evaluation_start": EVALUATION_START,
            "evaluation_end": EVALUATION_END,
        },
        "reward_model": {
            "family": "L2 logistic regression",
            "solver": "newton-cholesky",
            "regularization": "l2",
            "l1_ratio": 0.0,
            "C": 1.0,
            "fit_intercept": True,
            "class_weight": None,
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


def _verify_archive(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = base.sha256_file(path)
    if actual != ARCHIVE_SHA256:
        raise ValueError(f"archive SHA256 mismatch: expected {ARCHIVE_SHA256}, got {actual}")


def _load_qualification_lock(protocol_dir: Path) -> dict[str, object]:
    payload = json.loads((protocol_dir / QUALIFIED_MODEL_LOCK_FILE).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Study 3 qualified-model lock must be a JSON object")
    if payload.get("status") != "training_gate_passed":
        raise RuntimeError("Study 3 evaluation is not authorized by the qualified-model lock")
    for key in (
        "evaluation_outcomes_loaded",
        "random_reference_outcomes_loaded",
        "ope_estimates_computed",
    ):
        if payload.get(key) is not False:
            raise RuntimeError(f"Study 3 qualified-model lock does not preserve {key}=false")
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
        raise TypeError("Study 3 qualified-model lock is missing training/runtime metadata")

    expected_sha = training.get("coefficient_sha256")
    if expected_sha != EXPECTED_QUALIFIED_COEFFICIENT_SHA256:
        raise RuntimeError("Study 3 qualified-model lock coefficient SHA is not frozen value")
    if coefficient_sha256 != expected_sha:
        raise RuntimeError("Study 3 coefficient SHA mismatch; evaluation is not authorized")
    if n_features != EXPECTED_QUALIFIED_FEATURES or training.get("n_features") != n_features:
        raise RuntimeError("Study 3 feature-count mismatch; evaluation is not authorized")
    if n_iter != EXPECTED_QUALIFIED_N_ITER or training.get("n_iter") != n_iter:
        raise RuntimeError("Study 3 optimizer-iteration mismatch; evaluation is not authorized")
    if warnings_captured or training.get("warnings") != []:
        raise RuntimeError("Study 3 optimizer warnings present; evaluation is not authorized")

    current_runtime = _runtime_versions()
    for key in ("python", "numpy", "pandas", "scipy", "scikit_learn"):
        if runtime.get(key) != current_runtime[key]:
            raise RuntimeError(f"Study 3 runtime mismatch for {key}; evaluation is not authorized")


def _freeze_linear_model(
    model: Any,
    training: Any,
    item_context: Any,
    scaling: Any,
    layout: Any,
) -> base.FrozenLinearPolicyModel:
    coefficient = np.asarray(model.coef_[0], dtype=float)
    if coefficient.size != layout.n_features:
        raise ValueError("Study 3 coefficient dimension does not match frozen layout")

    user_coefficients: list[np.ndarray] = []
    for column_index, offset in enumerate(layout.user_offsets):
        width = len(training.user_levels[column_index])
        user_coefficients.append(coefficient[offset : offset + width].copy())

    item0 = np.where(
        np.isfinite(item_context.item_feature_0),
        item_context.item_feature_0,
        scaling.item_median,
    )
    action_base = np.full(ACTION_COUNT, float(model.intercept_[0]), dtype=float)
    for column_index, offset in enumerate(layout.item_offsets):
        codes = item_context.categorical_codes[:, column_index]
        action_base += coefficient[offset + codes]
    action_base += coefficient[layout.item_numeric_column] * (
        (item0 - scaling.item_mean) / scaling.item_scale
    )

    digest = hashlib.sha256()
    digest.update(np.asarray(model.intercept_, dtype=np.float64).tobytes())
    digest.update(np.asarray(model.coef_, dtype=np.float64).tobytes())
    maps = tuple(
        {value: index for index, value in enumerate(levels)}
        for levels in training.user_levels
    )
    return base.FrozenLinearPolicyModel(
        user_coefficients=tuple(user_coefficients),
        action_base_logit=action_base,
        affinity_coefficient=float(coefficient[layout.affinity_column]),
        affinity_median=scaling.affinity_median,
        affinity_mean=scaling.affinity_mean,
        affinity_scale=scaling.affinity_scale,
        user_level_maps=maps,
        coefficient_sha256=digest.hexdigest(),
        n_iter=int(model.n_iter_[0]),
    )


def _iter_evaluation_slices(
    archive: ZipFile,
    *,
    usecols: list[str],
    chunk_size: int,
):
    member = "open_bandit_dataset/bts/women/women.csv"
    position_one_seen = 0
    filled = 0
    with archive.open(member) as handle:
        reader = pd.read_csv(handle, usecols=usecols, chunksize=chunk_size)
        for chunk in reader:
            selected = chunk.loc[chunk["position"] == 1]
            if selected.empty:
                continue
            selected_count = len(selected)
            first_eval = max(0, TRAIN_ROWS - position_one_seen)
            position_one_seen += selected_count
            if first_eval >= selected_count:
                continue
            selected = selected.iloc[first_eval:]
            remaining = EVALUATION_ROWS - filled
            if len(selected) > remaining:
                selected = selected.iloc[:remaining]
            if selected.empty:
                continue
            filled += len(selected)
            yield selected
            if filled == EVALUATION_ROWS:
                break
    if filled != EVALUATION_ROWS:
        raise ValueError(f"expected {EVALUATION_ROWS} Study 3 evaluation rows, found {filled}")


def _validate_evaluation_window_without_outcomes(archive: ZipFile, *, chunk_size: int) -> None:
    first_timestamp: pd.Timestamp | None = None
    last_timestamp: pd.Timestamp | None = None
    for selected in _iter_evaluation_slices(
        archive,
        usecols=["timestamp", "position"],
        chunk_size=chunk_size,
    ):
        parsed = pd.to_datetime(selected["timestamp"], utc=True)
        if first_timestamp is None:
            first_timestamp = parsed.iloc[0]
        last_timestamp = parsed.iloc[-1]
    if first_timestamp != pd.Timestamp(EVALUATION_START):
        raise ValueError("Study 3 evaluation first timestamp does not match frozen lock")
    if last_timestamp != pd.Timestamp(EVALUATION_END):
        raise ValueError("Study 3 evaluation last timestamp does not match frozen lock")


def _evaluate_bts(
    archive: ZipFile,
    model: base.FrozenLinearPolicyModel,
    *,
    chunk_size: int,
) -> base.EvaluationArrays:
    usecols = [
        "timestamp",
        "position",
        "item_id",
        "click",
        "propensity_score",
        *USER_COLUMNS,
        *AFFINITY_COLUMNS,
    ]
    timestamps = np.empty(EVALUATION_ROWS, dtype=np.int64)
    reward = np.empty(EVALUATION_ROWS, dtype=float)
    propensity = np.empty(EVALUATION_ROWS, dtype=float)
    q_logged = np.empty(EVALUATION_ROWS, dtype=float)
    uniform_dm = np.empty(EVALUATION_ROWS, dtype=float)
    challenger_dm = np.empty(EVALUATION_ROWS, dtype=float)
    uniform_weights = np.empty(EVALUATION_ROWS, dtype=float)
    challenger_weights = np.empty(EVALUATION_ROWS, dtype=float)

    filled = 0
    for selected in _iter_evaluation_slices(
        archive,
        usecols=usecols,
        chunk_size=chunk_size,
    ):
        take = len(selected)
        stop = filled + take
        action = np.asarray(selected["item_id"].to_numpy(), dtype=np.int16)
        if np.any((action < 0) | (action >= ACTION_COUNT)):
            raise ValueError("Study 3 evaluation action outside frozen catalog 0..45")
        pscore = np.asarray(selected["propensity_score"].to_numpy(), dtype=float)
        if np.any(~np.isfinite(pscore)) or np.any(pscore <= 0.0):
            raise ValueError("Study 3 evaluation propensities must be finite and positive")

        user_contribution = _user_logit_contribution(selected, model)
        affinity = np.asarray(selected.loc[:, AFFINITY_COLUMNS].to_numpy(), dtype=float)
        affinity = np.where(np.isfinite(affinity), affinity, model.affinity_median)
        affinity_z = (affinity - model.affinity_mean) / model.affinity_scale
        logits = (
            user_contribution[:, None]
            + model.action_base_logit[None, :]
            + model.affinity_coefficient * affinity_z
        )
        q_all = expit(logits)
        row_index = np.arange(take)
        q_log = q_all[row_index, action]
        q_mean = np.mean(q_all, axis=1)
        best_action = np.argmax(q_all, axis=1).astype(np.int16)
        q_best = q_all[row_index, best_action]
        challenger_mean = (1.0 - CHALLENGER_EPSILON) * q_best + CHALLENGER_EPSILON * q_mean
        challenger_logged_probability = (
            CHALLENGER_EPSILON * UNIFORM_TARGET_PROBABILITY
            + (1.0 - CHALLENGER_EPSILON) * (action == best_action)
        )

        parsed = pd.to_datetime(selected["timestamp"], utc=True)
        timestamps[filled:stop] = np.asarray(
            parsed.dt.as_unit("ns").astype("int64").to_numpy(), dtype=np.int64
        )
        reward[filled:stop] = np.asarray(selected["click"].to_numpy(), dtype=float)
        propensity[filled:stop] = pscore
        q_logged[filled:stop] = q_log
        uniform_dm[filled:stop] = q_mean
        challenger_dm[filled:stop] = challenger_mean
        uniform_weights[filled:stop] = importance_weights(
            np.full(take, UNIFORM_TARGET_PROBABILITY, dtype=float), pscore
        )
        challenger_weights[filled:stop] = importance_weights(
            challenger_logged_probability.astype(float), pscore
        )
        filled = stop

    if timestamps[0] != int(pd.Timestamp(EVALUATION_START).value):
        raise ValueError("Study 3 evaluation first timestamp does not match frozen lock")
    if timestamps[-1] != int(pd.Timestamp(EVALUATION_END).value):
        raise ValueError("Study 3 evaluation last timestamp does not match frozen lock")
    return base.EvaluationArrays(
        timestamps_ns=timestamps,
        reward=reward,
        propensity=propensity,
        q_logged=q_logged,
        uniform_dm=uniform_dm,
        challenger_dm=challenger_dm,
        uniform_weights=uniform_weights,
        challenger_weights=challenger_weights,
    )


def _user_logit_contribution(frame: Any, model: base.FrozenLinearPolicyModel) -> np.ndarray:
    contribution = np.zeros(len(frame), dtype=float)
    for column_index, column in enumerate(USER_COLUMNS):
        values = frame[column].astype("string").fillna("__MISSING__").astype(str).to_numpy()
        mapping = model.user_level_maps[column_index]
        coefficient = model.user_coefficients[column_index]
        contribution += np.fromiter(
            (coefficient[mapping[value]] if value in mapping else 0.0 for value in values),
            dtype=float,
            count=len(frame),
        )
    return contribution


def _random_reference(archive: ZipFile, *, chunk_size: int) -> dict[str, object]:
    member = "open_bandit_dataset/random/women/women.csv"
    usecols = ["timestamp", "position", "item_id", "click"]
    total_reward = 0.0
    row_count = 0
    actions: set[int] = set()
    with archive.open(member) as handle:
        reader = pd.read_csv(handle, usecols=usecols, chunksize=chunk_size)
        for chunk in reader:
            selected = chunk.loc[chunk["position"] == 1]
            if selected.empty:
                continue
            parsed = pd.to_datetime(selected["timestamp"], utc=True)
            mask = (parsed >= pd.Timestamp(EVALUATION_START)) & (
                parsed <= pd.Timestamp(EVALUATION_END)
            )
            selected = selected.loc[mask]
            if selected.empty:
                continue
            row_count += len(selected)
            total_reward += float(selected["click"].sum())
            actions.update(int(value) for value in selected["item_id"].unique())
    if row_count != RANDOM_REFERENCE_ROWS:
        raise ValueError(
            f"expected {RANDOM_REFERENCE_ROWS} Study 3 Random reference rows, found {row_count}"
        )
    if actions != set(ACTION_IDS):
        raise ValueError("Study 3 Random reference window does not retain all 46 actions")
    return {
        "row_count": row_count,
        "action_count": len(actions),
        "value": total_reward / float(row_count),
        "window_start": EVALUATION_START,
        "window_end": EVALUATION_END,
    }


def protocol_hash_study3(protocol_dir: Path) -> str:
    """Hash the complete frozen Study 3 prospective and qualification chain."""
    names = (
        "study3_selected_source_lock_v2_2.json",
        "study3_temporal_audit_lock_v2_3.json",
        "study3_temporal_split_lock_v2_4.json",
        "study3_training_qualification_lock_v2_5.json",
        QUALIFIED_MODEL_LOCK_FILE,
        AUTHORIZATION_FILE,
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
