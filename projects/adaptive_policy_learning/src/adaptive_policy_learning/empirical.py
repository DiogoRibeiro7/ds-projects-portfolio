"""Frozen empirical Open Bandit Dataset evaluation pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from zipfile import ZipFile

import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from scipy.special import expit
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from adaptive_policy_learning.obd import sha256_file
from adaptive_policy_learning.ope import (
    direct_method,
    doubly_robust,
    importance_weights,
    ips,
    overlap_diagnostics,
    promotion_decision,
    snips,
)

ACTION_IDS = np.arange(80, dtype=np.int16)
USER_COLUMNS = tuple(f"user_feature_{i}" for i in range(4))
AFFINITY_COLUMNS = tuple(f"user-item_affinity_{i}" for i in range(80))
ITEM_CATEGORICAL_COLUMNS = ("item_feature_1", "item_feature_2", "item_feature_3")
PRIMARY_POSITION = 1
UNIFORM_TARGET_PROBABILITY = 1.0 / 80.0
CHALLENGER_EPSILON = 0.10
TRAIN_ROWS = 2_882_936
EVALUATION_ROWS = 1_235_545
RANDOM_REFERENCE_ROWS = 137_634
EVALUATION_START = "2019-11-28 16:55:17.867529+00:00"
EVALUATION_END = "2019-11-30 23:59:59.920907+00:00"
ARCHIVE_SHA256 = "e8ec18196582a5937381a1776382ca940689b90a18d2dcd1fb635be6df614d78"


@dataclass(frozen=True)
class ItemContext:
    """Frozen action features from the archived BTS item context."""

    item_feature_0: np.ndarray
    categorical_codes: np.ndarray
    categorical_levels: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class TrainingArrays:
    """Compact training-only arrays needed to construct the sparse design matrix."""

    user_codes: np.ndarray
    user_levels: tuple[tuple[str, ...], ...]
    item_ids: np.ndarray
    affinity: np.ndarray
    reward: np.ndarray


@dataclass(frozen=True)
class NumericScaling:
    """Training-only numeric preprocessing state."""

    item_median: float
    item_mean: float
    item_scale: float
    affinity_median: float
    affinity_mean: float
    affinity_scale: float


@dataclass(frozen=True)
class FeatureLayout:
    """Column offsets in the frozen sparse logistic-regression design."""

    user_offsets: tuple[int, ...]
    item_offsets: tuple[int, ...]
    item_numeric_column: int
    affinity_column: int
    n_features: int


@dataclass(frozen=True)
class FrozenLinearPolicyModel:
    """Minimal fitted state needed for deterministic candidate-action scoring."""

    user_coefficients: tuple[np.ndarray, ...]
    action_base_logit: np.ndarray
    affinity_coefficient: float
    affinity_median: float
    affinity_mean: float
    affinity_scale: float
    user_level_maps: tuple[Mapping[str, int], ...]
    coefficient_sha256: str
    n_iter: int


@dataclass(frozen=True)
class EvaluationArrays:
    """Row-wise evaluation quantities required for OPE and paired bootstrap."""

    timestamps_ns: np.ndarray
    reward: np.ndarray
    propensity: np.ndarray
    q_logged: np.ndarray
    uniform_dm: np.ndarray
    challenger_dm: np.ndarray
    uniform_weights: np.ndarray
    challenger_weights: np.ndarray


def run_primary_ope(
    archive_path: Path,
    output_path: Path,
    *,
    code_sha: str,
    protocol_dir: Path,
    chunk_size: int = 100_000,
) -> dict[str, object]:
    """Execute the frozen primary OPE study against the official OBD archive."""
    _verify_archive(archive_path)
    design_hash = protocol_hash(protocol_dir)

    with ZipFile(archive_path) as archive:
        item_context = _load_item_context(archive)
        training = _extract_training_arrays(archive, chunk_size=chunk_size)

    scaling = _training_numeric_scaling(training, item_context)
    x_train, layout = _build_training_matrix(training, item_context, scaling)
    model = _fit_reward_model(x_train, training.reward)
    frozen_model = _freeze_linear_model(model, training, item_context, scaling, layout)
    del x_train, training

    if frozen_model.n_iter >= 200:
        raise RuntimeError("reward model reached max_iter=200; empirical OPE is not authorized")

    with ZipFile(archive_path) as archive:
        evaluation = _evaluate_bts(
            archive,
            frozen_model,
            item_context,
            chunk_size=chunk_size,
        )
        random_reference = _random_reference(archive, chunk_size=chunk_size)

    uniform = _policy_summary(
        evaluation,
        evaluation.uniform_weights,
        evaluation.uniform_dm,
    )
    challenger = _policy_summary(
        evaluation,
        evaluation.challenger_weights,
        evaluation.challenger_dm,
    )
    uniform_sensitivity = _clipping_sensitivity(
        evaluation,
        evaluation.uniform_weights,
        evaluation.uniform_dm,
    )
    challenger_sensitivity = _clipping_sensitivity(
        evaluation,
        evaluation.challenger_weights,
        evaluation.challenger_dm,
    )

    bts_observed_value = float(np.mean(evaluation.reward))
    reference_value = float(random_reference["value"])
    benchmark_errors = _benchmark_errors(uniform, reference_value)

    challenger_dr_contribution = (
        evaluation.challenger_dm
        + evaluation.challenger_weights * (evaluation.reward - evaluation.q_logged)
    )
    paired_difference = challenger_dr_contribution - evaluation.reward
    bootstrap = moving_block_bootstrap_mean(
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
        float(bootstrap["lower"]),
        challenger_overlap.ess_fraction,
        minimum_ess_fraction=0.10,
    )

    result: dict[str, object] = {
        "status": "success",
        "code_sha": code_sha,
        "design_hash": design_hash,
        "source_archive_sha256": ARCHIVE_SHA256,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "frozen_sample": {
            "action_count": 80,
            "target_probability": UNIFORM_TARGET_PROBABILITY,
            "bts_training_rows": TRAIN_ROWS,
            "bts_evaluation_rows": EVALUATION_ROWS,
            "random_reference_rows": RANDOM_REFERENCE_ROWS,
            "evaluation_start": EVALUATION_START,
            "evaluation_end": EVALUATION_END,
        },
        "reward_model": {
            "family": "L2 logistic regression",
            "solver": "saga",
            "C": 1.0,
            "max_iter": 200,
            "n_iter": frozen_model.n_iter,
            "coefficient_sha256": frozen_model.coefficient_sha256,
            "n_features": layout.n_features,
        },
        "random_reference": random_reference,
        "uniform_random_benchmark": {
            "estimators": uniform,
            "errors_vs_random_reference": benchmark_errors,
            "overlap": _diagnostics_dict(
                overlap_diagnostics(evaluation.propensity, evaluation.uniform_weights)
            ),
            "clipping_sensitivity": uniform_sensitivity,
        },
        "challenger": {
            "estimators": challenger,
            "bts_observed_value": bts_observed_value,
            "dr_minus_bts_point_difference": float(np.mean(paired_difference)),
            "paired_moving_block_bootstrap": bootstrap,
            "overlap": _diagnostics_dict(challenger_overlap),
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


def protocol_hash(protocol_dir: Path) -> str:
    """Hash the frozen protocol chain in deterministic filename order."""
    names = (
        "design_lock.json",
        "source_contract_amendment_v0_2.json",
        "source_contract_amendment_v0_3.json",
        "source_contract_amendment_v0_4.json",
        "source_contract_amendment_v0_5.json",
        "temporal_split_lock_v0_6.json",
        "empirical_model_lock_v0_7.json",
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


def moving_block_bootstrap_mean(
    timestamps_ns: np.ndarray,
    values: np.ndarray,
    *,
    block_length_hours: int,
    replications: int,
    seed: int,
    confidence_level: float,
) -> dict[str, object]:
    """Paired moving-block bootstrap over chronologically ordered rows."""
    ts = np.asarray(timestamps_ns, dtype=np.int64)
    x = np.asarray(values, dtype=float)
    if ts.ndim != 1 or x.shape != ts.shape or ts.size == 0:
        raise ValueError("timestamps and values must be aligned non-empty vectors")
    if np.any(ts[1:] < ts[:-1]):
        raise ValueError("timestamps must be non-decreasing")
    if block_length_hours <= 0 or replications <= 0:
        raise ValueError("bootstrap block length and replication count must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence level must lie in (0, 1)")

    block_ns = np.int64(block_length_hours) * np.int64(3_600_000_000_000)
    latest_start = ts[-1] - block_ns
    eligible_count = int(np.searchsorted(ts, latest_start, side="right"))
    if eligible_count <= 0:
        raise ValueError("evaluation interval is shorter than the requested block length")

    prefix = np.concatenate((np.array([0.0]), np.cumsum(x, dtype=float)))
    rng = np.random.default_rng(seed)
    replicate_means = np.empty(replications, dtype=float)
    n = int(x.size)

    for replication in range(replications):
        remaining = n
        total = 0.0
        while remaining > 0:
            start = int(rng.integers(0, eligible_count))
            end_time = ts[start] + block_ns
            end = int(np.searchsorted(ts, end_time, side="left"))
            block_count = end - start
            if block_count <= 0:
                raise RuntimeError("moving-block bootstrap produced an empty block")
            take = min(block_count, remaining)
            total += float(prefix[start + take] - prefix[start])
            remaining -= take
        replicate_means[replication] = total / float(n)

    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(replicate_means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "method": "paired moving-block bootstrap",
        "block_length_hours": block_length_hours,
        "block_interval": "[start_timestamp, start_timestamp + 24 hours)",
        "start_sampling": "uniform over eligible evaluation-row start indices",
        "replications": replications,
        "seed": seed,
        "confidence_level": confidence_level,
        "point_estimate": float(np.mean(x)),
        "lower": float(lower),
        "upper": float(upper),
    }


def _verify_archive(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != ARCHIVE_SHA256:
        raise ValueError(f"archive SHA256 mismatch: expected {ARCHIVE_SHA256}, got {actual}")


def _load_item_context(archive: ZipFile) -> ItemContext:
    member = "open_bandit_dataset/bts/all/item_context.csv"
    with archive.open(member) as handle:
        frame = pd.read_csv(handle, index_col=0)
    if frame["item_id"].tolist() != list(range(80)):
        raise ValueError("BTS item-context catalog must be exact IDs 0..79")

    item0 = np.asarray(
        pd.to_numeric(frame["item_feature_0"], errors="coerce").to_numpy(), dtype=float
    )
    categorical_codes = np.empty((80, len(ITEM_CATEGORICAL_COLUMNS)), dtype=np.int16)
    categorical_levels: list[tuple[str, ...]] = []
    for column_index, column in enumerate(ITEM_CATEGORICAL_COLUMNS):
        values = _categorical_values(frame[column])
        levels = tuple(sorted(set(values.tolist())))
        mapping = {value: index for index, value in enumerate(levels)}
        categorical_codes[:, column_index] = np.fromiter(
            (mapping[str(value)] for value in values),
            dtype=np.int16,
            count=80,
        )
        categorical_levels.append(levels)
    return ItemContext(
        item_feature_0=item0,
        categorical_codes=categorical_codes,
        categorical_levels=tuple(categorical_levels),
    )


def _extract_training_arrays(archive: ZipFile, *, chunk_size: int) -> TrainingArrays:
    member = "open_bandit_dataset/bts/all/all.csv"
    usecols = ["position", "item_id", "click", *USER_COLUMNS, *AFFINITY_COLUMNS]
    user_codes = np.empty((TRAIN_ROWS, len(USER_COLUMNS)), dtype=np.int16)
    item_ids = np.empty(TRAIN_ROWS, dtype=np.int16)
    affinity = np.empty(TRAIN_ROWS, dtype=float)
    reward = np.empty(TRAIN_ROWS, dtype=np.uint8)
    level_maps: list[dict[str, int]] = [dict() for _ in USER_COLUMNS]

    filled = 0
    with archive.open(member) as handle:
        reader = pd.read_csv(handle, usecols=usecols, chunksize=chunk_size)
        for chunk in reader:
            selected = chunk.loc[chunk["position"] == PRIMARY_POSITION]
            if selected.empty:
                continue
            take = min(len(selected), TRAIN_ROWS - filled)
            if take <= 0:
                break
            selected = selected.iloc[:take]
            stop = filled + take
            ids = np.asarray(selected["item_id"].to_numpy(), dtype=np.int16)
            if np.any((ids < 0) | (ids >= 80)):
                raise ValueError("training action outside archived catalog 0..79")
            item_ids[filled:stop] = ids
            reward[filled:stop] = np.asarray(selected["click"].to_numpy(), dtype=np.uint8)

            affinity_matrix = np.asarray(
                selected.loc[:, AFFINITY_COLUMNS].to_numpy(), dtype=float
            )
            affinity[filled:stop] = affinity_matrix[np.arange(take), ids]

            for column_index, column in enumerate(USER_COLUMNS):
                values = _categorical_values(selected[column])
                mapping = level_maps[column_index]
                for value in pd.unique(values):
                    key = str(value)
                    if key not in mapping:
                        mapping[key] = len(mapping)
                user_codes[filled:stop, column_index] = np.fromiter(
                    (mapping[str(value)] for value in values),
                    dtype=np.int16,
                    count=take,
                )
            filled = stop
            if filled == TRAIN_ROWS:
                break

    if filled != TRAIN_ROWS:
        raise ValueError(f"expected {TRAIN_ROWS} training rows, found {filled}")

    levels_out: list[tuple[str, ...]] = []
    for column_index, old_mapping in enumerate(level_maps):
        levels = tuple(sorted(old_mapping))
        new_mapping = {value: index for index, value in enumerate(levels)}
        remap = np.empty(len(old_mapping), dtype=np.int16)
        for value, old_code in old_mapping.items():
            remap[old_code] = new_mapping[value]
        user_codes[:, column_index] = remap[user_codes[:, column_index]]
        levels_out.append(levels)

    return TrainingArrays(
        user_codes=user_codes,
        user_levels=tuple(levels_out),
        item_ids=item_ids,
        affinity=affinity,
        reward=reward,
    )


def _training_numeric_scaling(
    training: TrainingArrays,
    item_context: ItemContext,
) -> NumericScaling:
    logged_item0 = item_context.item_feature_0[training.item_ids]
    item_median, item_mean, item_scale = _median_mean_scale(logged_item0)
    affinity_median, affinity_mean, affinity_scale = _median_mean_scale(training.affinity)
    return NumericScaling(
        item_median=item_median,
        item_mean=item_mean,
        item_scale=item_scale,
        affinity_median=affinity_median,
        affinity_mean=affinity_mean,
        affinity_scale=affinity_scale,
    )


def _median_mean_scale(values: np.ndarray) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError("numeric feature has no finite training values")
    median = float(np.median(finite))
    imputed = np.where(np.isfinite(array), array, median)
    mean = float(np.mean(imputed))
    scale = float(np.std(imputed, ddof=0))
    if not math.isfinite(scale) or scale == 0.0:
        scale = 1.0
    return median, mean, scale


def _build_training_matrix(
    training: TrainingArrays,
    item_context: ItemContext,
    scaling: NumericScaling,
) -> tuple[Any, FeatureLayout]:
    user_offsets: list[int] = []
    item_offsets: list[int] = []
    offset = 0
    for levels in training.user_levels:
        user_offsets.append(offset)
        offset += len(levels)
    for levels in item_context.categorical_levels:
        item_offsets.append(offset)
        offset += len(levels)
    item_numeric_column = offset
    affinity_column = offset + 1
    n_features = offset + 2
    layout = FeatureLayout(
        user_offsets=tuple(user_offsets),
        item_offsets=tuple(item_offsets),
        item_numeric_column=item_numeric_column,
        affinity_column=affinity_column,
        n_features=n_features,
    )

    n = TRAIN_ROWS
    entries_per_row = len(USER_COLUMNS) + len(ITEM_CATEGORICAL_COLUMNS) + 2
    nnz = n * entries_per_row
    indices = np.empty(nnz, dtype=np.int32)
    data = np.ones(nnz, dtype=float)
    indptr = np.arange(0, nnz + 1, entries_per_row, dtype=np.int64)

    slot = 0
    for column_index, column_offset in enumerate(layout.user_offsets):
        indices[slot::entries_per_row] = column_offset + training.user_codes[:, column_index]
        slot += 1
    for column_index, column_offset in enumerate(layout.item_offsets):
        indices[slot::entries_per_row] = (
            column_offset + item_context.categorical_codes[training.item_ids, column_index]
        )
        slot += 1

    logged_item0 = item_context.item_feature_0[training.item_ids]
    logged_item0 = np.where(np.isfinite(logged_item0), logged_item0, scaling.item_median)
    indices[slot::entries_per_row] = layout.item_numeric_column
    data[slot::entries_per_row] = (logged_item0 - scaling.item_mean) / scaling.item_scale
    slot += 1

    affinity = np.where(
        np.isfinite(training.affinity),
        training.affinity,
        scaling.affinity_median,
    )
    indices[slot::entries_per_row] = layout.affinity_column
    data[slot::entries_per_row] = (
        affinity - scaling.affinity_mean
    ) / scaling.affinity_scale

    matrix = sparse.csr_matrix(
        (data, indices, indptr),
        shape=(n, n_features),
        dtype=float,
    )
    return matrix, layout


def _fit_reward_model(x_train: Any, y_train: np.ndarray) -> Any:
    model = LogisticRegression(
        solver="saga",
        penalty="l2",
        C=1.0,
        fit_intercept=True,
        max_iter=200,
        tol=0.0001,
        random_state=20260831,
        class_weight=None,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_train, y_train)
    if any(issubclass(warning.category, ConvergenceWarning) for warning in caught):
        raise RuntimeError("reward model failed to converge at frozen max_iter=200")
    if model.classes_.tolist() != [0, 1]:
        raise ValueError("reward model must fit binary click classes [0, 1]")
    return model


def _freeze_linear_model(
    model: Any,
    training: TrainingArrays,
    item_context: ItemContext,
    scaling: NumericScaling,
    layout: FeatureLayout,
) -> FrozenLinearPolicyModel:
    coefficient = np.asarray(model.coef_[0], dtype=float)
    if coefficient.size != layout.n_features:
        raise ValueError("reward-model coefficient dimension does not match frozen layout")

    user_coefficients: list[np.ndarray] = []
    for column_index, offset in enumerate(layout.user_offsets):
        width = len(training.user_levels[column_index])
        user_coefficients.append(coefficient[offset : offset + width].copy())

    item0 = np.where(
        np.isfinite(item_context.item_feature_0),
        item_context.item_feature_0,
        scaling.item_median,
    )
    action_base = np.full(80, float(model.intercept_[0]), dtype=float)
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
    return FrozenLinearPolicyModel(
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


def _evaluate_bts(
    archive: ZipFile,
    model: FrozenLinearPolicyModel,
    item_context: ItemContext,
    *,
    chunk_size: int,
) -> EvaluationArrays:
    del item_context
    member = "open_bandit_dataset/bts/all/all.csv"
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

    position_one_seen = 0
    filled = 0
    with archive.open(member) as handle:
        reader = pd.read_csv(handle, usecols=usecols, chunksize=chunk_size)
        for chunk in reader:
            selected = chunk.loc[chunk["position"] == PRIMARY_POSITION]
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
            take = len(selected)
            if take == 0:
                continue
            stop = filled + take

            action = np.asarray(selected["item_id"].to_numpy(), dtype=np.int16)
            if np.any((action < 0) | (action >= 80)):
                raise ValueError("evaluation action outside archived catalog 0..79")
            pscore = np.asarray(selected["propensity_score"].to_numpy(), dtype=float)
            if np.any(~np.isfinite(pscore)) or np.any(pscore <= 0.0):
                raise ValueError("evaluation propensities must be finite and positive")

            user_contribution = _user_logit_contribution(selected, model)
            affinity = np.asarray(
                selected.loc[:, AFFINITY_COLUMNS].to_numpy(), dtype=float
            )
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
            challenger_mean = (1.0 - CHALLENGER_EPSILON) * q_best + (
                CHALLENGER_EPSILON * q_mean
            )
            challenger_logged_probability = (
                CHALLENGER_EPSILON * UNIFORM_TARGET_PROBABILITY
                + (1.0 - CHALLENGER_EPSILON) * (action == best_action)
            )

            timestamps[filled:stop] = np.asarray(
                pd.to_datetime(selected["timestamp"], utc=True).astype("int64").to_numpy(),
                dtype=np.int64,
            )
            reward[filled:stop] = np.asarray(selected["click"].to_numpy(), dtype=float)
            propensity[filled:stop] = pscore
            q_logged[filled:stop] = q_log
            uniform_dm[filled:stop] = q_mean
            challenger_dm[filled:stop] = challenger_mean
            uniform_weights[filled:stop] = importance_weights(
                np.full(take, UNIFORM_TARGET_PROBABILITY, dtype=float),
                pscore,
            )
            challenger_weights[filled:stop] = importance_weights(
                challenger_logged_probability.astype(float),
                pscore,
            )
            filled = stop
            if filled == EVALUATION_ROWS:
                break

    if filled != EVALUATION_ROWS:
        raise ValueError(f"expected {EVALUATION_ROWS} evaluation rows, found {filled}")
    if timestamps[0] != int(pd.Timestamp(EVALUATION_START).value):
        raise ValueError("evaluation first timestamp does not match frozen temporal lock")
    if timestamps[-1] != int(pd.Timestamp(EVALUATION_END).value):
        raise ValueError("evaluation last timestamp does not match frozen temporal lock")
    return EvaluationArrays(
        timestamps_ns=timestamps,
        reward=reward,
        propensity=propensity,
        q_logged=q_logged,
        uniform_dm=uniform_dm,
        challenger_dm=challenger_dm,
        uniform_weights=uniform_weights,
        challenger_weights=challenger_weights,
    )


def _user_logit_contribution(
    frame: Any,
    model: FrozenLinearPolicyModel,
) -> np.ndarray:
    contribution = np.zeros(len(frame), dtype=float)
    for column_index, column in enumerate(USER_COLUMNS):
        values = _categorical_values(frame[column])
        mapping = model.user_level_maps[column_index]
        coefficient = model.user_coefficients[column_index]
        contribution += np.fromiter(
            (
                coefficient[mapping[str(value)]] if str(value) in mapping else 0.0
                for value in values
            ),
            dtype=float,
            count=len(frame),
        )
    return contribution


def _random_reference(archive: ZipFile, *, chunk_size: int) -> dict[str, object]:
    member = "open_bandit_dataset/random/all/all.csv"
    usecols = ["timestamp", "position", "item_id", "click"]
    total_reward = 0.0
    row_count = 0
    actions: set[int] = set()
    with archive.open(member) as handle:
        reader = pd.read_csv(handle, usecols=usecols, chunksize=chunk_size)
        for chunk in reader:
            selected = chunk.loc[chunk["position"] == PRIMARY_POSITION]
            if selected.empty:
                continue
            timestamp = selected["timestamp"].astype(str)
            selected = selected.loc[
                (timestamp >= EVALUATION_START) & (timestamp <= EVALUATION_END)
            ]
            if selected.empty:
                continue
            row_count += len(selected)
            total_reward += float(selected["click"].sum())
            actions.update(int(value) for value in selected["item_id"].unique())

    if row_count != RANDOM_REFERENCE_ROWS:
        raise ValueError(
            f"expected {RANDOM_REFERENCE_ROWS} Random reference rows, found {row_count}"
        )
    if actions != set(range(80)):
        raise ValueError("Random reference window does not retain all 80 actions")
    return {
        "row_count": row_count,
        "action_count": len(actions),
        "value": total_reward / float(row_count),
        "window_start": EVALUATION_START,
        "window_end": EVALUATION_END,
    }


def _policy_summary(
    evaluation: EvaluationArrays,
    weights: np.ndarray,
    dm_values: np.ndarray,
) -> dict[str, float]:
    return {
        "ips": ips(evaluation.reward, weights),
        "snips": snips(evaluation.reward, weights),
        "direct_method": direct_method(dm_values),
        "doubly_robust": doubly_robust(
            evaluation.reward,
            weights,
            evaluation.q_logged,
            dm_values,
        ),
    }


def _clipping_sensitivity(
    evaluation: EvaluationArrays,
    primary_weights: np.ndarray,
    dm_values: np.ndarray,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for cap in (5.0, 10.0, 20.0):
        clipped = np.minimum(primary_weights, cap)
        result[str(int(cap))] = {
            "ips": ips(evaluation.reward, clipped),
            "snips": snips(evaluation.reward, clipped),
            "direct_method": direct_method(dm_values),
            "doubly_robust": doubly_robust(
                evaluation.reward,
                clipped,
                evaluation.q_logged,
                dm_values,
            ),
            "overlap": _diagnostics_dict(
                overlap_diagnostics(evaluation.propensity, clipped)
            ),
        }
    return result


def _benchmark_errors(
    estimators: Mapping[str, float],
    reference: float,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, estimate in estimators.items():
        absolute = abs(float(estimate) - reference)
        result[name] = {
            "absolute_error": absolute,
            "relative_error": absolute / reference if reference > 0.0 else None,
        }
    return result


def _diagnostics_dict(value: Any) -> dict[str, object]:
    return {
        "n": int(value.n),
        "ess": float(value.ess),
        "ess_fraction": float(value.ess_fraction),
        "propensity_min": float(value.propensity_min),
        "propensity_median": float(value.propensity_median),
        "propensity_max": float(value.propensity_max),
        "weight_p50": float(value.weight_p50),
        "weight_p90": float(value.weight_p90),
        "weight_p95": float(value.weight_p95),
        "weight_p99": float(value.weight_p99),
        "weight_max": float(value.weight_max),
    }


def _categorical_values(series: Any) -> np.ndarray:
    values = series.astype("string").fillna("__MISSING__")
    return np.asarray(values.astype(str).to_numpy(), dtype=object)
