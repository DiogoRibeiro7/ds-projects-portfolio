"""Training-only qualification gate for Adaptive Policy Learning Study 3."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import warnings
from dataclasses import dataclass
from io import TextIOWrapper
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from adaptive_policy_learning.obd import sha256_file

CAMPAIGN = "women"
ACTION_COUNT = 46
ACTION_IDS = tuple(range(ACTION_COUNT))
USER_COLUMNS = tuple(f"user_feature_{i}" for i in range(4))
AFFINITY_COLUMNS = tuple(f"user-item_affinity_{i}" for i in range(ACTION_COUNT))
ITEM_CATEGORICAL_COLUMNS = ("item_feature_1", "item_feature_2", "item_feature_3")
PRIMARY_POSITION = "1"
TRAIN_ROWS = 1_811_697
TRAIN_LAST_TIMESTAMP = "2019-11-28 15:23:48.607846+00:00"
EVALUATION_FIRST_TIMESTAMP = "2019-11-28 15:23:48.989271+00:00"
ARCHIVE_SHA256 = "e8ec18196582a5937381a1776382ca940689b90a18d2dcd1fb635be6df614d78"
MAX_ITER = 100


@dataclass(frozen=True)
class ItemContext:
    item_feature_0: np.ndarray
    categorical_codes: np.ndarray
    categorical_levels: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class TrainingArrays:
    user_codes: np.ndarray
    user_levels: tuple[tuple[str, ...], ...]
    item_ids: np.ndarray
    affinity: np.ndarray
    reward: np.ndarray
    final_timestamp: str


@dataclass(frozen=True)
class NumericScaling:
    item_median: float
    item_mean: float
    item_scale: float
    affinity_median: float
    affinity_mean: float
    affinity_scale: float


@dataclass(frozen=True)
class FeatureLayout:
    user_offsets: tuple[int, ...]
    item_offsets: tuple[int, ...]
    item_numeric_column: int
    affinity_column: int
    n_features: int


def run_study3_training_qualification(
    archive_path: Path,
    output_path: Path,
    *,
    code_sha: str,
    protocol_dir: Path,
) -> dict[str, object]:
    """Fit the frozen Study 3 reward model using BTS training outcomes only."""
    _verify_archive(archive_path)
    design_hash = protocol_hash(protocol_dir)

    with ZipFile(archive_path) as archive:
        item_context = _load_item_context(archive)
        training = _extract_training_arrays(archive)

    scaling = _training_numeric_scaling(training, item_context)
    x_train, layout = _build_training_matrix(training, item_context, scaling)
    model, captured = _fit_reward_model(x_train, training.reward)
    coefficient_sha256 = _coefficient_sha256(model)

    result: dict[str, object] = {
        "status": "training_gate_passed",
        "study": "Adaptive Policy Learning Study 3",
        "protocol_version": "2.5-study3-training-only-qualification",
        "code_sha": code_sha,
        "protocol_hash": design_hash,
        "source_archive_sha256": ARCHIVE_SHA256,
        "campaign": CAMPAIGN,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "training": {
            "rows": TRAIN_ROWS,
            "final_timestamp": training.final_timestamp,
            "action_count": ACTION_COUNT,
            "n_features": layout.n_features,
            "solver": "newton-cholesky",
            "regularization": "l2",
            "l1_ratio": 0.0,
            "C": 1.0,
            "tol": 0.0001,
            "max_iter": MAX_ITER,
            "n_iter": int(model.n_iter_[0]),
            "coefficient_sha256": coefficient_sha256,
            "warnings": captured,
        },
        "evaluation_outcomes_loaded": False,
        "random_reference_outcomes_loaded": False,
        "ope_estimates_computed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _verify_archive(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != ARCHIVE_SHA256:
        raise ValueError(f"archive SHA256 mismatch: expected {ARCHIVE_SHA256}, got {actual}")


def _load_item_context(archive: ZipFile) -> ItemContext:
    member = "open_bandit_dataset/bts/women/item_context.csv"
    with archive.open(member) as handle:
        frame = pd.read_csv(handle, index_col=0)
    if frame["item_id"].tolist() != list(ACTION_IDS):
        raise ValueError("Study 3 item-context catalog must be exact IDs 0..45")

    item0 = np.asarray(pd.to_numeric(frame["item_feature_0"], errors="coerce"), dtype=float)
    categorical_codes = np.empty((ACTION_COUNT, len(ITEM_CATEGORICAL_COLUMNS)), dtype=np.int16)
    categorical_levels: list[tuple[str, ...]] = []
    for column_index, column in enumerate(ITEM_CATEGORICAL_COLUMNS):
        values = _categorical_values(frame[column])
        levels = tuple(sorted(set(values)))
        mapping = {value: index for index, value in enumerate(levels)}
        categorical_codes[:, column_index] = np.fromiter(
            (mapping[value] for value in values), dtype=np.int16, count=ACTION_COUNT
        )
        categorical_levels.append(levels)
    return ItemContext(item0, categorical_codes, tuple(categorical_levels))


def _extract_training_arrays(archive: ZipFile) -> TrainingArrays:
    """Read click only after a row is proven to belong to frozen position-1 training."""
    member = "open_bandit_dataset/bts/women/women.csv"
    user_codes = np.empty((TRAIN_ROWS, len(USER_COLUMNS)), dtype=np.int16)
    item_ids = np.empty(TRAIN_ROWS, dtype=np.int16)
    affinity = np.empty(TRAIN_ROWS, dtype=float)
    reward = np.empty(TRAIN_ROWS, dtype=np.uint8)
    level_maps: list[dict[str, int]] = [{} for _ in USER_COLUMNS]

    filled = 0
    final_timestamp = ""
    with archive.open(member) as raw:
        reader = csv.DictReader(TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
        for row in reader:
            if str(row["position"]).strip() != PRIMARY_POSITION:
                continue
            if filled == TRAIN_ROWS:
                # This is the first evaluation position-1 row. Its click field is deliberately not accessed.
                timestamp = str(row["timestamp"]).strip()
                if timestamp != EVALUATION_FIRST_TIMESTAMP:
                    raise ValueError("first evaluation timestamp does not match frozen Study 3 boundary")
                break

            timestamp = str(row["timestamp"]).strip()
            item_id = int(row["item_id"])
            if item_id not in ACTION_IDS:
                raise ValueError("training action outside frozen Study 3 catalog 0..45")
            click = int(row["click"])
            if click not in (0, 1):
                raise ValueError("training click must be binary")

            item_ids[filled] = item_id
            reward[filled] = click
            affinity[filled] = _finite_or_nan(row[AFFINITY_COLUMNS[item_id]])
            for column_index, column in enumerate(USER_COLUMNS):
                value = _categorical_value(row[column])
                mapping = level_maps[column_index]
                if value not in mapping:
                    mapping[value] = len(mapping)
                user_codes[filled, column_index] = mapping[value]
            final_timestamp = timestamp
            filled += 1
        else:
            raise ValueError("BTS women log ended before first frozen evaluation row")

    if filled != TRAIN_ROWS:
        raise ValueError(f"expected {TRAIN_ROWS} Study 3 training rows, found {filled}")
    if final_timestamp != TRAIN_LAST_TIMESTAMP:
        raise ValueError("last training timestamp does not match frozen Study 3 boundary")

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
        final_timestamp=final_timestamp,
    )


def _categorical_value(value: object) -> str:
    if value is None or str(value).strip() == "" or str(value).lower() == "nan":
        return "__MISSING__"
    return str(value)


def _categorical_values(series: pd.Series) -> list[str]:
    return [_categorical_value(value) for value in series.tolist()]


def _finite_or_nan(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return math.nan
    return parsed if math.isfinite(parsed) else math.nan


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


def _training_numeric_scaling(training: TrainingArrays, item_context: ItemContext) -> NumericScaling:
    logged_item0 = item_context.item_feature_0[training.item_ids]
    item_median, item_mean, item_scale = _median_mean_scale(logged_item0)
    affinity_median, affinity_mean, affinity_scale = _median_mean_scale(training.affinity)
    return NumericScaling(
        item_median, item_mean, item_scale, affinity_median, affinity_mean, affinity_scale
    )


def _build_training_matrix(
    training: TrainingArrays, item_context: ItemContext, scaling: NumericScaling
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
    layout = FeatureLayout(tuple(user_offsets), tuple(item_offsets), offset, offset + 1, offset + 2)

    entries_per_row = len(USER_COLUMNS) + len(ITEM_CATEGORICAL_COLUMNS) + 2
    nnz = TRAIN_ROWS * entries_per_row
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

    affinity_values = np.where(np.isfinite(training.affinity), training.affinity, scaling.affinity_median)
    indices[slot::entries_per_row] = layout.affinity_column
    data[slot::entries_per_row] = (affinity_values - scaling.affinity_mean) / scaling.affinity_scale

    matrix = sparse.csr_matrix(
        (data, indices, indptr), shape=(TRAIN_ROWS, layout.n_features), dtype=float
    )
    return matrix, layout


def _fit_reward_model(x_train: Any, y_train: np.ndarray) -> tuple[LogisticRegression, list[str]]:
    model = LogisticRegression(
        solver="newton-cholesky",
        l1_ratio=0.0,
        C=1.0,
        fit_intercept=True,
        max_iter=MAX_ITER,
        tol=0.0001,
        class_weight=None,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(x_train, y_train)

    messages = [f"{warning.category.__name__}: {warning.message}" for warning in caught]
    if any(issubclass(warning.category, ConvergenceWarning) for warning in caught):
        raise RuntimeError("Study 3 training gate failed: ConvergenceWarning")
    if any(
        token in str(warning.message).lower()
        for warning in caught
        for token in ("fallback", "ill-conditioned", "singular", "hessian")
    ):
        raise RuntimeError("Study 3 training gate failed: numerical fallback or instability warning")
    if model.classes_.tolist() != [0, 1]:
        raise RuntimeError("Study 3 training gate failed: classes must be [0, 1]")
    if int(model.n_iter_[0]) >= MAX_ITER:
        raise RuntimeError("Study 3 training gate failed: reached max_iter=100")
    if not np.all(np.isfinite(model.coef_)) or not np.all(np.isfinite(model.intercept_)):
        raise RuntimeError("Study 3 training gate failed: non-finite coefficients")
    return model, messages


def _coefficient_sha256(model: LogisticRegression) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(model.intercept_, dtype=np.float64).tobytes())
    digest.update(np.asarray(model.coef_, dtype=np.float64).tobytes())
    return digest.hexdigest()


def protocol_hash(protocol_dir: Path) -> str:
    """Hash the frozen Study 3 training protocol chain."""
    names = (
        "study3_selected_source_lock_v2_2.json",
        "study3_temporal_audit_lock_v2_3.json",
        "study3_temporal_split_lock_v2_4.json",
        "study3_training_qualification_lock_v2_5.json",
    )
    digest = hashlib.sha256()
    for name in names:
        path = protocol_dir / name
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
