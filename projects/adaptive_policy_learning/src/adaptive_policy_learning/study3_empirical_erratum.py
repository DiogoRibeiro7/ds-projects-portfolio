"""Execution errata and retry provenance for Study 3 primary OPE."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.special import expit

from adaptive_policy_learning import empirical as base
from adaptive_policy_learning import study3_empirical as original
from adaptive_policy_learning.ope import importance_weights
from adaptive_policy_learning.study3_training import (
    ACTION_COUNT,
    ACTION_IDS,
    AFFINITY_COLUMNS,
    USER_COLUMNS,
)

PROTOCOL_VERSION = "2.10-study3-primary-ope-execution-retry"
ERRATUM_FILE = "study3_primary_ope_execution_erratum_v2_9.json"
RETRY_FILE = "study3_primary_ope_execution_retry_v2_10.json"


def _parse_mixed_utc(values: pd.Series) -> pd.Series:
    """Parse the mixed fractional/non-fractional ISO-8601 forms in frozen OBD logs."""
    return pd.to_datetime(values, utc=True, format="mixed")


def _validate_evaluation_window_without_outcomes(
    archive: ZipFile,
    *,
    chunk_size: int,
) -> None:
    """Validate frozen BTS boundaries without including click in the reader."""
    first_timestamp: pd.Timestamp | None = None
    last_timestamp: pd.Timestamp | None = None
    for selected in original._iter_evaluation_slices(
        archive,
        usecols=("timestamp", "position"),
        chunk_size=chunk_size,
    ):
        parsed = _parse_mixed_utc(selected["timestamp"])
        if first_timestamp is None:
            first_timestamp = parsed.iloc[0]
        last_timestamp = parsed.iloc[-1]
    if first_timestamp != pd.Timestamp(original.EVALUATION_START):
        raise ValueError("Study 3 evaluation first timestamp does not match frozen lock")
    if last_timestamp != pd.Timestamp(original.EVALUATION_END):
        raise ValueError("Study 3 evaluation last timestamp does not match frozen lock")


def _evaluate_bts(
    archive: ZipFile,
    model: base.FrozenLinearPolicyModel,
    *,
    chunk_size: int,
) -> base.EvaluationArrays:
    """Evaluate the frozen BTS window with mixed-format timestamp parsing only."""
    usecols = (
        "timestamp",
        "position",
        "item_id",
        "click",
        "propensity_score",
        *USER_COLUMNS,
        *AFFINITY_COLUMNS,
    )
    timestamps = np.empty(original.EVALUATION_ROWS, dtype=np.int64)
    reward = np.empty(original.EVALUATION_ROWS, dtype=float)
    propensity = np.empty(original.EVALUATION_ROWS, dtype=float)
    q_logged = np.empty(original.EVALUATION_ROWS, dtype=float)
    uniform_dm = np.empty(original.EVALUATION_ROWS, dtype=float)
    challenger_dm = np.empty(original.EVALUATION_ROWS, dtype=float)
    uniform_weights = np.empty(original.EVALUATION_ROWS, dtype=float)
    challenger_weights = np.empty(original.EVALUATION_ROWS, dtype=float)

    filled = 0
    for selected in original._iter_evaluation_slices(
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

        user_contribution = original._user_logit_contribution(selected, model)
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
        challenger_mean = (
            (1.0 - original.CHALLENGER_EPSILON) * q_best
            + original.CHALLENGER_EPSILON * q_mean
        )
        challenger_logged_probability = (
            original.CHALLENGER_EPSILON * original.UNIFORM_TARGET_PROBABILITY
            + (1.0 - original.CHALLENGER_EPSILON) * (action == best_action)
        )

        parsed = _parse_mixed_utc(selected["timestamp"])
        timestamps[filled:stop] = np.asarray(
            parsed.dt.as_unit("ns").astype("int64").to_numpy(), dtype=np.int64
        )
        reward[filled:stop] = np.asarray(selected["click"].to_numpy(), dtype=float)
        propensity[filled:stop] = pscore
        q_logged[filled:stop] = q_log
        uniform_dm[filled:stop] = q_mean
        challenger_dm[filled:stop] = challenger_mean
        uniform_weights[filled:stop] = importance_weights(
            np.full(take, original.UNIFORM_TARGET_PROBABILITY, dtype=float), pscore
        )
        challenger_weights[filled:stop] = importance_weights(
            challenger_logged_probability.astype(float), pscore
        )
        filled = stop

    if timestamps[0] != int(pd.Timestamp(original.EVALUATION_START).value):
        raise ValueError("Study 3 evaluation first timestamp does not match frozen lock")
    if timestamps[-1] != int(pd.Timestamp(original.EVALUATION_END).value):
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


def _random_reference(archive: ZipFile, *, chunk_size: int) -> dict[str, object]:
    """Read the frozen Random-reference window with mixed-format timestamp parsing."""
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
            parsed = _parse_mixed_utc(selected["timestamp"])
            mask = (parsed >= pd.Timestamp(original.EVALUATION_START)) & (
                parsed <= pd.Timestamp(original.EVALUATION_END)
            )
            selected = selected.loc[mask]
            if selected.empty:
                continue
            row_count += len(selected)
            total_reward += float(selected["click"].sum())
            actions.update(int(value) for value in selected["item_id"].unique())
    if row_count != original.RANDOM_REFERENCE_ROWS:
        raise ValueError(
            f"expected {original.RANDOM_REFERENCE_ROWS} Study 3 Random reference rows, "
            f"found {row_count}"
        )
    if actions != set(ACTION_IDS):
        raise ValueError("Study 3 Random reference window does not retain all 46 actions")
    return {
        "row_count": row_count,
        "action_count": len(actions),
        "value": total_reward / float(row_count),
        "window_start": original.EVALUATION_START,
        "window_end": original.EVALUATION_END,
    }


def protocol_hash_study3_erratum(protocol_dir: Path) -> str:
    """Hash the frozen Study 3 chain including execution erratum and retry provenance."""
    names = (
        "study3_selected_source_lock_v2_2.json",
        "study3_temporal_audit_lock_v2_3.json",
        "study3_temporal_split_lock_v2_4.json",
        "study3_training_qualification_lock_v2_5.json",
        original.QUALIFIED_MODEL_LOCK_FILE,
        original.AUTHORIZATION_FILE,
        ERRATUM_FILE,
        RETRY_FILE,
    )
    digest = hashlib.sha256()
    for name in names:
        path = protocol_dir / name
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def run_study3_primary_ope_erratum(
    archive_path: Path,
    output_path: Path,
    *,
    code_sha: str,
    protocol_dir: Path,
    chunk_size: int = 100_000,
) -> dict[str, object]:
    """Run Study 3 with the timestamp erratum and recorded execution retry only."""
    old_validate = original._validate_evaluation_window_without_outcomes
    old_evaluate = original._evaluate_bts
    old_random = original._random_reference
    old_hash = original.protocol_hash_study3
    original._validate_evaluation_window_without_outcomes = (
        _validate_evaluation_window_without_outcomes
    )
    original._evaluate_bts = _evaluate_bts
    original._random_reference = _random_reference
    original.protocol_hash_study3 = protocol_hash_study3_erratum
    try:
        result = original.run_study3_primary_ope(
            archive_path,
            output_path,
            code_sha=code_sha,
            protocol_dir=protocol_dir,
            chunk_size=chunk_size,
        )
    finally:
        original._validate_evaluation_window_without_outcomes = old_validate
        original._evaluate_bts = old_evaluate
        original._random_reference = old_random
        original.protocol_hash_study3 = old_hash

    result["protocol_version"] = PROTOCOL_VERSION
    result["execution_erratum"] = {
        "file": ERRATUM_FILE,
        "failed_workflow_run_id": 34035420561,
        "scientific_design_changes": False,
        "timestamp_parser": "pandas format='mixed', utc=true",
    }
    result["execution_retry"] = {
        "file": RETRY_FILE,
        "failed_workflow_run_id": 34038554590,
        "scientific_design_changes": False,
        "reason": "static Ruff gate failed before dataset download or OPE execution",
    }
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result
