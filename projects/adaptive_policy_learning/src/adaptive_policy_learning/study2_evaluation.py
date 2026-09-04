"""Study 2 evaluation compatibility shim for pandas 3 datetime resolution."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.special import expit

from adaptive_policy_learning import empirical as base
from adaptive_policy_learning.ope import importance_weights


def _timestamps_to_ns(values: pd.Series) -> np.ndarray:
    """Convert timestamp strings to explicit nanosecond epoch integers."""
    parsed = pd.to_datetime(values, utc=True)
    return np.asarray(parsed.dt.as_unit("ns").astype("int64").to_numpy(), dtype=np.int64)


def _iter_evaluation_slices(
    archive: ZipFile,
    *,
    usecols: Sequence[str],
    chunk_size: int,
) -> Iterator[pd.DataFrame]:
    """Yield the frozen BTS evaluation rows using one shared selection contract."""
    if "position" not in usecols:
        raise ValueError("evaluation slice reader requires the position column")

    member = "open_bandit_dataset/bts/all/all.csv"
    position_one_seen = 0
    filled = 0
    with archive.open(member) as handle:
        reader = pd.read_csv(handle, usecols=list(usecols), chunksize=chunk_size)
        for chunk in reader:
            selected = chunk.loc[chunk["position"] == base.PRIMARY_POSITION]
            if selected.empty:
                continue

            selected_count = len(selected)
            first_eval = max(0, base.TRAIN_ROWS - position_one_seen)
            position_one_seen += selected_count
            if first_eval >= selected_count:
                continue

            selected = selected.iloc[first_eval:]
            remaining = base.EVALUATION_ROWS - filled
            if len(selected) > remaining:
                selected = selected.iloc[:remaining]
            if selected.empty:
                continue

            filled += len(selected)
            yield selected
            if filled == base.EVALUATION_ROWS:
                break

    if filled != base.EVALUATION_ROWS:
        raise ValueError(
            f"expected {base.EVALUATION_ROWS} evaluation rows, found {filled}"
        )


def _validate_evaluation_window_without_outcomes(
    archive: ZipFile,
    *,
    chunk_size: int,
) -> None:
    """Validate the frozen evaluation boundaries before reading click outcomes."""
    first_timestamp: pd.Timestamp | None = None
    last_timestamp: pd.Timestamp | None = None

    for selected in _iter_evaluation_slices(
        archive,
        usecols=("timestamp", "position"),
        chunk_size=chunk_size,
    ):
        parsed = pd.to_datetime(selected["timestamp"], utc=True)
        if first_timestamp is None:
            first_timestamp = parsed.iloc[0]
        last_timestamp = parsed.iloc[-1]

    if first_timestamp != pd.Timestamp(base.EVALUATION_START):
        raise ValueError("evaluation first timestamp does not match frozen temporal lock")
    if last_timestamp != pd.Timestamp(base.EVALUATION_END):
        raise ValueError("evaluation last timestamp does not match frozen temporal lock")


def evaluate_bts_study2(
    archive: ZipFile,
    model: base.FrozenLinearPolicyModel,
    item_context: base.ItemContext,
    *,
    chunk_size: int,
) -> base.EvaluationArrays:
    """Evaluate BTS with explicit ns timestamps after an outcome-free boundary pass."""
    del item_context
    _validate_evaluation_window_without_outcomes(archive, chunk_size=chunk_size)

    usecols = (
        "timestamp",
        "position",
        "item_id",
        "click",
        "propensity_score",
        *base.USER_COLUMNS,
        *base.AFFINITY_COLUMNS,
    )
    timestamps = np.empty(base.EVALUATION_ROWS, dtype=np.int64)
    reward = np.empty(base.EVALUATION_ROWS, dtype=float)
    propensity = np.empty(base.EVALUATION_ROWS, dtype=float)
    q_logged = np.empty(base.EVALUATION_ROWS, dtype=float)
    uniform_dm = np.empty(base.EVALUATION_ROWS, dtype=float)
    challenger_dm = np.empty(base.EVALUATION_ROWS, dtype=float)
    uniform_weights = np.empty(base.EVALUATION_ROWS, dtype=float)
    challenger_weights = np.empty(base.EVALUATION_ROWS, dtype=float)

    filled = 0
    for selected in _iter_evaluation_slices(
        archive,
        usecols=usecols,
        chunk_size=chunk_size,
    ):
        take = len(selected)
        stop = filled + take

        action = np.asarray(selected["item_id"].to_numpy(), dtype=np.int16)
        if np.any((action < 0) | (action >= 80)):
            raise ValueError("evaluation action outside archived catalog 0..79")
        pscore = np.asarray(selected["propensity_score"].to_numpy(), dtype=float)
        if np.any(~np.isfinite(pscore)) or np.any(pscore <= 0.0):
            raise ValueError("evaluation propensities must be finite and positive")

        user_contribution = base._user_logit_contribution(selected, model)
        affinity = np.asarray(
            selected.loc[:, base.AFFINITY_COLUMNS].to_numpy(), dtype=float
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
        challenger_mean = (1.0 - base.CHALLENGER_EPSILON) * q_best + (
            base.CHALLENGER_EPSILON * q_mean
        )
        challenger_logged_probability = (
            base.CHALLENGER_EPSILON * base.UNIFORM_TARGET_PROBABILITY
            + (1.0 - base.CHALLENGER_EPSILON) * (action == best_action)
        )

        timestamps[filled:stop] = _timestamps_to_ns(selected["timestamp"])
        reward[filled:stop] = np.asarray(selected["click"].to_numpy(), dtype=float)
        propensity[filled:stop] = pscore
        q_logged[filled:stop] = q_log
        uniform_dm[filled:stop] = q_mean
        challenger_dm[filled:stop] = challenger_mean
        uniform_weights[filled:stop] = importance_weights(
            np.full(take, base.UNIFORM_TARGET_PROBABILITY, dtype=float),
            pscore,
        )
        challenger_weights[filled:stop] = importance_weights(
            challenger_logged_probability.astype(float),
            pscore,
        )
        filled = stop

    if timestamps[0] != int(pd.Timestamp(base.EVALUATION_START).value):
        raise ValueError("evaluation first timestamp does not match frozen temporal lock")
    if timestamps[-1] != int(pd.Timestamp(base.EVALUATION_END).value):
        raise ValueError("evaluation last timestamp does not match frozen temporal lock")

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
