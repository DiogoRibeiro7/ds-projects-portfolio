"""Training-only qualification gate for Adaptive Policy Learning Study 2."""

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

STUDY2_MAX_ITER = 100
STUDY2_PROTOCOL_FILE = "study2_design_lock_v1_0.json"


def run_study2_training_qualification(
    archive_path: Path,
    output_path: Path,
    *,
    code_sha: str,
    protocol_dir: Path,
    chunk_size: int = 100_000,
) -> dict[str, object]:
    """Fit the Study 2 reward model on BTS training rows only."""
    base._verify_archive(archive_path)
    protocol_hash = _study2_protocol_hash(protocol_dir)

    with ZipFile(archive_path) as archive:
        item_context = base._load_item_context(archive)
        training = base._extract_training_arrays(archive, chunk_size=chunk_size)

    scaling = base._training_numeric_scaling(training, item_context)
    x_train, layout = base._build_training_matrix(training, item_context, scaling)
    model, captured = _fit_study2_reward_model(x_train, training.reward)
    frozen = base._freeze_linear_model(model, training, item_context, scaling, layout)

    result: dict[str, object] = {
        "status": "training_gate_passed",
        "study": "Adaptive Policy Learning Study 2",
        "protocol_version": "1.0-study2-prospective-lock",
        "code_sha": code_sha,
        "protocol_hash": protocol_hash,
        "source_archive_sha256": base.ARCHIVE_SHA256,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "training": {
            "rows": base.TRAIN_ROWS,
            "n_features": layout.n_features,
            "solver": "newton-cholesky",
            "C": 1.0,
            "tol": 0.0001,
            "max_iter": STUDY2_MAX_ITER,
            "n_iter": frozen.n_iter,
            "coefficient_sha256": frozen.coefficient_sha256,
            "warnings": captured,
        },
        "evaluation_outcomes_loaded": False,
        "random_reference_outcomes_loaded": False,
        "ope_estimates_computed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _fit_study2_reward_model(
    x_train: Any,
    y_train: np.ndarray,
) -> tuple[LogisticRegression, list[str]]:
    model = LogisticRegression(
        solver="newton-cholesky",
        penalty="l2",
        C=1.0,
        fit_intercept=True,
        max_iter=STUDY2_MAX_ITER,
        tol=0.0001,
        class_weight=None,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(x_train, y_train)

    messages = [f"{warning.category.__name__}: {warning.message}" for warning in caught]
    if any(issubclass(warning.category, ConvergenceWarning) for warning in caught):
        raise RuntimeError("Study 2 training gate failed: ConvergenceWarning")
    if any(
        token in str(warning.message).lower()
        for warning in caught
        for token in ("fallback", "ill-conditioned", "singular", "hessian")
    ):
        raise RuntimeError("Study 2 training gate failed: numerical fallback or instability warning")
    if model.classes_.tolist() != [0, 1]:
        raise RuntimeError("Study 2 training gate failed: classes must be [0, 1]")
    if int(model.n_iter_[0]) >= STUDY2_MAX_ITER:
        raise RuntimeError("Study 2 training gate failed: reached max_iter=100")
    if not np.all(np.isfinite(model.coef_)) or not np.all(np.isfinite(model.intercept_)):
        raise RuntimeError("Study 2 training gate failed: non-finite coefficients")
    return model, messages


def _study2_protocol_hash(protocol_dir: Path) -> str:
    digest = hashlib.sha256()
    for name in (
        "design_lock.json",
        "obd_source_lock.json",
        "temporal_split_lock_v0_6.json",
        "empirical_model_lock_v0_7.json",
        "optimizer_amendment_v0_8.json",
        "primary_empirical_terminal_status_v0_8.json",
        STUDY2_PROTOCOL_FILE,
    ):
        path = protocol_dir / name
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
