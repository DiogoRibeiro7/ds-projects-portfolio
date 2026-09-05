from __future__ import annotations

import csv
import warnings
from io import StringIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from adaptive_policy_learning import study3_training as s3


def _item_context_csv() -> str:
    buffer = StringIO()
    fieldnames = [
        "",
        "item_id",
        "item_feature_0",
        "item_feature_1",
        "item_feature_2",
        "item_feature_3",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for item_id in range(s3.ACTION_COUNT):
        writer.writerow(
            {
                "": item_id,
                "item_id": item_id,
                "item_feature_0": float(item_id),
                "item_feature_1": f"a{item_id % 2}",
                "item_feature_2": f"b{item_id % 3}",
                "item_feature_3": f"c{item_id % 4}",
            }
        )
    return buffer.getvalue()


def _women_csv(rows: list[dict[str, object]]) -> str:
    buffer = StringIO()
    fieldnames = [
        "timestamp",
        "item_id",
        "position",
        "click",
        *s3.USER_COLUMNS,
        *s3.AFFINITY_COLUMNS,
    ]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _row(timestamp: str, item_id: int, click: object) -> dict[str, object]:
    row: dict[str, object] = {
        "timestamp": timestamp,
        "item_id": item_id,
        "position": 1,
        "click": click,
    }
    for column in s3.USER_COLUMNS:
        row[column] = "u"
    for index, column in enumerate(s3.AFFINITY_COLUMNS):
        row[column] = float(index)
    return row


def test_training_extractor_does_not_convert_first_evaluation_click(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(s3, "TRAIN_ROWS", 2)
    monkeypatch.setattr(s3, "TRAIN_LAST_TIMESTAMP", "2019-11-28 00:00:02+00:00")
    monkeypatch.setattr(s3, "EVALUATION_FIRST_TIMESTAMP", "2019-11-28 00:00:03+00:00")

    archive_path = tmp_path / "obd.zip"
    rows = [
        _row("2019-11-28 00:00:01+00:00", 0, 0),
        _row("2019-11-28 00:00:02+00:00", 1, 1),
        _row("2019-11-28 00:00:03+00:00", 2, "NOT_AN_OUTCOME"),
    ]
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("open_bandit_dataset/bts/women/item_context.csv", _item_context_csv())
        archive.writestr("open_bandit_dataset/bts/women/women.csv", _women_csv(rows))

    with ZipFile(archive_path) as archive:
        training = s3._extract_training_arrays(archive)

    assert training.reward.tolist() == [0, 1]
    assert training.final_timestamp == "2019-11-28 00:00:02+00:00"


class _FakeModel:
    classes_ = np.array([0, 1])
    n_iter_ = np.array([3])
    coef_ = np.array([[1.0]])
    intercept_ = np.array([0.0])

    def fit(self, x: object, y: object) -> _FakeModel:
        return self


def _fit_with_model(monkeypatch: pytest.MonkeyPatch, model: object) -> None:
    monkeypatch.setattr(s3, "LogisticRegression", lambda **kwargs: model)
    s3._fit_reward_model(np.ones((2, 1)), np.array([0, 1], dtype=np.uint8))


def test_fit_reward_model_rejects_convergence_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeModel(_FakeModel):
        def fit(self, x: object, y: object) -> FakeModel:
            warnings.warn("did not converge", ConvergenceWarning)
            return self

    monkeypatch.setattr(s3, "LogisticRegression", lambda **kwargs: FakeModel())
    with pytest.raises(RuntimeError, match="ConvergenceWarning"):
        s3._fit_reward_model(np.ones((2, 1)), np.array([0, 1], dtype=np.uint8))


@pytest.mark.parametrize("message", ["singular Hessian", "ill-conditioned system", "solver fallback"])
def test_fit_reward_model_rejects_numerical_warning(
    message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeModel(_FakeModel):
        def fit(self, x: object, y: object) -> FakeModel:
            warnings.warn(message, RuntimeWarning)
            return self

    monkeypatch.setattr(s3, "LogisticRegression", lambda **kwargs: FakeModel())
    with pytest.raises(RuntimeError, match="numerical fallback or instability warning"):
        s3._fit_reward_model(np.ones((2, 1)), np.array([0, 1], dtype=np.uint8))


def test_fit_reward_model_rejects_wrong_classes(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeModel(_FakeModel):
        classes_ = np.array([0, 2])

    monkeypatch.setattr(s3, "LogisticRegression", lambda **kwargs: FakeModel())
    with pytest.raises(RuntimeError, match=r"classes must be \[0, 1\]"):
        s3._fit_reward_model(np.ones((2, 1)), np.array([0, 1], dtype=np.uint8))


def test_fit_reward_model_rejects_iteration_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeModel(_FakeModel):
        n_iter_ = np.array([s3.MAX_ITER])

    monkeypatch.setattr(s3, "LogisticRegression", lambda **kwargs: FakeModel())
    with pytest.raises(RuntimeError, match="reached max_iter=100"):
        s3._fit_reward_model(np.ones((2, 1)), np.array([0, 1], dtype=np.uint8))


def test_fit_reward_model_rejects_nonfinite_coefficients(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeModel(_FakeModel):
        coef_ = np.array([[np.nan]])

    monkeypatch.setattr(s3, "LogisticRegression", lambda **kwargs: FakeModel())
    with pytest.raises(RuntimeError, match="non-finite coefficients"):
        s3._fit_reward_model(np.ones((2, 1)), np.array([0, 1], dtype=np.uint8))
