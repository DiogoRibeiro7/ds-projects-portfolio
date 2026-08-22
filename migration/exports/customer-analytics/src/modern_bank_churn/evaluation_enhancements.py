from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score


class ModelEvaluator:
    """Compatibility evaluator expected by unit tests and docs examples."""

    def evaluate(self, model, X, y) -> dict[str, float]:
        y_pred = model.predict(X)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X)[:, 1]
            auc_roc = float(roc_auc_score(y, y_prob))
        else:
            auc_roc = float(roc_auc_score(y, y_pred))
        return {
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred, zero_division=0)),
            "recall": float(recall_score(y, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y, y_pred, zero_division=0)),
            "auc_roc": auc_roc,
        }

    def cross_validate(self, model, X, y, cv: int = 5) -> dict[str, object]:
        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
        return {
            "scores": scores.tolist(),
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
        }

    def evaluate_metric(self, model, X, y, metric: str = "accuracy") -> float:
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
        if metric == "accuracy":
            return float(accuracy_score(y, y_pred))
        if metric == "precision":
            return float(precision_score(y, y_pred, zero_division=0))
        if metric == "recall":
            return float(recall_score(y, y_pred, zero_division=0))
        if metric in {"f1", "f1_score"}:
            return float(f1_score(y, y_pred, zero_division=0))
        if metric in {"roc_auc", "auc_roc"}:
            return float(roc_auc_score(y, y_prob if y_prob is not None else y_pred))
        raise ValueError(f"Unsupported metric: {metric}")

    def get_confusion_matrix(self, model, X, y) -> np.ndarray:
        return confusion_matrix(y, model.predict(X))

    def get_classification_report(self, model, X, y) -> dict[str, object]:
        return classification_report(y, model.predict(X), output_dict=True, zero_division=0)

    def get_roc_curve(self, model, X, y):
        scores = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X)
        return roc_curve(y, scores)

    def get_feature_importance(self, model, feature_names: Sequence[str]) -> pd.DataFrame:
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            importances = np.zeros(len(feature_names), dtype=float)
        frame = pd.DataFrame({"feature": list(feature_names), "importance": importances})
        return frame.sort_values("importance", ascending=False).reset_index(drop=True)

    def calculate_metrics(self, y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float | None]:
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)
        return {
            "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
            "precision": float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
            "recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
            "f1_score": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
        }
