from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any

import pandas as pd


class ProductionPipeline:
    """Lightweight production wrapper used by tests and examples."""

    def __init__(
        self,
        model,
        feature_names: list[str],
        preprocessing_params: dict[str, Any] | None = None,
        version: str = "0.1.0",
        enable_monitoring: bool = False,
    ) -> None:
        self.model = model
        self.feature_names = feature_names
        self.preprocessing_params = preprocessing_params or {}
        self.version = version
        self.enable_monitoring = enable_monitoring
        self._prediction_count = 0
        self._total_prediction_time = 0.0

    def _validate_input(self, X: pd.DataFrame) -> None:
        missing = [name for name in self.feature_names if name not in X.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate_input(X)
        out = X[self.feature_names].copy()
        scaler = self.preprocessing_params.get("scaler")
        if scaler is not None:
            transformed = scaler.transform(out)
            out = pd.DataFrame(transformed, columns=self.feature_names, index=out.index)
        return out

    def _record_metrics(self, n_rows: int, elapsed: float) -> None:
        if not self.enable_monitoring:
            return
        self._prediction_count += n_rows
        self._total_prediction_time += elapsed

    def predict(self, X: pd.DataFrame):
        start = time.perf_counter()
        prepared = self._prepare(X)
        preds = self.model.predict(prepared)
        self._record_metrics(len(prepared), time.perf_counter() - start)
        return preds

    def predict_batch(self, X: pd.DataFrame):
        return self.predict(X)

    def predict_proba(self, X: pd.DataFrame):
        prepared = self._prepare(X)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(prepared)
        preds = self.model.predict(prepared)
        return pd.get_dummies(preds).reindex(columns=[0, 1], fill_value=0).to_numpy()

    def get_monitoring_metrics(self) -> dict[str, float]:
        avg = (
            self._total_prediction_time / self._prediction_count
            if self._prediction_count > 0
            else 0.0
        )
        return {
            "prediction_count": self._prediction_count,
            "avg_prediction_time": float(avg),
            "total_prediction_time": float(self._total_prediction_time),
        }

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "ProductionPipeline":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError("Loaded object is not ProductionPipeline")
        return obj
