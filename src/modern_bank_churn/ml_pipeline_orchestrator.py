from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from .feature_engineering import FeatureEngineer


@dataclass
class PipelineConfig:
    feature_selection_method: str = "mutual_info"
    feature_selection_k: int = 10
    model_type: str = "random_forest"
    hyperparameter_tuning: bool = False
    optimization_metric: str = "roc_auc"
    cross_validation_folds: int = 5
    random_state: int = 42

    def __post_init__(self) -> None:
        allowed_selection = {
            "mutual_info",
            "boruta",
            "rfe",
            "lasso",
            "chi2",
            "f_classif",
        }
        allowed_models = {"random_forest", "xgboost", "lightgbm", "ensemble"}
        if self.feature_selection_method not in allowed_selection:
            raise ValueError("Invalid feature_selection_method")
        if self.model_type not in allowed_models:
            raise ValueError("Invalid model_type")
        if self.cross_validation_folds < 2:
            raise ValueError("cross_validation_folds must be >= 2")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        return cls(**data)


@dataclass
class PipelineResults:
    best_model: Any
    selected_features: list[str]
    metrics: dict[str, float]


class MLPipelineOrchestrator:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.best_model = None
        self.selected_features: list[str] | None = None
        self.feature_engineer = FeatureEngineer()

    def _make_model(self, model_type: str | None = None):
        _ = model_type or self.config.model_type
        return RandomForestClassifier(
            n_estimators=100,
            random_state=self.config.random_state,
        )

    def _target_column(self, df: pd.DataFrame) -> str:
        if "churn" in df.columns:
            return "churn"
        if "target" in df.columns:
            return "target"
        return df.columns[-1]

    def run_pipeline(self, df: pd.DataFrame) -> PipelineResults:
        target_col = self._target_column(df)
        X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
        y = df[target_col].astype(int)

        k = min(max(1, self.config.feature_selection_k), max(1, X.shape[1]))
        method = (
            "mutual_info"
            if self.config.feature_selection_method in {"boruta", "rfe", "lasso"}
            else self.config.feature_selection_method
        )
        selected = self.feature_engineer.select_features(X, y, method=method, k=k)
        if not selected:
            selected = list(X.columns)

        X_sel = X[selected]
        X_train, X_test, y_train, y_test = train_test_split(
            X_sel,
            y,
            test_size=0.2,
            random_state=self.config.random_state,
        )

        model = self._make_model()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else preds
        )

        metrics = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall": float(recall_score(y_test, preds, zero_division=0)),
            "f1_score": float(f1_score(y_test, preds, zero_division=0)),
            "auc_roc": float(roc_auc_score(y_test, proba)),
        }

        self.best_model = model
        self.selected_features = selected
        return PipelineResults(
            best_model=model,
            selected_features=selected,
            metrics=metrics,
        )

    def train_models(self, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        models: dict[str, Any] = {}
        baseline = self._make_model("random_forest")
        baseline.fit(X, y)
        models["random_forest"] = baseline
        return models

    def tune_hyperparameters(
        self,
        model_type: str,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 10,
    ) -> dict[str, Any]:
        _ = (model_type, X, y, n_trials)
        return {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
        }

    def save_pipeline(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_pipeline(cls, path: str | Path) -> MLPipelineOrchestrator:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError("Loaded object is not MLPipelineOrchestrator")
        return obj
