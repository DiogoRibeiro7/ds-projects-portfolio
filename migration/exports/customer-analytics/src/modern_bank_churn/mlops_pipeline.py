from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from .feature_engineering import FeatureEngineer


LOGGER = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Runtime configuration for preparing churn datasets."""

    raw_data_path: str = "data/raw/bank_churn.csv"
    processed_data_path: str = "data/processed/bank_churn_processed.csv"
    random_state: int = 42
    train_size: float = 0.8
    min_rows: int = 100


@dataclass
class TrainingConfig:
    """Runtime configuration for model training."""

    model_random_state: int = 42
    n_estimators: int = 100
    min_accuracy: float = 0.0


class DataValidator:
    """Validate a dataset before it enters the MLOps pipeline."""

    def __init__(self, config: DataConfig):
        self.config = config

    def validate(self, df: pd.DataFrame) -> dict[str, Any]:
        errors: list[str] = []
        if len(df) < self.config.min_rows:
            errors.append(f"Dataset too small: {len(df)} rows")
        if df.isna().all(axis=1).any():
            errors.append("Dataset contains fully empty rows")

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "n_rows": int(len(df)),
            "n_columns": int(df.shape[1]),
            "is_valid": not errors,
            "errors": errors,
            "columns": list(df.columns),
        }


class DataPipeline:
    """Simple feature engineering and persistence pipeline."""

    def __init__(self, config: DataConfig):
        self.config = config
        self.feature_engineer = FeatureEngineer()
        self.validator = DataValidator(config)

    def run(self, df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
        if df is None:
            df = self._load_data()
        report = self.validator.validate(df)
        if not report["is_valid"]:
            raise ValueError(f"Invalid dataset: {report['errors']}")
        cleaned = self._clean_data(df)
        engineered = self.feature_engineer.create_features(cleaned)
        engineered.to_csv(self.config.processed_data_path, index=False)
        return engineered, {
            "feature_columns": [c for c in engineered.columns if c != "Churn"],
            "validation": report,
        }

    def _load_data(self) -> pd.DataFrame:
        return pd.read_csv(self.config.raw_data_path)

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        cleaned = df.drop_duplicates().copy()
        for col in cleaned.columns:
            if pd.api.types.is_numeric_dtype(cleaned[col]):
                cleaned[col] = cleaned[col].fillna(cleaned[col].median())
            else:
                mode = cleaned[col].mode(dropna=True)
                fill = mode.iloc[0] if not mode.empty else ""
                cleaned[col] = cleaned[col].fillna(fill)
        return cleaned


class TrainingPipeline:
    """Train a churn model and return metrics plus artifacts metadata."""

    def __init__(self, data_config: DataConfig, training_config: TrainingConfig):
        self.data_config = data_config
        self.training_config = training_config
        self.data_pipeline = DataPipeline(data_config)

    def run(self, df: pd.DataFrame | None = None) -> dict[str, Any]:
        dataset, pipeline_report = self.data_pipeline.run(df)
        target_col = self._resolve_target(dataset)
        X = dataset.drop(columns=[target_col])
        y = dataset[target_col].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            train_size=self.data_config.train_size,
            random_state=self.data_config.random_state,
            stratify=y,
        )

        model = RandomForestClassifier(
            random_state=self.training_config.model_random_state,
            n_estimators=self.training_config.n_estimators,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds

        metrics = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall": float(recall_score(y_test, preds, zero_division=0)),
            "f1": float(f1_score(y_test, preds, zero_division=0)),
        }
        if len(set(y_test)) > 1:
            metrics["mean_proba"] = float(probs.mean())

        if metrics["accuracy"] < self.training_config.min_accuracy:
            LOGGER.warning("Model accuracy below configured threshold")

        return {
            "model": model,
            "feature_names": list(X.columns),
            "target_column": target_col,
            "metrics": metrics,
            "pipeline_report": pipeline_report,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _resolve_target(self, df: pd.DataFrame) -> str:
        if "churn" in df.columns:
            return "churn"
        if "Churn" in df.columns:
            return "Churn"
        if "target" in df.columns:
            return "target"
        return df.columns[-1]


class ModelServer:
    """Minimal prediction wrapper suitable for integration tests and examples."""

    def __init__(self, model, feature_names: list[str]):
        self.model = model
        self.feature_names = feature_names

    def predict(self, payload: dict[str, Any] | pd.DataFrame) -> list[int] | list[float]:
        if isinstance(payload, dict):
            frame = pd.DataFrame([payload])
        else:
            frame = payload.copy()
        frame = frame[self.feature_names]
        if hasattr(self.model, "predict"):
            return list(map(int, self.model.predict(frame)))
        raise RuntimeError("Model does not implement predict")

    def score(self, payload: dict[str, Any] | pd.DataFrame) -> list[float]:
        if not hasattr(self.model, "predict_proba"):
            raise RuntimeError("Model does not expose probabilities")
        if isinstance(payload, dict):
            frame = pd.DataFrame([payload])
        else:
            frame = payload.copy()
        frame = frame[self.feature_names]
        return list(map(float, self.model.predict_proba(frame)[:, 1]))


def run_training_pipeline(
    data_path: str | None = None,
    train_size: float = 0.8,
    random_state: int = 42,
) -> dict[str, Any]:
    """Convenience wrapper around :class:`TrainingPipeline`."""
    data_config = DataConfig(
        raw_data_path=data_path or "data/raw/bank_churn.csv",
        train_size=train_size,
        random_state=random_state,
    )
    training_config = TrainingConfig()
    pipeline = TrainingPipeline(data_config, training_config)
    return pipeline.run()


def start_api_server(*, host: str = "127.0.0.1", port: int = 8000) -> str:
    """Start a lightweight API surface for manual experimentation."""
    return json.dumps({"status": "ready", "host": host, "port": port})


def pipeline_healthcheck() -> dict[str, Any]:
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": [
            "data_pipeline",
            "training_pipeline",
            "model_server",
        ],
    }


if __name__ == "__main__":
    run_training_pipeline()
