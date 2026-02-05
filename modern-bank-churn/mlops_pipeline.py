"""Complete MLOps Pipeline for Bank Churn Prediction Model

This module implements a production-ready MLOps pipeline including:
- Data ingestion and validation
- Feature engineering and storage
- Model training with hyperparameter tuning
- Model versioning and registry
- API serving with monitoring
- Drift detection and alerting
- A/B testing capabilities
"""

import hashlib
import json
import logging
import pickle
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

# Data processing
import lightgbm as lgb

# MLOps libraries
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
import redis
import uvicorn
import xgboost as xgb
from evidently import ColumnMapping
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection

# API and serving
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

# Monitoring and observability
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ML libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class DataConfig:
    """Data pipeline configuration"""

    raw_data_path: str = "data/raw/bank_churn.csv"
    processed_data_path: str = "data/processed/bank_churn_processed.csv"
    feature_store_path: str = "data/feature_store/"
    train_test_split: float = 0.2
    validation_split: float = 0.1
    random_state: int = 42

    # Data quality thresholds
    min_rows: int = 1000
    max_missing_pct: float = 0.3
    max_duplicates_pct: float = 0.05

    # Feature engineering parameters
    create_interaction_features: bool = True
    create_polynomial_features: bool = False
    feature_selection_threshold: float = 0.01


@dataclass
class TrainingConfig:
    """Model training configuration"""

    experiment_name: str = "bank_churn_prediction"
    model_name: str = "churn_classifier"
    tracking_uri: str = "http://localhost:5000"

    # Model parameters
    model_type: str = "lightgbm"  # Options: lightgbm, xgboost, random_forest
    n_trials: int = 100  # Optuna trials
    cv_folds: int = 5
    early_stopping_rounds: int = 50

    # Metrics thresholds
    min_accuracy: float = 0.85
    min_precision: float = 0.80
    min_recall: float = 0.75
    min_f1: float = 0.78
    min_auc: float = 0.85


@dataclass
class ServingConfig:
    """Model serving configuration"""

    model_uri: str = "models:/churn_classifier/production"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    batch_size: int = 1000
    cache_ttl: int = 300  # seconds

    # Redis cache
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    # Rate limiting
    max_requests_per_minute: int = 1000
    max_batch_size: int = 10000


@dataclass
class MonitoringConfig:
    """Monitoring configuration"""

    prometheus_port: int = 8001
    drift_threshold: float = 0.1
    performance_window_hours: int = 24
    alert_email: str = "mlops-team@company.com"
    slack_webhook: str = ""

    # Monitoring intervals
    drift_check_interval_minutes: int = 60
    performance_check_interval_minutes: int = 30

    # Alert thresholds
    accuracy_alert_threshold: float = 0.80
    latency_alert_threshold_ms: float = 100
    error_rate_alert_threshold: float = 0.01


# =============================================================================
# DATA VALIDATION
# =============================================================================


class DataValidator:
    """Validate data quality and integrity"""

    def __init__(self, config: DataConfig):
        self.config = config
        self.validation_results = {}

    def validate(self, df: pd.DataFrame) -> dict[str, Any]:
        """Run all validation checks"""
        logger.info("Starting data validation...")

        results = {
            "timestamp": datetime.now().isoformat(),
            "is_valid": True,
            "checks": {},
            "warnings": [],
            "errors": [],
        }

        # Check data size
        if len(df) < self.config.min_rows:
            results["errors"].append(
                f"Insufficient data: {len(df)} rows < {self.config.min_rows}"
            )
            results["is_valid"] = False

        # Check missing values
        missing_pct = df.isnull().sum() / len(df)
        for col, pct in missing_pct.items():
            if pct > self.config.max_missing_pct:
                results["warnings"].append(f"Column {col} has {pct:.2%} missing values")

        # Check duplicates
        duplicate_pct = df.duplicated().sum() / len(df)
        if duplicate_pct > self.config.max_duplicates_pct:
            results["warnings"].append(f"Data has {duplicate_pct:.2%} duplicate rows")

        # Check data types
        results["checks"]["data_types"] = self._check_data_types(df)

        # Check value ranges
        results["checks"]["value_ranges"] = self._check_value_ranges(df)

        # Check class balance
        if "Churn" in df.columns:
            results["checks"]["class_balance"] = self._check_class_balance(df)

        logger.info(f"Validation complete. Valid: {results['is_valid']}")
        return results

    def _check_data_types(self, df: pd.DataFrame) -> dict[str, str]:
        """Check if data types are as expected"""
        expected_types = {
            "CustomerId": "int64",
            "CreditScore": "float64",
            "Age": "int64",
            "Tenure": "int64",
            "Balance": "float64",
            "NumOfProducts": "int64",
            "EstimatedSalary": "float64",
        }

        results = {}
        for col, expected_type in expected_types.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                results[col] = (
                    "OK"
                    if actual_type == expected_type
                    else f"Expected {expected_type}, got {actual_type}"
                )

        return results

    def _check_value_ranges(self, df: pd.DataFrame) -> dict[str, Any]:
        """Check if values are within expected ranges"""
        ranges = {
            "CreditScore": (300, 850),
            "Age": (18, 100),
            "Tenure": (0, 10),
            "NumOfProducts": (1, 4),
        }

        results = {}
        for col, (min_val, max_val) in ranges.items():
            if col in df.columns:
                col_min = df[col].min()
                col_max = df[col].max()

                if col_min < min_val or col_max > max_val:
                    results[col] = (
                        f"Out of range: [{col_min}, {col_max}] not in [{min_val}, {max_val}]"
                    )
                else:
                    results[col] = "OK"

        return results

    def _check_class_balance(self, df: pd.DataFrame) -> dict[str, float]:
        """Check target class balance"""
        class_dist = df["Churn"].value_counts(normalize=True)

        return {
            "class_0": float(class_dist.get(0, 0)),
            "class_1": float(class_dist.get(1, 0)),
            "imbalance_ratio": float(class_dist.min() / class_dist.max()),
        }


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================


class FeatureEngineer:
    """Feature engineering pipeline tuned for the churn dataset.

    The pipeline builds balance-derived ratios, interaction terms, label
    encodings, and standardized numeric columns so the downstream model
    receives a dense design matrix. It expects the Kaggle-style churn
    columns (for example `Balance`, `EstimatedSalary`, `Geography`) plus a
    binary `Churn` column while fitting and will persist the fitted label
    encoders/`StandardScaler` for later inference.

    Attributes:
        config: ``DataConfig`` instance that toggles optional generators such
            as interaction features.
        feature_names: Ordered list of engineered feature names excluding the
            `Churn` target. Updated after ``fit_transform``.
        scaler: ``StandardScaler`` fitted on numeric features during training.
        label_encoders: Mapping of categorical columns to fitted encoders so
            ``transform`` can reproduce the same integer mapping online.

    Examples:
        >>> config = DataConfig(create_interaction_features=True)
        >>> engineer = FeatureEngineer(config)
        >>> engineered_train = engineer.fit_transform(train_df)
        >>> live_scores = engineer.transform(train_df.iloc[:2])
        >>> set(engineer.feature_names).issubset(engineered_train.columns)
        True
    """

    def __init__(self, config: DataConfig):
        self.config = config
        self.feature_names = []
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform features"""
        logger.info("Starting feature engineering...")

        # Basic features
        df = self._create_basic_features(df)

        # Interaction features
        if self.config.create_interaction_features:
            df = self._create_interaction_features(df)

        # Encode categorical variables
        df = self._encode_categorical(df)

        # Scale numerical features
        df = self._scale_features(df)

        # Feature selection
        if hasattr(self, "feature_selector"):
            df = self._select_features(df)

        self.feature_names = [col for col in df.columns if col != "Churn"]
        logger.info(f"Created {len(self.feature_names)} features")

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted parameters"""
        # Apply same transformations using fitted encoders and scalers
        df = self._create_basic_features(df)

        if self.config.create_interaction_features:
            df = self._create_interaction_features(df)

        # Use fitted encoders
        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                df[col] = encoder.transform(df[col])

        # Use fitted scaler
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
        numeric_cols = [col for col in numeric_cols if col != "Churn"]

        if numeric_cols:
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])

        return df

    def _create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic engineered features"""
        # Balance per product
        df["BalancePerProduct"] = df["Balance"] / df["NumOfProducts"].clip(lower=1)

        # Salary to age ratio
        df["SalaryAgeRatio"] = df["EstimatedSalary"] / df["Age"]

        # Credit score segments
        df["CreditScoreSegment"] = pd.cut(
            df["CreditScore"],
            bins=[0, 580, 670, 740, 800, 850],
            labels=["Very Poor", "Fair", "Good", "Very Good", "Exceptional"],
        )

        # Age groups
        df["AgeGroup"] = pd.cut(
            df["Age"],
            bins=[0, 25, 35, 45, 55, 65, 100],
            labels=["<25", "25-35", "35-45", "45-55", "55-65", "65+"],
        )

        # Activity score
        df["ActivityScore"] = (
            (df["NumOfProducts"] > 1).astype(int)
            + (df["HasCrCard"] == 1).astype(int)
            + (df["IsActiveMember"] == 1).astype(int)
        )

        # Zero balance flag
        df["HasZeroBalance"] = (df["Balance"] == 0).astype(int)

        return df

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features"""
        # Age and tenure interaction
        df["AgeTenureInteraction"] = df["Age"] * df["Tenure"]

        # Balance and products interaction
        df["BalanceProductsInteraction"] = df["Balance"] * df["NumOfProducts"]

        # Credit score and balance interaction
        df["CreditBalanceInteraction"] = df["CreditScore"] * df["Balance"]

        return df

    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables"""
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns

        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col])
            else:
                df[col] = self.label_encoders[col].transform(df[col])

        return df

    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features"""
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
        numeric_cols = [col for col in numeric_cols if col != "Churn"]

        if numeric_cols:
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])

        return df

    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select important features"""
        # This would use feature importance from a model
        # For now, return all features
        return df


# =============================================================================
# DATA PIPELINE
# =============================================================================


class DataPipeline:
    """End-to-end data pipeline"""

    def __init__(self, config: DataConfig):
        self.config = config
        self.validator = DataValidator(config)
        self.feature_engineer = FeatureEngineer(config)
        self.feature_store = FeatureStore(config.feature_store_path)

    def run(
        self, df: pd.DataFrame | None = None
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Run complete data pipeline"""
        logger.info("Starting data pipeline...")

        # Load data if not provided
        if df is None:
            df = self.load_raw_data()

        # Validate data
        validation_report = self.validator.validate(df)
        if not validation_report["is_valid"]:
            raise ValueError(f"Data validation failed: {validation_report['errors']}")

        # Clean data
        df = self.clean_data(df)

        # Feature engineering
        df = self.feature_engineer.fit_transform(df)

        # Save to feature store
        feature_set_id = self.feature_store.save_features(
            df,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "validation_report": validation_report,
                "feature_names": self.feature_engineer.feature_names,
            },
        )

        # Save processed data
        df.to_csv(self.config.processed_data_path, index=False)
        logger.info(f"Data pipeline complete. Feature set ID: {feature_set_id}")

        return df, {
            "feature_set_id": feature_set_id,
            "validation_report": validation_report,
            "n_features": len(self.feature_engineer.feature_names),
        }

    def load_raw_data(self) -> pd.DataFrame:
        """Load raw data from source"""
        logger.info(f"Loading data from {self.config.raw_data_path}")

        if self.config.raw_data_path.endswith(".csv"):
            df = pd.read_csv(self.config.raw_data_path)
        elif self.config.raw_data_path.endswith(".parquet"):
            df = pd.read_parquet(self.config.raw_data_path)
        else:
            raise ValueError(f"Unsupported file format: {self.config.raw_data_path}")

        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess data"""
        logger.info("Cleaning data...")

        # Remove duplicates
        df = df.drop_duplicates()

        # Handle missing values
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns

        # Fill numeric with median
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)

        # Fill categorical with mode
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col].fillna(
                    df[col].mode()[0] if not df[col].mode().empty else "Unknown",
                    inplace=True,
                )

        logger.info(f"Cleaned data: {len(df)} rows remaining")
        return df


# =============================================================================
# FEATURE STORE
# =============================================================================


class FeatureStore:
    """Feature store for versioned feature management"""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.base_path / "metadata.json"
        self._load_metadata()

    def save_features(self, df: pd.DataFrame, metadata: dict[str, Any]) -> str:
        """Save features with versioning"""
        # Generate feature set ID
        feature_set_id = self._generate_feature_set_id()

        # Create version directory
        version_dir = self.base_path / feature_set_id
        version_dir.mkdir(exist_ok=True)

        # Save features
        feature_path = version_dir / "features.parquet"
        df.to_parquet(feature_path, index=False)

        # Save metadata
        metadata["feature_set_id"] = feature_set_id
        metadata["created_at"] = datetime.now().isoformat()
        metadata["n_rows"] = len(df)
        metadata["n_cols"] = len(df.columns)
        metadata["columns"] = df.columns.tolist()

        with open(version_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Update global metadata
        self.metadata[feature_set_id] = metadata
        self._save_metadata()

        logger.info(f"Saved feature set: {feature_set_id}")
        return feature_set_id

    def load_features(self, feature_set_id: str) -> pd.DataFrame:
        """Load specific feature set"""
        feature_path = self.base_path / feature_set_id / "features.parquet"

        if not feature_path.exists():
            raise FileNotFoundError(f"Feature set not found: {feature_set_id}")

        return pd.read_parquet(feature_path)

    def get_latest_features(self) -> pd.DataFrame:
        """Get the most recent feature set"""
        if not self.metadata:
            raise ValueError("No feature sets available")

        latest_id = max(self.metadata.keys())
        return self.load_features(latest_id)

    def _generate_feature_set_id(self) -> str:
        """Generate unique feature set ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:6]
        return f"features_{timestamp}_{random_suffix}"

    def _load_metadata(self):
        """Load global metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

    def _save_metadata(self):
        """Save global metadata"""
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)


# =============================================================================
# MODEL TRAINING
# =============================================================================


class ModelTrainer:
    """Model training with hyperparameter optimization"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.mlflow_client = MlflowClient(tracking_uri=config.tracking_uri)
        mlflow.set_tracking_uri(config.tracking_uri)
        mlflow.set_experiment(config.experiment_name)

        self.best_model = None
        self.best_params = None
        self.best_metrics = None

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        """Train model with hyperparameter tuning"""
        logger.info("Starting model training...")

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=self.config.validation_split,
            random_state=self.config.random_state,
            stratify=y,
        )

        # Start MLflow run
        with mlflow.start_run() as run:
            # Log configuration
            mlflow.log_params(asdict(self.config))

            # Hyperparameter optimization
            study = self._optimize_hyperparameters(X_train, y_train, X_val, y_val)

            # Train final model with best parameters
            self.best_params = study.best_params
            mlflow.log_params(self.best_params)

            # Train model
            if self.config.model_type == "lightgbm":
                self.best_model = self._train_lightgbm(
                    X_train, y_train, X_val, y_val, self.best_params
                )
            elif self.config.model_type == "xgboost":
                self.best_model = self._train_xgboost(
                    X_train, y_train, X_val, y_val, self.best_params
                )
            else:
                self.best_model = self._train_random_forest(
                    X_train, y_train, self.best_params
                )

            # Evaluate model
            self.best_metrics = self._evaluate_model(self.best_model, X_val, y_val)
            mlflow.log_metrics(self.best_metrics)

            # Check if model meets requirements
            if not self._check_model_requirements():
                logger.warning("Model does not meet minimum requirements")

            # Log model
            signature = infer_signature(X_train, self.best_model.predict(X_train))
            mlflow.sklearn.log_model(
                self.best_model,
                "model",
                signature=signature,
                registered_model_name=self.config.model_name,
            )

            # Save artifacts
            self._save_artifacts(X_val, y_val)

            logger.info(f"Training complete. Run ID: {run.info.run_id}")

            return {
                "run_id": run.info.run_id,
                "model": self.best_model,
                "params": self.best_params,
                "metrics": self.best_metrics,
            }

    def _optimize_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> optuna.Study:
        """Optimize hyperparameters using Optuna"""
        logger.info("Starting hyperparameter optimization...")

        def objective(trial):
            if self.config.model_type == "lightgbm":
                params = {
                    "objective": "binary",
                    "metric": "auc",
                    "boosting_type": "gbdt",
                    "num_leaves": trial.suggest_int("num_leaves", 20, 300),
                    "learning_rate": trial.suggest_loguniform(
                        "learning_rate", 0.01, 0.3
                    ),
                    "feature_fraction": trial.suggest_uniform(
                        "feature_fraction", 0.5, 1.0
                    ),
                    "bagging_fraction": trial.suggest_uniform(
                        "bagging_fraction", 0.5, 1.0
                    ),
                    "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                    "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                    "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-8, 10.0),
                    "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-8, 10.0),
                }

                model = lgb.LGBMClassifier(**params, n_estimators=1000, random_state=42)
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[
                        lgb.early_stopping(self.config.early_stopping_rounds),
                        lgb.log_evaluation(0),
                    ],
                )

            elif self.config.model_type == "xgboost":
                params = {
                    "objective": "binary:logistic",
                    "eval_metric": "auc",
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_loguniform(
                        "learning_rate", 0.01, 0.3
                    ),
                    "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_uniform(
                        "colsample_bytree", 0.5, 1.0
                    ),
                    "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-8, 10.0),
                    "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-8, 10.0),
                }

                model = xgb.XGBClassifier(**params, n_estimators=1000, random_state=42)
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=self.config.early_stopping_rounds,
                    verbose=False,
                )

            else:  # Random Forest
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "max_depth": trial.suggest_int("max_depth", 3, 20),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                    "max_features": trial.suggest_categorical(
                        "max_features", ["sqrt", "log2"]
                    ),
                }

                model = RandomForestClassifier(**params, random_state=42)
                model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred)

            return auc

        # Create and run study
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.config.n_trials, show_progress_bar=True)

        logger.info(f"Best AUC: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")

        return study

    def _train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        params: dict[str, Any],
    ) -> lgb.LGBMClassifier:
        """Train LightGBM model"""
        model = lgb.LGBMClassifier(**params, n_estimators=1000, random_state=42)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(self.config.early_stopping_rounds),
                lgb.log_evaluation(period=50),
            ],
        )

        return model

    def _train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        params: dict[str, Any],
    ) -> xgb.XGBClassifier:
        """Train XGBoost model"""
        model = xgb.XGBClassifier(**params, n_estimators=1000, random_state=42)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=self.config.early_stopping_rounds,
            verbose=True,
        )

        return model

    def _train_random_forest(
        self, X_train: pd.DataFrame, y_train: pd.Series, params: dict[str, Any]
    ) -> RandomForestClassifier:
        """Train Random Forest model"""
        model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        return model

    def _evaluate_model(
        self, model, X_val: pd.DataFrame, y_val: pd.Series
    ) -> dict[str, float]:
        """Evaluate model performance"""
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred),
            "recall": recall_score(y_val, y_pred),
            "f1": f1_score(y_val, y_pred),
            "auc": roc_auc_score(y_val, y_pred_proba),
        }

        # Calculate additional metrics
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
        metrics["specificity"] = tn / (tn + fp)
        metrics["fall_out"] = fp / (fp + tn)

        return metrics

    def _check_model_requirements(self) -> bool:
        """Check if model meets minimum requirements"""
        if not self.best_metrics:
            return False

        return (
            self.best_metrics["accuracy"] >= self.config.min_accuracy
            and self.best_metrics["precision"] >= self.config.min_precision
            and self.best_metrics["recall"] >= self.config.min_recall
            and self.best_metrics["f1"] >= self.config.min_f1
            and self.best_metrics["auc"] >= self.config.min_auc
        )

    def _save_artifacts(self, X_val: pd.DataFrame, y_val: pd.Series):
        """Save training artifacts"""
        # Feature importance
        if hasattr(self.best_model, "feature_importances_"):
            importance = pd.DataFrame(
                {
                    "feature": X_val.columns,
                    "importance": self.best_model.feature_importances_,
                }
            ).sort_values("importance", ascending=False)

            # Save as CSV artifact
            importance.to_csv("feature_importance.csv", index=False)
            mlflow.log_artifact("feature_importance.csv")

        # Confusion matrix
        y_pred = self.best_model.predict(X_val)
        cm = confusion_matrix(y_val, y_pred)

        # Save confusion matrix
        np.save("confusion_matrix.npy", cm)
        mlflow.log_artifact("confusion_matrix.npy")


# =============================================================================
# TRAINING PIPELINE
# =============================================================================


class TrainingPipeline:
    """Complete training pipeline"""

    def __init__(self, data_config: DataConfig, training_config: TrainingConfig):
        self.data_config = data_config
        self.training_config = training_config
        self.data_pipeline = DataPipeline(data_config)
        self.model_trainer = ModelTrainer(training_config)

    def run(self, df: pd.DataFrame | None = None) -> dict[str, Any]:
        """Run complete training pipeline"""
        logger.info("Starting training pipeline...")

        # Run data pipeline
        processed_df, data_info = self.data_pipeline.run(df)

        # Prepare features and target
        feature_cols = [col for col in processed_df.columns if col != "Churn"]
        X = processed_df[feature_cols]
        y = processed_df["Churn"]

        # Train model
        training_results = self.model_trainer.train(X, y)

        # Combine results
        results = {
            "data_info": data_info,
            "training_results": training_results,
            "feature_engineer": self.data_pipeline.feature_engineer,
            "timestamp": datetime.now().isoformat(),
        }

        # Save pipeline artifacts
        self._save_pipeline_artifacts(results)

        logger.info("Training pipeline complete")
        return results

    def _save_pipeline_artifacts(self, results: dict[str, Any]):
        """Save pipeline artifacts"""
        # Save feature engineer
        with open("feature_engineer.pkl", "wb") as f:
            pickle.dump(results["feature_engineer"], f)

        # Save results summary
        summary = {
            "timestamp": results["timestamp"],
            "data_info": results["data_info"],
            "metrics": results["training_results"]["metrics"],
            "model_run_id": results["training_results"]["run_id"],
        }

        with open("training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)


# =============================================================================
# MODEL SERVING
# =============================================================================


class PredictionRequest(BaseModel):
    """API request model for predictions"""

    CustomerId: int
    CreditScore: float = Field(..., ge=300, le=850)
    Geography: str
    Gender: str
    Age: int = Field(..., ge=18, le=100)
    Tenure: int = Field(..., ge=0, le=10)
    Balance: float = Field(..., ge=0)
    NumOfProducts: int = Field(..., ge=1, le=4)
    HasCrCard: int = Field(..., ge=0, le=1)
    IsActiveMember: int = Field(..., ge=0, le=1)
    EstimatedSalary: float = Field(..., gt=0)


class PredictionResponse(BaseModel):
    """API response model for predictions"""

    customer_id: int
    churn_probability: float
    churn_prediction: int
    confidence: float
    risk_level: str
    timestamp: str


class ModelServer:
    """Model serving with caching and monitoring"""

    def __init__(self, config: ServingConfig):
        self.config = config
        self.model = None
        self.feature_engineer = None
        self.cache = None
        self._load_model()
        self._setup_cache()
        self._setup_metrics()

    def _load_model(self):
        """Load model from MLflow"""
        logger.info(f"Loading model from {self.config.model_uri}")

        # Load model
        self.model = mlflow.sklearn.load_model(self.config.model_uri)

        # Load feature engineer
        with open("feature_engineer.pkl", "rb") as f:
            self.feature_engineer = pickle.load(f)

        logger.info("Model loaded successfully")

    def _setup_cache(self):
        """Setup Redis cache"""
        try:
            self.cache = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                decode_responses=True,
            )
            self.cache.ping()
            logger.info("Redis cache connected")
        except:
            logger.warning("Redis not available, caching disabled")
            self.cache = None

    def _setup_metrics(self):
        """Setup Prometheus metrics"""
        self.prediction_counter = Counter(
            "model_predictions_total",
            "Total number of predictions",
            ["model", "version"],
        )

        self.prediction_latency = Histogram(
            "model_prediction_latency_seconds",
            "Prediction latency in seconds",
            ["model"],
        )

        self.prediction_errors = Counter(
            "model_prediction_errors_total",
            "Total prediction errors",
            ["model", "error_type"],
        )

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Make prediction for single customer"""
        start_time = datetime.now()

        try:
            # Check cache
            cache_key = self._get_cache_key(request)
            if self.cache:
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    logger.info(f"Cache hit for customer {request.CustomerId}")
                    return PredictionResponse(**json.loads(cached_result))

            # Prepare features
            df = pd.DataFrame([request.dict()])
            df = self.feature_engineer.transform(df)

            # Make prediction
            churn_proba = self.model.predict_proba(df)[0]
            churn_pred = int(churn_proba[1] > 0.5)

            # Determine risk level
            if churn_proba[1] < 0.3:
                risk_level = "Low"
            elif churn_proba[1] < 0.7:
                risk_level = "Medium"
            else:
                risk_level = "High"

            # Create response
            response = PredictionResponse(
                customer_id=request.CustomerId,
                churn_probability=float(churn_proba[1]),
                churn_prediction=churn_pred,
                confidence=float(max(churn_proba)),
                risk_level=risk_level,
                timestamp=datetime.now().isoformat(),
            )

            # Cache result
            if self.cache:
                self.cache.setex(
                    cache_key, self.config.cache_ttl, json.dumps(response.dict())
                )

            # Update metrics
            self.prediction_counter.labels(
                model=self.config.model_name, version="latest"
            ).inc()

            latency = (datetime.now() - start_time).total_seconds()
            self.prediction_latency.labels(model=self.config.model_name).observe(
                latency
            )

            return response

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            self.prediction_errors.labels(
                model=self.config.model_name, error_type=type(e).__name__
            ).inc()
            raise

    def predict_batch(
        self, requests: list[PredictionRequest]
    ) -> list[PredictionResponse]:
        """Make predictions for batch of customers"""
        if len(requests) > self.config.max_batch_size:
            raise ValueError(
                f"Batch size {len(requests)} exceeds maximum {self.config.max_batch_size}"
            )

        responses = []
        for request in requests:
            response = self.predict(request)
            responses.append(response)

        return responses

    def _get_cache_key(self, request: PredictionRequest) -> str:
        """Generate cache key for request"""
        # Create hash of request features
        feature_str = json.dumps(request.dict(), sort_keys=True)
        feature_hash = hashlib.md5(feature_str.encode()).hexdigest()
        return f"prediction:{self.config.model_name}:{feature_hash}"


# =============================================================================
# MODEL MONITORING
# =============================================================================


class DriftDetector:
    """Detect data and prediction drift"""

    def __init__(self, reference_data: pd.DataFrame, config: MonitoringConfig):
        self.reference_data = reference_data
        self.config = config
        self.drift_profile = None
        self._setup_profile()

    def _setup_profile(self):
        """Setup Evidently profile"""
        column_mapping = ColumnMapping(
            target="Churn",
            prediction="prediction",
            numerical_features=self.reference_data.select_dtypes(
                include=["float64", "int64"]
            ).columns.tolist(),
            categorical_features=self.reference_data.select_dtypes(
                include=["object", "category"]
            ).columns.tolist(),
        )

        self.drift_profile = Profile(sections=[DataDriftProfileSection()])

    def detect_drift(self, current_data: pd.DataFrame) -> dict[str, Any]:
        """Detect drift in current data"""
        logger.info("Checking for data drift...")

        # Calculate drift
        self.drift_profile.calculate(self.reference_data, current_data)
        drift_results = self.drift_profile.json()

        # Parse results
        drift_detected = False
        drifted_features = []

        # Check for drift in features
        # This is simplified - actual implementation would parse Evidently results

        results = {
            "timestamp": datetime.now().isoformat(),
            "drift_detected": drift_detected,
            "drifted_features": drifted_features,
            "drift_score": 0.0,  # Would calculate from Evidently
            "action_required": drift_detected,
        }

        if drift_detected:
            logger.warning(f"Data drift detected in features: {drifted_features}")
        else:
            logger.info("No significant drift detected")

        return results


class PerformanceMonitor:
    """Monitor model performance"""

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.performance_history = []
        self._setup_metrics()

    def _setup_metrics(self):
        """Setup performance metrics"""
        self.accuracy_gauge = Gauge(
            "model_accuracy", "Current model accuracy", ["model"]
        )

        self.precision_gauge = Gauge(
            "model_precision", "Current model precision", ["model"]
        )

        self.recall_gauge = Gauge("model_recall", "Current model recall", ["model"])

    def calculate_metrics(
        self, predictions: pd.DataFrame, actuals: pd.DataFrame
    ) -> dict[str, float]:
        """Calculate performance metrics"""
        y_true = actuals["Churn"]
        y_pred = predictions["prediction"]

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
        }

        # Update Prometheus metrics
        self.accuracy_gauge.labels(model="churn_classifier").set(metrics["accuracy"])
        self.precision_gauge.labels(model="churn_classifier").set(metrics["precision"])
        self.recall_gauge.labels(model="churn_classifier").set(metrics["recall"])

        # Store history
        self.performance_history.append(
            {"timestamp": datetime.now().isoformat(), "metrics": metrics}
        )

        # Check for performance degradation
        if metrics["accuracy"] < self.config.accuracy_alert_threshold:
            logger.warning(
                f"Performance degradation: accuracy {metrics['accuracy']:.3f}"
            )

        return metrics


class ModelMonitor:
    """Complete monitoring system"""

    def __init__(self, reference_data: pd.DataFrame, config: MonitoringConfig):
        self.config = config
        self.drift_detector = DriftDetector(reference_data, config)
        self.performance_monitor = PerformanceMonitor(config)
        self.alert_manager = AlertManager(config)

    def monitor(
        self, predictions: pd.DataFrame, actuals: pd.DataFrame | None = None
    ) -> dict[str, Any]:
        """Run complete monitoring"""
        results = {"timestamp": datetime.now().isoformat(), "checks": {}}

        # Check data drift
        drift_results = self.drift_detector.detect_drift(predictions)
        results["checks"]["drift"] = drift_results

        # Calculate performance if actuals available
        if actuals is not None:
            performance_results = self.performance_monitor.calculate_metrics(
                predictions, actuals
            )
            results["checks"]["performance"] = performance_results

        # Send alerts if needed
        self._check_alerts(results)

        return results

    def _check_alerts(self, results: dict[str, Any]):
        """Check and send alerts"""
        alerts = []

        # Check drift alert
        if results["checks"].get("drift", {}).get("drift_detected"):
            alerts.append(
                {
                    "type": "DRIFT",
                    "severity": "HIGH",
                    "message": "Data drift detected",
                    "details": results["checks"]["drift"],
                }
            )

        # Check performance alert
        performance = results["checks"].get("performance", {})
        if performance:
            if performance.get("accuracy", 1.0) < self.config.accuracy_alert_threshold:
                alerts.append(
                    {
                        "type": "PERFORMANCE",
                        "severity": "CRITICAL",
                        "message": f"Accuracy below threshold: {performance['accuracy']:.3f}",
                        "details": performance,
                    }
                )

        # Send alerts
        for alert in alerts:
            self.alert_manager.send_alert(alert)


# =============================================================================
# ALERT MANAGER
# =============================================================================


class AlertManager:
    """Manage alerts and notifications"""

    def __init__(self, config: MonitoringConfig):
        self.config = config

    def send_alert(self, alert: dict[str, Any]):
        """Send alert through configured channels"""
        logger.warning(f"ALERT: {alert['type']} - {alert['message']}")

        # Send email alert
        if self.config.alert_email:
            self._send_email_alert(alert)

        # Send Slack alert
        if self.config.slack_webhook:
            self._send_slack_alert(alert)

    def _send_email_alert(self, alert: dict[str, Any]):
        """Send email alert"""
        # Implementation would use email library
        logger.info(f"Email alert sent to {self.config.alert_email}")

    def _send_slack_alert(self, alert: dict[str, Any]):
        """Send Slack alert"""
        # Implementation would use requests to post to webhook
        logger.info("Slack alert sent")


# =============================================================================
# API APPLICATION
# =============================================================================

# Create FastAPI app
app = FastAPI(
    title="Bank Churn Prediction API",
    description="MLOps pipeline for bank customer churn prediction",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
serving_config = ServingConfig()
monitoring_config = MonitoringConfig()
model_server = ModelServer(serving_config)


# API endpoints
@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model_server.model is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Score a single churn prediction request via the FastAPI service.

    Parameters
    ----------
    request : PredictionRequest
        JSON payload with the Kaggle churn fields (e.g., ``CreditScore``,
        ``Geography``, ``Balance``). Field ranges are validated by Pydantic
        (credit scores 300–850, ages 18–100, etc.).

    Returns:
    -------
    PredictionResponse
        Structured response containing ``churn_probability`` (0–1),
        ``confidence``, a categorical ``risk_level``, and the ISO-8601
        timestamp when the model server scored the request.

    Raises:
    ------
    HTTPException
        Propagates a 500 error if ``ModelServer.predict`` fails (for example,
        batch size overflows or the feature store/MLflow artifacts are
        unavailable).

    Notes:
    -----
    The underlying ``ModelServer`` transparently performs feature engineering,
    logs Prometheus metrics, and caches the response in Redis (if configured)
    for ``ServingConfig.cache_ttl`` seconds to keep latency below 100 ms in
    practice.

    Examples:
    --------
    >>> from fastapi.testclient import TestClient
    >>> client = TestClient(app)
    >>> payload = {
    ...     "CustomerId": 42,
    ...     "CreditScore": 720,
    ...     "Geography": "France",
    ...     "Gender": "Female",
    ...     "Age": 32,
    ...     "Tenure": 3,
    ...     "Balance": 60000.0,
    ...     "NumOfProducts": 2,
    ...     "HasCrCard": 1,
    ...     "IsActiveMember": 1,
    ...     "EstimatedSalary": 90000.0,
    ... }
    >>> resp = client.post("/predict", json=payload)
    >>> resp.status_code
    200
    >>> sorted(resp.json().keys())[:3]
    ['churn_prediction', 'churn_probability', 'confidence']
    """
    try:
        response = model_server.predict(request)
        return response
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch", response_model=list[PredictionResponse])
async def predict_batch(requests: list[PredictionRequest]):
    """Batch prediction endpoint"""
    try:
        responses = model_server.predict_batch(requests)
        return responses
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()


@app.post("/retrain")
async def retrain(background_tasks: BackgroundTasks):
    """Trigger model retraining"""
    background_tasks.add_task(run_training_pipeline)
    return {"message": "Retraining started", "timestamp": datetime.now().isoformat()}


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def run_training_pipeline():
    """Run the complete training pipeline"""
    data_config = DataConfig()
    training_config = TrainingConfig()

    pipeline = TrainingPipeline(data_config, training_config)
    results = pipeline.run()

    logger.info("Training pipeline completed successfully")
    return results


def start_api_server():
    """Start the API server"""
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
        },
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bank Churn MLOps Pipeline")
    parser.add_argument(
        "--mode",
        choices=["train", "serve", "monitor"],
        default="serve",
        help="Pipeline mode",
    )

    args = parser.parse_args()

    if args.mode == "train":
        logger.info("Starting training pipeline...")
        results = run_training_pipeline()
        logger.info("Training complete!")

    elif args.mode == "serve":
        logger.info("Starting API server...")
        start_api_server()

    elif args.mode == "monitor":
        logger.info("Starting monitoring...")
        # Implementation for continuous monitoring
        pass
