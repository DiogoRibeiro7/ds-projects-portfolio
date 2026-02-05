"""
AutoML Orchestrator with support for multiple frameworks and automated pipeline generation.
"""

import os
import time
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import optuna
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.optuna import OptunaSearch

# AutoML frameworks
try:
    import autogluon.tabular as ag
    from autogluon.tabular import TabularDataset, TabularPredictor

    AUTOGLUON_AVAILABLE = True
except ImportError:
    AUTOGLUON_AVAILABLE = False

try:
    import h2o
    from h2o.automl import H2OAutoML

    H2O_AVAILABLE = True
except ImportError:
    H2O_AVAILABLE = False

try:
    import flaml
    from flaml import AutoML as FLAMLAutoML

    FLAML_AVAILABLE = True
except ImportError:
    FLAML_AVAILABLE = False

try:
    from pycaret.classification import *
    from pycaret.regression import *

    PYCARET_AVAILABLE = True
except ImportError:
    PYCARET_AVAILABLE = False

# Neural architecture search
try:
    import nni
    from nni.experiment import Experiment

    NNI_AVAILABLE = True
except ImportError:
    NNI_AVAILABLE = False

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import (
    VotingClassifier,
    VotingRegressor,
    StackingClassifier,
    StackingRegressor,
)

logger = logging.getLogger(__name__)


@dataclass
class AutoMLConfig:
    """Configuration for AutoML orchestrator."""

    task_type: str = "classification"  # 'classification' or 'regression'
    frameworks: List[str] = field(default_factory=lambda: ["autogluon", "h2o", "flaml"])
    time_limit: int = 3600  # seconds
    optimization_metric: str = "auto"
    ensemble_strategy: str = "auto"  # 'voting', 'stacking', 'blending', 'auto'
    use_gpu: bool = False
    n_jobs: int = -1
    verbosity: int = 2
    save_path: str = "./automl_models"
    experiment_name: str = "automl_experiment"

    # Neural architecture search
    enable_nas: bool = False
    nas_time_limit: int = 7200
    nas_search_space: Optional[Dict] = None

    # Feature engineering
    auto_feature_engineering: bool = True
    feature_selection_method: str = "auto"
    max_features: Optional[int] = None

    # Hyperparameter optimization
    hpo_method: str = "bayesian"  # 'grid', 'random', 'bayesian', 'hyperband'
    n_trials: int = 100

    # Distributed computing
    use_ray: bool = False
    ray_resources: Optional[Dict] = None

    # Model interpretability
    enable_interpretability: bool = True
    shap_samples: int = 100


class AutoMLOrchestrator:
    """
    Advanced AutoML orchestrator with multiple framework support.
    """

    def __init__(self, config: AutoMLConfig):
        """Initialize AutoML orchestrator."""
        self.config = config
        self.models = {}
        self.results = {}
        self.best_model = None
        self.feature_importance = None
        self.ensemble = None

        # Initialize Ray if needed
        if config.use_ray and not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        # Create save directory
        os.makedirs(config.save_path, exist_ok=True)

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO if self.config.verbosity > 0 else logging.WARNING,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
    ) -> "AutoMLOrchestrator":
        """
        Fit AutoML models using multiple frameworks.

        Parameters
        ----------
        X : pd.DataFrame
            Training features
        y : pd.Series
            Training labels
        X_valid : pd.DataFrame, optional
            Validation features
        y_valid : pd.Series, optional
            Validation labels

        Returns
        -------
        self
            Fitted orchestrator
        """
        logger.info(f"Starting AutoML with frameworks: {self.config.frameworks}")

        # Prepare data
        X_train, y_train = self._prepare_data(X, y)

        # Split validation set if not provided
        if X_valid is None:
            X_train, X_valid, y_train, y_valid = train_test_split(
                X_train,
                y_train,
                test_size=0.2,
                random_state=42,
                stratify=y_train if self.config.task_type == "classification" else None,
            )

        # Run AutoML frameworks
        framework_results = {}

        if self.config.use_ray:
            # Distributed training with Ray
            framework_results = self._fit_distributed(
                X_train, y_train, X_valid, y_valid
            )
        else:
            # Sequential or parallel training
            if "autogluon" in self.config.frameworks and AUTOGLUON_AVAILABLE:
                framework_results["autogluon"] = self._fit_autogluon(
                    X_train, y_train, X_valid, y_valid
                )

            if "h2o" in self.config.frameworks and H2O_AVAILABLE:
                framework_results["h2o"] = self._fit_h2o(
                    X_train, y_train, X_valid, y_valid
                )

            if "flaml" in self.config.frameworks and FLAML_AVAILABLE:
                framework_results["flaml"] = self._fit_flaml(
                    X_train, y_train, X_valid, y_valid
                )

            if "pycaret" in self.config.frameworks and PYCARET_AVAILABLE:
                framework_results["pycaret"] = self._fit_pycaret(
                    X_train, y_train, X_valid, y_valid
                )

        # Neural Architecture Search if enabled
        if self.config.enable_nas and NNI_AVAILABLE:
            framework_results["nas"] = self._run_nas(X_train, y_train, X_valid, y_valid)

        # Select best model
        self.results = framework_results
        self.best_model = self._select_best_model(framework_results, X_valid, y_valid)

        # Create ensemble if specified
        if self.config.ensemble_strategy != "none":
            self.ensemble = self._create_ensemble(framework_results, X_train, y_train)

        # Generate feature importance
        if self.config.enable_interpretability:
            self.feature_importance = self._calculate_feature_importance(X_train)

        logger.info("AutoML training completed")
        return self

    def _prepare_data(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for AutoML."""
        X_prepared = X.copy()
        y_prepared = y.copy()

        # Handle categorical variables
        for col in X_prepared.select_dtypes(include=["object"]).columns:
            le = LabelEncoder()
            X_prepared[col] = le.fit_transform(X_prepared[col].fillna("missing"))

        # Handle missing values
        X_prepared = X_prepared.fillna(X_prepared.mean())

        # Feature engineering if enabled
        if self.config.auto_feature_engineering:
            X_prepared = self._engineer_features(X_prepared)

        return X_prepared, y_prepared

    def _engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Automated feature engineering."""
        import featuretools as ft
        from featuretools.primitives import (
            Sum,
            Mean,
            Max,
            Min,
            Std,
            Count,
            PercentTrue,
            Mode,
            NumUnique,
        )

        # Create entity set
        es = ft.EntitySet(id="automl_data")

        # Add dataframe as entity
        es = es.add_dataframe(
            dataframe_name="data", dataframe=X, index="index", make_index=True
        )

        # Generate features
        feature_matrix, feature_defs = ft.dfs(
            entityset=es,
            target_dataframe_name="data",
            agg_primitives=[Sum, Mean, Max, Min, Std],
            trans_primitives=[],
            max_depth=2,
            n_jobs=self.config.n_jobs,
        )

        return feature_matrix

    def _fit_autogluon(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
    ) -> Dict:
        """Fit AutoGluon model."""
        logger.info("Training AutoGluon model...")

        # Prepare data
        train_data = TabularDataset(pd.concat([X_train, y_train], axis=1))
        valid_data = TabularDataset(pd.concat([X_valid, y_valid], axis=1))

        # Configure predictor
        predictor = TabularPredictor(
            label=y_train.name,
            problem_type=self.config.task_type,
            eval_metric=self.config.optimization_metric
            if self.config.optimization_metric != "auto"
            else None,
            path=os.path.join(self.config.save_path, "autogluon"),
        )

        # Train
        predictor.fit(
            train_data,
            tuning_data=valid_data,
            time_limit=self.config.time_limit,
            presets="best_quality",
            num_gpus=1 if self.config.use_gpu else 0,
            verbosity=self.config.verbosity,
        )

        # Get results
        leaderboard = predictor.leaderboard(valid_data, silent=True)
        best_model_name = leaderboard.iloc[0]["model"]

        return {
            "model": predictor,
            "score": leaderboard.iloc[0]["score_val"],
            "leaderboard": leaderboard,
            "best_model": best_model_name,
        }

    def _fit_h2o(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
    ) -> Dict:
        """Fit H2O AutoML model."""
        logger.info("Training H2O AutoML model...")

        # Initialize H2O
        h2o.init(nthreads=self.config.n_jobs, max_mem_size="4G")

        # Convert to H2O frames
        train = h2o.H2OFrame(pd.concat([X_train, y_train], axis=1))
        valid = h2o.H2OFrame(pd.concat([X_valid, y_valid], axis=1))

        # Set target column
        x = X_train.columns.tolist()
        y = y_train.name

        # For classification, convert target to factor
        if self.config.task_type == "classification":
            train[y] = train[y].asfactor()
            valid[y] = valid[y].asfactor()

        # Run AutoML
        aml = H2OAutoML(
            max_runtime_secs=self.config.time_limit,
            max_models=20,
            seed=42,
            verbosity="info" if self.config.verbosity > 0 else "warn",
            nfolds=5,
            balance_classes=True
            if self.config.task_type == "classification"
            else False,
            stopping_metric="AUTO",
            sort_metric="AUTO",
        )

        aml.train(x=x, y=y, training_frame=train, validation_frame=valid)

        # Get results
        lb = aml.leaderboard
        best_model = aml.leader

        return {
            "model": aml,
            "score": best_model.model_performance(valid).auc()
            if self.config.task_type == "classification"
            else best_model.model_performance(valid).rmse(),
            "leaderboard": lb,
            "best_model": best_model,
        }

    def _fit_flaml(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
    ) -> Dict:
        """Fit FLAML AutoML model."""
        logger.info("Training FLAML AutoML model...")

        automl = FLAMLAutoML()

        # Configure settings
        settings = {
            "time_budget": self.config.time_limit,
            "metric": self.config.optimization_metric
            if self.config.optimization_metric != "auto"
            else ("roc_auc" if self.config.task_type == "classification" else "r2"),
            "task": self.config.task_type,
            "n_jobs": self.config.n_jobs,
            "eval_method": "cv",
            "n_splits": 5,
            "verbose": self.config.verbosity,
            "seed": 42,
        }

        # Train
        automl.fit(X_train, y_train, X_val=X_valid, y_val=y_valid, **settings)

        # Get results
        score = automl.score(X_valid, y_valid)

        return {
            "model": automl,
            "score": score,
            "best_model": automl.best_model_for_estimator(automl.best_estimator),
            "best_config": automl.best_config,
            "best_loss": automl.best_loss,
        }

    def _fit_pycaret(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
    ) -> Dict:
        """Fit PyCaret AutoML model."""
        logger.info("Training PyCaret AutoML model...")

        # Combine features and target
        train_data = pd.concat([X_train, y_train], axis=1)

        if self.config.task_type == "classification":
            from pycaret.classification import setup, compare_models, finalize_model
        else:
            from pycaret.regression import setup, compare_models, finalize_model

        # Setup
        exp = setup(
            train_data,
            target=y_train.name,
            train_size=0.8,
            session_id=42,
            use_gpu=self.config.use_gpu,
            silent=self.config.verbosity == 0,
            n_jobs=self.config.n_jobs,
        )

        # Compare models
        best_model = compare_models(
            budget_time=self.config.time_limit / 60,  # Convert to minutes
            fold=5,
        )

        # Finalize model
        final_model = finalize_model(best_model)

        # Score on validation set
        predictions = final_model.predict(X_valid)

        if self.config.task_type == "classification":
            score = accuracy_score(y_valid, predictions)
        else:
            score = r2_score(y_valid, predictions)

        return {"model": final_model, "score": score, "best_model": best_model}

    def _run_nas(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
    ) -> Dict:
        """Run Neural Architecture Search."""
        logger.info("Running Neural Architecture Search...")

        # Define search space
        search_space = self.config.nas_search_space or {
            "n_layers": {"_type": "choice", "_value": [2, 3, 4, 5]},
            "n_units": {"_type": "choice", "_value": [32, 64, 128, 256]},
            "dropout_rate": {"_type": "uniform", "_value": [0.0, 0.5]},
            "learning_rate": {"_type": "loguniform", "_value": [0.0001, 0.1]},
            "batch_size": {"_type": "choice", "_value": [16, 32, 64, 128]},
        }

        # Create experiment
        experiment = Experiment("local")
        experiment.config.trial_command = "python nas_trial.py"
        experiment.config.trial_code_directory = "."
        experiment.config.search_space = search_space
        experiment.config.tuner.name = "TPE"
        experiment.config.tuner.class_args["optimize_mode"] = "maximize"
        experiment.config.max_trial_number = self.config.n_trials
        experiment.config.trial_concurrency = self.config.n_jobs

        # Run experiment
        experiment.run(port=8080, wait_completion=True)

        # Get best architecture
        best_params = experiment.get_job_statistics()

        return {"best_params": best_params, "experiment_id": experiment.id}

    def _fit_distributed(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
    ) -> Dict:
        """Distributed training using Ray."""
        logger.info("Running distributed training with Ray...")

        @ray.remote
        def train_framework(framework: str, X_train, y_train, X_valid, y_valid, config):
            """Train a single framework in distributed manner."""
            orchestrator = AutoMLOrchestrator(config)

            if framework == "autogluon" and AUTOGLUON_AVAILABLE:
                return orchestrator._fit_autogluon(X_train, y_train, X_valid, y_valid)
            elif framework == "h2o" and H2O_AVAILABLE:
                return orchestrator._fit_h2o(X_train, y_train, X_valid, y_valid)
            elif framework == "flaml" and FLAML_AVAILABLE:
                return orchestrator._fit_flaml(X_train, y_train, X_valid, y_valid)
            else:
                return None

        # Launch parallel training
        futures = []
        for framework in self.config.frameworks:
            future = train_framework.remote(
                framework, X_train, y_train, X_valid, y_valid, self.config
            )
            futures.append((framework, future))

        # Collect results
        results = {}
        for framework, future in futures:
            result = ray.get(future)
            if result:
                results[framework] = result

        return results

    def _select_best_model(
        self, framework_results: Dict, X_valid: pd.DataFrame, y_valid: pd.Series
    ) -> Any:
        """Select the best model from all frameworks."""
        best_score = -np.inf if self.config.task_type == "classification" else np.inf
        best_model = None
        best_framework = None

        for framework, result in framework_results.items():
            score = result.get("score", 0)

            if self.config.task_type == "classification":
                if score > best_score:
                    best_score = score
                    best_model = result["model"]
                    best_framework = framework
            else:
                if score < best_score:
                    best_score = score
                    best_model = result["model"]
                    best_framework = framework

        logger.info(f"Best model: {best_framework} with score: {best_score}")
        return best_model

    def _create_ensemble(
        self, framework_results: Dict, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Any:
        """Create ensemble from multiple models."""
        logger.info(f"Creating ensemble with strategy: {self.config.ensemble_strategy}")

        models = [(name, result["model"]) for name, result in framework_results.items()]

        if self.config.ensemble_strategy == "voting":
            if self.config.task_type == "classification":
                ensemble = VotingClassifier(estimators=models, voting="soft")
            else:
                ensemble = VotingRegressor(estimators=models)

        elif self.config.ensemble_strategy == "stacking":
            from sklearn.linear_model import LogisticRegression, Ridge

            if self.config.task_type == "classification":
                ensemble = StackingClassifier(
                    estimators=models, final_estimator=LogisticRegression()
                )
            else:
                ensemble = StackingRegressor(estimators=models, final_estimator=Ridge())

        elif self.config.ensemble_strategy == "blending":
            # Custom blending implementation
            ensemble = self._create_blending_ensemble(models, X_train, y_train)

        else:  # auto
            # Choose based on model performance variance
            ensemble = self._auto_select_ensemble(models, X_train, y_train)

        # Fit ensemble
        ensemble.fit(X_train, y_train)

        return ensemble

    def _create_blending_ensemble(
        self, models: List, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Any:
        """Create blending ensemble."""
        from sklearn.model_selection import KFold
        from sklearn.linear_model import LogisticRegression, Ridge

        n_splits = 5
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Generate blend features
        blend_features = np.zeros((len(X_train), len(models)))

        for i, (name, model) in enumerate(models):
            for train_idx, val_idx in kf.split(X_train):
                X_fold_train = X_train.iloc[train_idx]
                y_fold_train = y_train.iloc[train_idx]
                X_fold_val = X_train.iloc[val_idx]

                model.fit(X_fold_train, y_fold_train)

                if self.config.task_type == "classification":
                    blend_features[val_idx, i] = model.predict_proba(X_fold_val)[:, 1]
                else:
                    blend_features[val_idx, i] = model.predict(X_fold_val)

        # Train meta-model
        if self.config.task_type == "classification":
            meta_model = LogisticRegression()
        else:
            meta_model = Ridge()

        meta_model.fit(blend_features, y_train)

        return meta_model

    def _auto_select_ensemble(
        self, models: List, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Any:
        """Automatically select ensemble strategy."""
        # Evaluate model diversity
        predictions = []

        for name, model in models:
            pred = model.predict(X_train)
            predictions.append(pred)

        # Calculate pairwise correlation
        corr_matrix = np.corrcoef(predictions)
        avg_corr = (corr_matrix.sum() - len(models)) / (len(models) * (len(models) - 1))

        # Low correlation -> voting, high correlation -> stacking
        if avg_corr < 0.7:
            logger.info("Models are diverse, using voting ensemble")
            if self.config.task_type == "classification":
                return VotingClassifier(estimators=models, voting="soft")
            else:
                return VotingRegressor(estimators=models)
        else:
            logger.info("Models are similar, using stacking ensemble")
            from sklearn.linear_model import LogisticRegression, Ridge

            if self.config.task_type == "classification":
                return StackingClassifier(
                    estimators=models, final_estimator=LogisticRegression()
                )
            else:
                return StackingRegressor(estimators=models, final_estimator=Ridge())

    def _calculate_feature_importance(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """Calculate feature importance using SHAP."""
        import shap

        logger.info("Calculating feature importance...")

        # Use SHAP for interpretability
        if hasattr(self.best_model, "predict"):
            explainer = shap.Explainer(self.best_model.predict, X_train)
            shap_values = explainer(
                X_train.sample(min(self.config.shap_samples, len(X_train)))
            )

            # Calculate mean absolute SHAP values
            importance = pd.DataFrame(
                {
                    "feature": X_train.columns,
                    "importance": np.abs(shap_values.values).mean(axis=0),
                }
            ).sort_values("importance", ascending=False)

            return importance

        return pd.DataFrame()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if self.ensemble is not None:
            return self.ensemble.predict(X)
        elif self.best_model is not None:
            return self.best_model.predict(X)
        else:
            raise ValueError("No model has been trained yet")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        if self.config.task_type != "classification":
            raise ValueError("predict_proba is only available for classification")

        if self.ensemble is not None:
            return self.ensemble.predict_proba(X)
        elif self.best_model is not None:
            return self.best_model.predict_proba(X)
        else:
            raise ValueError("No model has been trained yet")

    def save(self, path: str):
        """Save the AutoML model."""
        save_dict = {
            "config": self.config,
            "best_model": self.best_model,
            "ensemble": self.ensemble,
            "results": self.results,
            "feature_importance": self.feature_importance,
        }

        joblib.dump(save_dict, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "AutoMLOrchestrator":
        """Load a saved AutoML model."""
        save_dict = joblib.load(path)

        orchestrator = cls(save_dict["config"])
        orchestrator.best_model = save_dict["best_model"]
        orchestrator.ensemble = save_dict["ensemble"]
        orchestrator.results = save_dict["results"]
        orchestrator.feature_importance = save_dict["feature_importance"]

        return orchestrator

    def generate_report(self) -> str:
        """Generate a comprehensive report of AutoML results."""
        report = []
        report.append("=" * 60)
        report.append("AutoML Orchestrator Report")
        report.append("=" * 60)

        # Configuration
        report.append(f"\nConfiguration:")
        report.append(f"  Task Type: {self.config.task_type}")
        report.append(f"  Frameworks: {', '.join(self.config.frameworks)}")
        report.append(f"  Time Limit: {self.config.time_limit}s")
        report.append(f"  Optimization Metric: {self.config.optimization_metric}")

        # Results
        report.append(f"\nFramework Results:")
        for framework, result in self.results.items():
            report.append(f"  {framework}: Score = {result.get('score', 'N/A')}")

        # Feature Importance
        if self.feature_importance is not None and len(self.feature_importance) > 0:
            report.append(f"\nTop 10 Important Features:")
            for idx, row in self.feature_importance.head(10).iterrows():
                report.append(f"  {row['feature']}: {row['importance']:.4f}")

        # Ensemble
        if self.ensemble is not None:
            report.append(f"\nEnsemble Strategy: {self.config.ensemble_strategy}")

        return "\n".join(report)
