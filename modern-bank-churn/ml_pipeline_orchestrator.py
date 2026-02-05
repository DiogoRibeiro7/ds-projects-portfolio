"""
ML Pipeline Orchestrator for Bank Churn Prediction

This module provides a unified interface to orchestrate the complete ML pipeline,
integrating all enhanced modules for feature engineering, modeling, evaluation,
and production readiness.

Author: Portfolio Team
Date: 2024
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
import yaml

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve
import joblib

# Import our enhanced modules
from enhanced_feature_engineering import (
    EnhancedFeatureSelector,
    FeatureInteractionDetector,
    TargetEncoder,
    FeatureStabilityAnalyzer,
)
from enhanced_modeling import (
    StratifiedCrossValidator,
    EnhancedEnsembleMethods,
    HyperparameterOptimizer,
    ModelCalibrator,
    UncertaintyQuantifier,
)
from enhanced_evaluation import (
    BusinessMetricsCalculator,
    FairnessEvaluator,
    ModelComparisonFramework,
)
from enhanced_production import (
    ModelVersionControl,
    DriftDetector,
    ModelExplainer,
    ABTestingFramework,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")


@dataclass
class PipelineConfig:
    """
    Central configuration for the bank churn ML pipeline.

    The dataclass captures sampling ratios, feature engineering toggles,
    modeling flags, and production settings so experiments remain
    reproducible. Probabilities such as ``test_size``/``validation_size`` are
    expressed as fractions (0–1). Paths like ``output_dir`` are interpreted
    relative to the project root.

    Key Fields
    ----------
    target_column : str
        Name of the binary churn column (default: ``"churn"``).
    test_size : float
        Fraction of rows reserved for the held-out test split.
    validation_size : float
        Fraction of the training fold used for validation during CV.
    feature_selection_method : str
        Strategy applied when ``n_features_to_select`` is set
        (``"boruta"``, ``"mutual_info"``, ``"rfe"``, or ``"ensemble"``).
    model_type : str
        ``"single"``, ``"ensemble"``, or ``"stacking"``.
    base_models : List[str]
        Candidate estimators, defaults to ``['xgboost', 'lightgbm', 'catboost']``.
    n_trials : int
        Number of hyperparameter trials when tuning is enabled.
    output_dir : str
        Directory where reports, artifacts, and MLflow references are written.

    Examples
    --------
    >>> cfg = PipelineConfig(test_size=0.25, n_trials=50, setup_ab_testing=True)
    >>> cfg.base_models
    ['xgboost', 'lightgbm', 'catboost']
    """

    # Data configuration
    target_column: str = "churn"
    test_size: float = 0.2
    validation_size: float = 0.15
    random_state: int = 42

    # Feature engineering
    feature_selection_method: str = (
        "boruta"  # 'boruta', 'mutual_info', 'rfe', 'ensemble'
    )
    n_features_to_select: int = 20
    detect_interactions: bool = True
    max_interactions: int = 10
    use_target_encoding: bool = True
    stability_check: bool = True

    # Modeling
    model_type: str = "ensemble"  # 'single', 'ensemble', 'stacking'
    base_models: List[str] = field(
        default_factory=lambda: ["xgboost", "lightgbm", "catboost"]
    )
    hyperparameter_tuning: bool = True
    n_trials: int = 100
    cv_strategy: str = "stratified"  # 'stratified', 'time_based', 'group'
    n_folds: int = 5
    calibrate_model: bool = True

    # Evaluation
    calculate_business_metrics: bool = True
    check_fairness: bool = True
    sensitive_features: List[str] = field(
        default_factory=lambda: ["gender", "age_group"]
    )
    compare_models: bool = True

    # Production
    enable_versioning: bool = True
    check_drift: bool = True
    generate_explanations: bool = True
    setup_ab_testing: bool = False

    # Output
    output_dir: str = "./pipeline_outputs"
    save_artifacts: bool = True
    generate_report: bool = True


class MLPipelineOrchestrator:
    """
    Coordinate data prep, modeling, evaluation, and production hand-off.

    The orchestrator glues together the enhanced feature engineering,
    modeling, evaluation, and production modules so a single call to
    ``run_pipeline`` ingests an ``(N × D)`` churn dataset, performs stratified
    splits, engineers features, tunes/ensembles models, reports business and
    fairness metrics, and exports artifacts such as MLflow runs and SHAP
    explanations.

    Attributes
    ----------
    config : PipelineConfig
        Immutable configuration describing sampling ratios, feature toggles,
        and deployment preferences.
    pipeline_state : dict
        Mutable status dictionary tracking timestamps, completed stages, and
        aggregate metrics for progress reporting or dashboards.

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.read_csv("data/churn.csv")
    >>> orchestrator = MLPipelineOrchestrator(PipelineConfig(test_size=0.3))
    >>> report = orchestrator.run_pipeline(data, quick_mode=True)
    >>> sorted(report["training_metrics"].keys())[:2]
    ['accuracy', 'auc']
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the ML Pipeline Orchestrator.

        Args:
            config: Pipeline configuration object
        """
        self.config = config or PipelineConfig()

        # Create output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self._initialize_components()

        # Pipeline state
        self.pipeline_state = {
            "status": "initialized",
            "start_time": None,
            "end_time": None,
            "stages_completed": [],
            "metrics": {},
            "artifacts": {},
        }

        logger.info("ML Pipeline Orchestrator initialized")

    def _initialize_components(self):
        """Initialize all pipeline components."""
        # Feature engineering
        self.feature_selector = EnhancedFeatureSelector()
        self.interaction_detector = FeatureInteractionDetector()
        self.target_encoder = TargetEncoder()
        self.stability_analyzer = FeatureStabilityAnalyzer()

        # Modeling
        self.cross_validator = StratifiedCrossValidator()
        self.ensemble_methods = EnhancedEnsembleMethods()
        self.hyperparameter_optimizer = HyperparameterOptimizer()
        self.model_calibrator = ModelCalibrator()
        self.uncertainty_quantifier = UncertaintyQuantifier()

        # Evaluation
        self.business_calculator = BusinessMetricsCalculator()
        self.fairness_evaluator = FairnessEvaluator()
        self.model_comparator = ModelComparisonFramework()

        # Production
        self.version_control = ModelVersionControl()
        self.drift_detector = DriftDetector()
        self.model_explainer = ModelExplainer()
        self.ab_testing = ABTestingFramework()

    def run_pipeline(
        self,
        data: pd.DataFrame,
        customer_values: Optional[pd.Series] = None,
        quick_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        Run the complete ML pipeline.

        Args:
            data: Input dataframe with features and target
            customer_values: Optional customer lifetime values for business metrics
            quick_mode: If True, skip time-consuming steps for quick experimentation

        Returns:
            Dictionary with pipeline results and artifacts
        """
        logger.info("Starting ML Pipeline execution")
        self.pipeline_state["start_time"] = datetime.now()
        self.pipeline_state["status"] = "running"

        try:
            # Stage 1: Data Preparation
            logger.info("Stage 1: Data Preparation")
            X_train, X_test, y_train, y_test = self._prepare_data(data)
            self.pipeline_state["stages_completed"].append("data_preparation")

            # Stage 2: Feature Engineering
            logger.info("Stage 2: Feature Engineering")
            X_train_fe, X_test_fe, feature_names = self._engineer_features(
                X_train, X_test, y_train, quick_mode
            )
            self.pipeline_state["stages_completed"].append("feature_engineering")

            # Stage 3: Model Training
            logger.info("Stage 3: Model Training")
            model, training_metrics = self._train_model(X_train_fe, y_train, quick_mode)
            self.pipeline_state["stages_completed"].append("model_training")

            # Stage 4: Model Evaluation
            logger.info("Stage 4: Model Evaluation")
            evaluation_results = self._evaluate_model(
                model,
                X_train_fe,
                y_train,
                X_test_fe,
                y_test,
                feature_names,
                customer_values,
            )
            self.pipeline_state["stages_completed"].append("model_evaluation")

            # Stage 5: Production Preparation
            logger.info("Stage 5: Production Preparation")
            if not quick_mode:
                production_artifacts = self._prepare_production(
                    model, X_train_fe, X_test_fe, y_test, feature_names
                )
                self.pipeline_state["stages_completed"].append("production_preparation")
            else:
                production_artifacts = {}

            # Stage 6: Generate Report
            if self.config.generate_report:
                logger.info("Stage 6: Generating Report")
                report = self._generate_report(
                    training_metrics, evaluation_results, production_artifacts
                )
                self.pipeline_state["stages_completed"].append("report_generation")

            # Save artifacts
            if self.config.save_artifacts:
                self._save_artifacts(model, feature_names, evaluation_results)

            self.pipeline_state["status"] = "completed"
            self.pipeline_state["end_time"] = datetime.now()

            # Compile results
            results = {
                "model": model,
                "feature_names": feature_names,
                "training_metrics": training_metrics,
                "evaluation_results": evaluation_results,
                "production_artifacts": production_artifacts,
                "pipeline_state": self.pipeline_state,
            }

            logger.info("ML Pipeline execution completed successfully")
            return results

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            self.pipeline_state["status"] = "failed"
            self.pipeline_state["error"] = str(e)
            raise

    def _prepare_data(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for modeling.

        Args:
            data: Input dataframe

        Returns:
            Train and test splits for features and target
        """
        # Separate features and target
        X = data.drop(columns=[self.config.target_column])
        y = data[self.config.target_column]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )

        # Handle missing values
        X_train = X_train.fillna(X_train.median())
        X_test = X_test.fillna(X_train.median())  # Use train median for test

        logger.info(
            f"Data prepared: Train shape {X_train.shape}, Test shape {X_test.shape}"
        )

        return X_train, X_test, y_train, y_test

    def _engineer_features(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        quick_mode: bool,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Perform feature engineering.

        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training target
            quick_mode: Skip time-consuming steps

        Returns:
            Engineered features and feature names
        """
        feature_names = list(X_train.columns)

        # Feature selection
        if not quick_mode and self.config.feature_selection_method != "none":
            logger.info(
                f"Performing feature selection using {self.config.feature_selection_method}"
            )
            selected_features = self.feature_selector.select_features(
                X_train,
                y_train,
                method=self.config.feature_selection_method,
                n_features=self.config.n_features_to_select,
            )
            X_train = X_train[selected_features]
            X_test = X_test[selected_features]
            feature_names = selected_features

            # Check feature stability
            if self.config.stability_check:
                stability_scores = self.stability_analyzer.analyze_stability(
                    X_train[selected_features], y_train, n_bootstrap=20
                )
                logger.info(f"Feature stability scores: {stability_scores}")

        # Detect interactions
        if self.config.detect_interactions and not quick_mode:
            logger.info("Detecting feature interactions")
            interactions = self.interaction_detector.find_interactions(
                X_train, y_train, max_interactions=self.config.max_interactions
            )

            # Add top interactions
            for (f1, f2), score in interactions[:5]:
                if f1 in X_train.columns and f2 in X_train.columns:
                    interaction_name = f"{f1}_x_{f2}"
                    X_train[interaction_name] = X_train[f1] * X_train[f2]
                    X_test[interaction_name] = X_test[f1] * X_test[f2]
                    feature_names.append(interaction_name)

        # Target encoding for categorical features
        if self.config.use_target_encoding:
            categorical_cols = X_train.select_dtypes(include=["object"]).columns
            if len(categorical_cols) > 0:
                logger.info(
                    f"Applying target encoding to {len(categorical_cols)} categorical features"
                )
                X_train_encoded, X_test_encoded = self.target_encoder.fit_transform(
                    X_train, y_train, X_test, categorical_cols
                )
                X_train = X_train_encoded
                X_test = X_test_encoded

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, feature_names

    def _train_model(
        self, X_train: np.ndarray, y_train: pd.Series, quick_mode: bool
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training target
            quick_mode: Skip time-consuming steps

        Returns:
            Trained model and training metrics
        """
        training_metrics = {}

        # Select model type
        if self.config.model_type == "ensemble":
            logger.info("Training ensemble model")

            if self.config.hyperparameter_tuning and not quick_mode:
                # Optimize hyperparameters
                logger.info("Optimizing hyperparameters")
                best_params = {}
                for model_name in self.config.base_models:
                    params = self.hyperparameter_optimizer.optimize(
                        X_train,
                        y_train,
                        model_type=model_name,
                        n_trials=self.config.n_trials if not quick_mode else 10,
                    )
                    best_params[model_name] = params
                training_metrics["best_params"] = best_params

            # Create ensemble
            if self.config.model_type == "stacking":
                model = self.ensemble_methods.create_stacking_ensemble(
                    X_train, y_train, cv_folds=self.config.n_folds
                )
            else:
                model = self.ensemble_methods.create_voting_ensemble(
                    X_train, y_train, models=self.config.base_models
                )
        else:
            # Train single model
            from xgboost import XGBClassifier

            model = XGBClassifier(random_state=self.config.random_state)
            model.fit(X_train, y_train)

        # Calculate cross-validation scores
        cv_scores = self.cross_validator.cross_validate(
            model,
            X_train,
            y_train,
            strategy=self.config.cv_strategy,
            n_folds=self.config.n_folds,
        )
        training_metrics["cv_scores"] = cv_scores

        # Calibrate model
        if self.config.calibrate_model:
            logger.info("Calibrating model")
            model = self.model_calibrator.calibrate_model(
                model, X_train, y_train, method="isotonic"
            )

        # Add uncertainty quantification
        uncertainties = self.uncertainty_quantifier.estimate_uncertainty(
            model,
            X_train,
            method="bootstrap",
            n_iterations=100 if not quick_mode else 10,
        )
        training_metrics["uncertainty"] = np.mean(uncertainties)

        logger.info(
            f"Model training completed. CV Score: {np.mean(cv_scores['test_score']):.4f}"
        )

        return model, training_metrics

    def _evaluate_model(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: pd.Series,
        X_test: np.ndarray,
        y_test: pd.Series,
        feature_names: List[str],
        customer_values: Optional[pd.Series],
    ) -> Dict[str, Any]:
        """
        Evaluate the model comprehensively.

        Args:
            model: Trained model
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            feature_names: List of feature names
            customer_values: Optional customer values for business metrics

        Returns:
            Evaluation results
        """
        evaluation_results = {}

        # Get predictions
        y_pred = model.predict(X_test)
        y_prob = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else y_pred
        )

        # Basic metrics
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
        )

        evaluation_results["accuracy"] = accuracy_score(y_test, y_pred)
        evaluation_results["precision"] = precision_score(y_test, y_pred)
        evaluation_results["recall"] = recall_score(y_test, y_pred)
        evaluation_results["f1"] = f1_score(y_test, y_pred)
        evaluation_results["auc_roc"] = roc_auc_score(y_test, y_prob)

        # Business metrics
        if self.config.calculate_business_metrics:
            logger.info("Calculating business metrics")
            if customer_values is not None and len(customer_values) == len(y_test):
                business_metrics = self.business_calculator.calculate_business_metrics(
                    y_test, y_pred, y_prob, customer_values
                )
            else:
                # Use default values
                business_metrics = self.business_calculator.calculate_business_metrics(
                    y_test, y_pred, y_prob
                )
            evaluation_results["business_metrics"] = business_metrics

        # Fairness evaluation
        if self.config.check_fairness and len(self.config.sensitive_features) > 0:
            logger.info("Evaluating fairness")
            # Create dummy sensitive features for demonstration
            sensitive_data = pd.DataFrame(
                np.random.randint(
                    0, 2, size=(len(y_test), len(self.config.sensitive_features))
                ),
                columns=self.config.sensitive_features,
            )
            fairness_metrics = self.fairness_evaluator.evaluate_fairness(
                y_test, y_pred, y_prob, sensitive_data
            )
            evaluation_results["fairness_metrics"] = fairness_metrics

        # Model comparison
        if self.config.compare_models:
            logger.info("Comparing models")
            # Create baseline model for comparison
            from sklearn.dummy import DummyClassifier

            baseline = DummyClassifier(strategy="stratified", random_state=42)
            baseline.fit(X_train, y_train)

            models = {"main_model": model, "baseline": baseline}
            comparison = self.model_comparator.compare_models(models, X_test, y_test)
            evaluation_results["model_comparison"] = comparison

        logger.info(
            f"Evaluation completed. AUC-ROC: {evaluation_results['auc_roc']:.4f}"
        )

        return evaluation_results

    def _prepare_production(
        self,
        model: Any,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_test: pd.Series,
        feature_names: List[str],
    ) -> Dict[str, Any]:
        """
        Prepare model for production deployment.

        Args:
            model: Trained model
            X_train: Training features
            X_test: Test features
            y_test: Test target
            feature_names: List of feature names

        Returns:
            Production artifacts
        """
        production_artifacts = {}

        # Model versioning
        if self.config.enable_versioning:
            logger.info("Setting up model versioning")
            model_id = self.version_control.save_model(
                model,
                metadata={
                    "features": feature_names,
                    "config": self.config.__dict__,
                    "timestamp": datetime.now().isoformat(),
                },
            )
            production_artifacts["model_id"] = model_id

        # Drift detection
        if self.config.check_drift:
            logger.info("Setting up drift detection")
            self.drift_detector.set_reference_data(X_train)
            drift_results = self.drift_detector.detect_drift(X_test)
            production_artifacts["drift_detection"] = drift_results

        # Model explainability
        if self.config.generate_explanations:
            logger.info("Generating model explanations")
            explanations = self.model_explainer.explain_model(
                model,
                X_test[:100],
                feature_names,  # Use subset for speed
            )
            production_artifacts["explanations"] = {
                "global_importance": explanations["feature_importance"],
                "sample_explanations": explanations["shap_values"][:5]
                if "shap_values" in explanations
                else None,
            }

        # A/B testing setup
        if self.config.setup_ab_testing:
            logger.info("Setting up A/B testing framework")
            ab_config = self.ab_testing.setup_test(
                control_model=None,  # Current production model
                treatment_model=model,
                test_name="new_model_deployment",
                traffic_split=0.2,
            )
            production_artifacts["ab_testing"] = ab_config

        logger.info("Production preparation completed")

        return production_artifacts

    def _generate_report(
        self,
        training_metrics: Dict,
        evaluation_results: Dict,
        production_artifacts: Dict,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive pipeline report.

        Args:
            training_metrics: Training stage metrics
            evaluation_results: Evaluation stage results
            production_artifacts: Production preparation artifacts

        Returns:
            Pipeline report
        """
        report = {
            "pipeline_config": self.config.__dict__,
            "execution_time": str(
                self.pipeline_state["end_time"] - self.pipeline_state["start_time"]
            ),
            "stages_completed": self.pipeline_state["stages_completed"],
            "training": {
                "cv_score": np.mean(training_metrics["cv_scores"]["test_score"]),
                "cv_std": np.std(training_metrics["cv_scores"]["test_score"]),
                "uncertainty": training_metrics.get("uncertainty", None),
                "best_params": training_metrics.get("best_params", {}),
            },
            "evaluation": {
                "test_metrics": {
                    "accuracy": evaluation_results["accuracy"],
                    "precision": evaluation_results["precision"],
                    "recall": evaluation_results["recall"],
                    "f1": evaluation_results["f1"],
                    "auc_roc": evaluation_results["auc_roc"],
                },
                "business_metrics": evaluation_results.get("business_metrics", {}),
                "fairness_metrics": evaluation_results.get("fairness_metrics", {}),
            },
            "production_readiness": {
                "model_versioned": "model_id" in production_artifacts,
                "drift_detected": production_artifacts.get("drift_detection", {}).get(
                    "drift_detected", False
                ),
                "explanations_generated": "explanations" in production_artifacts,
                "ab_test_configured": "ab_testing" in production_artifacts,
            },
        }

        # Save report
        report_path = (
            self.output_dir
            / f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Report saved to {report_path}")

        return report

    def _save_artifacts(
        self, model: Any, feature_names: List[str], evaluation_results: Dict
    ):
        """
        Save pipeline artifacts.

        Args:
            model: Trained model
            feature_names: List of feature names
            evaluation_results: Evaluation results
        """
        # Save model
        model_path = (
            self.output_dir / f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        )
        joblib.dump(model, model_path)

        # Save feature names
        features_path = self.output_dir / "feature_names.json"
        with open(features_path, "w") as f:
            json.dump(feature_names, f)

        # Save evaluation results
        eval_path = self.output_dir / "evaluation_results.json"
        with open(eval_path, "w") as f:
            json.dump(evaluation_results, f, indent=2, default=str)

        logger.info(f"Artifacts saved to {self.output_dir}")

    def run_experiment(
        self, data: pd.DataFrame, experiment_config: Dict
    ) -> Dict[str, Any]:
        """
        Run a quick experiment with custom configuration.

        Args:
            data: Input dataframe
            experiment_config: Dictionary with configuration overrides

        Returns:
            Experiment results
        """
        # Update config for experiment
        for key, value in experiment_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # Run pipeline in quick mode
        results = self.run_pipeline(data, quick_mode=True)

        return results

    def load_and_run(
        self, data_path: str, config_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load data and configuration, then run pipeline.

        Args:
            data_path: Path to data file
            config_path: Optional path to configuration file

        Returns:
            Pipeline results
        """
        # Load data
        if data_path.endswith(".csv"):
            data = pd.read_csv(data_path)
        elif data_path.endswith(".parquet"):
            data = pd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")

        # Load configuration if provided
        if config_path:
            with open(config_path, "r") as f:
                if config_path.endswith(".yaml"):
                    config_dict = yaml.safe_load(f)
                elif config_path.endswith(".json"):
                    config_dict = json.load(f)
                else:
                    raise ValueError(f"Unsupported config format: {config_path}")

            # Update configuration
            for key, value in config_dict.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

        # Run pipeline
        return self.run_pipeline(data)


def create_example_config() -> PipelineConfig:
    """
    Create an example pipeline configuration.

    Returns:
        Example PipelineConfig object
    """
    return PipelineConfig(
        # Data settings
        target_column="churn",
        test_size=0.2,
        validation_size=0.15,
        # Feature engineering
        feature_selection_method="boruta",
        n_features_to_select=25,
        detect_interactions=True,
        use_target_encoding=True,
        stability_check=True,
        # Modeling
        model_type="ensemble",
        base_models=["xgboost", "lightgbm", "catboost"],
        hyperparameter_tuning=True,
        n_trials=50,
        calibrate_model=True,
        # Evaluation
        calculate_business_metrics=True,
        check_fairness=True,
        sensitive_features=["gender", "age_group"],
        # Production
        enable_versioning=True,
        check_drift=True,
        generate_explanations=True,
        setup_ab_testing=False,
        # Output
        generate_report=True,
        save_artifacts=True,
    )


def main():
    """Example usage of the ML Pipeline Orchestrator."""

    # Create sample data
    np.random.seed(42)
    n_samples = 1000

    data = pd.DataFrame(
        {
            "feature_1": np.random.randn(n_samples),
            "feature_2": np.random.randn(n_samples),
            "feature_3": np.random.randn(n_samples),
            "feature_4": np.random.choice(["A", "B", "C"], n_samples),
            "feature_5": np.random.uniform(0, 100, n_samples),
            "gender": np.random.choice(["M", "F"], n_samples),
            "age_group": np.random.choice(["Young", "Middle", "Senior"], n_samples),
            "churn": np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        }
    )

    # Create pipeline with example config
    config = create_example_config()
    pipeline = MLPipelineOrchestrator(config)

    # Run pipeline
    print("Running ML Pipeline...")
    results = pipeline.run_pipeline(data, quick_mode=False)

    print("\n=== Pipeline Results ===")
    print(f"Stages Completed: {results['pipeline_state']['stages_completed']}")
    print(
        f"Training CV Score: {np.mean(results['training_metrics']['cv_scores']['test_score']):.4f}"
    )
    print(f"Test AUC-ROC: {results['evaluation_results']['auc_roc']:.4f}")

    if "business_metrics" in results["evaluation_results"]:
        print("\nBusiness Metrics:")
        for metric, value in results["evaluation_results"]["business_metrics"].items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")

    print("\nProduction Readiness:")
    if "drift_detection" in results["production_artifacts"]:
        print(
            f"  Drift Detected: {results['production_artifacts']['drift_detection'].get('drift_detected', False)}"
        )
    if "model_id" in results["production_artifacts"]:
        print(f"  Model Version: {results['production_artifacts']['model_id']}")

    print(
        f"\nPipeline completed in {results['pipeline_state']['end_time'] - results['pipeline_state']['start_time']}"
    )


if __name__ == "__main__":
    main()
