"""Advanced Modeling Techniques for Bank Churn Prediction.
Implements ensemble methods, hyperparameter tuning, calibration, and uncertainty quantification.
"""

import warnings
from dataclasses import dataclass

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score

warnings.filterwarnings("ignore")


@dataclass
class ModelResult:
    """Container for model results."""

    model_name: str
    cv_scores: np.ndarray
    mean_score: float
    std_score: float
    predictions: np.ndarray
    probabilities: np.ndarray
    feature_importance: pd.DataFrame = None
    calibration_score: float = None
    uncertainty: np.ndarray = None


class StratifiedCrossValidator:
    """Advanced cross-validation with proper stratification."""

    def __init__(self, n_splits: int = 5, random_state: int = 42):
        """Initialize cross-validator.

        Args:
            n_splits: Number of CV folds
            random_state: Random seed
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.cv = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )

    def validate_model(
        self, model, X: pd.DataFrame, y: pd.Series, scoring: str = "roc_auc"
    ) -> ModelResult:
        """Perform stratified cross-validation.

        Args:
            model: Scikit-learn compatible model
            X: Feature DataFrame
            y: Target Series
            scoring: Scoring metric

        Returns:
            ModelResult with CV scores
        """
        # Cross-validation scores
        cv_scores = cross_val_score(model, X, y, cv=self.cv, scoring=scoring, n_jobs=-1)

        # Cross-validation predictions
        if hasattr(model, "predict_proba"):
            probabilities = cross_val_predict(
                model, X, y, cv=self.cv, method="predict_proba", n_jobs=-1
            )[:, 1]
        else:
            probabilities = cross_val_predict(
                model, X, y, cv=self.cv, method="decision_function", n_jobs=-1
            )

        predictions = cross_val_predict(model, X, y, cv=self.cv, n_jobs=-1)

        # Feature importance (if available)
        feature_importance = None
        if hasattr(model, "feature_importances_"):
            model.fit(X, y)
            feature_importance = pd.DataFrame(
                {"feature": X.columns, "importance": model.feature_importances_}
            ).sort_values("importance", ascending=False)

        return ModelResult(
            model_name=type(model).__name__,
            cv_scores=cv_scores,
            mean_score=cv_scores.mean(),
            std_score=cv_scores.std(),
            predictions=predictions,
            probabilities=probabilities,
            feature_importance=feature_importance,
        )

    def nested_cv(
        self,
        model,
        param_grid: dict,
        X: pd.DataFrame,
        y: pd.Series,
        scoring: str = "roc_auc",
    ) -> tuple[float, dict]:
        """Nested cross-validation for unbiased performance estimation.

        Args:
            model: Base model
            param_grid: Parameter grid for tuning
            X: Feature DataFrame
            y: Target Series
            scoring: Scoring metric

        Returns:
            Unbiased score and best parameters
        """
        from sklearn.model_selection import GridSearchCV

        outer_scores = []
        best_params_list = []

        for train_idx, test_idx in self.cv.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Inner CV for hyperparameter tuning
            inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            grid_search = GridSearchCV(
                model, param_grid, cv=inner_cv, scoring=scoring, n_jobs=-1
            )
            grid_search.fit(X_train, y_train)

            # Evaluate on outer fold
            score = grid_search.score(X_test, y_test)
            outer_scores.append(score)
            best_params_list.append(grid_search.best_params_)

        # Most common best parameters
        from collections import Counter

        param_counter = Counter(str(p) for p in best_params_list)
        best_params = eval(param_counter.most_common(1)[0][0])

        return np.mean(outer_scores), best_params


class EnsembleMethods:
    """Advanced ensemble methods including stacking and blending."""

    def __init__(
        self, base_models: list = None, meta_model=None, use_probabilities: bool = True
    ):
        """Initialize ensemble methods.

        Args:
            base_models: List of base models
            meta_model: Meta-learner for stacking
            use_probabilities: Use probabilities instead of predictions
        """
        if base_models is None:
            base_models = [
                RandomForestClassifier(n_estimators=100, random_state=42),
                GradientBoostingClassifier(n_estimators=100, random_state=42),
                lgb.LGBMClassifier(n_estimators=100, random_state=42, verbosity=-1),
                xgb.XGBClassifier(
                    n_estimators=100, random_state=42, use_label_encoder=False
                ),
            ]

        if meta_model is None:
            meta_model = LogisticRegression(random_state=42)

        self.base_models = base_models
        self.meta_model = meta_model
        self.use_probabilities = use_probabilities
        self.stacking_model = None
        self.voting_model = None
        self.blending_weights = None

    def create_stacking_ensemble(self) -> StackingClassifier:
        """Create stacking ensemble.

        Returns:
            StackingClassifier
        """
        estimators = [(f"model_{i}", model) for i, model in enumerate(self.base_models)]

        self.stacking_model = StackingClassifier(
            estimators=estimators,
            final_estimator=self.meta_model,
            cv=5,  # Use cross-validation to train meta-model
            stack_method="predict_proba" if self.use_probabilities else "predict",
            n_jobs=-1,
        )

        return self.stacking_model

    def create_voting_ensemble(self, voting: str = "soft") -> VotingClassifier:
        """Create voting ensemble.

        Args:
            voting: 'hard' for majority voting, 'soft' for probability averaging

        Returns:
            VotingClassifier
        """
        estimators = [(f"model_{i}", model) for i, model in enumerate(self.base_models)]

        self.voting_model = VotingClassifier(
            estimators=estimators, voting=voting, n_jobs=-1
        )

        return self.voting_model

    def blend_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> np.ndarray:
        """Blend models using validation set.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target

        Returns:
            Blending weights
        """
        from scipy.optimize import minimize

        # Train base models
        predictions = []
        for model in self.base_models:
            model.fit(X_train, y_train)
            if hasattr(model, "predict_proba"):
                pred = model.predict_proba(X_val)[:, 1]
            else:
                pred = model.predict(X_val)
            predictions.append(pred)

        predictions = np.array(predictions).T

        # Optimize blending weights
        def loss_function(weights):
            blended = np.dot(predictions, weights)
            # Log loss
            epsilon = 1e-15
            blended = np.clip(blended, epsilon, 1 - epsilon)
            return -np.mean(y_val * np.log(blended) + (1 - y_val) * np.log(1 - blended))

        # Constraints: weights sum to 1, all non-negative
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(len(self.base_models))]

        # Initial weights (uniform)
        initial_weights = np.ones(len(self.base_models)) / len(self.base_models)

        # Optimize
        result = minimize(
            loss_function,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        self.blending_weights = result.x
        return self.blending_weights

    def predict_blended(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using blended ensemble.

        Args:
            X: Feature DataFrame

        Returns:
            Blended predictions
        """
        if self.blending_weights is None:
            raise ValueError("Must call blend_models first")

        predictions = []
        for model in self.base_models:
            if hasattr(model, "predict_proba"):
                pred = model.predict_proba(X)[:, 1]
            else:
                pred = model.predict(X)
            predictions.append(pred)

        predictions = np.array(predictions).T
        blended = np.dot(predictions, self.blending_weights)

        return blended


class HyperparameterOptimizer:
    """Advanced hyperparameter optimization using Optuna."""

    def __init__(
        self, model_type: str = "lightgbm", n_trials: int = 100, cv_folds: int = 5
    ):
        """Initialize optimizer.

        Args:
            model_type: Type of model to optimize
            n_trials: Number of optimization trials
            cv_folds: Number of CV folds
        """
        self.model_type = model_type
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.study = None
        self.best_params = None
        self.best_model = None

    def optimize(self, X: pd.DataFrame, y: pd.Series, scoring: str = "roc_auc") -> dict:
        """Optimize hyperparameters.

        Args:
            X: Feature DataFrame
            y: Target Series
            scoring: Scoring metric

        Returns:
            Best parameters
        """

        def objective(trial):
            if self.model_type == "lightgbm":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "max_depth": trial.suggest_int("max_depth", 3, 20),
                    "learning_rate": trial.suggest_loguniform(
                        "learning_rate", 0.001, 0.3
                    ),
                    "num_leaves": trial.suggest_int("num_leaves", 10, 300),
                    "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                    "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_uniform(
                        "colsample_bytree", 0.5, 1.0
                    ),
                    "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-8, 10.0),
                    "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-8, 10.0),
                    "random_state": 42,
                    "verbosity": -1,
                }
                model = lgb.LGBMClassifier(**params)

            elif self.model_type == "xgboost":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "max_depth": trial.suggest_int("max_depth", 3, 20),
                    "learning_rate": trial.suggest_loguniform(
                        "learning_rate", 0.001, 0.3
                    ),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                    "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_uniform(
                        "colsample_bytree", 0.5, 1.0
                    ),
                    "gamma": trial.suggest_loguniform("gamma", 1e-8, 1.0),
                    "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-8, 10.0),
                    "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-8, 10.0),
                    "random_state": 42,
                    "use_label_encoder": False,
                    "eval_metric": "logloss",
                }
                model = xgb.XGBClassifier(**params)

            elif self.model_type == "random_forest":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "max_depth": trial.suggest_int("max_depth", 3, 50),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                    "max_features": trial.suggest_categorical(
                        "max_features", ["sqrt", "log2", None]
                    ),
                    "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
                    "random_state": 42,
                    "n_jobs": -1,
                }
                model = RandomForestClassifier(**params)

            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

            # Cross-validation
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

            return scores.mean()

        # Create and run study
        self.study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
        )
        self.study.optimize(objective, n_trials=self.n_trials, n_jobs=1)

        # Get best parameters
        self.best_params = self.study.best_params

        # Train best model
        if self.model_type == "lightgbm":
            self.best_model = lgb.LGBMClassifier(**self.best_params)
        elif self.model_type == "xgboost":
            self.best_model = xgb.XGBClassifier(**self.best_params)
        elif self.model_type == "random_forest":
            self.best_model = RandomForestClassifier(**self.best_params)

        self.best_model.fit(X, y)

        return self.best_params

    def get_optimization_history(self) -> pd.DataFrame:
        """Get optimization history as DataFrame."""
        if self.study is None:
            return pd.DataFrame()

        history = []
        for trial in self.study.trials:
            trial_data = {"trial": trial.number, "value": trial.value}
            trial_data.update(trial.params)
            history.append(trial_data)

        return pd.DataFrame(history)


class ModelCalibrator:
    """Model calibration for reliable probability estimates."""

    def __init__(self, method: str = "isotonic", cv_folds: int = 3):
        """Initialize calibrator.

        Args:
            method: Calibration method ('isotonic' or 'sigmoid')
            cv_folds: Number of CV folds for calibration
        """
        self.method = method
        self.cv_folds = cv_folds
        self.calibrated_model = None
        self.calibration_scores = {}

    def calibrate_model(self, model, X: pd.DataFrame, y: pd.Series):
        """Calibrate model probabilities.

        Args:
            model: Base model to calibrate
            X: Feature DataFrame
            y: Target Series

        Returns:
            Calibrated model
        """
        self.calibrated_model = CalibratedClassifierCV(
            model, method=self.method, cv=self.cv_folds
        )
        self.calibrated_model.fit(X, y)

        return self.calibrated_model

    def evaluate_calibration(
        self, X: pd.DataFrame, y: pd.Series, n_bins: int = 10
    ) -> dict:
        """Evaluate calibration quality.

        Args:
            X: Feature DataFrame
            y: Target Series
            n_bins: Number of bins for calibration plot

        Returns:
            Calibration metrics
        """
        if self.calibrated_model is None:
            raise ValueError("Must calibrate model first")

        probabilities = self.calibrated_model.predict_proba(X)[:, 1]

        # Calculate calibration metrics
        from sklearn.calibration import calibration_curve

        fraction_pos, mean_pred = calibration_curve(
            y, probabilities, n_bins=n_bins, strategy="uniform"
        )

        # Expected Calibration Error (ECE)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0
        for bin_lower, bin_upper, frac_pos, pred_prob in zip(
            bin_lowers, bin_uppers, fraction_pos, mean_pred, strict=False
        ):
            # Proportion of samples in bin
            in_bin = np.logical_and(
                probabilities > bin_lower, probabilities <= bin_upper
            )
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                # Calibration error for this bin
                bin_error = np.abs(frac_pos - pred_prob)
                ece += prop_in_bin * bin_error

        # Brier Score
        from sklearn.metrics import brier_score_loss

        brier_score = brier_score_loss(y, probabilities)

        self.calibration_scores = {
            "ece": ece,
            "brier_score": brier_score,
            "fraction_positives": fraction_pos,
            "mean_predictions": mean_pred,
        }

        return self.calibration_scores


class UncertaintyQuantifier:
    """Quantify prediction uncertainty."""

    def __init__(self, n_estimators: int = 100, method: str = "bootstrap"):
        """Initialize uncertainty quantifier.

        Args:
            n_estimators: Number of estimators for uncertainty
            method: Method for uncertainty ('bootstrap' or 'dropout')
        """
        self.n_estimators = n_estimators
        self.method = method
        self.models = []
        self.uncertainty_scores = None

    def fit(self, base_model, X: pd.DataFrame, y: pd.Series):
        """Fit uncertainty estimator.

        Args:
            base_model: Base model to use
            X: Feature DataFrame
            y: Target Series
        """
        if self.method == "bootstrap":
            # Train multiple models on bootstrap samples
            for i in range(self.n_estimators):
                # Bootstrap sample
                indices = np.random.choice(len(X), size=len(X), replace=True)
                X_boot = X.iloc[indices]
                y_boot = y.iloc[indices]

                # Clone and train model
                from sklearn.base import clone

                model = clone(base_model)
                model.fit(X_boot, y_boot)
                self.models.append(model)

        elif self.method == "dropout":
            # For neural networks with MC Dropout
            # This would require a neural network with dropout layers
            raise NotImplementedError("Dropout method requires neural network")

        else:
            raise ValueError(f"Unknown method: {self.method}")

    def predict_with_uncertainty(
        self, X: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Make predictions with uncertainty estimates.

        Args:
            X: Feature DataFrame

        Returns:
            Predictions and uncertainty estimates
        """
        if not self.models:
            raise ValueError("Must fit uncertainty estimator first")

        # Get predictions from all models
        predictions = []
        for model in self.models:
            if hasattr(model, "predict_proba"):
                pred = model.predict_proba(X)[:, 1]
            else:
                pred = model.predict(X)
            predictions.append(pred)

        predictions = np.array(predictions)

        # Mean prediction
        mean_prediction = predictions.mean(axis=0)

        # Uncertainty (standard deviation)
        uncertainty = predictions.std(axis=0)

        # Epistemic uncertainty (model uncertainty)
        epistemic = uncertainty

        # Aleatoric uncertainty (data uncertainty) - approximation
        aleatoric = np.mean([pred * (1 - pred) for pred in predictions], axis=0)

        self.uncertainty_scores = {
            "total": uncertainty,
            "epistemic": epistemic,
            "aleatoric": aleatoric,
        }

        return mean_prediction, uncertainty

    def get_confidence_intervals(
        self, X: pd.DataFrame, confidence: float = 0.95
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get confidence intervals for predictions.

        Args:
            X: Feature DataFrame
            confidence: Confidence level

        Returns:
            Lower and upper bounds
        """
        if not self.models:
            raise ValueError("Must fit uncertainty estimator first")

        # Get predictions from all models
        predictions = []
        for model in self.models:
            if hasattr(model, "predict_proba"):
                pred = model.predict_proba(X)[:, 1]
            else:
                pred = model.predict(X)
            predictions.append(pred)

        predictions = np.array(predictions)

        # Calculate percentiles
        alpha = (1 - confidence) / 2
        lower = np.percentile(predictions, alpha * 100, axis=0)
        upper = np.percentile(predictions, (1 - alpha) * 100, axis=0)

        return lower, upper


class AdvancedModelPipeline:
    """Complete advanced modeling pipeline."""

    def __init__(self):
        self.cv_validator = StratifiedCrossValidator()
        self.ensemble = EnsembleMethods()
        self.optimizer = HyperparameterOptimizer()
        self.calibrator = ModelCalibrator()
        self.uncertainty = UncertaintyQuantifier()
        self.results = {}

    def run_complete_pipeline(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        optimize_hyperparams: bool = True,
        use_ensemble: bool = True,
        calibrate: bool = True,
        quantify_uncertainty: bool = True,
    ) -> dict:
        """Run complete modeling pipeline.

        Args:
            X: Feature DataFrame
            y: Target Series
            optimize_hyperparams: Whether to optimize hyperparameters
            use_ensemble: Whether to use ensemble methods
            calibrate: Whether to calibrate probabilities
            quantify_uncertainty: Whether to quantify uncertainty

        Returns:
            Dictionary with all results
        """
        results = {}

        # 1. Hyperparameter optimization
        if optimize_hyperparams:
            print("Optimizing hyperparameters...")
            best_params = self.optimizer.optimize(X, y)
            results["best_params"] = best_params
            results["optimization_history"] = self.optimizer.get_optimization_history()
            base_model = self.optimizer.best_model
        else:
            base_model = lgb.LGBMClassifier(
                n_estimators=100, random_state=42, verbosity=-1
            )
            base_model.fit(X, y)

        # 2. Cross-validation
        print("Running cross-validation...")
        cv_result = self.cv_validator.validate_model(base_model, X, y)
        results["cv_result"] = cv_result

        # 3. Ensemble methods
        if use_ensemble:
            print("Creating ensemble...")
            stacking = self.ensemble.create_stacking_ensemble()
            ensemble_result = self.cv_validator.validate_model(stacking, X, y)
            results["ensemble_result"] = ensemble_result

        # 4. Calibration
        if calibrate:
            print("Calibrating model...")
            self.calibrator.calibrate_model(base_model, X, y)
            calibration_scores = self.calibrator.evaluate_calibration(X, y)
            results["calibration_scores"] = calibration_scores

        # 5. Uncertainty quantification
        if quantify_uncertainty:
            print("Quantifying uncertainty...")
            self.uncertainty.fit(base_model, X, y)
            predictions, uncertainty = self.uncertainty.predict_with_uncertainty(X)
            results["uncertainty"] = {
                "predictions": predictions,
                "uncertainty": uncertainty,
                "scores": self.uncertainty.uncertainty_scores,
            }

        self.results = results
        return results

    def save_pipeline(self, filepath: str):
        """Save pipeline to file."""
        joblib.dump(self, filepath)

    @staticmethod
    def load_pipeline(filepath: str):
        """Load pipeline from file."""
        return joblib.load(filepath)


# Example usage
if __name__ == "__main__":
    # Create sample data
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42
    )
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(20)])
    y = pd.Series(y)

    # Run pipeline
    pipeline = AdvancedModelPipeline()
    results = pipeline.run_complete_pipeline(
        X,
        y,
        optimize_hyperparams=False,  # Set to True for real use
        use_ensemble=True,
        calibrate=True,
        quantify_uncertainty=True,
    )

    print("\nPipeline Results:")
    print(
        f"CV Score: {results['cv_result'].mean_score:.4f} (+/- {results['cv_result'].std_score:.4f})"
    )
    if "ensemble_result" in results:
        print(f"Ensemble Score: {results['ensemble_result'].mean_score:.4f}")
    if "calibration_scores" in results:
        print(f"Calibration ECE: {results['calibration_scores']['ece']:.4f}")
    print("\nPipeline complete!")
