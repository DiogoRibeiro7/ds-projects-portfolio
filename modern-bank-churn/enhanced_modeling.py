"""Enhanced Modeling for Bank Churn Prediction.
Advanced modeling with proper cross-validation, ensemble methods,
hyperparameter tuning, calibration, and uncertainty quantification.
"""

import warnings
from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class ModelResult:
    """Container for comprehensive model results."""

    model_name: str
    model: Any
    cv_scores: np.ndarray
    test_score: float
    predictions: np.ndarray
    prediction_probas: np.ndarray
    confidence_intervals: np.ndarray
    calibration_metrics: dict[str, float]
    best_params: dict[str, Any] = field(default_factory=dict)
    feature_importance: pd.DataFrame = None


class StratifiedCrossValidator:
    """Enhanced cross-validation with proper stratification and
    multiple validation strategies.
    """

    def __init__(
        self,
        n_splits: int = 5,
        validation_strategy: str = "stratified_kfold",
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        """Initialize cross-validator.

        Args:
            n_splits: Number of CV folds
            validation_strategy: Type of validation strategy
            test_size: Size of test set for holdout
            random_state: Random seed
        """
        self.n_splits = n_splits
        self.validation_strategy = validation_strategy
        self.test_size = test_size
        self.random_state = random_state
        self.cv_results_ = {}

    def validate(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        scoring: str | list[str] = "roc_auc",
        return_predictions: bool = True,
    ) -> dict[str, Any]:
        """Perform cross-validation with multiple metrics.

        Args:
            model: Model to validate
            X: Features
            y: Target
            scoring: Scoring metric(s)
            return_predictions: Whether to return out-of-fold predictions

        Returns:
            Dictionary with CV results
        """
        if isinstance(scoring, str):
            scoring = [scoring]

        # Initialize results
        results = {
            "scores": {},
            "predictions": None,
            "prediction_probas": None,
            "feature_importance": [],
        }

        # Get cross-validator
        if self.validation_strategy == "stratified_kfold":
            cv = StratifiedKFold(
                n_splits=self.n_splits, shuffle=True, random_state=self.random_state
            )
        elif self.validation_strategy == "repeated_stratified_kfold":
            from sklearn.model_selection import RepeatedStratifiedKFold

            cv = RepeatedStratifiedKFold(
                n_splits=self.n_splits, n_repeats=3, random_state=self.random_state
            )
        elif self.validation_strategy == "time_series_split":
            from sklearn.model_selection import TimeSeriesSplit

            cv = TimeSeriesSplit(n_splits=self.n_splits)
        else:
            raise ValueError(f"Unknown validation strategy: {self.validation_strategy}")

        # Perform cross-validation for each metric
        for metric in scoring:
            scores = cross_val_score(model, X, y, cv=cv, scoring=metric, n_jobs=-1)
            results["scores"][metric] = {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "values": scores,
                "ci_lower": np.percentile(scores, 2.5),
                "ci_upper": np.percentile(scores, 97.5),
            }

        # Get out-of-fold predictions if requested
        if return_predictions:
            results["predictions"] = cross_val_predict(
                clone(model), X, y, cv=cv, method="predict", n_jobs=-1
            )

            if hasattr(model, "predict_proba"):
                results["prediction_probas"] = cross_val_predict(
                    clone(model), X, y, cv=cv, method="predict_proba", n_jobs=-1
                )

        # Get feature importance from each fold
        if hasattr(model, "feature_importances_") or hasattr(model, "coef_"):
            for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
                X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                fold_model = clone(model)
                fold_model.fit(X_train, y_train)

                if hasattr(fold_model, "feature_importances_"):
                    importance = fold_model.feature_importances_
                elif hasattr(fold_model, "coef_"):
                    importance = np.abs(fold_model.coef_.flatten())
                else:
                    importance = np.zeros(X.shape[1])

                results["feature_importance"].append(importance)

            # Average feature importance across folds
            results["feature_importance"] = pd.DataFrame(
                np.mean(results["feature_importance"], axis=0),
                index=X.columns,
                columns=["importance"],
            ).sort_values("importance", ascending=False)

        return results


class EnhancedEnsembleMethods:
    """Advanced ensemble methods including stacking, blending,
    and custom ensemble strategies.
    """

    def __init__(self, base_models: list[tuple[str, BaseEstimator]] = None):
        """Initialize ensemble methods.

        Args:
            base_models: List of (name, model) tuples
        """
        if base_models is None:
            base_models = self._get_default_base_models()

        self.base_models = base_models
        self.stacked_model_ = None
        self.blended_model_ = None
        self.ensemble_weights_ = None

    def _get_default_base_models(self) -> list[tuple[str, BaseEstimator]]:
        """Get default base models for ensemble."""
        return [
            ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
            ("gb", GradientBoostingClassifier(n_estimators=100, random_state=42)),
            ("lgb", lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)),
            (
                "xgb",
                xgb.XGBClassifier(
                    n_estimators=100, random_state=42, use_label_encoder=False
                ),
            ),
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ]

    def create_stacking_ensemble(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        meta_model: BaseEstimator = None,
        cv_folds: int = 5,
        use_probas: bool = True,
    ) -> BaseEstimator:
        """Create stacking ensemble with cross-validation.

        Args:
            X: Training features
            y: Training target
            meta_model: Meta-learner model
            cv_folds: Number of CV folds for stacking
            use_probas: Whether to use probabilities as meta-features

        Returns:
            Fitted stacking ensemble
        """
        from sklearn.ensemble import StackingClassifier

        if meta_model is None:
            meta_model = LogisticRegression(max_iter=1000, random_state=42)

        # Create stacking classifier
        self.stacked_model_ = StackingClassifier(
            estimators=self.base_models,
            final_estimator=meta_model,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
            stack_method="predict_proba" if use_probas else "predict",
            n_jobs=-1,
        )

        # Fit stacking ensemble
        self.stacked_model_.fit(X, y)

        return self.stacked_model_

    def create_blending_ensemble(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        meta_model: BaseEstimator = None,
    ) -> BaseEstimator:
        """Create blending ensemble using validation set.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            meta_model: Meta-learner model

        Returns:
            Fitted blending ensemble
        """
        if meta_model is None:
            meta_model = LogisticRegression(max_iter=1000, random_state=42)

        # Train base models on training set
        blend_features_train = []
        blend_features_val = []

        for name, model in self.base_models:
            # Fit on training set
            model.fit(X_train, y_train)

            # Get predictions on validation set
            if hasattr(model, "predict_proba"):
                val_preds = model.predict_proba(X_val)[:, 1]
                train_preds = model.predict_proba(X_train)[:, 1]
            else:
                val_preds = model.predict(X_val)
                train_preds = model.predict(X_train)

            blend_features_val.append(val_preds)
            blend_features_train.append(train_preds)

        # Create blend features
        X_blend_train = np.column_stack(blend_features_train)
        X_blend_val = np.column_stack(blend_features_val)

        # Train meta-model
        self.blended_model_ = clone(meta_model)
        self.blended_model_.fit(X_blend_val, y_val)

        # Store base models for prediction
        self.base_models_fitted_ = [model for _, model in self.base_models]

        return self.blended_model_

    def create_weighted_ensemble(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        optimization_metric: str = "roc_auc",
        cv_folds: int = 5,
    ) -> VotingClassifier:
        """Create weighted voting ensemble with optimized weights.

        Args:
            X: Training features
            y: Training target
            optimization_metric: Metric to optimize weights
            cv_folds: Number of CV folds

        Returns:
            Weighted voting ensemble
        """
        # Get out-of-fold predictions for each model
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        oof_predictions = []

        for name, model in self.base_models:
            if hasattr(model, "predict_proba"):
                oof_pred = cross_val_predict(
                    clone(model), X, y, cv=cv, method="predict_proba", n_jobs=-1
                )[:, 1]
            else:
                oof_pred = cross_val_predict(
                    clone(model), X, y, cv=cv, method="predict", n_jobs=-1
                )

            oof_predictions.append(oof_pred)

        # Optimize weights
        self.ensemble_weights_ = self._optimize_weights(
            oof_predictions, y, optimization_metric
        )

        # Create weighted voting classifier
        weighted_ensemble = VotingClassifier(
            estimators=self.base_models,
            voting="soft",
            weights=self.ensemble_weights_,
            n_jobs=-1,
        )

        weighted_ensemble.fit(X, y)

        return weighted_ensemble

    def _optimize_weights(
        self, predictions: list[np.ndarray], y_true: np.ndarray, metric: str = "roc_auc"
    ) -> list[float]:
        """Optimize ensemble weights using Optuna."""

        def objective(trial):
            # Sample weights
            weights = []
            for i in range(len(predictions)):
                weights.append(trial.suggest_float(f"weight_{i}", 0.0, 1.0))

            # Normalize weights
            weights = np.array(weights)
            weights = weights / weights.sum()

            # Calculate weighted prediction
            weighted_pred = np.zeros_like(predictions[0])
            for i, pred in enumerate(predictions):
                weighted_pred += weights[i] * pred

            # Calculate metric
            if metric == "roc_auc":
                return roc_auc_score(y_true, weighted_pred)
            elif metric == "log_loss":
                return -log_loss(y_true, weighted_pred)
            else:
                raise ValueError(f"Unknown metric: {metric}")

        # Optimize
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=100, show_progress_bar=False)

        # Get best weights
        best_weights = []
        for i in range(len(predictions)):
            best_weights.append(study.best_params[f"weight_{i}"])

        # Normalize
        best_weights = np.array(best_weights)
        best_weights = best_weights / best_weights.sum()

        return best_weights.tolist()


class HyperparameterOptimizer:
    """Advanced hyperparameter optimization using Optuna with
    multiple optimization strategies.
    """

    def __init__(
        self,
        optimization_strategy: str = "bayesian",
        n_trials: int = 100,
        cv_folds: int = 5,
        scoring: str = "roc_auc",
        random_state: int = 42,
    ):
        """Initialize hyperparameter optimizer.

        Args:
            optimization_strategy: Optimization strategy
            n_trials: Number of optimization trials
            cv_folds: Number of CV folds
            scoring: Scoring metric
            random_state: Random seed
        """
        self.optimization_strategy = optimization_strategy
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.random_state = random_state
        self.best_params_ = {}
        self.optimization_history_ = []

    def optimize(
        self,
        model_class: str,
        X: pd.DataFrame,
        y: pd.Series,
        param_space: dict[str, Any] = None,
    ) -> tuple[BaseEstimator, dict[str, Any]]:
        """Optimize hyperparameters for given model.

        Args:
            model_class: Type of model ('rf', 'gb', 'lgb', 'xgb')
            X: Training features
            y: Training target
            param_space: Custom parameter space

        Returns:
            Tuple of (optimized model, best parameters)
        """
        if param_space is None:
            param_space = self._get_default_param_space(model_class)

        # Create objective function
        def objective(trial):
            # Sample parameters
            params = {}
            for param_name, param_config in param_space.items():
                if param_config["type"] == "int":
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config["low"],
                        param_config["high"],
                        step=param_config.get("step", 1),
                    )
                elif param_config["type"] == "float":
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config["low"],
                        param_config["high"],
                        log=param_config.get("log", False),
                    )
                elif param_config["type"] == "categorical":
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config["choices"]
                    )

            # Create model
            model = self._create_model(model_class, params)

            # Cross-validation
            cv = StratifiedKFold(
                n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
            )

            scores = cross_val_score(
                model, X, y, cv=cv, scoring=self.scoring, n_jobs=-1
            )

            return np.mean(scores)

        # Create study with appropriate sampler
        if self.optimization_strategy == "bayesian":
            sampler = optuna.samplers.TPESampler(seed=self.random_state)
        elif self.optimization_strategy == "random":
            sampler = optuna.samplers.RandomSampler(seed=self.random_state)
        elif self.optimization_strategy == "grid":
            sampler = optuna.samplers.GridSampler(param_space)
        else:
            raise ValueError(
                f"Unknown optimization strategy: {self.optimization_strategy}"
            )

        study = optuna.create_study(direction="maximize", sampler=sampler)

        # Optimize
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)

        # Get best parameters
        self.best_params_ = study.best_params
        self.optimization_history_ = study.trials_dataframe()

        # Create final model with best parameters
        best_model = self._create_model(model_class, self.best_params_)
        best_model.fit(X, y)

        return best_model, self.best_params_

    def _get_default_param_space(self, model_class: str) -> dict[str, Any]:
        """Get default parameter space for model class."""
        if model_class == "rf":
            return {
                "n_estimators": {"type": "int", "low": 50, "high": 300, "step": 50},
                "max_depth": {"type": "int", "low": 3, "high": 20},
                "min_samples_split": {"type": "int", "low": 2, "high": 20},
                "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
                "max_features": {
                    "type": "categorical",
                    "choices": ["sqrt", "log2", None],
                },
            }
        elif model_class == "lgb":
            return {
                "n_estimators": {"type": "int", "low": 50, "high": 500, "step": 50},
                "max_depth": {"type": "int", "low": 3, "high": 15},
                "learning_rate": {
                    "type": "float",
                    "low": 0.01,
                    "high": 0.3,
                    "log": True,
                },
                "num_leaves": {"type": "int", "low": 20, "high": 300},
                "min_child_samples": {"type": "int", "low": 10, "high": 100},
                "subsample": {"type": "float", "low": 0.5, "high": 1.0},
                "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
            }
        elif model_class == "xgb":
            return {
                "n_estimators": {"type": "int", "low": 50, "high": 500, "step": 50},
                "max_depth": {"type": "int", "low": 3, "high": 15},
                "learning_rate": {
                    "type": "float",
                    "low": 0.01,
                    "high": 0.3,
                    "log": True,
                },
                "subsample": {"type": "float", "low": 0.5, "high": 1.0},
                "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
                "gamma": {"type": "float", "low": 0, "high": 5},
                "min_child_weight": {"type": "int", "low": 1, "high": 10},
            }
        else:
            raise ValueError(f"Unknown model class: {model_class}")

    def _create_model(self, model_class: str, params: dict[str, Any]) -> BaseEstimator:
        """Create model instance with given parameters."""
        if model_class == "rf":
            return RandomForestClassifier(
                **params, random_state=self.random_state, n_jobs=-1
            )
        elif model_class == "lgb":
            return lgb.LGBMClassifier(
                **params, random_state=self.random_state, verbose=-1
            )
        elif model_class == "xgb":
            return xgb.XGBClassifier(
                **params, random_state=self.random_state, use_label_encoder=False
            )
        else:
            raise ValueError(f"Unknown model class: {model_class}")


class ModelCalibrator:
    """Model calibration for improved probability estimates.
    """

    def __init__(
        self,
        method: str = "isotonic",
        cv_folds: int = 3,
    ):
        """Initialize calibrator.

        Args:
            method: Calibration method ('sigmoid' or 'isotonic')
            cv_folds: Number of CV folds for calibration
        """
        self.method = method
        self.cv_folds = cv_folds
        self.calibrated_model_ = None
        self.calibration_metrics_ = {}

    def calibrate(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> CalibratedClassifierCV:
        """Calibrate model probabilities.

        Args:
            model: Model to calibrate
            X: Training features
            y: Training target

        Returns:
            Calibrated model
        """
        # Create calibrated classifier
        self.calibrated_model_ = CalibratedClassifierCV(
            model,
            method=self.method,
            cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42),
            n_jobs=-1,
        )

        # Fit calibrated model
        self.calibrated_model_.fit(X, y)

        # Calculate calibration metrics
        self._calculate_calibration_metrics(X, y)

        return self.calibrated_model_

    def _calculate_calibration_metrics(self, X: pd.DataFrame, y: pd.Series):
        """Calculate calibration metrics."""
        # Get predictions
        y_prob = self.calibrated_model_.predict_proba(X)[:, 1]

        # Brier score
        brier = brier_score_loss(y, y_prob)

        # Expected Calibration Error (ECE)
        ece = self._calculate_ece(y, y_prob)

        # Maximum Calibration Error (MCE)
        mce = self._calculate_mce(y, y_prob)

        self.calibration_metrics_ = {"brier_score": brier, "ece": ece, "mce": mce}

    def _calculate_ece(
        self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
    ) -> float:
        """Calculate Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0

        for i in range(n_bins):
            mask = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
            if mask.sum() > 0:
                bin_accuracy = y_true[mask].mean()
                bin_confidence = y_prob[mask].mean()
                ece += mask.sum() * np.abs(bin_accuracy - bin_confidence)

        return ece / len(y_true)

    def _calculate_mce(
        self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
    ) -> float:
        """Calculate Maximum Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        mce = 0

        for i in range(n_bins):
            mask = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
            if mask.sum() > 0:
                bin_accuracy = y_true[mask].mean()
                bin_confidence = y_prob[mask].mean()
                mce = max(mce, np.abs(bin_accuracy - bin_confidence))

        return mce


class UncertaintyQuantifier:
    """Uncertainty quantification for model predictions.
    """

    def __init__(
        self,
        method: str = "bootstrap",
        n_estimators: int = 100,
        confidence_level: float = 0.95,
    ):
        """Initialize uncertainty quantifier.

        Args:
            method: Method for uncertainty quantification
            n_estimators: Number of estimators for ensemble methods
            confidence_level: Confidence level for intervals
        """
        self.method = method
        self.n_estimators = n_estimators
        self.confidence_level = confidence_level
        self.models_ = []

    def fit(self, model_class: BaseEstimator, X: pd.DataFrame, y: pd.Series):
        """Fit uncertainty quantification.

        Args:
            model_class: Base model class
            X: Training features
            y: Training target
        """
        if self.method == "bootstrap":
            self._fit_bootstrap(model_class, X, y)
        elif self.method == "dropout":
            self._fit_dropout(model_class, X, y)
        elif self.method == "ensemble":
            self._fit_ensemble(model_class, X, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _fit_bootstrap(self, model_class: BaseEstimator, X: pd.DataFrame, y: pd.Series):
        """Fit using bootstrap aggregation."""
        n_samples = len(X)

        for i in range(self.n_estimators):
            # Bootstrap sample
            idx = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X.iloc[idx]
            y_boot = y.iloc[idx]

            # Train model
            model = clone(model_class)
            model.fit(X_boot, y_boot)
            self.models_.append(model)

    def _fit_ensemble(self, model_class: BaseEstimator, X: pd.DataFrame, y: pd.Series):
        """Fit using ensemble with different random seeds."""
        for i in range(self.n_estimators):
            # Clone model with different random state
            model = clone(model_class)
            if hasattr(model, "random_state"):
                model.random_state = i

            model.fit(X, y)
            self.models_.append(model)

    def _fit_dropout(self, model_class: BaseEstimator, X: pd.DataFrame, y: pd.Series):
        """Fit using dropout (for neural networks)."""
        # This would require neural network implementation
        # For now, fallback to ensemble
        self._fit_ensemble(model_class, X, y)

    def predict_with_uncertainty(
        self, X: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Make predictions with uncertainty estimates.

        Args:
            X: Features to predict

        Returns:
            Tuple of (predictions, lower_bound, upper_bound)
        """
        # Get predictions from all models
        predictions = []

        for model in self.models_:
            if hasattr(model, "predict_proba"):
                pred = model.predict_proba(X)[:, 1]
            else:
                pred = model.predict(X)
            predictions.append(pred)

        predictions = np.array(predictions)

        # Calculate statistics
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)

        # Calculate confidence intervals
        alpha = 1 - self.confidence_level
        lower_bound = np.percentile(predictions, alpha / 2 * 100, axis=0)
        upper_bound = np.percentile(predictions, (1 - alpha / 2) * 100, axis=0)

        return mean_pred, lower_bound, upper_bound

    def get_prediction_intervals(self, X: pd.DataFrame) -> pd.DataFrame:
        """Get prediction intervals as DataFrame.

        Args:
            X: Features to predict

        Returns:
            DataFrame with predictions and intervals
        """
        mean_pred, lower_bound, upper_bound = self.predict_with_uncertainty(X)

        return pd.DataFrame(
            {
                "prediction": mean_pred,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "uncertainty": upper_bound - lower_bound,
            }
        )


if __name__ == "__main__":
    print("Enhanced Modeling Module Loaded Successfully")
    print("=" * 60)
    print("Available Classes:")
    print("  - StratifiedCrossValidator: Enhanced cross-validation")
    print("  - EnhancedEnsembleMethods: Stacking, blending, weighted ensembles")
    print("  - HyperparameterOptimizer: Optuna-based hyperparameter tuning")
    print("  - ModelCalibrator: Probability calibration")
    print("  - UncertaintyQuantifier: Prediction uncertainty estimation")
