"""
Advanced Feature Selection Methods

Comprehensive feature selection techniques for optimal model performance.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Tuple, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, SelectFpr, SelectFdr, SelectFwe,
    RFE, RFECV, SelectFromModel, VarianceThreshold,
    mutual_info_classif, mutual_info_regression,
    f_classif, f_regression, chi2
)
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor
)
from sklearn.linear_model import (
    LassoCV, RidgeCV, ElasticNetCV,
    LogisticRegressionCV
)
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Optional imports
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from boruta import BorutaPy
    BORUTA_AVAILABLE = True
except ImportError:
    BORUTA_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class AdvancedFeatureSelector:
    """Advanced feature selection with multiple strategies."""

    def __init__(self, task_type: str = 'classification',
                 selection_methods: List[str] = None,
                 max_features: Optional[int] = None,
                 cv_folds: int = 5,
                 verbosity: int = 1):
        """
        Initialize advanced feature selector.

        Args:
            task_type: 'classification' or 'regression'
            selection_methods: List of methods to use
            max_features: Maximum number of features to select
            cv_folds: Number of CV folds for evaluation
            verbosity: Logging verbosity
        """
        self.task_type = task_type
        self.selection_methods = selection_methods or ['importance', 'mutual_info', 'lasso']
        self.max_features = max_features
        self.cv_folds = cv_folds
        self.verbosity = verbosity
        self.feature_scores = {}
        self.selected_features = {}
        self.selectors = {}

    def fit_select(self, X: pd.DataFrame, y: pd.Series,
                   return_scores: bool = False) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
        """
        Fit selectors and select features.

        Args:
            X: Feature dataframe
            y: Target series
            return_scores: Whether to return feature scores

        Returns:
            Selected features dataframe and optionally scores
        """
        # Ensure numeric data
        X_numeric = X.select_dtypes(include=[np.number])

        results = {}
        for method in self.selection_methods:
            if self.verbosity >= 1:
                print(f"Applying {method} selection...")

            selected_features, scores = self._apply_selection_method(
                X_numeric, y, method
            )

            results[method] = {
                'features': selected_features,
                'scores': scores
            }

        # Combine results
        final_features = self._combine_selections(results, X_numeric)

        if return_scores:
            return X[final_features], self.feature_scores
        else:
            return X[final_features]

    def _apply_selection_method(self, X: pd.DataFrame, y: pd.Series,
                               method: str) -> Tuple[List[str], Dict[str, float]]:
        """Apply specific selection method."""
        if method == 'variance':
            return self._variance_selection(X)

        elif method == 'correlation':
            return self._correlation_selection(X, y)

        elif method == 'mutual_info':
            return self._mutual_info_selection(X, y)

        elif method == 'importance':
            return self._importance_selection(X, y)

        elif method == 'lasso':
            return self._lasso_selection(X, y)

        elif method == 'rfe':
            return self._rfe_selection(X, y)

        elif method == 'boruta':
            return self._boruta_selection(X, y)

        elif method == 'shap':
            return self._shap_selection(X, y)

        elif method == 'forward':
            return self._forward_selection(X, y)

        elif method == 'backward':
            return self._backward_selection(X, y)

        elif method == 'genetic':
            return self._genetic_selection(X, y)

        else:
            raise ValueError(f"Unknown selection method: {method}")

    def _variance_selection(self, X: pd.DataFrame,
                           threshold: float = 0.01) -> Tuple[List[str], Dict[str, float]]:
        """Select features based on variance."""
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X)

        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()

        scores = dict(zip(X.columns, selector.variances_))

        self.selectors['variance'] = selector
        return selected_features, scores

    def _correlation_selection(self, X: pd.DataFrame, y: pd.Series,
                              threshold: float = 0.9) -> Tuple[List[str], Dict[str, float]]:
        """Select features based on correlation."""
        # Calculate correlation with target
        correlations = X.corrwith(y).abs()

        # Remove highly correlated features
        corr_matrix = X.corr().abs()
        upper_tri = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

        to_drop = set()
        for col_idx in range(len(corr_matrix.columns)):
            if col_idx >= len(corr_matrix.columns):
                break

            col = corr_matrix.columns[col_idx]
            correlated_features = corr_matrix.index[
                (corr_matrix[col] > threshold) & upper_tri[:, col_idx]
            ].tolist()

            # Keep feature with higher correlation to target
            if correlated_features:
                all_features = [col] + correlated_features
                target_corrs = correlations[all_features]
                keep_feature = target_corrs.idxmax()

                for feat in all_features:
                    if feat != keep_feature:
                        to_drop.add(feat)

        selected_features = [col for col in X.columns if col not in to_drop]
        scores = correlations.to_dict()

        return selected_features, scores

    def _mutual_info_selection(self, X: pd.DataFrame, y: pd.Series) -> Tuple[List[str], Dict[str, float]]:
        """Select features using mutual information."""
        if self.task_type == 'classification':
            mi_scores = mutual_info_classif(X, y)
        else:
            mi_scores = mutual_info_regression(X, y)

        scores = dict(zip(X.columns, mi_scores))

        # Select top features
        if self.max_features:
            k = min(self.max_features, len(X.columns))
        else:
            k = len(X.columns) // 2

        selector = SelectKBest(
            mutual_info_classif if self.task_type == 'classification' else mutual_info_regression,
            k=k
        )
        selector.fit(X, y)

        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()

        self.selectors['mutual_info'] = selector
        return selected_features, scores

    def _importance_selection(self, X: pd.DataFrame, y: pd.Series) -> Tuple[List[str], Dict[str, float]]:
        """Select features using tree-based importance."""
        if self.task_type == 'classification':
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

        model.fit(X, y)

        importances = model.feature_importances_
        scores = dict(zip(X.columns, importances))

        # Select features
        if self.max_features:
            threshold = np.sort(importances)[-self.max_features]
            selector = SelectFromModel(model, threshold=threshold, prefit=True)
        else:
            selector = SelectFromModel(model, prefit=True)

        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()

        self.selectors['importance'] = selector
        return selected_features, scores

    def _lasso_selection(self, X: pd.DataFrame, y: pd.Series) -> Tuple[List[str], Dict[str, float]]:
        """Select features using Lasso regularization."""
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if self.task_type == 'classification':
            model = LogisticRegressionCV(
                cv=self.cv_folds,
                penalty='l1',
                solver='liblinear',
                random_state=42,
                n_jobs=-1
            )
        else:
            model = LassoCV(cv=self.cv_folds, random_state=42, n_jobs=-1)

        model.fit(X_scaled, y)

        # Get coefficients
        if hasattr(model, 'coef_'):
            if len(model.coef_.shape) > 1:
                coefs = np.abs(model.coef_).mean(axis=0)
            else:
                coefs = np.abs(model.coef_)
        else:
            coefs = np.zeros(X.shape[1])

        scores = dict(zip(X.columns, coefs))

        # Select non-zero coefficients
        selected_mask = coefs != 0
        selected_features = X.columns[selected_mask].tolist()

        # Limit to max_features if specified
        if self.max_features and len(selected_features) > self.max_features:
            top_indices = np.argsort(coefs)[-self.max_features:]
            selected_features = X.columns[top_indices].tolist()

        return selected_features, scores

    def _rfe_selection(self, X: pd.DataFrame, y: pd.Series) -> Tuple[List[str], Dict[str, float]]:
        """Select features using Recursive Feature Elimination."""
        if self.task_type == 'classification':
            estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        else:
            estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)

        if self.max_features:
            n_features = min(self.max_features, X.shape[1])
        else:
            n_features = X.shape[1] // 2

        selector = RFE(estimator, n_features_to_select=n_features)
        selector.fit(X, y)

        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()

        # Create scores from ranking
        scores = dict(zip(X.columns, 1 / selector.ranking_))

        self.selectors['rfe'] = selector
        return selected_features, scores

    def _boruta_selection(self, X: pd.DataFrame, y: pd.Series) -> Tuple[List[str], Dict[str, float]]:
        """Select features using Boruta algorithm."""
        if not BORUTA_AVAILABLE:
            print("Boruta not available. Using importance selection instead.")
            return self._importance_selection(X, y)

        if self.task_type == 'classification':
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

        boruta_selector = BorutaPy(
            rf,
            n_estimators='auto',
            verbose=0,
            random_state=42,
            max_iter=100
        )

        boruta_selector.fit(X.values, y.values)

        selected_mask = boruta_selector.support_
        selected_features = X.columns[selected_mask].tolist()

        # Get feature importance scores
        scores = {}
        if hasattr(boruta_selector, 'ranking_'):
            scores = dict(zip(X.columns, 1 / boruta_selector.ranking_))

        self.selectors['boruta'] = boruta_selector
        return selected_features, scores

    def _shap_selection(self, X: pd.DataFrame, y: pd.Series) -> Tuple[List[str], Dict[str, float]]:
        """Select features using SHAP values."""
        if not SHAP_AVAILABLE:
            print("SHAP not available. Using importance selection instead.")
            return self._importance_selection(X, y)

        # Train model
        if self.task_type == 'classification':
            model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        else:
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)

        model.fit(X, y)

        # Calculate SHAP values
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)

        # Get feature importance from SHAP
        if len(shap_values.shape) == 3:  # Multi-class
            shap_importance = np.abs(shap_values.values).mean(axis=(0, 2))
        else:
            shap_importance = np.abs(shap_values.values).mean(axis=0)

        scores = dict(zip(X.columns, shap_importance))

        # Select top features
        if self.max_features:
            k = min(self.max_features, len(X.columns))
        else:
            k = len(X.columns) // 2

        top_indices = np.argsort(shap_importance)[-k:]
        selected_features = X.columns[top_indices].tolist()

        return selected_features, scores

    def _forward_selection(self, X: pd.DataFrame, y: pd.Series,
                          max_features: Optional[int] = None) -> Tuple[List[str], Dict[str, float]]:
        """Forward feature selection."""
        if max_features is None:
            max_features = self.max_features or X.shape[1] // 2

        selected_features = []
        remaining_features = list(X.columns)
        scores = {}

        # Get base model
        if self.task_type == 'classification':
            model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            scoring = 'accuracy'
        else:
            model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            scoring = 'r2'

        for _ in range(min(max_features, len(X.columns))):
            best_score = -np.inf
            best_feature = None

            for feature in remaining_features:
                current_features = selected_features + [feature]
                score = cross_val_score(
                    model, X[current_features], y,
                    cv=self.cv_folds, scoring=scoring, n_jobs=-1
                ).mean()

                if score > best_score:
                    best_score = score
                    best_feature = feature

            if best_feature:
                selected_features.append(best_feature)
                remaining_features.remove(best_feature)
                scores[best_feature] = best_score

                if self.verbosity >= 2:
                    print(f"Added {best_feature}, Score: {best_score:.4f}")

        return selected_features, scores

    def _backward_selection(self, X: pd.DataFrame, y: pd.Series,
                           min_features: int = 1) -> Tuple[List[str], Dict[str, float]]:
        """Backward feature selection."""
        selected_features = list(X.columns)
        scores = {}

        # Get base model
        if self.task_type == 'classification':
            model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            scoring = 'accuracy'
        else:
            model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            scoring = 'r2'

        # Calculate initial score
        best_score = cross_val_score(
            model, X[selected_features], y,
            cv=self.cv_folds, scoring=scoring, n_jobs=-1
        ).mean()

        while len(selected_features) > min_features:
            worst_score = np.inf
            worst_feature = None

            for feature in selected_features:
                current_features = [f for f in selected_features if f != feature]

                if not current_features:
                    continue

                score = cross_val_score(
                    model, X[current_features], y,
                    cv=self.cv_folds, scoring=scoring, n_jobs=-1
                ).mean()

                if score > best_score or (score == best_score and score < worst_score):
                    worst_score = score
                    worst_feature = feature

            if worst_feature and worst_score >= best_score * 0.95:  # Allow 5% degradation
                selected_features.remove(worst_feature)
                scores[worst_feature] = worst_score
                best_score = worst_score

                if self.verbosity >= 2:
                    print(f"Removed {worst_feature}, Score: {worst_score:.4f}")
            else:
                break

        # Create final scores
        for feature in selected_features:
            if feature not in scores:
                scores[feature] = best_score

        return selected_features, scores

    def _genetic_selection(self, X: pd.DataFrame, y: pd.Series,
                          population_size: int = 50,
                          generations: int = 20) -> Tuple[List[str], Dict[str, float]]:
        """Genetic algorithm for feature selection."""
        n_features = X.shape[1]

        if self.max_features:
            target_features = min(self.max_features, n_features)
        else:
            target_features = n_features // 2

        # Initialize population
        population = []
        for _ in range(population_size):
            individual = np.random.choice([0, 1], size=n_features, p=[0.5, 0.5])
            # Ensure at least some features are selected
            if individual.sum() == 0:
                individual[np.random.choice(n_features, target_features)] = 1
            population.append(individual)

        # Get model for evaluation
        if self.task_type == 'classification':
            model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            scoring = 'accuracy'
        else:
            model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            scoring = 'r2'

        best_individual = None
        best_score = -np.inf

        for generation in range(generations):
            # Evaluate population
            scores = []
            for individual in population:
                if individual.sum() == 0:
                    scores.append(-np.inf)
                    continue

                selected_cols = X.columns[individual.astype(bool)]
                score = cross_val_score(
                    model, X[selected_cols], y,
                    cv=3, scoring=scoring, n_jobs=-1  # Use fewer folds for speed
                ).mean()

                # Penalize for too many features
                penalty = abs(individual.sum() - target_features) * 0.001
                score -= penalty

                scores.append(score)

                if score > best_score:
                    best_score = score
                    best_individual = individual.copy()

            # Selection
            scores = np.array(scores)
            if scores.max() > -np.inf:
                probabilities = np.exp(scores - scores.max())
                probabilities /= probabilities.sum()
            else:
                probabilities = np.ones(population_size) / population_size

            # Create new population
            new_population = []

            # Elitism - keep best individual
            new_population.append(best_individual)

            while len(new_population) < population_size:
                # Selection
                parent1_idx = np.random.choice(population_size, p=probabilities)
                parent2_idx = np.random.choice(population_size, p=probabilities)

                parent1 = population[parent1_idx]
                parent2 = population[parent2_idx]

                # Crossover
                crossover_point = np.random.randint(1, n_features)
                child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])

                # Mutation
                mutation_rate = 0.01
                mutation_mask = np.random.random(n_features) < mutation_rate
                child[mutation_mask] = 1 - child[mutation_mask]

                new_population.append(child)

            population = new_population

            if self.verbosity >= 2 and generation % 5 == 0:
                print(f"Generation {generation}, Best Score: {best_score:.4f}")

        # Get selected features
        selected_features = X.columns[best_individual.astype(bool)].tolist()

        # Create scores
        feature_scores = dict(zip(X.columns, best_individual))

        return selected_features, feature_scores

    def _combine_selections(self, results: Dict, X: pd.DataFrame) -> List[str]:
        """Combine results from multiple selection methods."""
        # Count how many methods selected each feature
        feature_counts = {}
        all_scores = {}

        for method, result in results.items():
            selected = result['features']
            scores = result['scores']

            for feature in selected:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1

            for feature, score in scores.items():
                if feature not in all_scores:
                    all_scores[feature] = []
                all_scores[feature].append(score)

        # Average scores across methods
        self.feature_scores = {
            feature: np.mean(scores)
            for feature, scores in all_scores.items()
        }

        # Select features that appear in multiple methods or have high scores
        min_methods = len(self.selection_methods) // 2  # At least half of methods

        selected = [
            feature for feature, count in feature_counts.items()
            if count >= min_methods
        ]

        # If not enough features, add by score
        if self.max_features and len(selected) < self.max_features:
            remaining = [f for f in self.feature_scores if f not in selected]
            remaining_sorted = sorted(remaining, key=lambda x: self.feature_scores[x], reverse=True)
            n_to_add = min(self.max_features - len(selected), len(remaining_sorted))
            selected.extend(remaining_sorted[:n_to_add])

        # If too many features, keep top by score
        elif self.max_features and len(selected) > self.max_features:
            selected = sorted(selected, key=lambda x: self.feature_scores[x], reverse=True)
            selected = selected[:self.max_features]

        self.selected_features['combined'] = selected

        if self.verbosity >= 1:
            print(f"Selected {len(selected)} features from {X.shape[1]} total features")

        return selected


class FeatureImportanceAnalyzer:
    """Analyze and visualize feature importance."""

    def __init__(self):
        self.importance_scores = {}
        self.model_importances = {}

    def analyze(self, X: pd.DataFrame, y: pd.Series,
               models: List[str] = None) -> pd.DataFrame:
        """
        Analyze feature importance using multiple models.

        Args:
            X: Feature dataframe
            y: Target series
            models: List of model types to use

        Returns:
            DataFrame with importance scores
        """
        if models is None:
            models = ['rf', 'gb', 'et', 'lr']

        for model_name in models:
            if model_name == 'rf':
                scores = self._random_forest_importance(X, y)
            elif model_name == 'gb':
                scores = self._gradient_boosting_importance(X, y)
            elif model_name == 'et':
                scores = self._extra_trees_importance(X, y)
            elif model_name == 'lr':
                scores = self._linear_importance(X, y)
            elif model_name == 'lgb' and LIGHTGBM_AVAILABLE:
                scores = self._lightgbm_importance(X, y)
            else:
                continue

            self.model_importances[model_name] = scores

        # Combine scores
        importance_df = pd.DataFrame(self.model_importances)

        # Add statistics
        importance_df['mean'] = importance_df.mean(axis=1)
        importance_df['std'] = importance_df.std(axis=1)
        importance_df['median'] = importance_df.median(axis=1)
        importance_df['max'] = importance_df.max(axis=1)
        importance_df['min'] = importance_df.min(axis=1)

        # Rank features
        importance_df['rank'] = importance_df['mean'].rank(ascending=False)

        return importance_df.sort_values('mean', ascending=False)

    def _random_forest_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Get Random Forest feature importance."""
        # Determine task type
        if y.dtype == 'object' or y.nunique() < 20:
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

        model.fit(X, y)
        return pd.Series(model.feature_importances_, index=X.columns)

    def _gradient_boosting_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Get Gradient Boosting feature importance."""
        if y.dtype == 'object' or y.nunique() < 20:
            model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        else:
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)

        model.fit(X, y)
        return pd.Series(model.feature_importances_, index=X.columns)

    def _extra_trees_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Get Extra Trees feature importance."""
        if y.dtype == 'object' or y.nunique() < 20:
            model = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            model = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)

        model.fit(X, y)
        return pd.Series(model.feature_importances_, index=X.columns)

    def _linear_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Get linear model feature importance."""
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if y.dtype == 'object' or y.nunique() < 20:
            model = LogisticRegressionCV(cv=5, random_state=42, n_jobs=-1, max_iter=1000)
        else:
            model = ElasticNetCV(cv=5, random_state=42, n_jobs=-1)

        model.fit(X_scaled, y)

        # Get coefficients
        if hasattr(model, 'coef_'):
            if len(model.coef_.shape) > 1:
                coefs = np.abs(model.coef_).mean(axis=0)
            else:
                coefs = np.abs(model.coef_)
        else:
            coefs = np.zeros(X.shape[1])

        return pd.Series(coefs, index=X.columns)

    def _lightgbm_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Get LightGBM feature importance."""
        if not LIGHTGBM_AVAILABLE:
            return pd.Series(index=X.columns)

        if y.dtype == 'object' or y.nunique() < 20:
            model = lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbosity=-1)
        else:
            model = lgb.LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbosity=-1)

        model.fit(X, y)
        return pd.Series(model.feature_importances_, index=X.columns)