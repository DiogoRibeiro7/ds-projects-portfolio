"""
Enhanced Evaluation for Bank Churn Prediction.
Custom business metrics, fairness evaluation, statistical testing,
and ROI-based model assessment.
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    brier_score_loss,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import permutation_test_score
from sklearn.utils import resample

warnings.filterwarnings("ignore")


@dataclass
class BusinessMetrics:
    """Container for business-oriented metrics."""

    revenue_impact: float
    cost_savings: float
    roi: float
    payback_period: float
    customer_lifetime_value_impact: float
    retention_rate_improvement: float
    profit_curve_auc: float
    cost_benefit_threshold: float


@dataclass
class FairnessMetrics:
    """Container for fairness and bias metrics."""

    demographic_parity_difference: Dict[str, float]
    equal_opportunity_difference: Dict[str, float]
    disparate_impact_ratio: Dict[str, float]
    average_odds_difference: Dict[str, float]
    individual_fairness_score: float
    group_fairness_score: float


@dataclass
class ModelComparisonResult:
    """Container for model comparison results."""

    model_rankings: pd.DataFrame
    statistical_significance: pd.DataFrame
    pairwise_differences: pd.DataFrame
    best_model: str
    confidence_intervals: Dict[str, Tuple[float, float]]


class BusinessMetricsCalculator:
    """
    Calculate business-oriented metrics for churn models.
    """

    def __init__(
        self,
        customer_value: float = 1000,
        retention_cost: float = 50,
        acquisition_cost: float = 200,
        discount_rate: float = 0.1,
        time_horizon: int = 12,
    ):
        """
        Initialize business metrics calculator.

        Args:
            customer_value: Average customer lifetime value
            retention_cost: Cost of retention intervention
            acquisition_cost: Cost to acquire new customer
            discount_rate: Discount rate for NPV calculations
            time_horizon: Time horizon in months
        """
        self.customer_value = customer_value
        self.retention_cost = retention_cost
        self.acquisition_cost = acquisition_cost
        self.discount_rate = discount_rate
        self.time_horizon = time_horizon

    def calculate_business_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        customer_values: np.ndarray = None,
    ) -> BusinessMetrics:
        """
        Calculate comprehensive business metrics.

        Args:
            y_true: True labels (1 = churned)
            y_pred: Predicted labels
            y_prob: Predicted probabilities
            customer_values: Individual customer values (optional)

        Returns:
            BusinessMetrics object
        """
        if customer_values is None:
            customer_values = np.full(len(y_true), self.customer_value)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Revenue impact
        # True positives: correctly identified churners who were retained
        revenue_saved = tp * np.mean(customer_values[y_true == 1])

        # False negatives: missed churners
        revenue_lost = fn * np.mean(customer_values[y_true == 1])

        # Cost of interventions
        intervention_cost = (tp + fp) * self.retention_cost

        # Net revenue impact
        revenue_impact = revenue_saved - revenue_lost - intervention_cost

        # Cost savings vs. no model
        no_model_loss = sum(customer_values[y_true == 1])
        cost_savings = revenue_impact + no_model_loss

        # ROI
        roi = (revenue_impact / (intervention_cost + 0.001)) * 100

        # Payback period (months)
        monthly_benefit = revenue_impact / self.time_horizon
        payback_period = intervention_cost / (monthly_benefit + 0.001)

        # Customer Lifetime Value impact
        retention_rate_with_model = tp / (tp + fn) if (tp + fn) > 0 else 0
        retention_rate_without = 0  # Assuming no intervention
        clv_impact = (
            retention_rate_with_model - retention_rate_without
        ) * self.customer_value

        # Retention rate improvement
        retention_improvement = retention_rate_with_model

        # Profit curve AUC
        profit_auc = self._calculate_profit_curve_auc(y_true, y_prob, customer_values)

        # Cost-benefit threshold
        cb_threshold = self._find_cost_benefit_threshold(
            y_true, y_prob, customer_values
        )

        return BusinessMetrics(
            revenue_impact=revenue_impact,
            cost_savings=cost_savings,
            roi=roi,
            payback_period=payback_period,
            customer_lifetime_value_impact=clv_impact,
            retention_rate_improvement=retention_improvement,
            profit_curve_auc=profit_auc,
            cost_benefit_threshold=cb_threshold,
        )

    def _calculate_profit_curve_auc(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        customer_values: np.ndarray,
    ) -> float:
        """Calculate area under profit curve."""
        # Sort by predicted probability
        sorted_indices = np.argsort(y_prob)[::-1]
        y_true_sorted = y_true[sorted_indices]
        values_sorted = customer_values[sorted_indices]

        # Calculate cumulative profit
        cumulative_profit = []
        cumulative_cost = []

        for i in range(1, len(y_true) + 1):
            # Profit from true positives
            profit = sum(values_sorted[:i][y_true_sorted[:i] == 1])
            # Cost of interventions
            cost = i * self.retention_cost

            cumulative_profit.append(profit - cost)
            cumulative_cost.append(i)

        # Normalize and calculate AUC
        max_profit = (
            sum(customer_values[y_true == 1]) - sum(y_true == 1) * self.retention_cost
        )
        normalized_profit = np.array(cumulative_profit) / (max_profit + 0.001)

        # Calculate AUC
        from sklearn.metrics import auc

        percentages = np.array(cumulative_cost) / len(y_true)
        profit_auc = auc(percentages, normalized_profit)

        return profit_auc

    def _find_cost_benefit_threshold(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        customer_values: np.ndarray,
    ) -> float:
        """Find optimal probability threshold based on cost-benefit analysis."""
        best_threshold = 0.5
        best_profit = -np.inf

        for threshold in np.arange(0.1, 0.9, 0.01):
            y_pred = (y_prob >= threshold).astype(int)

            # Calculate profit
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

            profit = (
                tp * np.mean(customer_values[y_true == 1])  # Saved revenue
                - (tp + fp) * self.retention_cost  # Intervention cost
                - fn * np.mean(customer_values[y_true == 1])  # Lost revenue
            )

            if profit > best_profit:
                best_profit = profit
                best_threshold = threshold

        return best_threshold

    def calculate_roi_metrics(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        thresholds: List[float] = None,
    ) -> pd.DataFrame:
        """
        Calculate ROI metrics at different thresholds.

        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            thresholds: List of thresholds to evaluate

        Returns:
            DataFrame with ROI metrics at each threshold
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 0.9, 0.1)

        roi_metrics = []

        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            # Business metrics
            revenue = tp * self.customer_value - fn * self.customer_value
            cost = (tp + fp) * self.retention_cost
            roi = ((revenue - cost) / cost * 100) if cost > 0 else 0

            roi_metrics.append(
                {
                    "threshold": threshold,
                    "precision": precision,
                    "recall": recall,
                    "revenue": revenue,
                    "cost": cost,
                    "roi": roi,
                    "profit": revenue - cost,
                }
            )

        return pd.DataFrame(roi_metrics)


class FairnessEvaluator:
    """
    Evaluate model fairness across different demographic groups.
    """

    def __init__(self, sensitive_attributes: List[str]):
        """
        Initialize fairness evaluator.

        Args:
            sensitive_attributes: List of sensitive attribute names
        """
        self.sensitive_attributes = sensitive_attributes
        self.fairness_metrics_ = {}

    def evaluate_fairness(
        self,
        X: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray = None,
    ) -> FairnessMetrics:
        """
        Evaluate model fairness across sensitive attributes.

        Args:
            X: Features including sensitive attributes
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)

        Returns:
            FairnessMetrics object
        """
        demographic_parity = {}
        equal_opportunity = {}
        disparate_impact = {}
        average_odds = {}

        for attr in self.sensitive_attributes:
            if attr not in X.columns:
                continue

            # Get unique groups
            groups = X[attr].unique()

            if len(groups) == 2:
                # Binary attribute
                group_0_mask = X[attr] == groups[0]
                group_1_mask = X[attr] == groups[1]

                # Demographic parity difference
                dp_diff = self._calculate_demographic_parity(
                    y_pred[group_0_mask], y_pred[group_1_mask]
                )
                demographic_parity[attr] = dp_diff

                # Equal opportunity difference
                eo_diff = self._calculate_equal_opportunity(
                    y_true[group_0_mask],
                    y_pred[group_0_mask],
                    y_true[group_1_mask],
                    y_pred[group_1_mask],
                )
                equal_opportunity[attr] = eo_diff

                # Disparate impact ratio
                di_ratio = self._calculate_disparate_impact(
                    y_pred[group_0_mask], y_pred[group_1_mask]
                )
                disparate_impact[attr] = di_ratio

                # Average odds difference
                ao_diff = self._calculate_average_odds(
                    y_true[group_0_mask],
                    y_pred[group_0_mask],
                    y_true[group_1_mask],
                    y_pred[group_1_mask],
                )
                average_odds[attr] = ao_diff

            else:
                # Multi-valued attribute - calculate pairwise metrics
                # For simplicity, using max difference
                max_dp = 0
                max_eo = 0
                min_di = 1
                max_ao = 0

                for i, group_i in enumerate(groups):
                    for j, group_j in enumerate(groups[i + 1 :], i + 1):
                        mask_i = X[attr] == group_i
                        mask_j = X[attr] == group_j

                        dp = abs(
                            self._calculate_demographic_parity(
                                y_pred[mask_i], y_pred[mask_j]
                            )
                        )
                        max_dp = max(max_dp, dp)

                        eo = abs(
                            self._calculate_equal_opportunity(
                                y_true[mask_i],
                                y_pred[mask_i],
                                y_true[mask_j],
                                y_pred[mask_j],
                            )
                        )
                        max_eo = max(max_eo, eo)

                        di = self._calculate_disparate_impact(
                            y_pred[mask_i], y_pred[mask_j]
                        )
                        min_di = min(min_di, di)

                        ao = abs(
                            self._calculate_average_odds(
                                y_true[mask_i],
                                y_pred[mask_i],
                                y_true[mask_j],
                                y_pred[mask_j],
                            )
                        )
                        max_ao = max(max_ao, ao)

                demographic_parity[attr] = max_dp
                equal_opportunity[attr] = max_eo
                disparate_impact[attr] = min_di
                average_odds[attr] = max_ao

        # Calculate overall fairness scores
        individual_fairness = self._calculate_individual_fairness(X, y_pred, y_prob)
        group_fairness = self._calculate_group_fairness(
            demographic_parity, equal_opportunity, disparate_impact
        )

        return FairnessMetrics(
            demographic_parity_difference=demographic_parity,
            equal_opportunity_difference=equal_opportunity,
            disparate_impact_ratio=disparate_impact,
            average_odds_difference=average_odds,
            individual_fairness_score=individual_fairness,
            group_fairness_score=group_fairness,
        )

    def _calculate_demographic_parity(
        self, y_pred_0: np.ndarray, y_pred_1: np.ndarray
    ) -> float:
        """Calculate demographic parity difference."""
        return abs(y_pred_0.mean() - y_pred_1.mean())

    def _calculate_equal_opportunity(
        self,
        y_true_0: np.ndarray,
        y_pred_0: np.ndarray,
        y_true_1: np.ndarray,
        y_pred_1: np.ndarray,
    ) -> float:
        """Calculate equal opportunity difference."""
        # True positive rate for each group
        tpr_0 = y_pred_0[y_true_0 == 1].mean() if sum(y_true_0 == 1) > 0 else 0
        tpr_1 = y_pred_1[y_true_1 == 1].mean() if sum(y_true_1 == 1) > 0 else 0

        return abs(tpr_0 - tpr_1)

    def _calculate_disparate_impact(
        self, y_pred_0: np.ndarray, y_pred_1: np.ndarray
    ) -> float:
        """Calculate disparate impact ratio."""
        rate_0 = y_pred_0.mean()
        rate_1 = y_pred_1.mean()

        if rate_1 == 0:
            return 0 if rate_0 == 0 else np.inf

        return min(rate_0 / rate_1, rate_1 / rate_0)

    def _calculate_average_odds(
        self,
        y_true_0: np.ndarray,
        y_pred_0: np.ndarray,
        y_true_1: np.ndarray,
        y_pred_1: np.ndarray,
    ) -> float:
        """Calculate average odds difference."""
        # TPR difference
        tpr_0 = y_pred_0[y_true_0 == 1].mean() if sum(y_true_0 == 1) > 0 else 0
        tpr_1 = y_pred_1[y_true_1 == 1].mean() if sum(y_true_1 == 1) > 0 else 0

        # FPR difference
        fpr_0 = y_pred_0[y_true_0 == 0].mean() if sum(y_true_0 == 0) > 0 else 0
        fpr_1 = y_pred_1[y_true_1 == 0].mean() if sum(y_true_1 == 0) > 0 else 0

        return (abs(tpr_0 - tpr_1) + abs(fpr_0 - fpr_1)) / 2

    def _calculate_individual_fairness(
        self, X: pd.DataFrame, y_pred: np.ndarray, y_prob: np.ndarray = None
    ) -> float:
        """Calculate individual fairness score."""
        # Simplified: check consistency for similar individuals
        # In practice, this would require a similarity metric

        # For demonstration, using random sampling
        n_samples = min(100, len(X))
        consistency_scores = []

        for _ in range(n_samples):
            idx1, idx2 = np.random.choice(len(X), 2, replace=False)

            # Calculate feature similarity (cosine similarity)
            from sklearn.metrics.pairwise import cosine_similarity

            if X.select_dtypes(include=[np.number]).shape[1] > 0:
                X_numeric = X.select_dtypes(include=[np.number])
                similarity = cosine_similarity(
                    X_numeric.iloc[[idx1]], X_numeric.iloc[[idx2]]
                )[0, 0]

                # Check prediction consistency
                if y_prob is not None:
                    pred_diff = abs(y_prob[idx1] - y_prob[idx2])
                else:
                    pred_diff = abs(y_pred[idx1] - y_pred[idx2])

                # Higher similarity should have lower prediction difference
                consistency = 1 - pred_diff if similarity > 0.8 else 1

                consistency_scores.append(consistency)

        return np.mean(consistency_scores) if consistency_scores else 1.0

    def _calculate_group_fairness(
        self,
        demographic_parity: Dict[str, float],
        equal_opportunity: Dict[str, float],
        disparate_impact: Dict[str, float],
    ) -> float:
        """Calculate overall group fairness score."""
        scores = []

        # Demographic parity score (lower is better, max diff = 1)
        if demographic_parity:
            dp_score = 1 - np.mean(list(demographic_parity.values()))
            scores.append(dp_score)

        # Equal opportunity score (lower is better, max diff = 1)
        if equal_opportunity:
            eo_score = 1 - np.mean(list(equal_opportunity.values()))
            scores.append(eo_score)

        # Disparate impact score (closer to 1 is better)
        if disparate_impact:
            di_score = np.mean(list(disparate_impact.values()))
            scores.append(di_score)

        return np.mean(scores) if scores else 0.0


class ModelComparisonFramework:
    """
    Framework for comparing multiple models with statistical significance testing.
    """

    def __init__(
        self,
        metrics: List[str] = None,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
    ):
        """
        Initialize model comparison framework.

        Args:
            metrics: List of metrics to compare
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for intervals
        """
        if metrics is None:
            metrics = ["roc_auc", "f1", "precision", "recall", "log_loss"]

        self.metrics = metrics
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level

    def compare_models(
        self,
        models: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int = 5,
    ) -> ModelComparisonResult:
        """
        Compare multiple models with statistical tests.

        Args:
            models: Dictionary of model_name: model pairs
            X: Features
            y: Target
            cv_folds: Number of CV folds

        Returns:
            ModelComparisonResult object
        """
        # Calculate metrics for each model
        model_scores = {}

        for model_name, model in models.items():
            scores = self._evaluate_model(model, X, y, cv_folds)
            model_scores[model_name] = scores

        # Create rankings
        rankings = self._create_rankings(model_scores)

        # Statistical significance testing
        significance_matrix = self._test_statistical_significance(models, X, y)

        # Pairwise differences
        pairwise_diff = self._calculate_pairwise_differences(model_scores)

        # Determine best model
        best_model = rankings.iloc[0]["model"]

        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(model_scores)

        return ModelComparisonResult(
            model_rankings=rankings,
            statistical_significance=significance_matrix,
            pairwise_differences=pairwise_diff,
            best_model=best_model,
            confidence_intervals=confidence_intervals,
        )

    def _evaluate_model(
        self, model: Any, X: pd.DataFrame, y: pd.Series, cv_folds: int
    ) -> Dict[str, float]:
        """Evaluate model on multiple metrics."""
        from sklearn.model_selection import cross_validate

        scoring = {
            "roc_auc": "roc_auc",
            "f1": "f1",
            "precision": "precision",
            "recall": "recall",
            "neg_log_loss": "neg_log_loss",
        }

        cv_results = cross_validate(
            model,
            X,
            y,
            cv=cv_folds,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1,
        )

        scores = {}
        for metric in self.metrics:
            if metric == "log_loss":
                scores[metric] = -np.mean(cv_results["test_neg_log_loss"])
            else:
                scores[metric] = np.mean(cv_results[f"test_{metric}"])

        return scores

    def _create_rankings(
        self, model_scores: Dict[str, Dict[str, float]]
    ) -> pd.DataFrame:
        """Create model rankings based on average rank across metrics."""
        rankings_data = []

        for model_name, scores in model_scores.items():
            avg_score = np.mean(list(scores.values()))
            rankings_data.append(
                {"model": model_name, "avg_score": avg_score, **scores}
            )

        rankings_df = pd.DataFrame(rankings_data)

        # Calculate rank for each metric
        for metric in self.metrics:
            if metric == "log_loss":
                # Lower is better for log_loss
                rankings_df[f"{metric}_rank"] = rankings_df[metric].rank(ascending=True)
            else:
                # Higher is better for other metrics
                rankings_df[f"{metric}_rank"] = rankings_df[metric].rank(
                    ascending=False
                )

        # Calculate average rank
        rank_columns = [f"{m}_rank" for m in self.metrics]
        rankings_df["avg_rank"] = rankings_df[rank_columns].mean(axis=1)

        # Sort by average rank
        rankings_df = rankings_df.sort_values("avg_rank")

        return rankings_df

    def _test_statistical_significance(
        self, models: Dict[str, Any], X: pd.DataFrame, y: pd.Series
    ) -> pd.DataFrame:
        """Test statistical significance between models."""
        model_names = list(models.keys())
        n_models = len(model_names)

        # Initialize significance matrix
        sig_matrix = pd.DataFrame(
            np.zeros((n_models, n_models)), index=model_names, columns=model_names
        )

        for i, model1_name in enumerate(model_names):
            for j, model2_name in enumerate(model_names):
                if i < j:
                    # Perform DeLong test or permutation test
                    p_value = self._delongs_test(
                        models[model1_name], models[model2_name], X, y
                    )

                    sig_matrix.loc[model1_name, model2_name] = p_value
                    sig_matrix.loc[model2_name, model1_name] = p_value

        return sig_matrix

    def _delongs_test(
        self, model1: Any, model2: Any, X: pd.DataFrame, y: pd.Series
    ) -> float:
        """
        Perform DeLong's test for comparing AUC scores.
        Simplified version using bootstrap.
        """
        # Get predictions
        from sklearn.model_selection import cross_val_predict

        y_prob1 = cross_val_predict(
            model1, X, y, cv=3, method="predict_proba", n_jobs=-1
        )[:, 1]
        y_prob2 = cross_val_predict(
            model2, X, y, cv=3, method="predict_proba", n_jobs=-1
        )[:, 1]

        # Bootstrap test
        differences = []

        for _ in range(self.n_bootstrap):
            idx = resample(range(len(y)), n_samples=len(y))
            auc1 = roc_auc_score(y.iloc[idx], y_prob1[idx])
            auc2 = roc_auc_score(y.iloc[idx], y_prob2[idx])
            differences.append(auc1 - auc2)

        # Calculate p-value
        differences = np.array(differences)
        p_value = 2 * min((differences > 0).mean(), (differences < 0).mean())

        return p_value

    def _calculate_pairwise_differences(
        self, model_scores: Dict[str, Dict[str, float]]
    ) -> pd.DataFrame:
        """Calculate pairwise differences in metrics."""
        model_names = list(model_scores.keys())
        n_models = len(model_names)

        differences = {}

        for metric in self.metrics:
            diff_matrix = pd.DataFrame(
                np.zeros((n_models, n_models)), index=model_names, columns=model_names
            )

            for i, model1 in enumerate(model_names):
                for j, model2 in enumerate(model_names):
                    diff = model_scores[model1][metric] - model_scores[model2][metric]
                    diff_matrix.loc[model1, model2] = diff

            differences[metric] = diff_matrix

        return differences

    def _calculate_confidence_intervals(
        self, model_scores: Dict[str, Dict[str, float]]
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for model scores."""
        confidence_intervals = {}

        for model_name, scores in model_scores.items():
            # Simplified: using normal approximation
            # In practice, would use bootstrap
            avg_score = np.mean(list(scores.values()))
            std_score = np.std(list(scores.values()))

            z_score = stats.norm.ppf((1 + self.confidence_level) / 2)
            margin = z_score * std_score

            confidence_intervals[model_name] = (avg_score - margin, avg_score + margin)

        return confidence_intervals


if __name__ == "__main__":
    print("Enhanced Evaluation Module Loaded Successfully")
    print("=" * 60)
    print("Available Classes:")
    print("  - BusinessMetricsCalculator: ROI and business-oriented metrics")
    print("  - FairnessEvaluator: Bias and fairness evaluation")
    print("  - ModelComparisonFramework: Statistical model comparison")
