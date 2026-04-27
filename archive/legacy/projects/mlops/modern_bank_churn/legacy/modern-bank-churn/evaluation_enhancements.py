"""Advanced Evaluation Framework for Bank Churn Models.
Implements custom business metrics, fairness evaluation, and ROI-based assessment.
"""

import warnings
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore")


@dataclass
class BusinessMetrics:
    """Container for business-oriented metrics."""

    revenue_impact: float
    cost_savings: float
    roi: float
    customer_lifetime_value: float
    retention_rate: float
    intervention_cost: float
    false_alarm_cost: float
    missed_churn_cost: float


@dataclass
class FairnessMetrics:
    """Container for fairness and bias metrics."""

    demographic_parity: dict[str, float]
    equal_opportunity: dict[str, float]
    calibration_by_group: dict[str, float]
    disparate_impact: dict[str, float]
    group_fairness_score: float


class CustomBusinessMetrics:
    """Calculate custom business metrics for churn prediction."""

    def __init__(
        self,
        avg_customer_value: float = 1000,
        retention_cost: float = 50,
        acquisition_cost: float = 200,
        intervention_success_rate: float = 0.3,
    ):
        """Initialize business metrics calculator.

        Args:
            avg_customer_value: Average customer lifetime value
            retention_cost: Cost of retention intervention
            acquisition_cost: Cost of acquiring new customer
            intervention_success_rate: Success rate of retention efforts
        """
        self.avg_customer_value = avg_customer_value
        self.retention_cost = retention_cost
        self.acquisition_cost = acquisition_cost
        self.intervention_success_rate = intervention_success_rate

    def calculate_roi(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None
    ) -> BusinessMetrics:
        """Calculate ROI and business metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities

        Returns:
            BusinessMetrics object
        """
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Revenue calculations
        # True Positives: Successfully retained customers
        revenue_saved = tp * self.avg_customer_value * self.intervention_success_rate

        # False Negatives: Missed churners (lost revenue)
        revenue_lost = fn * self.avg_customer_value

        # False Positives: Unnecessary interventions
        unnecessary_cost = fp * self.retention_cost

        # True Positives: Intervention costs for actual churners
        necessary_cost = tp * self.retention_cost

        # Total impact
        total_benefit = revenue_saved - revenue_lost
        total_cost = unnecessary_cost + necessary_cost

        # ROI
        roi = (total_benefit - total_cost) / (total_cost + 1e-10)

        # Cost of different error types
        false_alarm_cost = fp * self.retention_cost
        missed_churn_cost = fn * (self.avg_customer_value - self.acquisition_cost)

        # Customer lifetime value impact
        if y_prob is not None:
            # Expected value calculation
            expected_value = np.mean(
                y_prob * self.avg_customer_value * self.intervention_success_rate
                - (1 - y_prob) * self.retention_cost
            )
        else:
            expected_value = self.avg_customer_value

        # Retention rate
        retention_rate = (tn + tp * self.intervention_success_rate) / len(y_true)

        return BusinessMetrics(
            revenue_impact=total_benefit,
            cost_savings=revenue_saved - total_cost,
            roi=roi,
            customer_lifetime_value=expected_value,
            retention_rate=retention_rate,
            intervention_cost=total_cost,
            false_alarm_cost=false_alarm_cost,
            missed_churn_cost=missed_churn_cost,
        )

    def optimal_threshold(
        self, y_true: np.ndarray, y_prob: np.ndarray, metric: str = "roi"
    ) -> tuple[float, float]:
        """Find optimal probability threshold based on business metric.

        Args:
            y_true: True labels
            y_prob: Prediction probabilities
            metric: Metric to optimize ('roi', 'profit', 'f1')

        Returns:
            Optimal threshold and metric value
        """
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_threshold = 0.5
        best_value = -np.inf

        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)

            if metric == "roi":
                business_metrics = self.calculate_roi(y_true, y_pred, y_prob)
                value = business_metrics.roi
            elif metric == "profit":
                business_metrics = self.calculate_roi(y_true, y_pred, y_prob)
                value = business_metrics.cost_savings
            elif metric == "f1":
                value = f1_score(y_true, y_pred)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            if value > best_value:
                best_value = value
                best_threshold = threshold

        return best_threshold, best_value

    def calculate_profit_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        budget_fractions: np.ndarray = None,
    ) -> dict:
        """Calculate profit curve for different intervention budgets.

        Args:
            y_true: True labels
            y_prob: Prediction probabilities
            budget_fractions: Fractions of customers to target

        Returns:
            Dictionary with profit curve data
        """
        if budget_fractions is None:
            budget_fractions = np.arange(0, 1.01, 0.01)

        n_samples = len(y_true)
        profits = []
        rois = []

        # Sort by probability (descending)
        sorted_indices = np.argsort(y_prob)[::-1]
        y_true_sorted = y_true[sorted_indices]
        y_prob_sorted = y_prob[sorted_indices]

        for fraction in budget_fractions:
            n_target = int(fraction * n_samples)

            if n_target == 0:
                profits.append(0)
                rois.append(0)
                continue

            # Target top n customers
            y_pred = np.zeros(n_samples)
            y_pred[sorted_indices[:n_target]] = 1

            # Calculate metrics
            business_metrics = self.calculate_roi(y_true, y_pred, y_prob)
            profits.append(business_metrics.cost_savings)
            rois.append(business_metrics.roi)

        return {
            "budget_fractions": budget_fractions,
            "profits": np.array(profits),
            "rois": np.array(rois),
            "max_profit": np.max(profits),
            "optimal_budget": budget_fractions[np.argmax(profits)],
        }


class FairnessEvaluator:
    """Evaluate model fairness and bias."""

    def __init__(self, protected_attributes: list[str]):
        """Initialize fairness evaluator.

        Args:
            protected_attributes: List of protected attribute names
        """
        self.protected_attributes = protected_attributes
        self.fairness_metrics = {}

    def evaluate_fairness(
        self,
        X: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray = None,
    ) -> FairnessMetrics:
        """Evaluate fairness metrics.

        Args:
            X: Feature DataFrame with protected attributes
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities

        Returns:
            FairnessMetrics object
        """
        demographic_parity = {}
        equal_opportunity = {}
        calibration_by_group = {}
        disparate_impact = {}

        for attr in self.protected_attributes:
            if attr not in X.columns:
                continue

            # Get unique groups
            groups = X[attr].unique()

            # Calculate metrics for each group
            group_metrics = {}
            for group in groups:
                mask = X[attr] == group
                group_y_true = y_true[mask]
                group_y_pred = y_pred[mask]

                if len(group_y_true) == 0:
                    continue

                # Positive prediction rate (for demographic parity)
                positive_rate = np.mean(group_y_pred)
                group_metrics[group] = {
                    "positive_rate": positive_rate,
                    "size": len(group_y_true),
                }

                # True positive rate (for equal opportunity)
                if np.sum(group_y_true) > 0:
                    tpr = recall_score(group_y_true, group_y_pred)
                    group_metrics[group]["tpr"] = tpr

                # Calibration (if probabilities available)
                if y_prob is not None:
                    group_y_prob = y_prob[mask]
                    # Mean calibration error
                    calibration_error = np.abs(
                        np.mean(group_y_prob) - np.mean(group_y_true)
                    )
                    group_metrics[group]["calibration_error"] = calibration_error

            # Calculate fairness metrics
            if len(group_metrics) >= 2:
                # Demographic parity: difference in positive prediction rates
                positive_rates = [m["positive_rate"] for m in group_metrics.values()]
                demographic_parity[attr] = np.max(positive_rates) - np.min(
                    positive_rates
                )

                # Equal opportunity: difference in TPR
                tprs = [m.get("tpr", 0) for m in group_metrics.values()]
                if any(tprs):
                    equal_opportunity[attr] = np.max(tprs) - np.min(tprs)

                # Disparate impact: ratio of positive rates
                if np.min(positive_rates) > 0:
                    disparate_impact[attr] = np.min(positive_rates) / np.max(
                        positive_rates
                    )
                else:
                    disparate_impact[attr] = 0

                # Calibration by group
                if y_prob is not None:
                    calib_errors = [
                        m.get("calibration_error", 0) for m in group_metrics.values()
                    ]
                    calibration_by_group[attr] = np.mean(calib_errors)

        # Overall fairness score (lower is better)
        fairness_scores = []
        if demographic_parity:
            fairness_scores.append(np.mean(list(demographic_parity.values())))
        if equal_opportunity:
            fairness_scores.append(np.mean(list(equal_opportunity.values())))
        if calibration_by_group:
            fairness_scores.append(np.mean(list(calibration_by_group.values())))

        group_fairness_score = np.mean(fairness_scores) if fairness_scores else 0

        return FairnessMetrics(
            demographic_parity=demographic_parity,
            equal_opportunity=equal_opportunity,
            calibration_by_group=calibration_by_group,
            disparate_impact=disparate_impact,
            group_fairness_score=group_fairness_score,
        )

    def bias_mitigation_recommendations(
        self, fairness_metrics: FairnessMetrics
    ) -> list[str]:
        """Provide recommendations for bias mitigation.

        Args:
            fairness_metrics: Calculated fairness metrics

        Returns:
            List of recommendations
        """
        recommendations = []

        # Check demographic parity
        for attr, value in fairness_metrics.demographic_parity.items():
            if value > 0.1:
                recommendations.append(
                    f"High demographic parity difference ({value:.3f}) for {attr}. "
                    f"Consider rebalancing training data or adjusting thresholds per group."
                )

        # Check equal opportunity
        for attr, value in fairness_metrics.equal_opportunity.items():
            if value > 0.1:
                recommendations.append(
                    f"Equal opportunity violation ({value:.3f}) for {attr}. "
                    f"Consider group-specific thresholds or fairness-aware training."
                )

        # Check disparate impact
        for attr, value in fairness_metrics.disparate_impact.items():
            if value < 0.8:  # 80% rule
                recommendations.append(
                    f"Potential disparate impact ({value:.3f}) for {attr}. "
                    f"Review feature importance and consider removing biased features."
                )

        # Check calibration
        for attr, value in fairness_metrics.calibration_by_group.items():
            if value > 0.05:
                recommendations.append(
                    f"Calibration issues ({value:.3f}) for {attr}. "
                    f"Consider group-specific calibration."
                )

        if not recommendations:
            recommendations.append("No significant fairness issues detected.")

        return recommendations


class ModelComparisonFramework:
    """Framework for comparing multiple models."""

    def __init__(self):
        self.models = {}
        self.results = {}
        self.comparison_df = None

    def add_model(
        self,
        name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray = None,
        model_object=None,
    ):
        """Add model to comparison.

        Args:
            name: Model name
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities
            model_object: Optional model object
        """
        self.models[name] = {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "model": model_object,
        }

    def compare_models(
        self,
        business_metrics_calculator: CustomBusinessMetrics = None,
        fairness_evaluator: FairnessEvaluator = None,
        X: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Compare all models.

        Args:
            business_metrics_calculator: Optional business metrics calculator
            fairness_evaluator: Optional fairness evaluator
            X: Optional feature DataFrame for fairness evaluation

        Returns:
            Comparison DataFrame
        """
        comparison_data = []

        for name, model_data in self.models.items():
            y_true = model_data["y_true"]
            y_pred = model_data["y_pred"]
            y_prob = model_data["y_prob"]

            # Basic metrics
            metrics = {
                "model": name,
                "accuracy": np.mean(y_true == y_pred),
                "precision": precision_score(y_true, y_pred),
                "recall": recall_score(y_true, y_pred),
                "f1": f1_score(y_true, y_pred),
            }

            # AUC if probabilities available
            if y_prob is not None:
                metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
                # Precision-Recall AUC
                precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
                metrics["pr_auc"] = auc(recall_vals, precision_vals)

            # Business metrics
            if business_metrics_calculator and y_prob is not None:
                business_metrics = business_metrics_calculator.calculate_roi(
                    y_true, y_pred, y_prob
                )
                metrics["roi"] = business_metrics.roi
                metrics["revenue_impact"] = business_metrics.revenue_impact
                metrics["cost_savings"] = business_metrics.cost_savings

            # Fairness metrics
            if fairness_evaluator and X is not None and y_prob is not None:
                fairness_metrics = fairness_evaluator.evaluate_fairness(
                    X, y_true, y_pred, y_prob
                )
                metrics["fairness_score"] = fairness_metrics.group_fairness_score

            comparison_data.append(metrics)

        self.comparison_df = pd.DataFrame(comparison_data)
        return self.comparison_df

    def statistical_significance_test(
        self,
        model1_name: str,
        model2_name: str,
        metric: str = "roc_auc",
        n_bootstrap: int = 1000,
    ) -> dict:
        """Test statistical significance of model differences.

        Args:
            model1_name: First model name
            model2_name: Second model name
            metric: Metric to compare
            n_bootstrap: Number of bootstrap samples

        Returns:
            Test results
        """
        if model1_name not in self.models or model2_name not in self.models:
            raise ValueError("Models not found")

        model1_data = self.models[model1_name]
        model2_data = self.models[model2_name]

        y_true = model1_data["y_true"]
        y_prob1 = model1_data["y_prob"]
        y_prob2 = model2_data["y_prob"]

        if y_prob1 is None or y_prob2 is None:
            raise ValueError("Probabilities required for significance testing")

        # Bootstrap test
        differences = []
        n_samples = len(y_true)

        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_prob1_boot = y_prob1[indices]
            y_prob2_boot = y_prob2[indices]

            # Calculate metric difference
            if metric == "roc_auc":
                score1 = roc_auc_score(y_true_boot, y_prob1_boot)
                score2 = roc_auc_score(y_true_boot, y_prob2_boot)
            elif metric == "pr_auc":
                p1, r1, _ = precision_recall_curve(y_true_boot, y_prob1_boot)
                score1 = auc(r1, p1)
                p2, r2, _ = precision_recall_curve(y_true_boot, y_prob2_boot)
                score2 = auc(r2, p2)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            differences.append(score2 - score1)

        differences = np.array(differences)

        # Calculate p-value (two-tailed)
        p_value = 2 * min(np.mean(differences <= 0), np.mean(differences >= 0))

        # Confidence interval
        ci_lower = np.percentile(differences, 2.5)
        ci_upper = np.percentile(differences, 97.5)

        return {
            "model1": model1_name,
            "model2": model2_name,
            "metric": metric,
            "mean_difference": np.mean(differences),
            "std_difference": np.std(differences),
            "p_value": p_value,
            "ci_95": (ci_lower, ci_upper),
            "significant": p_value < 0.05,
        }

    def rank_models(self, weights: dict[str, float] = None) -> pd.DataFrame:
        """Rank models based on weighted metrics.

        Args:
            weights: Dictionary of metric weights

        Returns:
            Ranked DataFrame
        """
        if self.comparison_df is None:
            raise ValueError("Must run compare_models first")

        if weights is None:
            # Default weights
            weights = {"roc_auc": 0.3, "f1": 0.2, "roi": 0.3, "fairness_score": 0.2}

        # Normalize scores (0-1 scale)
        normalized_df = self.comparison_df.copy()
        for col in self.comparison_df.columns:
            if col == "model":
                continue
            if col in normalized_df.columns:
                # For fairness_score, lower is better
                if col == "fairness_score":
                    normalized_df[col] = 1 - (
                        (normalized_df[col] - normalized_df[col].min())
                        / (normalized_df[col].max() - normalized_df[col].min() + 1e-10)
                    )
                else:
                    normalized_df[col] = (
                        normalized_df[col] - normalized_df[col].min()
                    ) / (normalized_df[col].max() - normalized_df[col].min() + 1e-10)

        # Calculate weighted score
        normalized_df["weighted_score"] = 0
        for metric, weight in weights.items():
            if metric in normalized_df.columns:
                normalized_df["weighted_score"] += normalized_df[metric] * weight

        # Rank
        normalized_df["rank"] = normalized_df["weighted_score"].rank(ascending=False)
        normalized_df = normalized_df.sort_values("rank")

        return normalized_df


class EvaluationVisualizer:
    """Visualization tools for model evaluation."""

    def __init__(self, figsize: tuple[int, int] = (15, 10)):
        """Initialize visualizer.

        Args:
            figsize: Figure size
        """
        self.figsize = figsize

    def plot_comprehensive_evaluation(
        self, y_true: np.ndarray, y_prob: np.ndarray, model_name: str = "Model"
    ) -> plt.Figure:
        """Create comprehensive evaluation plot.

        Args:
            y_true: True labels
            y_prob: Prediction probabilities
            model_name: Model name

        Returns:
            Figure object
        """
        fig, axes = plt.subplots(2, 3, figsize=self.figsize)

        # 1. ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)

        axes[0, 0].plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
        axes[0, 0].plot([0, 1], [0, 1], "k--", alpha=0.5)
        axes[0, 0].set_xlabel("False Positive Rate")
        axes[0, 0].set_ylabel("True Positive Rate")
        axes[0, 0].set_title("ROC Curve")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)

        axes[0, 1].plot(recall, precision, label=f"PR (AUC = {pr_auc:.3f})")
        axes[0, 1].set_xlabel("Recall")
        axes[0, 1].set_ylabel("Precision")
        axes[0, 1].set_title("Precision-Recall Curve")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Calibration Plot
        from sklearn.calibration import calibration_curve

        fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)

        axes[0, 2].plot(mean_pred, fraction_pos, "o-", label=model_name)
        axes[0, 2].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
        axes[0, 2].set_xlabel("Mean Predicted Probability")
        axes[0, 2].set_ylabel("Fraction of Positives")
        axes[0, 2].set_title("Calibration Plot")
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # 4. Prediction Distribution
        axes[1, 0].hist(
            y_prob[y_true == 0], bins=30, alpha=0.5, label="Negative", color="blue"
        )
        axes[1, 0].hist(
            y_prob[y_true == 1], bins=30, alpha=0.5, label="Positive", color="red"
        )
        axes[1, 0].set_xlabel("Prediction Probability")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].set_title("Prediction Distribution by Class")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 5. Confusion Matrix
        y_pred = (y_prob >= 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred)

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1, 1])
        axes[1, 1].set_xlabel("Predicted")
        axes[1, 1].set_ylabel("Actual")
        axes[1, 1].set_title("Confusion Matrix")

        # 6. Threshold Analysis
        thresholds = np.arange(0.1, 0.9, 0.05)
        f1_scores = []
        precision_scores = []
        recall_scores = []

        for threshold in thresholds:
            y_pred_thresh = (y_prob >= threshold).astype(int)
            f1_scores.append(f1_score(y_true, y_pred_thresh))
            precision_scores.append(precision_score(y_true, y_pred_thresh))
            recall_scores.append(recall_score(y_true, y_pred_thresh))

        axes[1, 2].plot(thresholds, f1_scores, label="F1 Score")
        axes[1, 2].plot(thresholds, precision_scores, label="Precision")
        axes[1, 2].plot(thresholds, recall_scores, label="Recall")
        axes[1, 2].set_xlabel("Threshold")
        axes[1, 2].set_ylabel("Score")
        axes[1, 2].set_title("Threshold Analysis")
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        plt.suptitle(f"{model_name} - Comprehensive Evaluation", fontsize=16)
        plt.tight_layout()

        return fig


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 1000

    # Simulate predictions from two models
    y_true = np.random.randint(0, 2, n_samples)
    y_prob_model1 = np.random.beta(2, 2, n_samples)
    y_prob_model2 = np.random.beta(2.5, 1.5, n_samples)

    # Add some correlation with true labels
    y_prob_model1[y_true == 1] += 0.2
    y_prob_model2[y_true == 1] += 0.25
    y_prob_model1 = np.clip(y_prob_model1, 0, 1)
    y_prob_model2 = np.clip(y_prob_model2, 0, 1)

    y_pred_model1 = (y_prob_model1 >= 0.5).astype(int)
    y_pred_model2 = (y_prob_model2 >= 0.5).astype(int)

    # Business metrics
    business_calculator = CustomBusinessMetrics()
    metrics1 = business_calculator.calculate_roi(y_true, y_pred_model1, y_prob_model1)
    print(f"Model 1 ROI: {metrics1.roi:.2f}")

    # Find optimal threshold
    optimal_threshold, optimal_roi = business_calculator.optimal_threshold(
        y_true, y_prob_model1, metric="roi"
    )
    print(f"Optimal threshold: {optimal_threshold:.2f}, ROI: {optimal_roi:.2f}")

    # Model comparison
    comparison = ModelComparisonFramework()
    comparison.add_model("Model_1", y_true, y_pred_model1, y_prob_model1)
    comparison.add_model("Model_2", y_true, y_pred_model2, y_prob_model2)

    comparison_df = comparison.compare_models(business_calculator)
    print("\nModel Comparison:")
    print(comparison_df)

    # Statistical significance
    sig_test = comparison.statistical_significance_test("Model_1", "Model_2")
    print("\nStatistical Significance Test:")
    print(f"P-value: {sig_test['p_value']:.4f}")
    print(f"Significant: {sig_test['significant']}")

    print("\nEvaluation complete!")
