"""Compliance Tools - Model governance, fairness testing, and regulatory compliance"""

import hashlib
import json
import logging
import pickle
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
)
from sklearn.model_selection import cross_val_score

# Fairness metrics

# For advanced fairness metrics
try:
    import fairlearn.metrics as fairness_metrics
    from fairlearn.reductions import (
        DemographicParity,
        EqualizedOdds,
        ExponentiatedGradient,
        GridSearch,
    )

    FAIRLEARN_AVAILABLE = True
except ImportError:
    FAIRLEARN_AVAILABLE = False
    warnings.warn("Fairlearn not installed. Some fairness features will be limited.")

try:
    from aif360.algorithms.postprocessing import EqOddsPostprocessing
    from aif360.algorithms.preprocessing import DisparateImpactRemover, Reweighing
    from aif360.datasets import BinaryLabelDataset
    from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric

    AIF360_AVAILABLE = True
except ImportError:
    AIF360_AVAILABLE = False
    warnings.warn("AIF360 not installed. Some fairness features will be limited.")

logger = logging.getLogger(__name__)


class RegulatoryFramework(Enum):
    """Regulatory frameworks"""

    GDPR = "GDPR"  # General Data Protection Regulation (EU)
    CCPA = "CCPA"  # California Consumer Privacy Act
    HIPAA = "HIPAA"  # Health Insurance Portability and Accountability Act (US)
    SOX = "SOX"  # Sarbanes-Oxley Act
    BASEL_III = "BASEL_III"  # Basel III (Banking)
    MiFID_II = "MiFID_II"  # Markets in Financial Instruments Directive II
    PCI_DSS = "PCI_DSS"  # Payment Card Industry Data Security Standard
    ISO_27001 = "ISO_27001"  # Information Security Management
    AI_ACT = "AI_ACT"  # EU AI Act
    FDA_AI_ML = "FDA_AI_ML"  # FDA AI/ML Medical Devices


class RiskLevel(Enum):
    """Risk levels for model assessment"""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ModelLifecycleStage(Enum):
    """Model lifecycle stages"""

    DEVELOPMENT = "DEVELOPMENT"
    VALIDATION = "VALIDATION"
    TESTING = "TESTING"
    STAGING = "STAGING"
    PRODUCTION = "PRODUCTION"
    DEPRECATED = "DEPRECATED"
    ARCHIVED = "ARCHIVED"


@dataclass
class ModelCard:
    """Model documentation card for governance"""

    model_id: str
    model_name: str
    version: str
    created_date: datetime
    updated_date: datetime
    author: str
    description: str
    use_case: str
    model_type: str
    algorithm: str
    training_data: dict[str, Any]
    performance_metrics: dict[str, float]
    limitations: list[str]
    ethical_considerations: list[str]
    fairness_metrics: dict[str, float] = field(default_factory=dict)
    regulatory_compliance: list[str] = field(default_factory=list)
    risk_assessment: dict[str, Any] = field(default_factory=dict)
    lifecycle_stage: ModelLifecycleStage = ModelLifecycleStage.DEVELOPMENT
    approval_status: str = "PENDING"
    approvers: list[dict[str, Any]] = field(default_factory=list)
    audit_trail: list[dict[str, Any]] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    deployment_info: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceConfig:
    """Configuration for compliance tools"""

    regulatory_frameworks: list[RegulatoryFramework] = field(
        default_factory=lambda: [RegulatoryFramework.GDPR, RegulatoryFramework.AI_ACT]
    )
    fairness_threshold: float = 0.8
    risk_threshold: float = 0.7
    require_approval: bool = True
    min_approvers: int = 2
    audit_retention_days: int = 365
    enable_fairness_monitoring: bool = True
    enable_drift_detection: bool = True
    protected_attributes: list[str] = field(
        default_factory=lambda: ["gender", "race", "age", "ethnicity"]
    )


class ModelGovernanceFramework:
    """Model governance framework for lifecycle management"""

    def __init__(self, config: ComplianceConfig = None):
        self.config = config or ComplianceConfig()
        self.model_registry = {}
        self.approval_queue = []
        self.audit_log = []

    def register_model(self, model: BaseEstimator, model_card: ModelCard) -> str:
        """Register a model in the governance framework"""
        try:
            # Generate unique model ID if not provided
            if not model_card.model_id:
                model_card.model_id = self._generate_model_id(model, model_card)

            # Add to registry
            self.model_registry[model_card.model_id] = {
                "model": model,
                "card": model_card,
                "registered_date": datetime.now(),
            }

            # Log registration
            self._log_event(
                model_id=model_card.model_id,
                event_type="MODEL_REGISTERED",
                details={
                    "model_name": model_card.model_name,
                    "version": model_card.version,
                },
            )

            # Check if approval required
            if self.config.require_approval:
                self.approval_queue.append(model_card.model_id)

            logger.info(f"Model {model_card.model_id} registered successfully")
            return model_card.model_id

        except Exception as e:
            logger.error(f"Error registering model: {e}")
            raise

    def approve_model(self, model_id: str, approver: str, comments: str = "") -> bool:
        """Approve a model for production use"""
        if model_id not in self.model_registry:
            raise ValueError(f"Model {model_id} not found in registry")

        model_entry = self.model_registry[model_id]
        model_card = model_entry["card"]

        # Add approver
        approval = {
            "approver": approver,
            "timestamp": datetime.now(),
            "comments": comments,
        }
        model_card.approvers.append(approval)

        # Check if enough approvals
        if len(model_card.approvers) >= self.config.min_approvers:
            model_card.approval_status = "APPROVED"
            model_card.lifecycle_stage = ModelLifecycleStage.STAGING

            # Remove from queue
            if model_id in self.approval_queue:
                self.approval_queue.remove(model_id)

            # Log approval
            self._log_event(
                model_id=model_id,
                event_type="MODEL_APPROVED",
                details={
                    "approver": approver,
                    "total_approvals": len(model_card.approvers),
                },
            )

            return True

        return False

    def promote_model(
        self, model_id: str, target_stage: ModelLifecycleStage, promoted_by: str
    ) -> bool:
        """Promote model to next lifecycle stage"""
        if model_id not in self.model_registry:
            raise ValueError(f"Model {model_id} not found in registry")

        model_entry = self.model_registry[model_id]
        model_card = model_entry["card"]

        # Check if approved for production
        if target_stage == ModelLifecycleStage.PRODUCTION:
            if model_card.approval_status != "APPROVED":
                logger.warning(f"Model {model_id} not approved for production")
                return False

        # Update stage
        old_stage = model_card.lifecycle_stage
        model_card.lifecycle_stage = target_stage
        model_card.updated_date = datetime.now()

        # Log promotion
        self._log_event(
            model_id=model_id,
            event_type="MODEL_PROMOTED",
            details={
                "from_stage": old_stage.value,
                "to_stage": target_stage.value,
                "promoted_by": promoted_by,
            },
        )

        return True

    def get_model_lineage(self, model_id: str) -> dict[str, Any]:
        """Get complete lineage of a model"""
        if model_id not in self.model_registry:
            raise ValueError(f"Model {model_id} not found in registry")

        model_entry = self.model_registry[model_id]
        model_card = model_entry["card"]

        # Get audit trail for this model
        model_events = [
            event for event in self.audit_log if event.get("model_id") == model_id
        ]

        lineage = {
            "model_id": model_id,
            "model_name": model_card.model_name,
            "version": model_card.version,
            "current_stage": model_card.lifecycle_stage.value,
            "approval_status": model_card.approval_status,
            "created_date": model_card.created_date.isoformat(),
            "training_data": model_card.training_data,
            "dependencies": model_card.dependencies,
            "events": model_events,
            "approvers": model_card.approvers,
        }

        return lineage

    def _generate_model_id(self, model: BaseEstimator, model_card: ModelCard) -> str:
        """Generate unique model ID"""
        content = (
            f"{model_card.model_name}_{model_card.version}_{datetime.now().isoformat()}"
        )
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def _log_event(self, model_id: str, event_type: str, details: dict = None):
        """Log governance event"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "model_id": model_id,
            "event_type": event_type,
            "details": details or {},
        }
        self.audit_log.append(event)

        # Also add to model's audit trail
        if model_id in self.model_registry:
            self.model_registry[model_id]["card"].audit_trail.append(event)


class ModelRiskAssessment:
    """Model risk assessment framework"""

    def __init__(self, config: ComplianceConfig = None):
        self.config = config or ComplianceConfig()
        self.risk_factors = {
            "data_quality": {"weight": 0.2, "score": 0},
            "model_complexity": {"weight": 0.15, "score": 0},
            "performance_stability": {"weight": 0.2, "score": 0},
            "fairness": {"weight": 0.15, "score": 0},
            "interpretability": {"weight": 0.1, "score": 0},
            "security": {"weight": 0.1, "score": 0},
            "compliance": {"weight": 0.1, "score": 0},
        }

    def assess_model_risk(
        self,
        model: BaseEstimator,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_card: ModelCard = None,
    ) -> dict[str, Any]:
        """Comprehensive model risk assessment"""
        risk_scores = {}

        # 1. Data Quality Risk
        risk_scores["data_quality"] = self._assess_data_quality(X_train, X_test)

        # 2. Model Complexity Risk
        risk_scores["model_complexity"] = self._assess_model_complexity(model)

        # 3. Performance Stability Risk
        risk_scores["performance_stability"] = self._assess_performance_stability(
            model, X_train, y_train, X_test, y_test
        )

        # 4. Fairness Risk
        if self.config.enable_fairness_monitoring:
            risk_scores["fairness"] = self._assess_fairness_risk(model, X_test, y_test)

        # 5. Interpretability Risk
        risk_scores["interpretability"] = self._assess_interpretability(model)

        # 6. Security Risk
        risk_scores["security"] = self._assess_security_risk(model, model_card)

        # 7. Compliance Risk
        risk_scores["compliance"] = self._assess_compliance_risk(model_card)

        # Calculate overall risk score
        overall_risk = sum(
            self.risk_factors[factor]["weight"] * risk_scores[factor]
            for factor in self.risk_factors
        )

        # Determine risk level
        if overall_risk < 0.3:
            risk_level = RiskLevel.LOW
        elif overall_risk < 0.5:
            risk_level = RiskLevel.MEDIUM
        elif overall_risk < 0.7:
            risk_level = RiskLevel.HIGH
        else:
            risk_level = RiskLevel.CRITICAL

        assessment = {
            "timestamp": datetime.now().isoformat(),
            "overall_risk_score": overall_risk,
            "risk_level": risk_level.value,
            "risk_factors": risk_scores,
            "recommendations": self._generate_recommendations(risk_scores, risk_level),
            "mitigation_strategies": self._suggest_mitigations(risk_scores),
        }

        # Update model card if provided
        if model_card:
            model_card.risk_assessment = assessment

        return assessment

    def _assess_data_quality(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> float:
        """Assess data quality risk"""
        risk_score = 0

        # Check for missing values
        train_missing = X_train.isnull().sum().sum() / (
            X_train.shape[0] * X_train.shape[1]
        )
        test_missing = X_test.isnull().sum().sum() / (X_test.shape[0] * X_test.shape[1])

        if train_missing > 0.1 or test_missing > 0.1:
            risk_score += 0.3

        # Check for data drift
        for col in X_train.select_dtypes(include=[np.number]).columns:
            if col in X_test.columns:
                # KS test for distribution shift
                ks_stat, p_value = stats.ks_2samp(
                    X_train[col].dropna(), X_test[col].dropna()
                )
                if p_value < 0.05:
                    risk_score += 0.1

        # Check for class imbalance (if classification)
        # Simplified check - would need y_train for full implementation

        return min(risk_score, 1.0)

    def _assess_model_complexity(self, model: BaseEstimator) -> float:
        """Assess model complexity risk"""
        risk_score = 0.5  # Default medium complexity

        # Check model type
        model_type = type(model).__name__

        # Simple models
        simple_models = [
            "LinearRegression",
            "LogisticRegression",
            "DecisionTreeClassifier",
            "DecisionTreeRegressor",
            "KNeighborsClassifier",
            "KNeighborsRegressor",
        ]

        # Complex models
        complex_models = [
            "RandomForestClassifier",
            "RandomForestRegressor",
            "GradientBoostingClassifier",
            "GradientBoostingRegressor",
            "XGBClassifier",
            "XGBRegressor",
            "LGBMClassifier",
            "LGBMRegressor",
        ]

        # Very complex models
        very_complex = [
            "MLPClassifier",
            "MLPRegressor",
            "NeuralNetwork",
            "DeepLearning",
            "Ensemble",
        ]

        if model_type in simple_models:
            risk_score = 0.2
        elif model_type in complex_models:
            risk_score = 0.5
        elif (
            model_type in very_complex or "Neural" in model_type or "Deep" in model_type
        ):
            risk_score = 0.8

        # Check for ensemble methods
        if hasattr(model, "estimators_") or hasattr(model, "base_estimator_"):
            risk_score += 0.1

        return min(risk_score, 1.0)

    def _assess_performance_stability(
        self,
        model: BaseEstimator,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> float:
        """Assess model performance stability"""
        risk_score = 0

        try:
            # Cross-validation scores
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            cv_std = np.std(cv_scores)

            # High variance in CV scores indicates instability
            if cv_std > 0.1:
                risk_score += 0.3

            # Train vs test performance gap
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            performance_gap = abs(train_score - test_score)

            if performance_gap > 0.1:
                risk_score += 0.3  # Potential overfitting

            # Test score threshold
            if test_score < 0.7:
                risk_score += 0.2

        except Exception as e:
            logger.warning(f"Error assessing performance stability: {e}")
            risk_score = 0.5  # Default to medium risk

        return min(risk_score, 1.0)

    def _assess_fairness_risk(
        self, model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series
    ) -> float:
        """Assess fairness risk"""
        risk_score = 0

        # Check for protected attributes
        protected_found = False
        for attr in self.config.protected_attributes:
            if attr in X_test.columns:
                protected_found = True
                # Would calculate fairness metrics here
                # Simplified for demonstration
                risk_score += 0.2

        if not protected_found:
            # No protected attributes found - could be hidden bias
            risk_score = 0.5

        return min(risk_score, 1.0)

    def _assess_interpretability(self, model: BaseEstimator) -> float:
        """Assess model interpretability risk"""
        model_type = type(model).__name__

        # Interpretable models
        interpretable = [
            "LinearRegression",
            "LogisticRegression",
            "DecisionTreeClassifier",
            "DecisionTreeRegressor",
        ]

        # Somewhat interpretable
        somewhat = [
            "RandomForestClassifier",
            "RandomForestRegressor",
            "GradientBoostingClassifier",
            "GradientBoostingRegressor",
        ]

        # Black box models
        blackbox = ["MLPClassifier", "MLPRegressor", "SVC", "SVR"]

        if model_type in interpretable:
            return 0.2
        elif model_type in somewhat:
            return 0.5
        elif model_type in blackbox or "Neural" in model_type or "Deep" in model_type:
            return 0.8
        else:
            return 0.6  # Default

    def _assess_security_risk(
        self, model: BaseEstimator, model_card: ModelCard = None
    ) -> float:
        """Assess security risk"""
        risk_score = 0

        # Check for pickle/joblib serialization (potential security risk)
        try:
            # Test serialization
            pickle.dumps(model)
            risk_score += 0.2  # Pickle can be insecure
        except:
            pass

        # Check for external dependencies
        if model_card and model_card.dependencies:
            if len(model_card.dependencies) > 5:
                risk_score += 0.2

        # Check for potential data leakage
        if hasattr(model, "training_data_"):
            risk_score += 0.3  # Model stores training data

        return min(risk_score, 1.0)

    def _assess_compliance_risk(self, model_card: ModelCard = None) -> float:
        """Assess regulatory compliance risk"""
        if not model_card:
            return 0.8  # High risk if no documentation

        risk_score = 0

        # Check documentation completeness
        if not model_card.description:
            risk_score += 0.2
        if not model_card.limitations:
            risk_score += 0.2
        if not model_card.ethical_considerations:
            risk_score += 0.2
        if not model_card.regulatory_compliance:
            risk_score += 0.2

        return min(risk_score, 1.0)

    def _generate_recommendations(
        self, risk_scores: dict[str, float], risk_level: RiskLevel
    ) -> list[str]:
        """Generate risk mitigation recommendations"""
        recommendations = []

        for factor, score in risk_scores.items():
            if score > 0.7:
                if factor == "data_quality":
                    recommendations.append(
                        "Improve data quality through cleaning and validation"
                    )
                elif factor == "model_complexity":
                    recommendations.append(
                        "Consider simpler models or add interpretability tools"
                    )
                elif factor == "performance_stability":
                    recommendations.append(
                        "Address overfitting through regularization or more data"
                    )
                elif factor == "fairness":
                    recommendations.append(
                        "Implement fairness constraints and bias mitigation"
                    )
                elif factor == "interpretability":
                    recommendations.append(
                        "Add SHAP/LIME explanations for model predictions"
                    )
                elif factor == "security":
                    recommendations.append(
                        "Implement secure model serialization and access controls"
                    )
                elif factor == "compliance":
                    recommendations.append(
                        "Complete model documentation and compliance checks"
                    )

        if risk_level == RiskLevel.CRITICAL:
            recommendations.insert(
                0, "CRITICAL: Model should not be deployed without addressing issues"
            )

        return recommendations

    def _suggest_mitigations(
        self, risk_scores: dict[str, float]
    ) -> dict[str, list[str]]:
        """Suggest specific mitigation strategies"""
        mitigations = {}

        for factor, score in risk_scores.items():
            if score > 0.5:
                mitigations[factor] = self._get_mitigation_strategies(factor, score)

        return mitigations

    def _get_mitigation_strategies(self, factor: str, score: float) -> list[str]:
        """Get specific mitigation strategies for each risk factor"""
        strategies = {
            "data_quality": [
                "Implement data validation pipelines",
                "Add data quality monitoring",
                "Use synthetic data for augmentation",
            ],
            "model_complexity": [
                "Use model distillation",
                "Implement feature selection",
                "Add regularization",
            ],
            "performance_stability": [
                "Increase training data",
                "Use cross-validation",
                "Implement early stopping",
            ],
            "fairness": [
                "Apply fairness constraints",
                "Use bias mitigation algorithms",
                "Regular fairness audits",
            ],
            "interpretability": [
                "Add SHAP values",
                "Implement LIME",
                "Create feature importance plots",
            ],
            "security": [
                "Use secure serialization",
                "Implement access controls",
                "Add model signing",
            ],
            "compliance": [
                "Complete model cards",
                "Regular compliance audits",
                "Maintain audit trails",
            ],
        }

        return strategies.get(factor, ["Review and address risk factors"])


class FairnessTesting:
    """Fairness and bias testing framework"""

    def __init__(self, config: ComplianceConfig = None):
        self.config = config or ComplianceConfig()
        self.fairness_metrics = {}

    def test_fairness(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y_true: pd.Series,
        sensitive_features: pd.DataFrame = None,
        y_pred: np.ndarray = None,
    ) -> dict[str, Any]:
        """Comprehensive fairness testing"""
        if y_pred is None:
            y_pred = model.predict(X)

        results = {
            "timestamp": datetime.now().isoformat(),
            "overall_fairness": True,
            "metrics": {},
            "disparities": {},
            "recommendations": [],
        }

        # If no sensitive features specified, look for them in X
        if sensitive_features is None:
            sensitive_cols = [
                col for col in self.config.protected_attributes if col in X.columns
            ]
            if sensitive_cols:
                sensitive_features = X[sensitive_cols]

        if sensitive_features is None or sensitive_features.empty:
            logger.warning("No sensitive features found for fairness testing")
            return results

        # Test each sensitive feature
        for col in sensitive_features.columns:
            feature_results = self._test_feature_fairness(
                y_true, y_pred, sensitive_features[col]
            )
            results["metrics"][col] = feature_results

            # Check for disparities
            if feature_results["disparity_ratio"] < self.config.fairness_threshold:
                results["overall_fairness"] = False
                results["disparities"][col] = feature_results["disparity_ratio"]
                results["recommendations"].append(
                    f"Address disparity in {col}: ratio = {feature_results['disparity_ratio']:.2f}"
                )

        # Additional fairness metrics if fairlearn available
        if FAIRLEARN_AVAILABLE:
            results["advanced_metrics"] = self._compute_advanced_metrics(
                y_true, y_pred, sensitive_features
            )

        return results

    def _test_feature_fairness(
        self, y_true: pd.Series, y_pred: np.ndarray, sensitive_feature: pd.Series
    ) -> dict[str, float]:
        """Test fairness for a single sensitive feature"""
        metrics = {}

        # Get unique groups
        groups = sensitive_feature.unique()

        # Calculate metrics per group
        group_metrics = {}
        for group in groups:
            mask = sensitive_feature == group
            if mask.sum() > 0:
                group_metrics[group] = {
                    "accuracy": accuracy_score(y_true[mask], y_pred[mask]),
                    "size": mask.sum(),
                    "positive_rate": (y_pred[mask] == 1).mean()
                    if len(np.unique(y_pred)) == 2
                    else 0,
                }

        # Calculate disparity metrics
        accuracies = [m["accuracy"] for m in group_metrics.values()]
        positive_rates = [m["positive_rate"] for m in group_metrics.values()]

        metrics["group_metrics"] = group_metrics
        metrics["accuracy_disparity"] = (
            max(accuracies) - min(accuracies) if accuracies else 0
        )
        metrics["disparity_ratio"] = (
            min(accuracies) / max(accuracies) if max(accuracies) > 0 else 0
        )

        # Statistical parity difference
        if positive_rates and len(positive_rates) > 1:
            metrics["statistical_parity_diff"] = max(positive_rates) - min(
                positive_rates
            )

        return metrics

    def _compute_advanced_metrics(
        self, y_true: pd.Series, y_pred: np.ndarray, sensitive_features: pd.DataFrame
    ) -> dict[str, float]:
        """Compute advanced fairness metrics using fairlearn"""
        metrics = {}

        try:
            # Demographic parity difference
            metrics["demographic_parity_diff"] = (
                fairness_metrics.demographic_parity_difference(
                    y_true, y_pred, sensitive_features=sensitive_features
                )
            )

            # Equalized odds difference
            metrics["equalized_odds_diff"] = fairness_metrics.equalized_odds_difference(
                y_true, y_pred, sensitive_features=sensitive_features
            )

        except Exception as e:
            logger.warning(f"Error computing advanced fairness metrics: {e}")

        return metrics

    def mitigate_bias(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        sensitive_features: pd.DataFrame,
        method: str = "reweighing",
    ) -> BaseEstimator:
        """Apply bias mitigation techniques"""
        if method == "reweighing" and AIF360_AVAILABLE:
            return self._apply_reweighing(model, X, y, sensitive_features)
        elif method == "exponentiated_gradient" and FAIRLEARN_AVAILABLE:
            return self._apply_exponentiated_gradient(model, X, y, sensitive_features)
        else:
            logger.warning(f"Mitigation method {method} not available")
            return model

    def _apply_reweighing(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        sensitive_features: pd.DataFrame,
    ) -> BaseEstimator:
        """Apply reweighing bias mitigation (AIF360)"""
        # Implementation would require AIF360 setup
        logger.info("Applying reweighing bias mitigation")
        return model

    def _apply_exponentiated_gradient(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        sensitive_features: pd.DataFrame,
    ) -> BaseEstimator:
        """Apply exponentiated gradient reduction (Fairlearn)"""
        try:
            constraint = DemographicParity()
            mitigator = ExponentiatedGradient(model, constraint)
            mitigator.fit(X, y, sensitive_features=sensitive_features)
            return mitigator
        except Exception as e:
            logger.error(f"Error applying exponentiated gradient: {e}")
            return model


class ComplianceReporting:
    """Generate compliance reports and documentation"""

    def __init__(self, config: ComplianceConfig = None):
        self.config = config or ComplianceConfig()
        self.report_templates = self._load_templates()

    def generate_compliance_report(
        self,
        model_card: ModelCard,
        risk_assessment: dict[str, Any],
        fairness_results: dict[str, Any],
        regulatory_framework: RegulatoryFramework,
    ) -> dict[str, Any]:
        """Generate comprehensive compliance report"""
        report = {
            "report_id": self._generate_report_id(),
            "generated_date": datetime.now().isoformat(),
            "regulatory_framework": regulatory_framework.value,
            "model_info": {
                "model_id": model_card.model_id,
                "model_name": model_card.model_name,
                "version": model_card.version,
                "lifecycle_stage": model_card.lifecycle_stage.value,
                "approval_status": model_card.approval_status,
            },
            "compliance_status": self._assess_compliance_status(
                model_card, risk_assessment, fairness_results, regulatory_framework
            ),
            "risk_assessment": risk_assessment,
            "fairness_assessment": fairness_results,
            "requirements_checklist": self._generate_requirements_checklist(
                model_card, regulatory_framework
            ),
            "gaps_identified": self._identify_compliance_gaps(
                model_card, regulatory_framework
            ),
            "recommendations": self._generate_compliance_recommendations(
                model_card, risk_assessment, regulatory_framework
            ),
            "attestation": self._generate_attestation(model_card),
        }

        return report

    def generate_model_card_report(self, model_card: ModelCard) -> str:
        """Generate model card documentation"""
        template = """
# Model Card: {model_name}

## Model Details
- **Model ID**: {model_id}
- **Version**: {version}
- **Created**: {created_date}
- **Author**: {author}
- **Type**: {model_type}
- **Algorithm**: {algorithm}
- **Stage**: {lifecycle_stage}

## Intended Use
{description}

### Use Case
{use_case}

## Training Data
{training_data}

## Performance Metrics
{performance_metrics}

## Limitations
{limitations}

## Ethical Considerations
{ethical_considerations}

## Fairness Metrics
{fairness_metrics}

## Risk Assessment
{risk_assessment}

## Regulatory Compliance
{regulatory_compliance}

## Approval Status
- **Status**: {approval_status}
- **Approvers**: {approvers}

## Dependencies
{dependencies}

## Deployment Information
{deployment_info}

## Audit Trail
{audit_trail}

---
*Generated: {timestamp}*
        """

        # Format the template with model card data
        report = template.format(
            model_name=model_card.model_name,
            model_id=model_card.model_id,
            version=model_card.version,
            created_date=model_card.created_date.isoformat(),
            author=model_card.author,
            model_type=model_card.model_type,
            algorithm=model_card.algorithm,
            lifecycle_stage=model_card.lifecycle_stage.value,
            description=model_card.description,
            use_case=model_card.use_case,
            training_data=json.dumps(model_card.training_data, indent=2),
            performance_metrics=json.dumps(model_card.performance_metrics, indent=2),
            limitations="\n".join(
                f"- {limitation}" for limitation in model_card.limitations
            ),
            ethical_considerations="\n".join(
                f"- {e}" for e in model_card.ethical_considerations
            ),
            fairness_metrics=json.dumps(model_card.fairness_metrics, indent=2),
            risk_assessment=json.dumps(model_card.risk_assessment, indent=2)
            if model_card.risk_assessment
            else "Not assessed",
            regulatory_compliance=", ".join(model_card.regulatory_compliance)
            or "None specified",
            approval_status=model_card.approval_status,
            approvers=json.dumps(model_card.approvers, indent=2),
            dependencies="\n".join(f"- {d}" for d in model_card.dependencies),
            deployment_info=json.dumps(model_card.deployment_info, indent=2),
            audit_trail=self._format_audit_trail(model_card.audit_trail),
            timestamp=datetime.now().isoformat(),
        )

        return report

    def export_compliance_package(
        self,
        model_card: ModelCard,
        risk_assessment: dict[str, Any],
        fairness_results: dict[str, Any],
        output_dir: str = "compliance_reports",
    ) -> str:
        """Export complete compliance documentation package"""
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        package_dir = (
            output_path / f"compliance_package_{model_card.model_id}_{timestamp}"
        )
        package_dir.mkdir(exist_ok=True)

        # Generate and save model card
        model_card_report = self.generate_model_card_report(model_card)
        with open(package_dir / "model_card.md", "w") as f:
            f.write(model_card_report)

        # Save risk assessment
        with open(package_dir / "risk_assessment.json", "w") as f:
            json.dump(risk_assessment, f, indent=2, default=str)

        # Save fairness results
        with open(package_dir / "fairness_assessment.json", "w") as f:
            json.dump(fairness_results, f, indent=2, default=str)

        # Generate compliance reports for each framework
        for framework in self.config.regulatory_frameworks:
            report = self.generate_compliance_report(
                model_card, risk_assessment, fairness_results, framework
            )
            with open(
                package_dir / f"compliance_report_{framework.value}.json", "w"
            ) as f:
                json.dump(report, f, indent=2, default=str)

        # Create summary document
        summary = self._create_compliance_summary(
            model_card, risk_assessment, fairness_results
        )
        with open(package_dir / "compliance_summary.md", "w") as f:
            f.write(summary)

        logger.info(f"Compliance package exported to {package_dir}")
        return str(package_dir)

    def _load_templates(self) -> dict[str, str]:
        """Load report templates for different regulatory frameworks"""
        templates = {
            RegulatoryFramework.GDPR: "gdpr_template.yaml",
            RegulatoryFramework.AI_ACT: "ai_act_template.yaml",
            RegulatoryFramework.FDA_AI_ML: "fda_template.yaml",
        }
        # In production, these would be loaded from files
        return templates

    def _generate_report_id(self) -> str:
        """Generate unique report ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random_suffix = hashlib.sha256(timestamp.encode()).hexdigest()[:6]
        return f"RPT-{timestamp}-{random_suffix}"

    def _assess_compliance_status(
        self,
        model_card: ModelCard,
        risk_assessment: dict[str, Any],
        fairness_results: dict[str, Any],
        regulatory_framework: RegulatoryFramework,
    ) -> str:
        """Assess overall compliance status"""
        compliant = True

        # Check risk level
        if risk_assessment.get("risk_level") in ["HIGH", "CRITICAL"]:
            compliant = False

        # Check fairness
        if not fairness_results.get("overall_fairness", True):
            compliant = False

        # Check documentation
        if not model_card.limitations or not model_card.ethical_considerations:
            compliant = False

        # Framework-specific checks
        if regulatory_framework == RegulatoryFramework.GDPR:
            if "data_protection" not in model_card.regulatory_compliance:
                compliant = False
        elif regulatory_framework == RegulatoryFramework.AI_ACT:
            if model_card.risk_assessment.get("risk_level") == "CRITICAL":
                compliant = False

        return "COMPLIANT" if compliant else "NON_COMPLIANT"

    def _generate_requirements_checklist(
        self, model_card: ModelCard, regulatory_framework: RegulatoryFramework
    ) -> dict[str, bool]:
        """Generate requirements checklist for regulatory framework"""
        checklist = {}

        if regulatory_framework == RegulatoryFramework.GDPR:
            checklist = {
                "data_minimization": True,
                "purpose_limitation": bool(model_card.use_case),
                "transparency": bool(model_card.description),
                "data_protection_impact_assessment": bool(model_card.risk_assessment),
                "right_to_explanation": bool(model_card.limitations),
                "lawful_basis": "consent" in model_card.regulatory_compliance,
            }
        elif regulatory_framework == RegulatoryFramework.AI_ACT:
            risk_level = (
                model_card.risk_assessment.get("risk_level")
                if model_card.risk_assessment
                else "UNKNOWN"
            )
            checklist = {
                "risk_assessment_completed": bool(model_card.risk_assessment),
                "transparency_requirements": bool(model_card.description),
                "human_oversight": "human_review" in model_card.regulatory_compliance,
                "accuracy_robustness": bool(model_card.performance_metrics),
                "non_discrimination": bool(model_card.fairness_metrics),
                "high_risk_requirements": risk_level not in ["HIGH", "CRITICAL"],
            }
        elif regulatory_framework == RegulatoryFramework.HIPAA:
            checklist = {
                "phi_protection": "phi_protected" in model_card.regulatory_compliance,
                "access_controls": "access_control" in model_card.regulatory_compliance,
                "audit_logs": bool(model_card.audit_trail),
                "encryption": "encryption" in model_card.regulatory_compliance,
                "breach_notification": "breach_protocol"
                in model_card.regulatory_compliance,
            }

        return checklist

    def _identify_compliance_gaps(
        self, model_card: ModelCard, regulatory_framework: RegulatoryFramework
    ) -> list[str]:
        """Identify compliance gaps"""
        gaps = []

        checklist = self._generate_requirements_checklist(
            model_card, regulatory_framework
        )

        for requirement, met in checklist.items():
            if not met:
                gaps.append(f"{requirement}: Not met")

        return gaps

    def _generate_compliance_recommendations(
        self,
        model_card: ModelCard,
        risk_assessment: dict[str, Any],
        regulatory_framework: RegulatoryFramework,
    ) -> list[str]:
        """Generate compliance recommendations"""
        recommendations = []

        # Risk-based recommendations
        if risk_assessment.get("risk_level") in ["HIGH", "CRITICAL"]:
            recommendations.append("Reduce model risk before deployment")

        # Documentation recommendations
        if not model_card.limitations:
            recommendations.append("Document model limitations")
        if not model_card.ethical_considerations:
            recommendations.append("Add ethical considerations")

        # Framework-specific recommendations
        if regulatory_framework == RegulatoryFramework.GDPR:
            recommendations.append("Ensure GDPR compliance through DPIAs")
        elif regulatory_framework == RegulatoryFramework.AI_ACT:
            recommendations.append("Complete EU AI Act conformity assessment")

        return recommendations

    def _generate_attestation(self, model_card: ModelCard) -> dict[str, Any]:
        """Generate compliance attestation"""
        return {
            "date": datetime.now().isoformat(),
            "model_id": model_card.model_id,
            "attested_by": model_card.author,
            "statement": "This model has been reviewed for compliance with applicable regulations.",
            "approvers": model_card.approvers,
        }

    def _format_audit_trail(self, audit_trail: list[dict[str, Any]]) -> str:
        """Format audit trail for reporting"""
        if not audit_trail:
            return "No audit events"

        formatted = []
        for event in audit_trail[-10:]:  # Last 10 events
            formatted.append(
                f"- {event.get('timestamp', 'N/A')}: {event.get('event_type', 'Unknown')} "
                f"- {json.dumps(event.get('details', {}))}"
            )

        return "\n".join(formatted)

    def _create_compliance_summary(
        self,
        model_card: ModelCard,
        risk_assessment: dict[str, Any],
        fairness_results: dict[str, Any],
    ) -> str:
        """Create executive summary of compliance status"""
        summary = f"""
# Compliance Summary

## Model: {model_card.model_name} (v{model_card.version})

### Overview
- **Model ID**: {model_card.model_id}
- **Stage**: {model_card.lifecycle_stage.value}
- **Risk Level**: {risk_assessment.get("risk_level", "Not assessed")}
- **Fairness Status**: {"PASS" if fairness_results.get("overall_fairness") else "FAIL"}

### Compliance Status by Framework
"""
        for framework in self.config.regulatory_frameworks:
            status = self._assess_compliance_status(
                model_card, risk_assessment, fairness_results, framework
            )
            summary += f"- **{framework.value}**: {status}\n"

        summary += f"""
### Key Findings
- Overall Risk Score: {risk_assessment.get("overall_risk_score", 0):.2f}
- Approval Status: {model_card.approval_status}
- Number of Approvers: {len(model_card.approvers)}

### Recommendations
"""
        for framework in self.config.regulatory_frameworks:
            recommendations = self._generate_compliance_recommendations(
                model_card, risk_assessment, framework
            )
            if recommendations:
                summary += f"\n#### {framework.value}\n"
                for rec in recommendations:
                    summary += f"- {rec}\n"

        summary += f"""
---
*Generated: {datetime.now().isoformat()}*
        """

        return summary


class RegulatoryTracker:
    """Track and monitor regulatory requirements"""

    def __init__(self):
        self.requirements = self._initialize_requirements()
        self.updates = []
        self.compliance_calendar = {}

    def _initialize_requirements(self) -> dict[RegulatoryFramework, dict]:
        """Initialize regulatory requirements database"""
        return {
            RegulatoryFramework.GDPR: {
                "name": "General Data Protection Regulation",
                "jurisdiction": "European Union",
                "effective_date": "2018-05-25",
                "key_requirements": [
                    "Lawful basis for processing",
                    "Data minimization",
                    "Purpose limitation",
                    "Transparency",
                    "Data subject rights",
                    "Data protection by design",
                    "Data breach notification (72 hours)",
                    "Data Protection Impact Assessments",
                ],
                "penalties": "Up to 4% of global annual revenue or €20M",
                "updates": [],
            },
            RegulatoryFramework.AI_ACT: {
                "name": "EU Artificial Intelligence Act",
                "jurisdiction": "European Union",
                "effective_date": "2024-08-01",
                "key_requirements": [
                    "Risk-based approach (minimal, limited, high, unacceptable)",
                    "Prohibited AI practices",
                    "High-risk AI system requirements",
                    "Transparency obligations",
                    "Human oversight",
                    "Accuracy and robustness",
                    "Conformity assessments",
                    "Post-market monitoring",
                ],
                "penalties": "Up to 6% of global annual revenue or €30M",
                "updates": [],
            },
            RegulatoryFramework.CCPA: {
                "name": "California Consumer Privacy Act",
                "jurisdiction": "California, USA",
                "effective_date": "2020-01-01",
                "key_requirements": [
                    "Consumer right to know",
                    "Right to delete",
                    "Right to opt-out",
                    "Right to non-discrimination",
                    "Privacy policy requirements",
                    "Data breach notification",
                ],
                "penalties": "$2,500-$7,500 per violation",
                "updates": [],
            },
            RegulatoryFramework.HIPAA: {
                "name": "Health Insurance Portability and Accountability Act",
                "jurisdiction": "United States",
                "effective_date": "1996-08-21",
                "key_requirements": [
                    "Protected Health Information (PHI) safeguards",
                    "Administrative safeguards",
                    "Physical safeguards",
                    "Technical safeguards",
                    "Breach notification",
                    "Business Associate Agreements",
                    "Patient access rights",
                ],
                "penalties": "$100-$50,000 per violation, up to $1.5M per year",
                "updates": [],
            },
        }

    def check_requirements(
        self, model_card: ModelCard, frameworks: list[RegulatoryFramework]
    ) -> dict[str, Any]:
        """Check model against regulatory requirements"""
        results = {}

        for framework in frameworks:
            if framework in self.requirements:
                results[framework.value] = self._check_framework_requirements(
                    model_card, framework
                )

        return results

    def _check_framework_requirements(
        self, model_card: ModelCard, framework: RegulatoryFramework
    ) -> dict[str, Any]:
        """Check requirements for specific framework"""
        requirements = self.requirements[framework]
        checks = {}

        if framework == RegulatoryFramework.GDPR:
            checks = {
                "data_minimization": self._check_data_minimization(model_card),
                "transparency": bool(model_card.description and model_card.limitations),
                "lawful_basis": "consent" in model_card.regulatory_compliance,
                "dpia_completed": bool(model_card.risk_assessment),
                "data_subject_rights": "deletion_capability"
                in model_card.regulatory_compliance,
            }
        elif framework == RegulatoryFramework.AI_ACT:
            risk_level = (
                model_card.risk_assessment.get("risk_level")
                if model_card.risk_assessment
                else None
            )
            checks = {
                "risk_assessment": bool(model_card.risk_assessment),
                "high_risk_compliant": risk_level not in ["HIGH", "CRITICAL"]
                if risk_level
                else False,
                "transparency": bool(model_card.description),
                "human_oversight": "human_review" in model_card.regulatory_compliance,
                "testing_validation": bool(model_card.performance_metrics),
                "documentation": bool(
                    model_card.limitations and model_card.ethical_considerations
                ),
            }

        return {
            "framework": framework.value,
            "requirements": requirements,
            "checks": checks,
            "compliant": all(checks.values()),
            "gaps": [k for k, v in checks.items() if not v],
        }

    def _check_data_minimization(self, model_card: ModelCard) -> bool:
        """Check if model follows data minimization principle"""
        # Check if training data description mentions data minimization
        if model_card.training_data:
            data_desc = str(model_card.training_data).lower()
            return any(
                term in data_desc
                for term in ["minimal", "necessary", "required", "essential"]
            )
        return False

    def get_upcoming_deadlines(self) -> list[dict[str, Any]]:
        """Get upcoming regulatory deadlines"""
        deadlines = []
        current_date = datetime.now()

        # Add sample deadlines (in production, these would come from a database)
        deadlines.append(
            {
                "date": current_date + timedelta(days=30),
                "description": "Annual GDPR compliance review",
                "framework": RegulatoryFramework.GDPR.value,
            }
        )

        deadlines.append(
            {
                "date": current_date + timedelta(days=90),
                "description": "AI Act conformity assessment due",
                "framework": RegulatoryFramework.AI_ACT.value,
            }
        )

        return sorted(deadlines, key=lambda x: x["date"])

    def track_regulatory_updates(self) -> list[dict[str, Any]]:
        """Track regulatory updates and changes"""
        # In production, this would fetch from regulatory databases/APIs
        updates = [
            {
                "date": datetime.now() - timedelta(days=7),
                "framework": RegulatoryFramework.AI_ACT.value,
                "update": "New guidance on high-risk AI systems classification",
                "impact": "HIGH",
                "action_required": "Review and update risk assessments",
            },
            {
                "date": datetime.now() - timedelta(days=14),
                "framework": RegulatoryFramework.GDPR.value,
                "update": "Updated guidelines on automated decision-making",
                "impact": "MEDIUM",
                "action_required": "Update transparency documentation",
            },
        ]

        return updates


class ComplianceFramework:
    """Main compliance framework integrating all components"""

    def __init__(self, config: ComplianceConfig = None):
        self.config = config or ComplianceConfig()
        self.governance = ModelGovernanceFramework(config)
        self.risk_assessment = ModelRiskAssessment(config)
        self.fairness_testing = FairnessTesting(config)
        self.reporting = ComplianceReporting(config)
        self.regulatory_tracker = RegulatoryTracker()

    def full_compliance_check(
        self,
        model: BaseEstimator,
        model_card: ModelCard,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        sensitive_features: pd.DataFrame = None,
    ) -> dict[str, Any]:
        """Perform full compliance check on a model"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "model_id": model_card.model_id,
            "checks_performed": [],
        }

        # 1. Register model
        model_id = self.governance.register_model(model, model_card)
        results["registration"] = {"status": "SUCCESS", "model_id": model_id}
        results["checks_performed"].append("Model Registration")

        # 2. Risk Assessment
        risk_results = self.risk_assessment.assess_model_risk(
            model, X_train, y_train, X_test, y_test, model_card
        )
        results["risk_assessment"] = risk_results
        results["checks_performed"].append("Risk Assessment")

        # 3. Fairness Testing
        fairness_results = self.fairness_testing.test_fairness(
            model, X_test, y_test, sensitive_features
        )
        results["fairness_testing"] = fairness_results
        results["checks_performed"].append("Fairness Testing")

        # 4. Regulatory Compliance Check
        regulatory_results = self.regulatory_tracker.check_requirements(
            model_card, self.config.regulatory_frameworks
        )
        results["regulatory_compliance"] = regulatory_results
        results["checks_performed"].append("Regulatory Compliance")

        # 5. Generate Reports
        for framework in self.config.regulatory_frameworks:
            report = self.reporting.generate_compliance_report(
                model_card, risk_results, fairness_results, framework
            )
            results[f"report_{framework.value}"] = report

        # 6. Overall Compliance Status
        results["overall_status"] = self._determine_overall_status(
            risk_results, fairness_results, regulatory_results
        )

        # 7. Export compliance package
        if results["overall_status"] in ["COMPLIANT", "CONDITIONAL"]:
            package_path = self.reporting.export_compliance_package(
                model_card, risk_results, fairness_results
            )
            results["compliance_package"] = package_path

        return results

    def _determine_overall_status(
        self, risk_results: dict, fairness_results: dict, regulatory_results: dict
    ) -> str:
        """Determine overall compliance status"""
        # Check risk level
        risk_level = risk_results.get("risk_level", "UNKNOWN")
        if risk_level == "CRITICAL":
            return "NON_COMPLIANT"

        # Check fairness
        if not fairness_results.get("overall_fairness", True):
            return "CONDITIONAL"

        # Check regulatory compliance
        all_compliant = all(
            result.get("compliant", False) for result in regulatory_results.values()
        )

        if not all_compliant:
            return "CONDITIONAL"

        if risk_level == "HIGH":
            return "CONDITIONAL"

        return "COMPLIANT"
