import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def make_synthetic_credit_data(
    n_samples: int = 5000, random_state: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=12,
        n_informative=7,
        n_redundant=2,
        n_clusters_per_class=2,
        weights=[0.82, 0.18],
        flip_y=0.02,
        class_sep=1.2,
        random_state=random_state,
    )

    columns = [
        "income",
        "debt_to_income",
        "credit_utilization",
        "open_accounts",
        "delinq_30d",
        "delinq_90d",
        "age",
        "employment_years",
        "loan_amount",
        "loan_term",
        "recent_inquiries",
        "savings_balance",
    ]

    df = pd.DataFrame(X, columns=columns)
    target = pd.Series(y, name="default")
    return df, target


def train_model(df: pd.DataFrame, target: pd.Series) -> dict:
    X_train, X_val, y_train, y_val = train_test_split(
        df,
        target,
        test_size=0.2,
        stratify=target,
        random_state=42,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_val_scaled)
    y_scores = model.predict_proba(X_val_scaled)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred, zero_division=0),
        "recall": recall_score(y_val, y_pred, zero_division=0),
        "f1": f1_score(y_val, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_val, y_scores),
        "pr_auc": average_precision_score(y_val, y_scores),
        "confusion_matrix": confusion_matrix(y_val, y_pred),
    }

    return metrics


def main() -> None:
    df, target = make_synthetic_credit_data()
    metrics = train_model(df, target)

    print("Credit Risk Model Metrics")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1: {metrics['f1']:.3f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
    print(f"PR-AUC: {metrics['pr_auc']:.3f}")
    print("Confusion Matrix:\n", metrics["confusion_matrix"])


if __name__ == "__main__":
    main()
