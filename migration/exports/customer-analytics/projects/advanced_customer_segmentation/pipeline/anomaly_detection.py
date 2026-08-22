"""Outlier and anomaly detection utilities for customer segmentation."""
import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_outliers(df: pd.DataFrame, feature_cols=None, contamination=0.1, random_state=42):
    """Detect outliers using Isolation Forest. Returns a DataFrame with an 'anomaly' column."""
    if feature_cols is None:
        ignored = {"customer_id", "cluster", "cluster_label"}
        feature_cols = [
            col
            for col in df.columns
            if col not in ignored and pd.api.types.is_numeric_dtype(df[col])
        ]
    iso = IsolationForest(contamination=contamination, random_state=random_state)
    preds = iso.fit_predict(df[feature_cols])
    df_out = df.copy()
    df_out["anomaly"] = (preds == -1).astype(int)
    return df_out
