"""Automated cluster labeling and summary statistics."""
import numpy as np
import pandas as pd


def label_clusters(df: pd.DataFrame, cluster_col: str = "cluster") -> pd.DataFrame:
    """Assigns descriptive labels to clusters based on mean feature values."""
    ignored = {"customer_id", cluster_col}
    feature_cols = [
        col
        for col in df.columns
        if col not in ignored and pd.api.types.is_numeric_dtype(df[col])
    ]
    cluster_means = df.groupby(cluster_col)[feature_cols].mean()
    labels = []
    for idx, row in cluster_means.iterrows():
        dominant_feature = row.abs().idxmax()
        label = f"{dominant_feature}_high" if row[dominant_feature] > row[feature_cols].mean() else f"{dominant_feature}_low"
        labels.append(label)
    label_map = {k: v for k, v in zip(cluster_means.index, labels)}
    df["cluster_label"] = df[cluster_col].map(label_map)
    return df

def cluster_summary_stats(df: pd.DataFrame, cluster_col: str = "cluster") -> pd.DataFrame:
    """Return summary statistics for each cluster with labels."""
    ignored = {"customer_id", cluster_col, "cluster_label"}
    feature_cols = [
        col
        for col in df.columns
        if col not in ignored and pd.api.types.is_numeric_dtype(df[col])
    ]
    summary = df.groupby([cluster_col, "cluster_label"])[feature_cols].agg(['mean', 'std', 'min', 'max', 'count'])
    return summary.reset_index()
