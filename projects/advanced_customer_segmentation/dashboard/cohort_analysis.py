"""Cohort analysis and segment profiling utilities."""
import pandas as pd


def cohort_summary(df: pd.DataFrame, cohort_col: str, metrics: list = None):
    """Return summary statistics for each cohort/segment."""
    if metrics is None:
        metrics = [col for col in df.columns if df[col].dtype in ['float64', 'int64'] and col not in ['customer_id', 'cluster']]
    summary = df.groupby(cohort_col)[metrics].agg(['mean', 'std', 'min', 'max', 'count'])
    return summary

def cluster_profile(df: pd.DataFrame):
    """Return descriptive statistics for each cluster."""
    metrics = [col for col in df.columns if df[col].dtype in ['float64', 'int64'] and col not in ['customer_id', 'cluster']]
    return df.groupby('cluster')[metrics].agg(['mean', 'std', 'min', 'max', 'count'])
    return df.groupby('cluster')[metrics].agg(['mean', 'std', 'min', 'max', 'count'])
