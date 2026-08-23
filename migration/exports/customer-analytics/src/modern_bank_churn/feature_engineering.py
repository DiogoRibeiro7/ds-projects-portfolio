from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif


class FeatureEngineer:
    """Minimal feature engineering API kept for test/runtime compatibility."""

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in out.select_dtypes(include=[np.number]).columns:
            if out[col].isna().all():
                out[col] = 0.0
        numeric = out.select_dtypes(include=[np.number])
        for col in numeric.columns:
            out[f"{col}_squared"] = numeric[col] ** 2
        return out

    def create_interaction_features(
        self, df: pd.DataFrame, cols: list[str]
    ) -> pd.DataFrame:
        out = df.copy()
        for i, left in enumerate(cols):
            for right in cols[i + 1 :]:
                out[f"{left}_x_{right}"] = out[left] * out[right]
        return out

    def create_polynomial_features(
        self, df: pd.DataFrame, degree: int = 2
    ) -> pd.DataFrame:
        out = df.copy()
        if degree < 2:
            return out
        numeric = out.select_dtypes(include=[np.number])
        for col in numeric.columns:
            for p in range(2, degree + 1):
                out[f"{col}^{p}"] = numeric[col] ** p
        return out

    def create_time_features(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        out = df.copy()
        dt = pd.to_datetime(out[date_col], errors="coerce")
        out[f"{date_col}_year"] = dt.dt.year
        out[f"{date_col}_month"] = dt.dt.month
        out[f"{date_col}_day"] = dt.dt.day
        out[f"{date_col}_dayofweek"] = dt.dt.dayofweek
        return out

    def select_features(
        self, X: pd.DataFrame, y: pd.Series, method: str = "mutual_info", k: int = 10
    ) -> list[str]:
        if X.empty:
            return []
        k = max(1, min(k, X.shape[1]))

        X_num = X.select_dtypes(include=[np.number]).copy()
        if X_num.empty:
            return list(X.columns[:k])

        if method == "mutual_info":
            scores = mutual_info_classif(X_num, y, random_state=42)
        elif method == "chi2":
            shifted = X_num.copy()
            mins = shifted.min(axis=0)
            for col in shifted.columns:
                if mins[col] < 0:
                    shifted[col] = shifted[col] - mins[col]
            scores, _ = chi2(shifted, y)
        elif method == "f_classif":
            scores, _ = f_classif(X_num, y)
        else:
            scores = mutual_info_classif(X_num, y, random_state=42)

        order = np.argsort(np.nan_to_num(scores))[::-1]
        return [X_num.columns[i] for i in order[:k]]

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in out.columns:
            if pd.api.types.is_numeric_dtype(out[col]):
                median = out[col].median()
                fill = 0.0 if pd.isna(median) else median
                out[col] = out[col].fillna(fill)
            else:
                mode = out[col].mode(dropna=True)
                fill = mode.iloc[0] if not mode.empty else ""
                out[col] = out[col].fillna(fill)
        return out
