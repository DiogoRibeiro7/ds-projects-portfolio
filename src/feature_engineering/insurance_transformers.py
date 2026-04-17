"""Insurance-specific sklearn-compatible feature transformers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class InsuranceFeatureBuilder(BaseEstimator, TransformerMixin):
    """Build deterministic insurance pricing features for tabular policy data."""

    def __init__(self) -> None:
        self.numeric_cols = ["age", "bmi", "children"]
        self.categorical_cols = ["sex", "smoker", "region"]

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "InsuranceFeatureBuilder":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        data = X.copy()
        data["age_bmi"] = data["age"] * data["bmi"] / 100
        data["bmi_squared"] = data["bmi"] ** 2
        data["smoker_age"] = np.where(data["smoker"] == "yes", data["age"], 0)
        data["children_rate"] = data["children"] / (data["age"] + 1)
        data["underage"] = (data["age"] < 25).astype(int)
        return data
