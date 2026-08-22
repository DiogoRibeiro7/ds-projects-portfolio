"""Feature engineering for advanced customer segmentation."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = ["age", "income", "spending_score", "loyalty_score"]
CATEGORICAL_FEATURES = ["gender", "region"]


def build_feature_pipeline():
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop="first", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    pipeline = build_feature_pipeline()
    features = pipeline.fit_transform(df)
    feature_names = (
        list(pipeline.named_transformers_["num"].get_feature_names_out(NUMERIC_FEATURES)) +
        list(pipeline.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES))
    )
    return pd.DataFrame(features, columns=feature_names)
