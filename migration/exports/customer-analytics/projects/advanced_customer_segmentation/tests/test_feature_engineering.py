import pandas as pd

from projects.advanced_customer_segmentation.pipeline import feature_engineering


def test_transform_features():
    df = pd.DataFrame({
        'age': [25, 30],
        'income': [50000, 60000],
        'spending_score': [60, 70],
        'loyalty_score': [0.8, 0.9],
        'gender': ['Male', 'Female'],
        'region': ['North', 'South']
    })
    features = feature_engineering.transform_features(df)
    assert features.shape[0] == 2
    assert features.shape[1] > 0
    assert features.shape[0] == 2
    assert features.shape[1] > 0
