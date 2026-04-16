import pandas as pd
from projects.advanced_customer_segmentation.pipeline import anomaly_detection

def test_detect_outliers():
    df = pd.DataFrame({
        'customer_id': [1, 2, 3, 4, 5],
        'age': [25, 30, 22, 45, 52],
        'income': [50000, 60000, 52000, 90000, 120000],
        'spending_score': [60, 70, 77, 40, 20],
        'loyalty_score': [0.8, 0.9, 0.9, 0.4, 0.2],
        'gender': ['Male', 'Female', 'Female', 'Male', 'Female'],
        'region': ['North', 'South', 'East', 'West', 'North'],
        'cluster': [0, 1, 0, 1, 1],
        'cluster_label': ['age_high', 'income_high', 'spending_score_high', 'income_high', 'income_high']
    })
    df_out = anomaly_detection.detect_outliers(df)
    assert 'anomaly' in df_out.columns
    assert set(df_out['anomaly'].unique()).issubset({0, 1})
