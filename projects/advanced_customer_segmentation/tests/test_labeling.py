import pandas as pd
from projects.advanced_customer_segmentation.pipeline import cluster_labeling

def test_label_clusters():
    df = pd.DataFrame({
        'customer_id': [1, 2, 3],
        'age': [25, 30, 22],
        'income': [50000, 60000, 52000],
        'spending_score': [60, 70, 77],
        'loyalty_score': [0.8, 0.9, 0.9],
        'gender': ['Male', 'Female', 'Female'],
        'region': ['North', 'South', 'East'],
        'cluster': [0, 1, 0]
    })
    df_labeled = cluster_labeling.label_clusters(df)
    assert 'cluster_label' in df_labeled.columns
    assert df_labeled['cluster_label'].notnull().all()
