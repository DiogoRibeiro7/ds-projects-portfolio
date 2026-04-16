import pandas as pd

from projects.advanced_customer_segmentation.pipeline import clustering


def test_select_best_clustering():
    df = pd.DataFrame({
        'age': [25, 30, 22, 45, 52],
        'income': [50000, 60000, 52000, 90000, 120000],
        'spending_score': [60, 70, 77, 40, 20],
        'loyalty_score': [0.8, 0.9, 0.9, 0.4, 0.2],
        'gender': ['Male', 'Female', 'Female', 'Male', 'Female'],
        'region': ['North', 'South', 'East', 'West', 'North']
    })
    preprocessor = clustering.build_feature_pipeline()
    X = preprocessor.fit_transform(df)
    model, best_k, best_score = clustering.select_best_clustering(X, cluster_range=(2, 3))
    assert best_k in [2, 3]
    assert best_score > 0
    assert best_k in [2, 3]
    assert best_score > 0
