import importlib.util
from pathlib import Path


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_customer_segmentation_smoke():
    path = Path("projects/machine_learning/customer_segmentation/train.py")
    module = _load_module("customer_segmentation_train", path)

    df = module.make_synthetic_customers(n_samples=300, random_state=7)
    df_clustered, inertia = module.cluster_customers(df, n_clusters=3)
    summary = module.summarize_clusters(df_clustered)

    assert df_clustered["cluster"].nunique() == 3
    assert summary["size"].sum() == len(df_clustered)
    assert inertia > 0


def test_credit_risk_modeling_smoke():
    path = Path("projects/machine_learning/credit_risk_modeling/train.py")
    module = _load_module("credit_risk_modeling_train", path)

    df, target = module.make_synthetic_credit_data(n_samples=1000, random_state=11)
    metrics = module.train_model(df, target)

    for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
        assert 0 <= metrics[key] <= 1

    assert metrics["confusion_matrix"].shape == (2, 2)


def test_recommendation_system_smoke():
    path = Path("projects/machine_learning/recommendation_system/train.py")
    module = _load_module("recommendation_system_train", path)

    interactions = module.make_interactions(n_users=40, n_items=30, random_state=13)
    train_interactions, test_interactions = module.train_test_split_interactions(
        interactions, random_state=13
    )
    hit_rate = module.evaluate_hit_rate(train_interactions, test_interactions, k=5)

    assert 0 <= hit_rate <= 1

    user_item, similarity = module.train_item_similarity(train_interactions)
    example_user = int(user_item.index[0])
    recs = module.recommend_items(user_item, similarity, example_user, k=5)
    assert len(recs) == 5


def test_sales_forecasting_smoke():
    path = Path("projects/time_series/sales_forecasting/train.py")
    module = _load_module("sales_forecasting_train", path)

    series = module.make_synthetic_sales(n_days=120, random_state=17)
    train, test = module.train_test_split(series, test_days=14)
    naive_forecast = module.forecast_naive(train, horizon=len(test))
    arima_forecast = module.forecast_arima(train, horizon=len(test))

    naive_metrics = module.calculate_metrics(test, naive_forecast)
    arima_metrics = module.calculate_metrics(test, arima_forecast)

    assert naive_metrics["mae"] > 0
    assert arima_metrics["mae"] > 0


def test_diff_in_diff_smoke():
    path = Path("projects/causal_inference/campaign_diff_in_diff/train.py")
    module = _load_module("diff_in_diff_train", path)

    df = module.make_panel_data(
        n_units=80, n_periods=6, treatment_effect=4.0, random_state=19
    )
    results = module.diff_in_diff(df)

    assert "effect" in results
    assert "p_value" in results
