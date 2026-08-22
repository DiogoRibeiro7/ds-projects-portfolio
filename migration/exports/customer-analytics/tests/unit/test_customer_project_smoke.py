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


def test_diff_in_diff_smoke():
    path = Path("projects/causal_inference/campaign_diff_in_diff/train.py")
    module = _load_module("diff_in_diff_train", path)

    df = module.make_panel_data(
        n_units=80, n_periods=6, treatment_effect=4.0, random_state=19
    )
    results = module.diff_in_diff(df)

    assert "effect" in results
    assert "p_value" in results
