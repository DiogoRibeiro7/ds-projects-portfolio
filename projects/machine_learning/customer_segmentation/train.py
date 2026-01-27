import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def make_synthetic_customers(n_samples: int = 1200, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    segments = [
        # high value, low frequency
        {
            "mean": [900, 2.0, 0.2, 10],
            "std": [120, 0.5, 0.1, 2],
            "n": int(n_samples * 0.2),
        },
        # mid value, high frequency
        {
            "mean": [300, 8.0, 0.6, 25],
            "std": [80, 1.5, 0.1, 4],
            "n": int(n_samples * 0.5),
        },
        # low value, low frequency, high churn risk
        {
            "mean": [120, 1.5, 0.8, 5],
            "std": [40, 0.5, 0.15, 2],
            "n": int(n_samples * 0.3),
        },
    ]

    rows = []
    for segment in segments:
        data = rng.normal(segment["mean"], segment["std"], size=(segment["n"], 4))
        rows.append(data)

    data = np.vstack(rows)
    df = pd.DataFrame(
        data,
        columns=["avg_order_value", "orders_per_month", "promo_usage", "site_visits"],
    )

    df["orders_per_month"] = df["orders_per_month"].clip(lower=0.5)
    df["promo_usage"] = df["promo_usage"].clip(0, 1)
    df["site_visits"] = df["site_visits"].clip(lower=1)

    return df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def cluster_customers(df: pd.DataFrame, n_clusters: int = 3) -> tuple[pd.DataFrame, float]:
    scaler = StandardScaler()
    features = scaler.fit_transform(df)

    model = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    labels = model.fit_predict(features)

    df_clustered = df.copy()
    df_clustered["cluster"] = labels

    return df_clustered, float(model.inertia_)


def summarize_clusters(df_clustered: pd.DataFrame) -> pd.DataFrame:
    summary = df_clustered.groupby("cluster").agg(
        size=("cluster", "size"),
        avg_order_value=("avg_order_value", "mean"),
        orders_per_month=("orders_per_month", "mean"),
        promo_usage=("promo_usage", "mean"),
        site_visits=("site_visits", "mean"),
    )
    return summary.sort_values("size", ascending=False)


def main() -> None:
    df = make_synthetic_customers()
    df_clustered, inertia = cluster_customers(df, n_clusters=3)
    summary = summarize_clusters(df_clustered)

    print("Cluster Summary")
    print(summary.round(2))
    print("\nInertia:", round(inertia, 2))


if __name__ == "__main__":
    main()
