"""Customer Segmentation ML Pipeline

Performs K-Means clustering on customer data and saves results for dashboard visualization.
"""
import pandas as pd
from sklearn.cluster import KMeans


def load_data(path: str) -> pd.DataFrame:
    """Load customer data from CSV."""
    return pd.read_csv(path)


def run_clustering(df: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    """Run K-Means clustering and return DataFrame with cluster labels."""
    features = df.select_dtypes(include=["float", "int"]).values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["cluster"] = kmeans.fit_predict(features)
    return df


def main():
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "customers_sample.csv")
    output_path = os.path.join(base_dir, "clustered_customers.csv")
    df = load_data(data_path)
    df_clustered = run_clustering(df)
    df_clustered.to_csv(output_path, index=False)
    print(f"Clustering complete. Results saved to {output_path}.")

if __name__ == "__main__":
    main()
