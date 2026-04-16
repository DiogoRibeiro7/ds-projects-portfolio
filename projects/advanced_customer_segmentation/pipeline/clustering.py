"""Advanced clustering pipeline with model selection and explainability."""
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import joblib
import shap
import numpy as np

from .feature_engineering import build_feature_pipeline


def select_best_clustering(X, cluster_range=(2, 6)):
    best_score = -1
    best_model = None
    best_k = None
    for k in range(cluster_range[0], cluster_range[1] + 1):
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels)
        if score > best_score:
            best_score = score
            best_model = model
            best_k = k
    return best_model, best_k, best_score


def run_pipeline(input_csv, output_csv, model_path, shap_path):
    df = pd.read_csv(input_csv)
    preprocessor = build_feature_pipeline()
    X = preprocessor.fit_transform(df)
    model, best_k, best_score = select_best_clustering(X)
    labels = model.predict(X)
    df['cluster'] = labels
    df.to_csv(output_csv, index=False)
    joblib.dump({'model': model, 'preprocessor': preprocessor}, model_path)
    # SHAP explainability (KernelExplainer for clustering)
    explainer = shap.KernelExplainer(model.predict, X)
    shap_values = explainer.shap_values(X[:100])  # Limit for speed
    np.save(shap_path, shap_values)
    return best_k, best_score

if __name__ == "__main__":
    best_k, best_score = run_pipeline(
        input_csv="../data/customers_sample_advanced.csv",
        output_csv="../data/customers_segmented.csv",
        model_path="../data/cluster_model.joblib",
        shap_path="../data/shap_values.npy"
    )
    print(f"Best K: {best_k}, Silhouette Score: {best_score:.3f}")
