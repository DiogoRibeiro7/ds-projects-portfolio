"""Advanced Streamlit dashboard for customer segmentation with explainability."""
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import shap
import numpy as np
import os
from .cohort_analysis import cohort_summary, cluster_profile
from ..pipeline.cluster_labeling import label_clusters, cluster_summary_stats
from ..pipeline.anomaly_detection import detect_outliers

st.set_page_config(page_title="Advanced Customer Segmentation", layout="wide")
st.title("👥 Advanced Customer Segmentation Dashboard")
# Interactive filtering
st.sidebar.header("Interactive Filtering")
gender_options = df["gender"].unique().tolist() if "gender" in df.columns else []
region_options = df["region"].unique().tolist() if "region" in df.columns else []
loyalty_min, loyalty_max = (df["loyalty_score"].min(), df["loyalty_score"].max()) if "loyalty_score" in df.columns else (0, 1)

selected_genders = st.sidebar.multiselect("Gender", gender_options, default=gender_options)
selected_regions = st.sidebar.multiselect("Region", region_options, default=region_options)
loyalty_range = st.sidebar.slider("Loyalty Score Range", float(loyalty_min), float(loyalty_max), (float(loyalty_min), float(loyalty_max)))

filtered_df = df.copy()
if "gender" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["gender"].isin(selected_genders)]
if "region" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["region"].isin(selected_regions)]
if "loyalty_score" in filtered_df.columns:
    filtered_df = filtered_df[(filtered_df["loyalty_score"] >= loyalty_range[0]) & (filtered_df["loyalty_score"] <= loyalty_range[1])]


# Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "..", "data")
segmented_csv = os.path.join(data_dir, "customers_segmented.csv")
model_path = os.path.join(data_dir, "cluster_model.joblib")
shap_path = os.path.join(data_dir, "shap_values.npy")


def process_external_data(uploaded_file):
    import tempfile
    import pandas as pd
    from ..pipeline.feature_engineering import build_feature_pipeline
    from ..pipeline.clustering import select_best_clustering
    from ..pipeline.cluster_labeling import label_clusters
    from ..pipeline.anomaly_detection import detect_outliers
    # Read uploaded CSV
    df = pd.read_csv(uploaded_file)
    # Feature engineering
    preprocessor = build_feature_pipeline()
    X = preprocessor.fit_transform(df)
    # Clustering
    model, best_k, best_score = select_best_clustering(X)
    df['cluster'] = model.predict(X)
    df = label_clusters(df)
    df = detect_outliers(df)
    return df

@st.cache_data
def load_data():
    return pd.read_csv(segmented_csv)

@st.cache_resource
def load_model():
    return joblib.load(model_path)

@st.cache_data
def load_shap():
    return np.load(shap_path, allow_pickle=True)


# External data upload
uploaded_file = st.sidebar.file_uploader("Upload your own customer CSV", type=["csv"])
if uploaded_file is not None:
    df = process_external_data(uploaded_file)
    st.success("External data loaded and processed!")
else:
    df = load_data()
    df = label_clusters(df)
    df = detect_outliers(df)
model_bundle = load_model()
shap_values = load_shap()

st.sidebar.header("Cluster Selection")
clusters = sorted(df["cluster"].unique())
selected_cluster = st.sidebar.selectbox("Select Cluster", clusters)

st.subheader(f"Cluster {selected_cluster} Overview")
st.dataframe(df[df["cluster"] == selected_cluster])
if st.checkbox("Highlight anomalies (outliers)"):
    st.dataframe(df[df["anomaly"] == 1])

st.write(f"Cluster label: {df[df['cluster'] == selected_cluster]['cluster_label'].iloc[0]}")

# Cohort analysis and profiling
# Cohort analysis and profiling
st.subheader("Cohort Analysis & Segment Profiling")
cohort_col = st.selectbox("Select cohort/segment column", [col for col in df.columns if col not in ["customer_id", "cluster"]])
st.write(f"Summary statistics by {cohort_col}:")
st.dataframe(cohort_summary(df, cohort_col))

st.write("Cluster profiling:")
st.dataframe(cluster_profile(df))

st.subheader("Cluster Summary Statistics (with Labels)")
st.dataframe(cluster_summary_stats(df))

st.subheader("Cluster Distribution")
fig = px.histogram(df, x="cluster", color="cluster", barmode="group")
st.plotly_chart(fig, use_container_width=True)

# Outlier distribution visualization
st.subheader("Outlier/Anomaly Distribution")
fig_outlier = px.histogram(df, x="anomaly", color="cluster", barmode="group", title="Anomaly Distribution by Cluster", labels={"anomaly": "Anomaly (1=Outlier)"})
st.plotly_chart(fig_outlier, use_container_width=True)


st.subheader("Feature Importance (SHAP)")
shap.summary_plot(shap_values, features=filtered_df.drop(["customer_id", "cluster"], axis=1, errors="ignore"), show=False)
st.pyplot(bbox_inches="tight")

# SHAP force plot for individual customers
st.subheader("SHAP Force Plot for Individual Customer")
if "customer_id" in filtered_df.columns:
    customer_ids = filtered_df["customer_id"].tolist()
    selected_customer = st.selectbox("Select Customer ID", customer_ids)
    idx = filtered_df.index[filtered_df["customer_id"] == selected_customer][0]
    st.write(f"Force plot for Customer ID: {selected_customer}")
    shap.initjs()
    import streamlit.components.v1 as components
    force_plot_html = shap.force_plot(
        shap_values.base_values[idx] if hasattr(shap_values, 'base_values') else shap_values[0][idx],
        shap_values[idx] if hasattr(shap_values, 'base_values') else shap_values[0][idx],
        filtered_df.drop(["customer_id", "cluster"], axis=1, errors="ignore").iloc[idx],
        matplotlib=False,
        show=False
    )
    components.html(shap.getjs() + force_plot_html.html(), height=300)

st.subheader("Download Segmented Data & Reports")
st.download_button("Download CSV", filtered_df.to_csv(index=False), file_name="customers_segmented.csv")

# Excel download
import io
excel_buffer = io.BytesIO()
with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
    filtered_df.to_excel(writer, index=False, sheet_name="Segmented Data")
    # Optionally add summary sheets
    cluster_stats = cluster_profile(filtered_df)
    cluster_stats.to_excel(writer, sheet_name="Cluster Profile")
st.download_button(
    label="Download Excel",
    data=excel_buffer.getvalue(),
    file_name="customers_segmented.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# PDF download (simple table report)
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
def create_pdf(dataframe):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    style = getSampleStyleSheet()
    title = Paragraph("Customer Segmentation Report", style["Title"])
    elements.append(title)
    # Table data
    data = [dataframe.columns.tolist()] + dataframe.head(30).values.tolist()
    t = Table(data)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
    ]))
    elements.append(t)
    doc.build(elements)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf
st.download_button(
    label="Download PDF",
    data=create_pdf(filtered_df),
    file_name="customers_segmented.pdf",
    mime="application/pdf"
)
