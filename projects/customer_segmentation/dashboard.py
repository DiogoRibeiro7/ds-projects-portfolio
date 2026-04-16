"""Streamlit Dashboard for Customer Segmentation"""
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")
st.title("🧑‍🤝‍🧑 Customer Segmentation Dashboard")


import os
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "clustered_customers.csv")
    return pd.read_csv(data_path)

df = load_data()

st.sidebar.header("Cluster Selection")
clusters = df["cluster"].unique()
selected_cluster = st.sidebar.selectbox("Select Cluster", clusters)

st.subheader(f"Cluster {selected_cluster} Overview")
st.dataframe(df[df["cluster"] == selected_cluster])

fig = px.scatter(df, x=df.columns[1], y=df.columns[2], color="cluster", title="Cluster Visualization")
st.plotly_chart(fig, use_container_width=True)
