"""
Data Quality Dashboard
Interactive dashboard for monitoring data quality, profiling, and lineage
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import sqlite3

# Import our quality modules
from src.data_quality.quality_framework import (
    DataQualityFramework, ValidationStatus, DataQualityLevel
)
from src.data_profiling.profiling_tools import (
    DataProfiler, DataLineageTracker, DataVersionManager, DataCatalog
)
from src.data_preprocessing.preprocessing_pipelines import (
    PreprocessingPipeline, PreprocessingConfig
)

# Configure Streamlit
st.set_page_config(
    page_title="Data Quality Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .success-metric {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .warning-metric {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    .error-metric {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
    div[data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)


class DataQualityDashboard:
    """Main dashboard application"""

    def __init__(self):
        self.quality_framework = DataQualityFramework()
        self.profiler = DataProfiler()
        self.lineage_tracker = DataLineageTracker()
        self.version_manager = DataVersionManager()
        self.catalog = DataCatalog()

        # Initialize session state
        if 'current_dataset' not in st.session_state:
            st.session_state.current_dataset = None
        if 'quality_reports' not in st.session_state:
            st.session_state.quality_reports = []
        if 'profiles' not in st.session_state:
            st.session_state.profiles = []

    def run(self):
        """Run the dashboard application"""
        st.title("🔍 Data Quality Management Dashboard")
        st.markdown("### Comprehensive Data Quality Monitoring & Analysis Platform")

        # Sidebar navigation
        with st.sidebar:
            st.header("Navigation")
            page = st.selectbox(
                "Select Module",
                ["Overview", "Data Profiling", "Quality Validation",
                 "Data Lineage", "Preprocessing", "Data Catalog", "Real-time Monitor"]
            )

            # Dataset selection
            st.header("Dataset")
            uploaded_file = st.file_uploader(
                "Upload Dataset",
                type=['csv', 'xlsx', 'parquet', 'json']
            )

            if uploaded_file:
                self.load_dataset(uploaded_file)

        # Main content area
        if page == "Overview":
            self.show_overview()
        elif page == "Data Profiling":
            self.show_profiling()
        elif page == "Quality Validation":
            self.show_validation()
        elif page == "Data Lineage":
            self.show_lineage()
        elif page == "Preprocessing":
            self.show_preprocessing()
        elif page == "Data Catalog":
            self.show_catalog()
        elif page == "Real-time Monitor":
            self.show_realtime_monitor()

    def load_dataset(self, uploaded_file):
        """Load dataset from uploaded file"""
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.parquet'):
                df = pd.read_parquet(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)
            else:
                st.error("Unsupported file format")
                return

            st.session_state.current_dataset = df
            st.session_state.dataset_name = uploaded_file.name
            st.success(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")

    def show_overview(self):
        """Display overview dashboard"""
        st.header("📊 Data Quality Overview")

        if st.session_state.current_dataset is None:
            st.warning("Please upload a dataset to begin")
            return

        df = st.session_state.current_dataset

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Records", f"{len(df):,}")

        with col2:
            st.metric("Total Features", f"{df.shape[1]}")

        with col3:
            missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
            st.metric("Missing Data", f"{missing_pct:.1f}%",
                     delta=f"{missing_pct - 5:.1f}%" if missing_pct > 5 else "Good")

        with col4:
            # Calculate quality score
            quality_score = 100 - missing_pct
            st.metric("Quality Score", f"{quality_score:.1f}%",
                     delta="Good" if quality_score > 80 else "Needs Improvement")

        # Data type distribution
        st.subheader("Data Type Distribution")
        type_dist = df.dtypes.value_counts()

        fig = px.pie(
            values=type_dist.values,
            names=type_dist.index.astype(str),
            title="Column Data Types"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Missing data heatmap
        st.subheader("Missing Data Analysis")
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)

        if len(missing_data) > 0:
            fig = px.bar(
                x=missing_data.values,
                y=missing_data.index,
                orientation='h',
                title="Missing Values by Column",
                labels={'x': 'Missing Count', 'y': 'Column'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No missing data detected!")

        # Statistical summary
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)

        # Correlation matrix for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            st.subheader("Feature Correlations")
            corr_matrix = df[numeric_cols].corr()

            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title="Correlation Matrix"
            )
            st.plotly_chart(fig, use_container_width=True)

    def show_profiling(self):
        """Display data profiling interface"""
        st.header("🔬 Data Profiling")

        if st.session_state.current_dataset is None:
            st.warning("Please upload a dataset first")
            return

        df = st.session_state.current_dataset

        # Profiling options
        col1, col2, col3 = st.columns(3)

        with col1:
            profile_type = st.selectbox(
                "Profile Type",
                ["Quick Profile", "Detailed Profile", "Column Analysis", "Distribution Analysis"]
            )

        with col2:
            generate_report = st.button("Generate Profile Report", type="primary")

        with col3:
            export_format = st.selectbox("Export Format", ["HTML", "JSON", "PDF"])

        if generate_report:
            with st.spinner("Generating profile..."):
                # Generate profile
                profile = self.profiler.profile_dataset(
                    df,
                    st.session_state.dataset_name,
                    detailed=(profile_type == "Detailed Profile")
                )
                st.session_state.profiles.append(profile)

                # Display profile results
                st.success("Profile generated successfully!")

                # Profile metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Shape", f"{profile.shape[0]} × {profile.shape[1]}")

                with col2:
                    st.metric("Memory Usage", f"{profile.memory_usage / 1024 / 1024:.2f} MB")

                with col3:
                    unique_avg = np.mean(list(profile.unique_values.values()))
                    st.metric("Avg Unique Values", f"{unique_avg:.0f}")

                with col4:
                    outlier_total = sum(profile.outliers.values())
                    st.metric("Total Outliers", f"{outlier_total:,}")

                # Column-wise profiling
                if profile_type in ["Column Analysis", "Detailed Profile"]:
                    st.subheader("Column Profiles")

                    selected_col = st.selectbox("Select Column", df.columns)

                    if selected_col:
                        col_data = df[selected_col]
                        col_stats = profile.statistics.get(selected_col, {})

                        # Display column statistics
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.write("**Basic Statistics**")
                            st.write(f"- Type: {profile.dtypes[selected_col]}")
                            st.write(f"- Missing: {profile.missing_values[selected_col] * 100:.1f}%")
                            st.write(f"- Unique: {profile.unique_values[selected_col]}")
                            st.write(f"- Outliers: {profile.outliers.get(selected_col, 0)}")

                        with col2:
                            if 'mean' in col_stats:
                                st.write("**Numeric Statistics**")
                                st.write(f"- Mean: {col_stats['mean']:.2f}")
                                st.write(f"- Std: {col_stats['std']:.2f}")
                                st.write(f"- Min: {col_stats['min']:.2f}")
                                st.write(f"- Max: {col_stats['max']:.2f}")

                        with col3:
                            if 'top' in col_stats:
                                st.write("**Categorical Statistics**")
                                st.write(f"- Top Value: {col_stats['top']}")
                                st.write(f"- Frequency: {col_stats['freq']}")

                        # Distribution plot
                        if pd.api.types.is_numeric_dtype(col_data):
                            fig = px.histogram(
                                col_data,
                                nbins=30,
                                title=f"Distribution of {selected_col}"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            value_counts = col_data.value_counts().head(20)
                            fig = px.bar(
                                x=value_counts.index,
                                y=value_counts.values,
                                title=f"Top Values in {selected_col}"
                            )
                            st.plotly_chart(fig, use_container_width=True)

    def show_validation(self):
        """Display quality validation interface"""
        st.header("✅ Quality Validation")

        if st.session_state.current_dataset is None:
            st.warning("Please upload a dataset first")
            return

        df = st.session_state.current_dataset

        # Validation configuration
        st.subheader("Validation Rules")

        col1, col2 = st.columns(2)

        with col1:
            completeness_threshold = st.slider(
                "Completeness Threshold (%)",
                min_value=0, max_value=100, value=95
            )

            check_outliers = st.checkbox("Check for Outliers", value=True)
            outlier_method = st.selectbox(
                "Outlier Detection Method",
                ["IQR", "Z-Score", "Isolation Forest"],
                disabled=not check_outliers
            )

        with col2:
            check_duplicates = st.checkbox("Check for Duplicates", value=True)
            check_correlations = st.checkbox("Check High Correlations", value=True)
            correlation_threshold = st.slider(
                "Correlation Threshold",
                min_value=0.0, max_value=1.0, value=0.95,
                disabled=not check_correlations
            )

        # Run validation
        if st.button("Run Validation", type="primary"):
            with st.spinner("Running validation checks..."):
                # Run validation
                report = self.quality_framework.run_validation(
                    df, dataset_name=st.session_state.dataset_name
                )
                st.session_state.quality_reports.append(report)

                # Display results
                st.success(f"Quality Score: {report.quality_score:.1f}%")

                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Checks", report.summary['total_checks'])

                with col2:
                    st.metric("Passed", report.summary['passed'],
                             delta=f"{report.summary['passed'] / report.summary['total_checks'] * 100:.0f}%")

                with col3:
                    st.metric("Failed", report.summary['failed'])

                with col4:
                    st.metric("Warnings", report.summary['warnings'])

                # Detailed results
                st.subheader("Validation Results")

                # Create results DataFrame
                results_data = []
                for result in report.validation_results:
                    results_data.append({
                        'Check': result.rule_name,
                        'Status': result.status.value,
                        'Severity': result.severity.value,
                        'Message': result.message
                    })

                results_df = pd.DataFrame(results_data)

                # Add status colors
                def highlight_status(row):
                    if row['Status'] == 'passed':
                        return ['background-color: #d4edda'] * len(row)
                    elif row['Status'] == 'failed':
                        return ['background-color: #f8d7da'] * len(row)
                    elif row['Status'] == 'warning':
                        return ['background-color: #fff3cd'] * len(row)
                    else:
                        return [''] * len(row)

                styled_df = results_df.style.apply(highlight_status, axis=1)
                st.dataframe(styled_df, use_container_width=True)

                # Recommendations
                if report.recommendations:
                    st.subheader("Recommendations")
                    for rec in report.recommendations:
                        st.info(f"💡 {rec}")

                # Export report
                st.subheader("Export Report")
                col1, col2, col3 = st.columns(3)

                with col1:
                    if st.button("Export as HTML"):
                        html_report = self.quality_framework.export_report(report, "html")
                        st.download_button(
                            "Download HTML Report",
                            html_report,
                            file_name="quality_report.html",
                            mime="text/html"
                        )

                with col2:
                    if st.button("Export as JSON"):
                        json_report = self.quality_framework.export_report(report, "json")
                        st.download_button(
                            "Download JSON Report",
                            json_report,
                            file_name="quality_report.json",
                            mime="application/json"
                        )

                with col3:
                    if st.button("Export as Markdown"):
                        md_report = self.quality_framework.export_report(report, "markdown")
                        st.download_button(
                            "Download Markdown Report",
                            md_report,
                            file_name="quality_report.md",
                            mime="text/markdown"
                        )

    def show_lineage(self):
        """Display data lineage visualization"""
        st.header("🔗 Data Lineage")

        # Lineage tracking
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Track Transformation")

            source_datasets = st.multiselect(
                "Source Datasets",
                ["raw_data", "external_api", "database_table", "file_upload"]
            )

            transformation_type = st.selectbox(
                "Transformation Type",
                ["Join", "Filter", "Aggregate", "Clean", "Transform", "Merge"]
            )

            output_name = st.text_input("Output Dataset Name")

        with col2:
            st.subheader("Transformation Details")

            operations = st.multiselect(
                "Operations Applied",
                ["Remove Duplicates", "Handle Missing", "Scale Features",
                 "Encode Categories", "Filter Rows", "Join Tables"]
            )

            created_by = st.text_input("Created By", value="Data Team")

        if st.button("Track Lineage", type="primary"):
            if source_datasets and output_name:
                # Track lineage
                lineage = self.lineage_tracker.track_transformation(
                    input_datasets=source_datasets,
                    output_dataset=output_name,
                    transformation={
                        'type': transformation_type,
                        'operations': operations
                    },
                    created_by=created_by
                )

                st.success(f"Lineage tracked: {lineage.lineage_id}")

        # Visualize lineage
        st.subheader("Lineage Visualization")

        # Create sample lineage graph
        fig = go.Figure()

        # Sample nodes and edges
        nodes = ["raw_data", "cleaned_data", "features", "model_input", "predictions"]
        edges = [
            ("raw_data", "cleaned_data"),
            ("cleaned_data", "features"),
            ("features", "model_input"),
            ("model_input", "predictions")
        ]

        # Add nodes
        for i, node in enumerate(nodes):
            fig.add_trace(go.Scatter(
                x=[i],
                y=[0],
                mode='markers+text',
                marker=dict(size=30, color='lightblue', line=dict(width=2)),
                text=[node],
                textposition="bottom center"
            ))

        # Add edges
        for edge in edges:
            start_idx = nodes.index(edge[0])
            end_idx = nodes.index(edge[1])
            fig.add_trace(go.Scatter(
                x=[start_idx, end_idx],
                y=[0, 0],
                mode='lines',
                line=dict(width=2, color='gray'),
                showlegend=False
            ))

        fig.update_layout(
            title="Data Lineage Graph",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=300
        )

        st.plotly_chart(fig, use_container_width=True)

    def show_preprocessing(self):
        """Display preprocessing pipeline interface"""
        st.header("🔧 Data Preprocessing Pipeline")

        if st.session_state.current_dataset is None:
            st.warning("Please upload a dataset first")
            return

        df = st.session_state.current_dataset

        # Preprocessing configuration
        st.subheader("Pipeline Configuration")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**Outlier Detection**")
            outlier_method = st.selectbox(
                "Method",
                ["isolation_forest", "iqr", "zscore", "lof", "dbscan"]
            )
            outlier_threshold = st.slider(
                "Contamination",
                min_value=0.01, max_value=0.2, value=0.05
            )

        with col2:
            st.write("**Missing Data**")
            imputation_strategy = st.selectbox(
                "Strategy",
                ["knn", "simple", "iterative", "matrix", "forward_fill"]
            )

        with col3:
            st.write("**Scaling**")
            scaling_method = st.selectbox(
                "Method",
                ["standard", "minmax", "robust", "quantile", "power"]
            )

        # Additional options
        col1, col2 = st.columns(2)

        with col1:
            feature_engineering = st.checkbox("Enable Feature Engineering", value=True)
            handle_imbalance = st.checkbox("Handle Class Imbalance", value=False)

        with col2:
            if handle_imbalance:
                imbalance_strategy = st.selectbox(
                    "Balancing Strategy",
                    ["smote", "adasyn", "random_under", "tomek"]
                )
            else:
                imbalance_strategy = "smote"

        # Run preprocessing
        if st.button("Run Preprocessing Pipeline", type="primary"):
            with st.spinner("Running preprocessing pipeline..."):
                # Create config
                config = PreprocessingConfig(
                    outlier_method=outlier_method,
                    outlier_threshold=outlier_threshold,
                    imputation_strategy=imputation_strategy,
                    scaling_method=scaling_method,
                    feature_engineering=feature_engineering,
                    handle_imbalance=handle_imbalance,
                    imbalance_strategy=imbalance_strategy
                )

                # Create pipeline
                pipeline = PreprocessingPipeline(config)

                # Separate features and target if target column exists
                if 'target' in df.columns:
                    X = df.drop('target', axis=1)
                    y = df['target']
                else:
                    X = df
                    y = None

                # Run preprocessing
                X_processed, y_processed = pipeline.fit_transform(X, y)

                # Display results
                st.success("Preprocessing completed!")

                # Show before/after comparison
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Original Data**")
                    st.write(f"- Shape: {X.shape}")
                    st.write(f"- Missing: {X.isnull().sum().sum()}")
                    st.write(f"- Memory: {X.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

                with col2:
                    st.write("**Processed Data**")
                    st.write(f"- Shape: {X_processed.shape}")
                    st.write(f"- Missing: {X_processed.isnull().sum().sum()}")
                    st.write(f"- Memory: {X_processed.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

                # Pipeline summary
                st.subheader("Pipeline Steps")
                summary = pipeline.get_pipeline_summary()
                for step in summary['steps']:
                    st.write(f"- {step[0]}: {step[1]}")

                # Download processed data
                st.subheader("Export Processed Data")
                csv = X_processed.to_csv(index=False)
                st.download_button(
                    "Download Processed Data",
                    csv,
                    file_name="processed_data.csv",
                    mime="text/csv"
                )

    def show_catalog(self):
        """Display data catalog interface"""
        st.header("📚 Data Catalog")

        # Register dataset
        st.subheader("Register Dataset")

        col1, col2 = st.columns(2)

        with col1:
            dataset_name = st.text_input("Dataset Name")
            dataset_desc = st.text_area("Description")
            owner = st.text_input("Owner", value="Data Team")

        with col2:
            tags = st.multiselect(
                "Tags",
                ["production", "staging", "test", "ml", "analytics", "raw", "processed"]
            )

        if st.button("Register Dataset", type="primary"):
            if dataset_name and dataset_desc:
                self.catalog.register_dataset(
                    dataset_id=f"ds_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    name=dataset_name,
                    description=dataset_desc,
                    owner=owner,
                    tags=tags
                )
                st.success(f"Dataset '{dataset_name}' registered successfully!")

        # Search catalog
        st.subheader("Search Catalog")

        search_query = st.text_input("Search datasets...")
        search_tags = st.multiselect("Filter by tags", ["production", "ml", "analytics"])

        if st.button("Search"):
            results = self.catalog.search_datasets(query=search_query, tags=search_tags)

            if results:
                st.write(f"Found {len(results)} datasets")

                # Display results
                for result in results:
                    with st.expander(f"📁 {result['name']}"):
                        st.write(f"**Description:** {result['description']}")
                        st.write(f"**Owner:** {result['owner']}")
                        st.write(f"**Created:** {result['created_date']}")
                        st.write(f"**Tags:** {', '.join(result['tags'])}")
                        st.write(f"**Quality Score:** {result['quality_score']:.1f}%")
                        st.write(f"**Usage Count:** {result['usage_count']}")
            else:
                st.info("No datasets found matching your criteria")

    def show_realtime_monitor(self):
        """Display real-time monitoring dashboard"""
        st.header("📡 Real-time Data Quality Monitor")

        # Create placeholder for real-time updates
        placeholder = st.empty()

        # Simulate real-time monitoring
        while True:
            with placeholder.container():
                # Current timestamp
                st.write(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                # Real-time metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    incoming_rate = np.random.randint(100, 1000)
                    st.metric("Records/sec", incoming_rate,
                             delta=f"{np.random.randint(-50, 50)}")

                with col2:
                    quality_score = np.random.uniform(85, 99)
                    st.metric("Quality Score", f"{quality_score:.1f}%",
                             delta=f"{np.random.uniform(-2, 2):.1f}%")

                with col3:
                    anomalies = np.random.randint(0, 10)
                    st.metric("Anomalies Detected", anomalies,
                             delta=f"{np.random.randint(-3, 3)}")

                with col4:
                    processing_time = np.random.uniform(10, 50)
                    st.metric("Avg Processing (ms)", f"{processing_time:.1f}",
                             delta=f"{np.random.uniform(-5, 5):.1f}")

                # Real-time charts
                st.subheader("Live Metrics")

                # Generate sample time series data
                time_points = pd.date_range(
                    end=datetime.now(),
                    periods=50,
                    freq='1min'
                )

                quality_scores = np.random.uniform(85, 99, 50)
                record_counts = np.random.randint(100, 1000, 50)

                # Create subplots
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=("Data Quality Score", "Incoming Records Rate")
                )

                # Quality score line
                fig.add_trace(
                    go.Scatter(x=time_points, y=quality_scores, mode='lines+markers',
                             name='Quality Score', line=dict(color='green')),
                    row=1, col=1
                )

                # Record count bars
                fig.add_trace(
                    go.Bar(x=time_points, y=record_counts, name='Records',
                          marker=dict(color='blue')),
                    row=2, col=1
                )

                fig.update_layout(height=600, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

                # Alert section
                st.subheader("Active Alerts")

                alerts = [
                    {"level": "warning", "message": "Missing data rate increased by 5%", "time": "2 mins ago"},
                    {"level": "info", "message": "New data source connected", "time": "5 mins ago"},
                    {"level": "error", "message": "Schema validation failed for column 'amount'", "time": "10 mins ago"}
                ]

                for alert in alerts:
                    if alert['level'] == 'error':
                        st.error(f"🔴 {alert['message']} ({alert['time']})")
                    elif alert['level'] == 'warning':
                        st.warning(f"🟡 {alert['message']} ({alert['time']})")
                    else:
                        st.info(f"🔵 {alert['message']} ({alert['time']})")

                # Break the loop for demo (remove in production)
                break


def main():
    """Main entry point"""
    dashboard = DataQualityDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()