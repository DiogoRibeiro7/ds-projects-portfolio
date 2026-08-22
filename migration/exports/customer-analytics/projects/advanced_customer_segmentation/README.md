# Advanced Customer Segmentation Project

This project demonstrates a professional, modular pipeline for customer segmentation using advanced feature engineering, model selection, and explainability. It includes an interactive Streamlit dashboard for data exploration and model interpretation.

## Features
- Data validation and feature engineering (scaling, encoding)
- Model selection (KMeans with silhouette scoring)
- Model explainability (SHAP values)
- Automated artifact saving (model, metrics, plots)
- Streamlit dashboard with:
  - Data exploration
  - Cluster analysis
  - SHAP feature importance
  - Downloadable segmented data

## Usage

1. **Run the clustering pipeline:**
   ```bash
   python -m projects.advanced_customer_segmentation.pipeline.clustering
   ```
2. **Launch the dashboard:**
   ```bash
   streamlit run projects/advanced_customer_segmentation/dashboard/app.py
   ```

## Requirements
- pandas
- scikit-learn
- joblib
- shap
- streamlit
- plotly

Install requirements with:
```bash
pip install pandas scikit-learn joblib shap streamlit plotly
```

## Testing
Test scaffolding is provided in `tests/` for pipeline components.
