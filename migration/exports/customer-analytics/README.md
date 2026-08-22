# Customer Analytics

Copy-first export of customer analytics material from `ds-projects-portfolio`.

This snapshot keeps the original portfolio files in place and packages the customer/churn components so they can be validated or promoted as a separate repository.

## Included

- `src/modern_bank_churn`: churn feature engineering, model orchestration, evaluation, production wrapper, and lightweight MLOps helpers.
- `projects/churn_prediction`: Telco churn notebooks and project README.
- `projects/advanced_customer_segmentation`: segmentation pipeline, dashboard code, and local tests.
- `projects/customer_segmentation`: simple segmentation pipeline/dashboard demo.
- `projects/machine_learning/customer_segmentation`: synthetic customer clustering smoke example.
- `projects/causal_inference/campaign_diff_in_diff`: campaign treatment-effect smoke example.
- `notebooks/customer_retention_uplift_modeling.ipynb`: customer retention/uplift notebook.

## Validate

```bash
python -m pip install -e ".[test]"
pytest
python -c "import nbformat; nbformat.read('notebooks/customer_retention_uplift_modeling.ipynb', as_version=4)"
```

Install `.[dashboard]` if you want to run Streamlit/SHAP dashboard components.
