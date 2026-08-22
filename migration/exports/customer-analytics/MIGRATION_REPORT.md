# Migration Report: customer-analytics

## Source

Copied from `ds-projects-portfolio` on branch `migration/copy-first-exports`.

## Policy

This is a non-destructive export. Original source files remain in the portfolio repository.

## Export Contents

- `src/modern_bank_churn`
- `projects/churn_prediction`
- `projects/advanced_customer_segmentation`
- `projects/customer_segmentation`
- `projects/machine_learning/customer_segmentation`
- `projects/causal_inference/campaign_diff_in_diff`
- `notebooks/customer_retention_uplift_modeling.ipynb`
- focused tests and packaging

## Export-Local Adaptations

- Added focused `pyproject.toml`, `.gitignore`, CI, and README.
- Added export-local pytest fixtures so `modern_bank_churn` tests can run without the full monorepo test stack.
- Trimmed copied smoke tests to the customer analytics examples included in this snapshot.
- Fixed the advanced segmentation anomaly detector default feature-column selection in the export copy.

## Validation

Run from this directory:

```bash
python -m pip install -e ".[test]"
pytest
python -c "import nbformat; nbformat.read('notebooks/customer_retention_uplift_modeling.ipynb', as_version=4)"
```

Validated during export creation:

- `python -m pip install -e ".[test,notebook]"`
- `pytest`: 51 passed
- notebook JSON validation: 1 valid notebook
