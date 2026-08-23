# API Overview

This section maps the maintained code surfaces that support the portfolio. It is
kept intentionally curated: source modules and project packages remain the
canonical implementation, while the docs highlight the parts a reviewer is most
likely to inspect.

```{toctree}
:maxdepth: 2
:caption: Maintained Guides

ml_api_usage
```

## Core Library Areas

- `src/statistics/`: experiment analysis, robust statistics, quantile models,
  and mixture-of-experts helpers.
- `src/data_processing/`: cleaning, benchmark, and optimization examples.
- `src/feature_engineering/`: reusable transformers and feature generation
  utilities.
- `src/genai/`: retrieval, chunking, prompt, evaluation, and guardrail helpers
  for GenAI portfolio notebooks.
- `src/time_series/`: anomaly and time-series utility functions.

## Portfolio Project Packages

- `projects/porto_lisbon_uhi_exposure/src/uhi_exposure/`: urban heat island
  exposure pipeline with tests and plotting utilities.
- `projects/portugal_gdp_bayesian_revision/src/pt_gdp_bayes/`: Bayesian GDP and
  population reconciliation workflow.
- `projects/pt_salary_gamma_distribution/src/pt_salary_gamma_distribution/`:
  salary-distribution extraction, fitting, plotting, and pipeline code.
- `projects/advanced_customer_segmentation/`: clustering, labeling, anomaly
  detection, and dashboard components.
- `projects/statistical_methods/`: advanced statistical tests, Bayesian A/B
  testing, causal inference, bandits, and validation suites.

## Service Example

The maintained API-facing example is the FastAPI inference service in
`src/api/ml_api.py`. See the [ML API usage guide](ml_api_usage.md) for local
usage commands and sample requests.
