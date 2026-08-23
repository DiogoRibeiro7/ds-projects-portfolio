# Data Science Projects Portfolio

![Fast CI](https://github.com/diogoribeiro7/ds-projects-portfolio/actions/workflows/ci.yml/badge.svg)

A structured monorepo for portfolio-grade data science work: reusable modeling
code, notebooks, reproducible demos, and production-oriented references.

## Start here (first 2 minutes)

1. Clone and bootstrap:

   ```bash
   git clone https://github.com/diogoribeiro7/ds-projects-portfolio.git
   cd ds-projects-portfolio
   python -m venv .venv
   .venv\Scripts\Activate.ps1  # Windows
   # or: source .venv/bin/activate  # macOS/Linux
   pip install -r requirements.txt
   ```

   Optional contributor stack:

   ```bash
   pip install -r requirements-dev.txt   # linting, type-checking, pre-commit, dev tooling
   ```

2. Run the portfolio demo:

   ```bash
   python examples/run_demo.py
   ```

3. Validate the repository check (optional but recommended):

   ```bash
   make check
   ```

4. Open a featured notebook to evaluate portfolio quality quickly:

   - [`notebooks/healthcare_analysis.ipynb`](notebooks/healthcare_analysis.ipynb)
   - [`notebooks/finance_credit_risk_stress_testing.ipynb`](notebooks/finance_credit_risk_stress_testing.ipynb)
   - [`examples/eda_end_to_end.ipynb`](examples/eda_end_to_end.ipynb)

## What this portfolio demonstrates

This repository is centered on practical projects where technical quality is
paired with clear storytelling:

- Modeling reliability: healthcare explainability, credit-risk modeling, forecasting, and fraud detection.
- Deployment-readiness: FastAPI workflows, MLOps drift management, and notebook-to-service transitions.
- AI engineering: RAG, retrieval/eval service patterns, and controlled guardrails.
- Decision quality: calibration, subgroup fairness, benchmarking, and governance-oriented reporting.

## Featured portfolio set

Use this short list first when reviewing:

- [`notebooks/healthcare_analysis.ipynb`](notebooks/healthcare_analysis.ipynb) (healthcare)  
- [`notebooks/life_science_medical_imaging_triage.ipynb`](notebooks/life_science_medical_imaging_triage.ipynb) (medical imaging)
- [`notebooks/insurance_data_science.ipynb`](notebooks/insurance_data_science.ipynb) (insurance governance)
- [`notebooks/finance_credit_risk_stress_testing.ipynb`](notebooks/finance_credit_risk_stress_testing.ipynb) (finance stress testing)
- [`notebooks/genai_rag_pipeline.ipynb`](notebooks/genai_rag_pipeline.ipynb) (GenAI production pipeline)
- [`notebooks/energy_load_probabilistic_forecasting.ipynb`](notebooks/energy_load_probabilistic_forecasting.ipynb) (time-series forecasting)
- [`notebooks/mlops_mlflow_drift_lifecycle.ipynb`](notebooks/mlops_mlflow_drift_lifecycle.ipynb) (MLOps lifecycle)

## Project index

Use this map when you want to jump directly to a project area:

- [churn_prediction](projects/churn_prediction/)
- [customer segmentation](projects/customer_segmentation/)
- [ab_testing](projects/ab_testing/)
- [statistical_methods](projects/statistical_methods/)
- [streamlit_apps](projects/streamlit_apps/)
- [dashboard_enhanced](projects/dashboard_enhanced/)
- [archive legacy MLOps notebooks](archive/legacy/projects/mlops/)
- [deep_learning](projects/deep_learning/)
- [nlp](projects/nlp/)
- [performance_optimization](projects/performance_optimization/)

The complete notebook catalog (including archive/reference notebooks) remains in
[`notebooks/README.md`](notebooks/README.md).

The machine-readable featured-project catalogue is in [`projects.yml`](projects.yml).
The portfolio structure is documented in [`docs/architecture.md`](docs/architecture.md).

## Standalone repositories

Selected projects are also available as focused standalone repositories:

- [experimentation-toolkit](https://github.com/DiogoRibeiro7/experimentation-toolkit)
- [genai-rag-engineering](https://github.com/DiogoRibeiro7/genai-rag-engineering)
- [portugal-gdp-bayesian-revision](https://github.com/DiogoRibeiro7/portugal-gdp-bayesian-revision)
- [porto-lisbon-uhi-exposure](https://github.com/DiogoRibeiro7/porto-lisbon-uhi-exposure)
- [city-wage-cost-global](https://github.com/DiogoRibeiro7/city-wage-cost-global)
- [pt-salary-gamma-distribution](https://github.com/DiogoRibeiro7/pt-salary-gamma-distribution)
- [customer-analytics](https://github.com/DiogoRibeiro7/customer-analytics)

## End-to-end analytics examples

- [`examples/eda_end_to_end.ipynb`](examples/eda_end_to_end.ipynb) for a complete
  exploratory analysis walkthrough.

## Repository layout

```text
.
├── src/                  # Core reusable packages
├── docs/                 # Portfolio documentation and API reference
├── notebooks/            # Portfolio notebooks (featured + archive)
├── projects/             # Project writeups and reference implementations
├── examples/             # Runnable portfolio demos
├── tests/                # Automated test suite
├── scripts/              # Repo tooling
├── deployment/           # Portfolio-facing deployment entry points
└── archive/              # Historical artifacts and legacy references
```

Repository scope contract is documented in
[`docs/PORTFOLIO_SCOPE.md`](docs/PORTFOLIO_SCOPE.md).

## Read next

- Portfolio visitors: [`docs/index.md`](docs/index.md) and
  [`docs/README_ENHANCED.md`](docs/README_ENHANCED.md)
- Contributors: [`docs/internal.md`](docs/internal.md),
  [`docs/contributor/development.md`](docs/contributor/development.md)
- If you spot an issue: open it on GitHub

This project is meant to stay portfolio-first: fast signal for reviewers, concise
entry paths, and clear separation between maintained portfolio content and archive
backlog.

