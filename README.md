# Data Science Projects Portfolio

![Fast CI](https://github.com/diogoribeiro7/ds-projects-portfolio/actions/workflows/ci.yml/badge.svg)

A curated data science portfolio with notebooks, reusable Python modules,
reproducible demos, and project writeups for applied analytics, experimentation,
machine learning, and AI engineering.

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

3. Run the repository checks:

   ```bash
   make check
   ```

4. Open a featured project or notebook for a quick review:

   - [`projects/adaptive_policy_learning/`](projects/adaptive_policy_learning/)
   - [`notebooks/healthcare_analysis.ipynb`](notebooks/healthcare_analysis.ipynb)
   - [`notebooks/finance_credit_risk_stress_testing.ipynb`](notebooks/finance_credit_risk_stress_testing.ipynb)
   - [`examples/eda_end_to_end.ipynb`](examples/eda_end_to_end.ipynb)

## What this portfolio demonstrates

This repository is organized around work that is useful to inspect in a hiring
or project-review context:

- Applied modeling: healthcare explainability, credit-risk stress testing,
  forecasting, churn, segmentation, and fraud examples.
- Experimentation and statistics: prospectively locked contextual-bandit studies,
  off-policy evaluation, independent-policy calibration, overlap diagnostics,
  A/B testing, power analysis, Bayesian methods, causal inference, and validation checks.
- Decision science: probabilistic forecasting linked to constrained optimization
  and evaluation through realised operational cost.
- Operational data science: model monitoring examples, reproducible demos,
  API-oriented workflows, provenance-aware execution, and clear testing conventions.
- AI engineering: RAG, retrieval evaluation, telemetry, and guardrail patterns
  that can be reviewed offline.

## Featured portfolio set

Use this short list first when reviewing:

- [`projects/adaptive_policy_learning/`](projects/adaptive_policy_learning/) (contextual bandits, OPE, prospective study design)
- [`notebooks/healthcare_analysis.ipynb`](notebooks/healthcare_analysis.ipynb) (healthcare)
- [`notebooks/life_science_medical_imaging_triage.ipynb`](notebooks/life_science_medical_imaging_triage.ipynb) (medical imaging)
- [`notebooks/insurance_data_science.ipynb`](notebooks/insurance_data_science.ipynb) (insurance governance)
- [`notebooks/finance_credit_risk_stress_testing.ipynb`](notebooks/finance_credit_risk_stress_testing.ipynb) (finance stress testing)
- [`notebooks/genai_rag_pipeline.ipynb`](notebooks/genai_rag_pipeline.ipynb) (RAG engineering)
- [`notebooks/energy_load_probabilistic_forecasting.ipynb`](notebooks/energy_load_probabilistic_forecasting.ipynb) (time-series forecasting)
- [`notebooks/mlops_mlflow_drift_lifecycle.ipynb`](notebooks/mlops_mlflow_drift_lifecycle.ipynb) (MLOps lifecycle)

## Featured completed study: adaptive policy learning

[`projects/adaptive_policy_learning/`](projects/adaptive_policy_learning/) is a completed
three-study contextual-bandit programme using the ZOZO Research Open Bandit Dataset.
It is designed around prospective protocol locks, training-only qualification gates,
independently logged Random-policy calibration, IPS/SNIPS/DM/DR estimation, overlap
diagnostics, moving-block bootstrap uncertainty, and a conservative deployment rule.

The scientific record deliberately retains negative and inconvenient results:

- **Study 1** terminated before OPE because its frozen SAGA reward model failed to
  converge under both prospectively allowed iteration budgets.
- **Study 2** produced a positive challenger DR-minus-BTS estimate, but the frozen
  promotion rule rejected deployment because the challenger ESS fraction was only
  about `0.20%`, far below the required `10%`.
- **Study 3** used a separately frozen `women` campaign and deterministic numerical
  execution. The challenger DR estimate was `0.0044247405` versus observed BTS
  `0.0062400025`; the paired 95% interval for DR-minus-BTS was
  `[-0.0049919409, -0.0009922200]`, and the challenger ESS fraction was only
  `0.0002268`. The terminal decision was **`do_not_promote`**.

The point of the project is not to manufacture a winning policy. It demonstrates how
prospective design, reproducibility controls, independent calibration, uncertainty,
and support diagnostics can prevent unsupported deployment claims. The full empirical
record is in [`projects/adaptive_policy_learning/RESULTS.md`](projects/adaptive_policy_learning/RESULTS.md).

## Project index

Use this map when you want to jump directly to a project area:

- [Adaptive policy learning and off-policy evaluation](projects/adaptive_policy_learning/)
- [Mobility demand and fleet optimization](projects/mobility_demand_optimization/)
- [Portugal housing, tourism and short-term accommodation](projects/portugal_housing_tourism/)
- [churn prediction](projects/churn_prediction/)
- [customer segmentation](projects/customer_segmentation/)
- [A/B testing notebooks](projects/ab_testing/)
- [statistical methods examples](projects/statistical_methods/)
- [feature engineering utilities](projects/feature_engineering/)
- [Streamlit app examples](projects/streamlit_apps/)
- [dashboard components](projects/dashboard_enhanced/)
- [deep learning notebooks](projects/deep_learning/)
- [NLP examples](projects/nlp/)
- [performance optimization examples](projects/performance_optimization/)

The complete active notebook catalog remains in [`notebooks/README.md`](notebooks/README.md).

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
├── notebooks/            # Portfolio notebooks
├── projects/             # Project writeups and reference implementations
├── examples/             # Runnable portfolio demos
├── tests/                # Automated test suite
└── scripts/              # Repo tooling
```

Repository scope contract is documented in
[`docs/PORTFOLIO_SCOPE.md`](docs/PORTFOLIO_SCOPE.md).

## Read next

- Portfolio visitors: [`docs/index.md`](docs/index.md) and
  [`docs/portfolio_overview.md`](docs/portfolio_overview.md)
- Contributors: [`docs/internal.md`](docs/internal.md),
  [`docs/contributor/development.md`](docs/contributor/development.md)
- If you spot an issue: open it on GitHub

This project is meant to stay portfolio-first: fast signal for reviewers, concise
entry paths, and clear separation between maintained portfolio content and
external historical backlog.
