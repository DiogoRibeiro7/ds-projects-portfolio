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

The complete notebook catalog (including archive/reference notebooks) remains in
[`notebooks/README.md`](notebooks/README.md).

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
├── deployment/           # Deployment examples and manifests
└── archive/              # Historical artifacts and legacy references
```

Repository scoping is documented in [`docs/contributor/REPO_STRUCTURE.md`](docs/contributor/REPO_STRUCTURE.md).

## Read next

- Portfolio visitors: [`docs/index.md`](docs/index.md) and
  [`docs/README_ENHANCED.md`](docs/README_ENHANCED.md)
- Contributors: [`docs/internal.md`](docs/internal.md),
  [`docs/contributor/development.md`](docs/contributor/development.md)
- If you spot an issue: open it on GitHub

This project is meant to stay portfolio-first: fast signal for reviewers, concise
entry paths, and clear separation between maintained portfolio content and archive
backlog.
