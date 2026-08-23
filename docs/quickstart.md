# Quickstart

This guide gets a reviewer or contributor from a fresh clone to the maintained
portfolio checks.

## Prerequisites

- Python 3.11 or 3.12
- Git
- Make, if you want to use the bundled repository targets

## Clone And Install

```bash
git clone https://github.com/DiogoRibeiro7/ds-projects-portfolio.git
cd ds-projects-portfolio
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

Install the default dependencies:

```bash
pip install -r requirements.txt
```

For contributor checks:

```bash
pip install -r requirements-dev.txt
```

## Run The Portfolio Demo

```bash
python examples/run_demo.py
```

This is the fastest executable check that the repository imports and core demo
path are healthy.

## Run Validation

```bash
make check
```

For documentation-only work:

```bash
cd docs
make html
```

## Review Featured Work

Start with the featured notebooks and catalogue:

- [`projects.yml`](https://github.com/DiogoRibeiro7/ds-projects-portfolio/blob/main/projects.yml)
- [`notebooks/healthcare_analysis.ipynb`](https://github.com/DiogoRibeiro7/ds-projects-portfolio/blob/main/notebooks/healthcare_analysis.ipynb)
- [`notebooks/finance_credit_risk_stress_testing.ipynb`](https://github.com/DiogoRibeiro7/ds-projects-portfolio/blob/main/notebooks/finance_credit_risk_stress_testing.ipynb)
- [`notebooks/mlops_mlflow_drift_lifecycle.ipynb`](https://github.com/DiogoRibeiro7/ds-projects-portfolio/blob/main/notebooks/mlops_mlflow_drift_lifecycle.ipynb)
- [`examples/eda_end_to_end.ipynb`](https://github.com/DiogoRibeiro7/ds-projects-portfolio/blob/main/examples/eda_end_to_end.ipynb)

## Next Steps

- Use [`usage.md`](usage.md) for copy-pasteable module examples.
- Use [`architecture.md`](architecture.md) to understand the repository model.
- Use [`PORTFOLIO_SCOPE.md`](PORTFOLIO_SCOPE.md) before adding, moving, or
  archiving portfolio material.
