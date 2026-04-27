# Data Science Projects Portfolio

![Fast CI](https://github.com/diogoribeiro7/ds-projects-portfolio/actions/workflows/ci.yml/badge.svg)

A structured monorepo for experimentation-heavy data science work: reusable analytics code, production-ready APIs, notebooks, dashboards, and project references.

Repository boundaries are enforced with an explicit structure contract in `docs/REPO_STRUCTURE.md`.

## What is included

- `src/` — core Python packages for APIs, data processing, feature engineering, modeling, and statistical utilities.
- `projects/` — portfolio projects, demos, and legacy reference implementations.
- `docs/` — authoring, architecture notes, API docs, and team workflows.
- `examples/` — executable demos and end-to-end notebooks.
- `notebooks/` — analysis notebooks for healthcare, experimentation, and model exploration.
- `scripts/` — utility scripts for tests, notebook maintenance, and repository workflows.

## Quick start

1. Create a Python environment

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
# or source .venv/bin/activate  # macOS/Linux
```

2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

3. Run the quality gate

```bash
make check
```

4. Run a demo

```bash
python examples/run_demo.py
```

## Recommended workflow

- Use `make lint` for style and static analysis
- Use `make typecheck` for `mypy` validation
- Use `make test` for core unit/integration/regression coverage
- Use `make docs` to build the documentation site under `docs/_build`

## Notebook support

This repo includes notebook-based exploration and analysis. For the healthcare SHAP notebook, use the pinned environment:

```bash
python -m venv .venv-healthcare
.venv-healthcare\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-notebook-healthcare-shap.txt
python -m ipykernel install --user --name healthcare-shap --display-name "Python (healthcare-shap)"
```

Open `notebooks/healthcare_analysis.ipynb` and select the `Python (healthcare-shap)` kernel.
See `notebooks/README.md` for a full notebook catalog (level, use case, runtime).

Additional domain notebooks:
- `notebooks/finance_credit_risk_stress_testing.ipynb`
- `notebooks/insurance_fraud_triage_optimization.ipynb`
- `notebooks/life_science_clinical_response_safety.ipynb`

Advanced domain notebooks:
- `notebooks/finance_market_risk_var_backtesting.ipynb`
- `notebooks/insurance_reserving_triangle_chainladder.ipynb`
- `notebooks/life_science_survival_censoring_analysis.ipynb`

## Useful scripts

- `scripts/run_tests.py` — run the standard Python test suite
- `scripts/run_notebook_tests.py` — validate and execute notebook workflows
- `scripts/fix_notebook_issues.py` — repair common notebook compatibility issues

## Selected project references

- `projects/machine_learning/customer_segmentation/`
- `projects/machine_learning/credit_risk_modeling/`
- `projects/machine_learning/recommendation_system/`
- `projects/time_series/sales_forecasting/`
- `src/modern_bank_churn/`
- `projects/streamlit_apps/`

Archived legacy project material lives under `archive/legacy/projects/`.

## Repository layout

```text
.
├── src/                       # reusable Python packages
├── projects/                  # portfolio projects and reference implementations
├── docs/                      # documentation and authoring source
├── examples/                  # runnable examples and notebooks
├── notebooks/                 # exploratory notebooks
├── scripts/                   # repo maintenance scripts
├── tests/                     # automated tests
├── deployment/                # deploy manifests and configs
└── tools/                     # developer tooling
```

## Active Quality Scope

The default quality gate (`make check` and CI lint/typecheck/test jobs) focuses on active code paths:

- `src/`
- `tools/`
- `tests/`
- `scripts/`

Portfolio/reference areas (`projects/`, `notebooks/`, `tutorials/`, parts of `deployment/`) are kept in the monorepo but are not quality-gated by default.

## Commands summary

Command | Purpose
--- | ---
`make format` | Auto-format code using Ruff
`make lint` | Run static analysis and lint checks
`make typecheck` | Run `mypy` for type validation
`make test` | Run unit/integration/regression tests
`make docs` | Build Sphinx documentation
`make build` | Build distribution artifacts
`make clean` | Clean caches, build artifacts, and docs output
`make check` | Run lint + typecheck + test

## Support

- Read `docs/index.md` to get started with the documentation
- Use `docs/development.md` for development workflows and contribution guidance
- Open a GitHub issue for questions or bugs
- Refer to `CODEOWNERS` for ownership and review guidance

---

This repository is designed to be a stable playground for data science experimentation and a reusable foundation for production-ready workflows.
