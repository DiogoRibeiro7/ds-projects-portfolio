# Portfolio Overview

This page expands the repository README for reviewers who want a structured tour
of the portfolio without reading every notebook or project folder.

## What To Review First

Use this order when time is limited:

1. [`README.md`](https://github.com/DiogoRibeiro7/ds-projects-portfolio/blob/main/README.md)
   for the high-level narrative and featured set.
2. [`projects.yml`](https://github.com/DiogoRibeiro7/ds-projects-portfolio/blob/main/projects.yml)
   for the machine-readable project catalogue.
3. Featured notebooks in `notebooks/` for modeling depth and communication.
4. `examples/run_demo.py` for a quick executable smoke path.
5. `src/`, `tests/`, and `docs/api/` for reusable code and validation quality.

## Featured Themes

The portfolio is organized around practical evidence of data science maturity:

- **Statistical rigor:** experimentation, power analysis, robust estimators,
  Bayesian methods, and uncertainty communication.
- **Applied modeling:** healthcare explainability, finance stress testing,
  customer analytics, geospatial exposure analysis, and forecasting.
- **AI engineering:** retrieval pipelines, evaluation patterns, guardrails, and
  service-oriented GenAI examples.
- **Production awareness:** reproducible demos, test coverage, service-oriented
  notebooks, and documentation that separates active assets from archived work.

## Repository Map

```text
.
├── README.md              # Main portfolio entry point
├── projects.yml           # Featured project catalogue
├── notebooks/             # Portfolio notebooks
├── projects/              # Project writeups and reference implementations
├── examples/              # Runnable demos and examples
├── src/                   # Reusable Python modules
├── tests/                 # Automated validation for active modules
└── docs/                  # Portfolio and contributor documentation
```

## Standalone Project Repositories

Some selected projects also have focused standalone repositories:

- [`experimentation-toolkit`](https://github.com/DiogoRibeiro7/experimentation-toolkit)
- [`genai-rag-engineering`](https://github.com/DiogoRibeiro7/genai-rag-engineering)
- [`portugal-gdp-bayesian-revision`](https://github.com/DiogoRibeiro7/portugal-gdp-bayesian-revision)
- [`porto-lisbon-uhi-exposure`](https://github.com/DiogoRibeiro7/porto-lisbon-uhi-exposure)
- [`city-wage-cost-global`](https://github.com/DiogoRibeiro7/city-wage-cost-global)
- [`pt-salary-gamma-distribution`](https://github.com/DiogoRibeiro7/pt-salary-gamma-distribution)
- [`customer-analytics`](https://github.com/DiogoRibeiro7/customer-analytics)

Those repositories are separate maintenance surfaces. This repository remains
the portfolio index and keeps active in-repo projects, notebooks, examples, and
supporting code.

## Validation

The default validation path is intentionally short:

```bash
python examples/run_demo.py
make check
```

For documentation changes:

```bash
cd docs
make html
```

For project-specific validation, use the README or notebook instructions in the
relevant project folder.
