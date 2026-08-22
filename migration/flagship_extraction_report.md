# Flagship Extraction Report

## Scope

This report covers the four high-value standalone project snapshots created under
`migration/exports/`:

| Portfolio source | Export snapshot |
|---|---|
| `projects/portugal_gdp_bayesian_revision` | `migration/exports/portugal-gdp-bayesian-revision` |
| `projects/porto_lisbon_uhi_exposure` | `migration/exports/porto-lisbon-uhi-exposure` |
| `projects/city_wage_cost_global` | `migration/exports/city-wage-cost-global` |
| `projects/pt_salary_gamma_distribution` | `migration/exports/pt-salary-gamma-distribution` |

The original decomposition prompt suggested the names `city-wage-cost-analysis` and
`portugal-salary-distribution`; the snapshots use the current project-aligned names
above. These are copy-first exports, not moves.

## Policy

No original project directories were deleted, moved, renamed, or rewritten. The portfolio
repository remains the home for active data science projects, and these snapshots are
standalone-ready copies for later repository creation.

## Export Summary

### Portugal GDP Bayesian Revision

Export: `migration/exports/portugal-gdp-bayesian-revision`

Copied:

- `src/pt_gdp_bayes`
- project tests
- project notebook
- `scripts/run_analysis.py`
- README, `pyproject.toml`, `requirements.txt`

Intentionally excluded:

- monorepo root packaging and dependency files
- unrelated notebooks, tests, and shared modules
- generated caches and local validation artifacts

Import/package changes:

- Kept canonical package namespace `pt_gdp_bayes`.
- Preserved project-local package/test layout.

Validation:

- `python -m pip install -e ".[test]"`
- `pytest tests/ -q`: 145 passed

### Porto Lisbon UHI Exposure

Export: `migration/exports/porto-lisbon-uhi-exposure`

Copied:

- `src/uhi_exposure`
- project tests
- project notebook
- `scripts/prepare_real_data.py`
- `scripts/run_analysis.py`
- demo plots
- README and `pyproject.toml`

Intentionally excluded:

- unrelated geospatial notebooks and root tests
- monorepo-level dependency/configuration files
- generated caches and local validation artifacts

Import/package changes:

- Kept canonical package namespace `uhi_exposure`.
- Export metadata was adjusted in the copy only to validate in the current Python 3.13
  workspace with modern NumPy, GeoPandas, and PyArrow wheels.

Validation:

- `python -m pip install -e .`
- `pytest tests/ -q`: 12 passed

### City Wage Cost Global

Export: `migration/exports/city-wage-cost-global`

Copied:

- four project notebooks
- README
- methodology document
- project `requirements.txt`

Intentionally excluded:

- monorepo root packaging
- unrelated economics notebooks
- generated caches and local validation artifacts

Import/package changes:

- No package namespace was introduced because this is a notebook-first project.
- Dependency validation used notebook JSON parsing and dependency import smoke checks.

Validation:

- `python -m pip install -r requirements.txt`
- notebook JSON validation: 4 valid notebooks
- dependency import smoke test passed for the required notebook stack

### Portugal Salary Gamma Distribution

Export: `migration/exports/pt-salary-gamma-distribution`

Copied:

- `src/pt_salary_gamma_distribution`
- project tests
- notebook and paired Python notebook script
- analysis and summary scripts
- notebook summary report
- README, ROADMAP, requirements, and `.gitignore`

Intentionally excluded:

- monorepo root packaging and dependency files
- unrelated Portugal economics notebooks
- generated caches and local validation artifacts

Import/package changes:

- Kept canonical package namespace `pt_salary_gamma_distribution`.
- Added export-only `pyproject.toml` because the portfolio project did not include
  standalone package metadata.
- Added export-only CI workflow.

Validation:

- `python -m pip install -e ".[test]"`
- notebook JSON validation: 1 valid notebook
- `pytest tests/ -q`: 12 passed

## Remaining Issues

- The exports are snapshots inside this repository. Creating separate GitHub repositories
  remains a manual or later automated step.
- Some validation commands were adapted from the original prompt where a project is
  notebook-first or has no meaningful `src` package.
- Root repository cleanup is intentionally out of scope. Originals remain here unless the
  user explicitly approves a later removal after destination repositories exist.

## Commit History

The flagship exports were added on the migration branch in:

- `ec79b7c Add copy-first decomposition exports`
- `c5bcd46 Add city wage cost export`
- `148771c Add Portugal salary distribution export`
