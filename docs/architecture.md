# Portfolio Architecture

This repository is the active portfolio surface for data science projects, notebooks,
examples, documentation, and selected reusable source modules.

## Repository Model

The active tree is intentionally organized for review:

- `README.md` provides the first-pass portfolio narrative.
- `projects.yml` is the machine-readable catalogue for featured work.
- `notebooks/`, `projects/`, and `examples/` contain the in-repository portfolio material.
- `src/`, `tests/`, `scripts/`, and `tools/` support reusable code and validation.
- `archive/` keeps historical/reference material away from the primary portfolio path.

## Standalone Repositories

Some portfolio projects also have focused standalone repositories with independent
dependencies and CI:

- `experimentation-toolkit`: `https://github.com/DiogoRibeiro7/experimentation-toolkit`
- `genai-rag-engineering`: `https://github.com/DiogoRibeiro7/genai-rag-engineering`
- `portugal-gdp-bayesian-revision`: `https://github.com/DiogoRibeiro7/portugal-gdp-bayesian-revision`
- `porto-lisbon-uhi-exposure`: `https://github.com/DiogoRibeiro7/porto-lisbon-uhi-exposure`
- `city-wage-cost-global`: `https://github.com/DiogoRibeiro7/city-wage-cost-global`
- `pt-salary-gamma-distribution`: `https://github.com/DiogoRibeiro7/pt-salary-gamma-distribution`
- `customer-analytics`: `https://github.com/DiogoRibeiro7/customer-analytics`

This repository remains the portfolio index and the home of active in-repo project files.
The standalone repositories are separate maintenance surfaces, not duplicated working
copies inside this repo.

## Catalogue

`projects.yml` is the source of truth for featured project metadata.

Required fields per project:

- `name`
- `repository`
- `category`
- `status`
- `summary`
- `methods`
- `technologies`
- `featured`

## Boundaries

Generated artifacts, caches, local runtime outputs, and migration workspaces are not part of
the active portfolio surface. Historical material can remain in Git history or `archive/`,
but the root tree should stay focused on project discovery and review.
