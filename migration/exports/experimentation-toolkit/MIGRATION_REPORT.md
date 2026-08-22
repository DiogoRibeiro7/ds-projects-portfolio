# Experimentation Toolkit Export Report

## Decision

Create a standalone-ready experimentation package as a copy-first export.
No source files, notebooks, or portfolio assets were moved or deleted from `ds-projects-portfolio`.

## Export Location

`migration/exports/experimentation-toolkit`

## Kept In Portfolio

- A/B testing notebooks and playbooks under `projects/ab_testing/`
- Portfolio apps under `projects/streamlit_apps/`
- Existing statistics and data-processing modules under `src/`
- Existing tests under `tests/`

## Exported Package Contents

- `experimentation_toolkit.statistics`
- `experimentation_toolkit.power`
- `experimentation_toolkit.validation`
- `experimentation_toolkit.variance_reduction`
- `experimentation_toolkit.diagnostics`
- `experimentation_toolkit.bandits`

## Validation Commands

Run from the export directory:

```bash
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
mypy src tests
pytest --cov=experimentation_toolkit --cov-report=term-missing
```
