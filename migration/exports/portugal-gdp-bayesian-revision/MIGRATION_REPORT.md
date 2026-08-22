# Portugal GDP Bayesian Revision Export Report

## Decision

Create a standalone-ready copy of the flagship project.
No source files, notebooks, data inputs, or portfolio assets were moved or deleted from
`ds-projects-portfolio`.

## Export Location

`migration/exports/portugal-gdp-bayesian-revision`

## Copied From Portfolio

- `projects/portugal_gdp_bayesian_revision/src/pt_gdp_bayes`
- `projects/portugal_gdp_bayesian_revision/tests`
- `projects/portugal_gdp_bayesian_revision/notebooks`
- `projects/portugal_gdp_bayesian_revision/data/known_observations.csv`
- `projects/portugal_gdp_bayesian_revision/data/source_catalog.json`
- `projects/portugal_gdp_bayesian_revision/scripts/run_analysis.py`
- Existing README, `pyproject.toml`, `requirements.txt`, and `.gitignore`

## Kept In Portfolio

The original `projects/portugal_gdp_bayesian_revision` directory remains in place and unchanged.

## Validation Commands

Run from the export directory:

```bash
python -m pip install -e ".[test]"
pytest tests/ -q
```
