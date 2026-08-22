# Experimentation Toolkit

Reusable A/B testing and experimentation utilities extracted from the data science portfolio.

This export is copy-first: the original portfolio files remain in `ds-projects-portfolio`.
The package is intended to become a standalone repository after review.

## Included Scope

- Frequentist two-proportion tests and Welch t-tests
- Bootstrap confidence intervals
- Two-proportion sample size and power calculations
- Sample-ratio mismatch checks
- CUPED variance reduction
- Small bandit policy helpers
- DataFrame group summaries for experiment diagnostics

## Source Provenance

The implementation was curated from these portfolio areas:

- `src/statistics/core.py`
- `src/data_processing/cleaning.py`
- `projects/ab_testing/core.py`
- A/B testing notebooks under `projects/ab_testing/`

Notebook storytelling, portfolio-specific apps, generated data, and unrelated ML/data-engineering code are intentionally not part of this package export.

## Development

```bash
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
mypy src tests
pytest --cov=experimentation_toolkit --cov-report=term-missing
```
