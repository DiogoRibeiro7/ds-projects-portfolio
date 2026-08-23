# CI Pipeline

This repository uses GitHub Actions to protect the active portfolio surface:
source modules, tests, examples, scripts, tools, docs, and selected notebooks.

## Required Pull Request Checks

The main workflow is `.github/workflows/ci.yml`.

It runs:

- Ruff format and lint checks.
- Mypy type checking for active Python modules and tools.
- Pytest on Python 3.11 and 3.12.
- Sphinx documentation build.

The workflow is intentionally portfolio-first. It validates maintained code and
documentation without presenting this repository as a deployment platform.

## Optional Deep Checks

Additional workflows cover deeper or less frequent checks:

- `.github/workflows/notebook-tests.yml`: notebook validation.
- `.github/workflows/codeql-analysis.yml`: security analysis.
- `.github/workflows/dependency-update.yml`: dependency maintenance.
- `.github/workflows/release.yml`: release hygiene.

These workflows are useful for maintenance, but the fast CI path remains the
default signal for normal portfolio cleanup and documentation changes.

## Local Validation

Run the same practical checks before opening a pull request:

```bash
make check
cd docs
make html
```

For focused Python changes:

```bash
ruff check src tests tools
mypy src tools
pytest tests/unit
```

## Maintenance Notes

- Keep required checks fast enough for small portfolio edits.
- Avoid adding generated outputs or local runtime state to CI.
- Prefer project-specific validation in each project README over one oversized
  repository-wide workflow.
