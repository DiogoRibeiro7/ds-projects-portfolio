# Contributor and Maintenance Documentation

This area is for contributors, maintainers, and release workflows. It contains
process and hygiene documentation; portfolio-first readers should use the
portfolio-facing docs section in [index.md](index.md).

```{toctree}
:maxdepth: 2
:hidden:

CONTRIBUTING_DOCS
development
DOCS_STYLE
CODE_QUALITY
DOC_COVERAGE
TESTING
TESTING_INFRASTRUCTURE
CI_CD_PIPELINE
REPO_STRUCTURE
ROADMAP
RELEASE
```

## Quick start for contributors

- Review scope and structure in [`docs/REPO_STRUCTURE.md`](REPO_STRUCTURE.md).
- Follow contribution workflow in [`docs/CONTRIBUTING_DOCS.md`](CONTRIBUTING_DOCS.md).
- Keep docs and runtime checks healthy using [`docs/development.md`](development.md).

## CI and checks for maintainers

- Use the fast path on PRs through [`.github/workflows/ci.yml`](.github/workflows/ci.yml).
- Run deep checks via:
  - [`notebook-tests.yml`](../.github/workflows/notebook-tests.yml)
  - [`codeql-analysis.yml`](../.github/workflows/codeql-analysis.yml)
- Deep checks are optional and can be started either from the Actions UI (`Run workflow`) or by adding the `run-deep-ci` label to the PR.

The repository uses this split to keep required checks fast while still allowing deeper validation before release.
