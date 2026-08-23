# Contributor and Maintenance Documentation

This area is for contributors and maintainers. It contains lightweight process
and hygiene documentation; portfolio-first readers should use the
portfolio-facing docs section in [index.md](index.md).

```{toctree}
:maxdepth: 2
:hidden:

contributor/CONTRIBUTING_DOCS
contributor/development
contributor/DOCS_STYLE
contributor/CODE_QUALITY
contributor/DOC_COVERAGE
contributor/TESTING
contributor/TESTING_INFRASTRUCTURE
contributor/CI_CD_PIPELINE
contributor/REPO_STRUCTURE
contributor/ROADMAP
```

## Quick start for contributors

- Review scope and structure in
  [`docs/contributor/REPO_STRUCTURE.md`](contributor/REPO_STRUCTURE.md).
- Follow contribution workflow in
  [`docs/contributor/CONTRIBUTING_DOCS.md`](contributor/CONTRIBUTING_DOCS.md).
- Keep docs and runtime checks healthy using
  [`docs/contributor/development.md`](contributor/development.md).

## CI and checks for maintainers

- Use the fast path on PRs through
  [`.github/workflows/ci.yml`](https://github.com/DiogoRibeiro7/ds-projects-portfolio/blob/main/.github/workflows/ci.yml).
- Run deep checks via:
  - [`notebook-tests.yml`](https://github.com/DiogoRibeiro7/ds-projects-portfolio/blob/main/.github/workflows/notebook-tests.yml)
  - [`codeql-analysis.yml`](https://github.com/DiogoRibeiro7/ds-projects-portfolio/blob/main/.github/workflows/codeql-analysis.yml)
- Deep checks are optional and can be started either from the Actions UI (`Run workflow`) or by adding the `run-deep-ci` label to the PR.

The repository uses this split to keep required checks fast while still allowing
deeper validation before larger portfolio updates.
