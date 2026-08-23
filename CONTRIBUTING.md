# Contributing

Thanks for helping improve the Data Science Projects Portfolio. This repository
is a public portfolio surface, so contributions should keep the maintained data
science projects, examples, and docs easy to review.

## 1. Environment Setup

```bash
git clone https://github.com/<org>/ds-projects-portfolio.git
cd ds-projects-portfolio
python -m venv .venv && . .venv/Scripts/activate  # use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
pre-commit install
```

Optional extras:

- `pip install -r requirements-dev.txt` for contributor tooling
  (lint, formatting, type-checking, docs tooling).
- `pip install -r requirements-test.txt` for notebook and perf tooling.
- `pip install -r requirements-notebook-healthcare-shap.txt` only when running
  the standalone healthcare SHAP notebook environment.

## 2. Useful Commands

Command | Purpose
--- | ---
`make format` | Apply in-place formatting with Ruff
`make lint` | Static analysis / lint checks (matches CI)
`make typecheck` | Run mypy on `tools/` and `src/`
`make test` | Fast pytest suite (unit + integration + regression)
`make docs` | Build HTML docs (`docs/_build`)
`make build` | Package distributions via `python -m build`
`make clean` | Remove caches, dist artifacts, docs build
`make check` | Combo of lint + typecheck + test (same as CI)
`make test-slow` | Optional long-running suite for local/deep validation
`python scripts/generate_regression_baselines.py` | Refresh regression snapshots when expected outputs change
`python examples/run_demo.py` | Smoke-test the stack end-to-end
`pre-commit run --all-files` | Run hook suite manually

Set `TEST_OPTS` to pass extra flags, e.g. `make test TEST_OPTS='-k feature_x'`.

## 3. Branch & PR Rules

1. Branch off `main`. Use the format `feature/<short-description>` or `fix/<short-description>`.
2. Keep PRs focused. Large cross-cutting changes should be split.
3. Every PR must:
   - include docs/tests for new behavior
   - pass `make check`
   - mention related issues (`Fixes #123`)
4. Request review from the CODEOWNERS group.
5. Squash & merge after approval.

## 4. Code Style

- **Python**: Ruff (PEP8 + repo-specific rules). No unused imports, keep type hints on public APIs.
- **Docstrings**: Google-style with Args/Returns/Raises. Include examples for
  high-usage functions; see `docs/contributor/DOCS_STYLE.md`.
- **Notebooks**: Clear markdown sections (“Setup”, “EDA”, “Modeling”, “Recommendations”). Remove noisy outputs before committing.
- **Config/JSON**: Keep sorted keys and trailing newlines.

## 5. Running Checks

Local checklist before pushing:

```bash
make check
python examples/run_demo.py
pre-commit run --all-files
```

For doc-heavy changes, also run `(cd docs && make html)`. Updating regression
baselines? Run the generator script and commit the refreshed files plus the
change that required them.

## 6. Filing Issues

When reporting a bug, include:

- Steps to reproduce
- Expected vs actual behavior
- Environment info (OS, Python version)
- Relevant logs/tracebacks

For feature requests, outline the use case and which module(s) are involved.

Thanks for contributing. See also:
[`docs/contributor/CODE_QUALITY.md`](docs/contributor/CODE_QUALITY.md) for detailed lint/format/type guidance.
