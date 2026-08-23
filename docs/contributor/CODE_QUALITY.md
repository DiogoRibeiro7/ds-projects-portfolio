# Code Quality Guide

This page summarizes the formatting, linting, and typing rules enforced locally (via `pre-commit`) and in CI.

## Toolchain

| Tool | Purpose | Config |
| --- | --- | --- |
| Ruff (`ruff format`, `ruff check`) | Formatter + linter (PEP8, pyupgrade, bugbear, import sorting) | `pyproject.toml` |
| Black | Secondary formatter for IDEs/CI compatibility | `pyproject.toml` |
| mypy | Static type checking (strict mode with selective ignores) | `pyproject.toml` |
| pytest | Unit/integration/regression tests with coverage | `pytest.ini` / `pyproject.toml` |
| Sphinx | Docs build | `docs/conf.py` |

## Workflow

1. **Install hooks**: `pre-commit install` (runs Ruff format/check, Black check, mypy, whitespace cleanup on every commit).
2. **Run the check suite** before pushing: `make check` for broad local linting,
   type checking, and tests.
3. **Optional**: `make docs` and `make test-slow` for deeper local validation.

## Common Issues & Fixes

| Issue | Fix |
| --- | --- |
| Ruff formatting violations | Run `make format` (or `ruff format .`) |
| Ruff lint errors (E/F/W, bugbear, imports) | Follow the message hint; prefer refactors over `# noqa`. If a rule is genuinely noisy, fine-tune it in `pyproject.toml`. |
| mypy “missing import” | Add the module to `[tool.mypy.overrides].ignore_missing_imports` when stubs don’t exist. |
| mypy strictness errors | Add precise type hints or use `typing.cast`. Avoid blanket `# type: ignore` unless unavoidable. |
| Pre-commit hook fails | Run `pre-commit run --all-files` to fix locally before committing. |

## Fast Reference Commands

```bash
make format    # ruff format
make lint      # ruff check
make typecheck # mypy src/
make test      # fast pytest suite
make check     # broad lint + typecheck + test suite
pre-commit run --all-files  # run hooks manually
```

Following these rules ensures local development, pre-commit hooks, and CI stay in sync and deterministic.
