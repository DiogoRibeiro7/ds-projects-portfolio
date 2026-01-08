# Data Science Projects Portfolio

## What & Why

This monorepo hosts the reusable code, notebooks, and APIs that the DS Platform team uses to prototype and productionize experimentation-heavy ML projects. It bundles:

- statistical tooling (`src/statistics/core.py`, `statistical_methods/`)
- data pipelines and model orchestrators (`src/data_processing/`, `modern-bank-churn/`)
- serving infrastructure (FastAPI app under `src/api/`)
- dashboards, docs, and notebooks

Use it to explore best practices, run demos, or build new project configurations.

## Install (Command 1)

```bash
pip install -r requirements-dev.txt
```

> Python 3.9–3.12 on Ubuntu/macOS/Windows is supported (see matrix below).

## Quickstart (Commands 2 & 3)

```bash
make check                 # lint + typecheck + fast tests (matches CI defaults)
python examples/run_demo.py  # minimal demo dataset + conversion analysis
```

These are the only three commands needed for a new contributor: install deps, run the quality gate, run the demo.

## Minimal Example

`examples/run_demo.py` builds a deterministic A/B dataset, cleans it, and runs `ExperimentAnalyzer.analyze_conversion`. Run:

```bash
python examples/run_demo.py
```

You will see the control/treatment conversion rates, lift, and p-value—handy for smoke testing the stack.

## Support Matrix

| Category | Supported | Notes |
| --- | --- | --- |
| Python | 3.9 / 3.10 / 3.11 / 3.12 | Matches GitHub Actions matrix |
| OS | Ubuntu-latest / macOS-latest / Windows-latest | CI validated |
| Tooling | pip, pytest 9.x, Ruff, nbval, Hypothesis | Installed via `requirements-dev.txt` |

## Project Status

Active. CI runs lint + unit/integration/regression on every PR and a nightly slow suite. Regression baselines live in `tests/regression/baselines/` and are regenerated with `python scripts/generate_regression_baselines.py`.

## Repository Layout

```
.
├── src/                       # Core Python packages (APIs, utilities, stats)
├── tests/                     # Unit, integration, regression, perf, notebooks
├── modern-bank-churn/         # MLOps/orchestration reference project
├── statistical_methods/       # Advanced stats utilities
├── docs/                      # Architecture notes & guides
├── scripts/                   # Operational scripts (baseline generation, etc.)
├── examples/                  # Runnable demos (see run_demo.py)
└── notebooks/                 # Analysis notebooks (see ab_testing/, etc.)
```

## Common Commands

Command | Description
--- | ---
`make format` | Auto-format the codebase with Ruff
`make lint` | Static analysis / style checks (Ruff)
`make typecheck` | Mypy across `tools/` and `src/`
`make test` | Fast pytest suite (unit + integration + regression, no `slow`)
`make docs` | Build HTML docs via Sphinx (outputs to `docs/_build`)
`make build` | Create distribution artifacts (`python -m build`)
`make clean` | Remove caches, build artifacts, docs output
`make check` | Runs `lint`, `typecheck`, and `test` — mirrors CI’s fast checks
`make test-slow` | Execute the `slow`-tagged tests (nightly/opt-in)

## Support / Questions

- Open a GitHub issue (tag `question` for routing).
- Ping the `#ds-platform` Slack channel.
- For emergencies, contact the CODEOWNERS (see `CODEOWNERS`).
- Need deeper lint/typecheck guidance? Read [docs/CODE_QUALITY.md](docs/CODE_QUALITY.md).

Happy hacking!
