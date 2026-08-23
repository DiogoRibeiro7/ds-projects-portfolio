# Installation

Follow these steps to get a fully reproducible environment for the portfolio.

## Prerequisites

- Python 3.10+ (matches `pyproject.toml`)
- Git

## Create an Environment

```bash
git clone https://github.com/diogoribeiro7/ds-projects-portfolio.git
cd ds-projects-portfolio

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

> Tip: use `pip install -e .` to work on the packaged modules located under
> `src/`.

## Optional Extras

| Need                | Command                                                                 |
| ------------------- | ----------------------------------------------------------------------- |
| Notebook workflow   | `pip install jupyterlab` (already included in `requirements.txt`)       |
| Local docs preview  | `pip install sphinx-autobuild && (cd docs && make livehtml)`            |
| GPU experimentation | Install CUDA/cuDNN per vendor instructions then enable PyTorch/TensorFlow |
| Test and benchmark stack | `pip install -r requirements-test.txt` |
| Docs tooling only   | `pip install -r docs/requirements-docs.txt`                            |

## Dependency File Guide

- `requirements.txt`: Primary user environment used by README and docs quick starts.
- `requirements-dev.txt`: Contributor/maintainer environment (lint,
  type-checking, pre-commit, and docs tooling).
- `requirements-test.txt`: Optional CI/test stack for notebook and benchmark runs.
- `requirements-core.txt`: Internal CI baseline for notebook validation jobs.
- `requirements-notebook-healthcare-shap.txt`: Optional notebook-only stack for
  the healthcare SHAP notebook.

## Validate the Environment

```bash
pre-commit install
pytest tests/unit -n auto
(cd docs && make html)
```

## Upgrades

Dependencies for portfolio users are pinned in `requirements.txt`. Run
`pip install --upgrade -r requirements.txt` to pick up patched releases, then
re-run the smoke tests above.
