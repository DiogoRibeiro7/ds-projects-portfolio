# Installation

Follow these steps to get a fully reproducible environment for the portfolio.

## Prerequisites

- Python 3.10+ (matches `pyproject.toml`)
- Git
- Optional: Docker + docker-compose for containerized workflows

## Create an Environment

```bash
git clone https://github.com/diogoribeiro7/ds-projects-portfolio.git
cd ds-projects-portfolio

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-test.txt
pip install -r docs/requirements-docs.txt
```

> Tip: use `pip install -e .` to work on the packaged modules located under
> `src/`.

## Optional Extras

| Need                | Command                                                                 |
| ------------------- | ----------------------------------------------------------------------- |
| Notebook workflow   | `pip install jupyterlab` (already included in `requirements.txt`)       |
| Local docs preview  | `pip install sphinx-autobuild && (cd docs && make livehtml)`            |
| GPU experimentation | Install CUDA/cuDNN per vendor instructions then enable PyTorch/TensorFlow |

## Validate the Environment

```bash
pre-commit install
pytest tests/unit -n auto
(cd docs && make html)
```

## Upgrades

Dependencies are pinned in the various `requirements-*.txt` files. Run
`pip install --upgrade -r requirements.txt` to pick up patched releases, then
re-run the smoke tests above.
