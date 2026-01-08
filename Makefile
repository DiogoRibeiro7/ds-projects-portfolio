PYTHON ?= python
PYTEST ?= pytest
MYPY ?= mypy
SPHINXBUILD ?= sphinx-build
TEST_OPTS ?=

.PHONY: check format lint typecheck test test-unit test-integration test-regression test-slow test-all docs build clean

format:
	ruff format .

lint:
	ruff check .

typecheck:
	$(MYPY) tools/ src/ --ignore-missing-imports

test:
	$(PYTEST) -m "(unit or integration or regression) and not slow" $(TEST_OPTS)

test-unit:
	$(PYTEST) -m "unit and not slow" $(TEST_OPTS)

test-integration:
	$(PYTEST) -m "integration and not slow" $(TEST_OPTS)

test-regression:
	$(PYTEST) -m "regression and not slow" $(TEST_OPTS)

test-slow:
	$(PYTEST) -m "slow" $(TEST_OPTS)

test-all:
	$(PYTEST) $(TEST_OPTS)

docs:
	$(SPHINXBUILD) -b html docs docs/_build

build:
	$(PYTHON) -m build

clean:
	rm -rf htmlcov .pytest_cache .mypy_cache .ruff_cache build dist docs/_build
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]"

check: lint typecheck test
