# Testing Strategy

This repository uses pytest with layered suites so that fast, deterministic tests run on every pull request and heavier flows run nightly or on demand.

## Test Layers

- **Unit (`tests/unit/`, `pytest -m unit`)** — Pure functions and isolated classes. These tests mock all I/O, seed RNGs, and finish in milliseconds.
- **Integration (`tests/integration/`, `pytest -m integration`)** — Multi-component flows such as API surfaces, caching layers, or orchestration pipelines. External services (DB/Redis/cloud) are mocked or replaced with in-memory fakes.
- **Regression (`tests/regression/`, `pytest -m regression`)** — Snapshot/binary compatibility checks that lock in bug fixes. These tests should reproduce previously fixed scenarios and assert on concrete outputs.
- **Slow (`pytest -m slow`)** — Optional tag layered on any test (unit/integration/regression) when it exceeds budget. Slow tests only run nightly or when manually requested.

Other dedicated directories (e.g., `tests/performance`, `tests/notebook_tests`) remain but are opt‑in.

## Deterministic Seeding

- `tests/conftest.py` seeds Faker (`Faker.seed(42)`) and sets deterministic NumPy draws inside fixtures.
- Property-based/Hypothesis tests honor the `HYPOTHESIS_PROFILE` env variable (default `dev`); CI switches to the stricter `ci` profile.
- When adding stochastic logic, call `np.random.seed` / `random.seed` inside fixtures or use `pytest`'s `monkeypatch` so that default runs are reproducible.
- `PYTHONHASHSEED=0` is set in CI to keep hashing deterministic.

## Performance Budget

- Default PR suite (`make test`) must finish in **<10 minutes** on GitHub Actions and <5 minutes locally on a modern laptop.
- Individual unit tests should complete in <1 second; integration/regression tests should stay under 30 seconds.
- Any test exceeding that budget must be marked `@pytest.mark.slow` and will only run on the nightly/manual CI job (`slow-tests`).

## Running Tests

```
make test                 # unit + integration + regression (excludes slow)
make test-unit            # just unit tests
make test-integration     # just integration tests
make test-regression      # regression/snapshot tests
make test-slow            # slow-only suite (nightly equivalent)
```

## Quickstart for local testing

To verify a clean local state quickly:

1. Install dev dependencies:

   ```bash
   python -m pip install -r requirements-dev.txt
   ```

2. Run the fast suite first:

   ```bash
   make test-unit
   ```

3. Expand when green:

   ```bash
   make test
   ```

All commands respect `PYTEST_ADDOPTS` / `TEST_OPTS`, so you can append flags via `make test TEST_OPTS='-k feature_x'`.

CI runs unit+integration+regression on every PR and executes the slow suite on the scheduled nightly / manual dispatch workflow. Notebook/performance suites still live in `.github/workflows/*.yml` and can be triggered independently.

## Updating Regression Baselines

1. Confirm the behavioral change is intentional and review the failing regression test output.
2. Run `python scripts/generate_regression_baselines.py` to regenerate the JSON snapshot(s) under `tests/regression/baselines/`. The script records metadata (seed, platform, Python/numpy versions) and prints a sha256 checksum for auditing.
3. Inspect the updated baseline file(s) — they are small JSON artifacts — and commit them alongside the code change plus a short explanation in your PR.
4. Re-run `make test-regression` (or the full `make test`) to ensure the suite passes with the refreshed baselines.
