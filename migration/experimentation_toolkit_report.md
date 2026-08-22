# Experimentation Toolkit Report

## Scope

This report covers the copy-first export at:

`migration/exports/experimentation-toolkit`

The export creates a focused Python package for controlled-experiment analysis. The
original portfolio files remain in place.

## Migrated Modules

| Export module | Purpose | Source provenance |
|---|---|---|
| `experimentation_toolkit.statistics` | two-proportion z-test, Welch t-test, bootstrap lift interval | curated from `src/statistics/core.py` and experiment examples |
| `experimentation_toolkit.power` | two-proportion sample size, approximate power, Cohen's h | curated from statistics/power-analysis material |
| `experimentation_toolkit.validation` | sample-ratio mismatch checks and binary metric validation | curated from experiment validation patterns |
| `experimentation_toolkit.variance_reduction` | CUPED adjustment and variance-reduction diagnostics | curated from A/B testing and statistical-methods material |
| `experimentation_toolkit.diagnostics` | group-level metric summaries | curated from lightweight data-processing/reporting helpers |
| `experimentation_toolkit.bandits` | epsilon-greedy, Thompson sampling, UCB1 helpers | curated from `projects/ab_testing` bandit material |

The package uses the namespace `experimentation_toolkit`; no `src.*` or `projects.*`
runtime imports are required in the export.

## Deliberately Discarded Scope

The export intentionally does not include:

- GenAI/RAG code.
- cloud, security, privacy, compliance, AutoML, API, or generic deployment code.
- churn/customer-specific code.
- broad data-engineering and data-quality frameworks.
- Streamlit apps and notebook storytelling.
- generated datasets, notebook outputs, caches, and local runtime artifacts.
- the root `ds-portfolio` package metadata and invalid `ds_portfolio.cli` console script.

The original A/B testing notebooks and apps remain in the portfolio repository as project
content. They were not copied wholesale into the package because the goal is a focused
library, not another mini-monorepo.

## Public API

The export defines an explicit package-level `__all__`:

- `BanditState`
- `BootstrapInterval`
- `CupedResult`
- `SampleRatioResult`
- `TestResult`
- `apply_cuped`
- `bootstrap_ci_diff`
- `cohens_h`
- `epsilon_greedy_arm`
- `power_two_proportions`
- `sample_ratio_mismatch`
- `sample_size_two_proportions`
- `summarize_groups`
- `thompson_beta_arm`
- `two_proportion_z_test`
- `ucb1_arm`
- `welch_t_test`

The API avoids wildcard exports and does not hide optional-import failures.

## Dependency Graph

Runtime dependencies are intentionally small:

- `numpy`
- `pandas`
- `scipy`

Development-only dependencies:

- `mypy`
- `pandas-stubs`
- `pytest`
- `pytest-cov`
- `ruff`

Internal dependency shape:

- top-level `experimentation_toolkit.__init__` re-exports selected public functions and result dataclasses.
- `statistics`, `power`, and `validation` depend on SciPy statistical functions.
- `variance_reduction`, `bandits`, and `diagnostics` depend on NumPy and/or pandas.
- no module depends on the portfolio root package, `src`, `projects`, notebooks, or generated artifacts.

## Tests And Validation

Validation performed from `migration/exports/experimentation-toolkit`:

- `python -m pip install -e ".[dev]"`
- `ruff check .`
- `ruff format --check .`
- `mypy src tests`
- `pytest --cov=experimentation_toolkit --cov-report=term-missing`

Result:

- 11 tests passed.
- coverage: 80%.

Test coverage includes:

- public API behavior for z-tests, bootstrap intervals, power/sample-size monotonicity, SRM checks, CUPED, group summaries, and Welch t-tests.
- bandit policy behavior and validation.
- experiment validation edge cases.

## Installation Test Status

The export was validated through editable installation and direct package import/use in
tests. The original prompt requested a wheel-build clean-environment installation test; that
has not been added as a dedicated automated test in the export. It remains a useful next
hardening step before publishing the toolkit as an independent repository.

## Unresolved Design Decisions

- Whether to include a small CLI for common experiment checks. The old root
  `ds-portfolio` console script was not preserved because it pointed to a non-existent
  package.
- Whether to add notebook examples to the export, or keep examples only in the portfolio.
- Whether to extend variance reduction beyond single-covariate CUPED.
- Whether to expose confidence-interval helpers separately from hypothesis-test result
  objects.
- Whether to package richer sequential testing and off-policy evaluation methods after
  reviewing the remaining A/B testing notebooks.

## History And Repository Status

This is a fresh snapshot export. Original source material remains in:

- `src/statistics`
- selected `src/data_processing`
- `projects/ab_testing`
- `projects/statistical_methods`
- `projects/streamlit_apps`
- root tests and examples

No portfolio files were deleted or moved as part of this export.
