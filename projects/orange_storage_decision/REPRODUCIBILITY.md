# Reproducibility and completion audit

## Scope and release status

The orange-selling project is **complete as a reproducible synthetic R case
study**. It is not a calibrated commercial decision system, a safety
certification or a claim that a particular sale policy improves real harvest
returns. Further work with real data would be a separate empirical study,
subject to the predeclared requirements in [PROTOCOL.md](PROTOCOL.md).

The [scientific synthesis](SCIENTIFIC_SYNTHESIS.md) is the consolidated
interpretation of the saved comparisons. There is no universal optimal sale
day: prices, deterioration rates, grade labels, logistics and the cost of
information change the decision. No further synthetic-model extensions are
required to close this methodological portfolio case study.

## Environment and checks

- Language: **R only**. The scripts use base R (including its standard
  `stats`, `utils`, `graphics` and `grDevices` packages); no Python, external
  packages, API keys, personal data or internet access are needed to run them.
- Execution directory: `projects/orange_storage_decision`, not the repository
  root. All R `source()` calls and result paths are relative to that folder.
- The GitHub Actions workflow [Orange Storage Decision (R)](../../.github/workflows/orange-storage-decision.yml)
  runs all model tests, the seeded demonstration, reporting smoke tests and
  the final evidence audit. Smoke tests run **smaller** panels than the saved
  research reports; passing CI is not a byte-for-byte full-study replay.
- Each archived report has a `settings.R` snapshot with generator inputs and
  the R version used when it was produced. Exact CSV equality can vary with
  numerical-library or R-version changes, so compare declared estimands,
  seeds, tolerance and reproducibility conditions rather than expecting a
  universal binary-identical environment.

Run from the project directory:

```sh
Rscript tests/test_release_audit.R
Rscript tests/test_model.R
Rscript scripts/demo.R

# Full test suite on a shell with Rscript available:
for test_script in tests/test_*.R; do Rscript "$test_script"; done
```

The release audit checks:

1. All 14 archived studies have nonempty reports, settings, parseable CSV
   evidence and figures.
2. Local Markdown links and image references resolve across project documents.
3. Seventeen selected headline figures in the scientific synthesis match the
   appropriate rows and columns of the **committed** CSV summaries to two
   decimal places. The audit rejects missing or ambiguous source rows.

These checks protect the integrity of the repository evidence, not the
statistical assumptions behind it or its applicability outside simulation.

## Archived numerical studies

Each report below has its own exact reproduction command, interpretation
limits and linked CSV/settings files. Its experiment uses the generator and
sample size given there; the reports are **not** observations from one common
benchmark dataset.

| Topic | Archived results | Runner |
| --- | --- | --- |
| Price forecast sensitivity | [Results](results/price_sensitivity/RESULTS.md) | `scripts/run_price_sensitivity.R` |
| Minimax regret over price scenarios | [Results](results/robust_policy/RESULTS.md) | `scripts/run_robust_policy.R` |
| Post-decision survival stress | [Results](results/survival_stress/RESULTS.md) | `scripts/run_survival_stress.R` |
| Inspection grading error | [Results](results/inspection_error/RESULTS.md) | `scripts/run_inspection_error.R` |
| Finite grading calibration bootstrap | [Results](results/calibration_uncertainty/RESULTS.md) | `scripts/run_calibration_uncertainty.R` |
| Shared-shape Weibull | [Results](results/shared_shape/RESULTS.md) | `scripts/run_shared_shape.R` |
| Model averaging | [Results](results/shape_ensemble/RESULTS.md) | `scripts/run_shape_ensemble.R` |
| Nondestructive inspection budgets | [Results](results/inspection_budget/RESULTS.md) | `scripts/run_inspection_budget.R` |
| Destructive inspection inventory ledger | [Results](results/inspection_removals/RESULTS.md) | `scripts/run_inspection_removals.R` |
| Economic value of destructive plans | [Results](results/inspection_plan_value/RESULTS.md) | `scripts/run_inspection_plan_value.R` |
| Frozen historical baseline | [Results](results/historical_baseline/RESULTS.md) | `scripts/run_historical_baseline.R` |
| Between-archive uncertainty | [Results](results/archive_uncertainty/RESULTS.md) | `scripts/run_archive_uncertainty.R` |
| Historical-model transfer under shift | [Results](results/historical_shift/RESULTS.md) | `scripts/run_historical_shift.R` |
| Regret over survival scenarios | [Results](results/survival_regret/RESULTS.md) | `scripts/run_survival_regret.R` |

To regenerate a full example **without overwriting tracked evidence**, pass
a fresh output directory to the report script (the examples below write to
ignored `results/replay_*` subdirectories):

```sh
Rscript scripts/run_price_sensitivity.R 200 results/replay_price_sensitivity
Rscript scripts/run_inspection_plan_value.R 200 results/replay_inspection_plan_value
Rscript scripts/run_historical_shift.R 30 results/replay_historical_shift 50
```

Delete or archive those generated directories only after checking the files
you need. The audit compares committed `results/<study>/` records; it does
not silently replace them with the reproduction run.

## Methods without archived full-size numerical comparisons

The project also includes testable R implementations and CI smoke-report
generation for the one-step adaptive decision, shelf-life stress of adaptive
policies, noisy adaptive labels, finite grader-rate calibration, calibration
grid resolution and Gaussian-quadrature reference. These extensions are
documented in [ADAPTIVE_POLICY.md](ADAPTIVE_POLICY.md),
[ADAPTIVE_STRESS.md](ADAPTIVE_STRESS.md),
[ADAPTIVE_LABEL_ERROR.md](ADAPTIVE_LABEL_ERROR.md),
[ADAPTIVE_CALIBRATION_UNCERTAINTY.md](ADAPTIVE_CALIBRATION_UNCERTAINTY.md),
[ADAPTIVE_GRID_RESOLUTION.md](ADAPTIVE_GRID_RESOLUTION.md) and
[ADAPTIVE_GAUSS_REFERENCE.md](ADAPTIVE_GAUSS_REFERENCE.md).

They are **methodological tools**, not further empirical results. Generating
a two- or three-lot CI smoke report cannot justify a statement that one
adaptive policy improves field returns, and it does not replace a large
independent validation study.

## Project handoff

- Study question, algorithms and assumptions: [README.md](README.md).
- Quantitative conclusions, including negative cases: [SCIENTIFIC_SYNTHESIS.md](SCIENTIFIC_SYNTHESIS.md).
- Data collection and prospective validation requirements: [PROTOCOL.md](PROTOCOL.md).
- Evidence integrity: `R/release_audit.R` and `tests/test_release_audit.R`.

The release is scoped to **transparent, reproducible synthetic analysis**.
The subsequently archived [real SIMA price source](MARKET_PRICES.md) and
[out-of-time quotation-forecast extension](ROLLING_PRICE_BACKTEST.md), together
with its [hypothetical publication-lag stress test](AVAILABILITY_STRESS.md),
and the [matched product-mix audit](COMPOSITION_AUDIT.md), are separate
exercises; none modifies or retrospectively validates this
completed postharvest decision-study release.
Before any operational recommendation, acquire representative fruit-level
and lot-level data, record real prices and costs, validate on later harvest
lots, and report calibration, realised returns and downside against a
prespecified operational baseline. If those criteria are not met, the case
study must remain labelled as synthetic and unfeatured.
