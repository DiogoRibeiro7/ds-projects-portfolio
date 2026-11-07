# Reproducibility and completion audit

## Scope and release status

The orange-selling project has **two completed, separate research tracks**:
the reproducible synthetic R selling-decision case study and a retrospective
study of observed wholesale orange quotations. The latter is consolidated in
the [empirical quotation synthesis](EMPIRICAL_SYNTHESIS.md); neither track
estimates realised producer returns or validates a commercial storage policy.
A prospective fruit-lot and actual-sale validation study is still required,
subject to the predeclared requirements in [PROTOCOL.md](PROTOCOL.md).

The [scientific synthesis](SCIENTIFIC_SYNTHESIS.md) is the consolidated
interpretation of the saved comparisons. There is no universal optimal sale
day: prices, deterioration rates, grade labels, logistics and the cost of
information change the decision. No further synthetic-model or retrospective
wholesale-quotation extensions are required to close the documented
**research phase**.

## Environment and checks

- Language: **R only**. The scripts use base R (including its standard
  `stats`, `utils`, `graphics` and `grDevices` packages); no Python, external
  packages, API keys, personal data or internet access are needed to run them.
- Execution directory: `projects/orange_storage_decision`, not the repository
  root. All R `source()` calls and result paths are relative to that folder.
- The GitHub Actions workflow [Orange Storage Decision (R)](../../.github/workflows/orange-storage-decision.yml)
  runs all model tests, the seeded demonstration, reporting smoke tests and
  both evidence audits. Smoke tests run **smaller** panels than the saved
  synthetic research reports; passing CI is not a byte-for-byte full-study
  replay.
- Each archived report has a `settings.R` snapshot with generator inputs and
  the R version used when it was produced. Exact CSV equality can vary with
  numerical-library or R-version changes, so compare declared estimands,
  seeds, tolerance and reproducibility conditions rather than expecting a
  universal binary-identical environment.

Run from the project directory:

```sh
Rscript tests/test_release_audit.R
Rscript tests/test_sima_empirical_closure.R
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

The empirical closure audit independently recomputes the declared data-source
size, 2026 starts, four-horizon maturity/match/missing denominators, complete-
case quotation movements, illustrative cost/retention scenarios, feasibility
thresholds and no-assumption identification bounds. It cross-checks the
linked methods from the same frozen wholesale source. These checks protect
the published evidence and calculations, **not** the statistical assumptions
behind a model or its applicability to real storage decisions.

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
- Synthetic quantitative conclusions, including negative results:
  [SCIENTIFIC_SYNTHESIS.md](SCIENTIFIC_SYNTHESIS.md).
- Recorded wholesale quotation evidence and its limitations:
  [EMPIRICAL_SYNTHESIS.md](EMPIRICAL_SYNTHESIS.md).
- Prospective data and validation requirements: [PROTOCOL.md](PROTOCOL.md).
- Evidence integrity: `R/release_audit.R`,
  `tests/test_release_audit.R` and `tests/test_sima_empirical_closure.R`.

### Distinct empirical-quotation evidence

The frozen [GPP SIMA source](MARKET_PRICES.md) covers **3,412** official
wholesale orange quotation rows (2025-01-02 through 2026-10-30); the
uncompressed CSV SHA-256 is
`b039f254b4fb18a41b7997208a43ceae46ee9f48db978934ab9fad6de9e24da0`.
Its [rolling-origin quotation backtest](ROLLING_PRICE_BACKTEST.md),
[hypothetical availability stress](AVAILABILITY_STRESS.md),
[composition audit](COMPOSITION_AUDIT.md),
[exact 52-week seasonal benchmark](SEASONAL_52_BENCHMARK.md) and
[level-drift audit](LEVEL_DRIFT_AUDIT.md) are price-only exercises.
The [exact 1–4-week forward quotation windows](FORWARD_PRICE_WINDOWS.md),
[assumed break-even arithmetic](QUOTE_BREAK_EVEN_SCENARIOS.md),
[continuous feasibility thresholds](QUOTE_FEASIBILITY_FRONTIER.md) and
[missing-endpoint identification bounds](QUOTE_MISSING_ENDPOINT_BOUNDS.md)
complete the narrower retrospective quotation analysis.

Among **485** observed 2026 starting panel-weeks, the horizon-eligible
denominators are **453, 442, 431 and 420**, with **20, 39, 58 and 77**
respectively missing exact later quotations. **343** starts have all four
future quotations, a *retrospectively selected* subset. A separate fixed
common-maturity denominator of **420** does not rely on future quote matches.
The empirical closure audit checks these numbers, corresponding quote changes,
hypothetical storage-friction arithmetic and worst/best-case missing bounds
without imputing a future price.

The GitHub Actions workflow recreates empirical CSVs, charts and reports as
named, **30-day artifacts**; those transient files are not a permanent
archived public dataset. The source snapshot and versioned R methods/tests
remain reproducible independently of artifact expiry. The original 14
synthetic archived studies and their baselines were not overwritten.

**Completion is limited to the reproducible research release.** Recorded
wholesale quotations are neither lot-level observations nor executed sales.
No storage-loss fraction, real storage cost or grower profit is estimated.
An operational application requires representative lot-level sampling, real
physical losses, dated information availability, an auditable cost ledger,
realised net returns and a prespecified prospective comparison on later
harvest lots. Until then the portfolio entry stays **unfeatured** and the
results must never be presented as a field-validated sale recommendation.
