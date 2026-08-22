# Roadmap: `01_salary_gamma_distribution_portugal`

This roadmap is specific to:

```text
notebooks/01_salary_gamma_distribution_portugal.ipynb
```

It now serves two purposes:

1. record what has already been implemented;
2. clarify what is still worth doing next.

## Status summary

Current notebook status:

- `Completed`
  grouped-data extraction and validation
- `Completed`
  grouped likelihood fits for Gamma, Lognormal, Weibull, and Generalized Gamma
- `Completed`
  grouped histogram, residual, and decile-fit diagnostics
- `Completed`
  minimum-wage, real-wage, and wage-compression interpretation layers
- `Completed`
  sensitivity checks for minimum-wage and top-bracket handling
- `Completed`
  grouped bootstrap parameter ranges for selected years
- `Completed`
  tail-only Lognormal-versus-Pareto comparison
- `Completed`
  README figure index and glossary
- `Partial`
  optional microdata bridge
- `Pending`
  policy-break overlays and richer two-part tail models

## Current state

The notebook already does the following:

- downloads and validates the official GEP workbook sources;
- extracts grouped salary-bracket and decile-summary tables;
- restricts the reproducible grouped-data window to `2007–2024`;
- documents bracket-definition changes by year;
- fits `Gamma`, `Lognormal`, `Weibull`, and `Generalized Gamma`;
- validates extraction against source cells;
- compares models with AIC, BIC, residuals, grouped histograms, and decile-fit checks;
- tests robustness to minimum-wage-bin and top-bracket treatment;
- adds real-wage and minimum-wage interpretation;
- adds wage-compression and harmonized share-shift plots;
- adds tail-only diagnostics and top-share fit checks;
- documents the optional microdata schema.

Current empirical result:

- `Lognormal` remains the consistent grouped-data winner in the present public-data setup.
- `Gamma` remains useful as a benchmark, but not as the best-supported full-distribution claim from the public grouped tables.

## Phase 1: tighten the current grouped-data model

Priority: high

Status:

- `Completed`
  add bracket-definition metadata by year
- `Completed`
  add sensitivity checks for the top open-ended bracket
- `Completed`
  add sensitivity checks for the exact minimum-wage bin
- `Completed`
  add body-only and alternative grouped-fit summaries
- `Completed`
  add bootstrap-style parameter ranges

Delivered outputs:

- `bracket_metadata_by_year.csv`
- `sensitivity_fit_results.csv`
- `sensitivity_summary.csv`
- `bootstrap_parameter_ranges.csv`

Remaining gaps:

- add a more explicit comparison between full-distribution fit and body-only fit in one compact summary table;
- add a threshold-stability diagnostic for the grouped open-top treatment.

## Phase 2: improve the economic interpretation

Priority: high

Status:

- `Completed`
  add nominal-versus-real comparisons
- `Completed`
  add explicit minimum-wage interpretation
- `Completed`
  add wage-compression metrics
- `Completed`
  add a “distribution changed where?” section
- `Pending`
  add explicit policy-break overlays

Delivered outputs:

- `real_wage_context.png`
- `minimum_wage_regime.png`
- `wage_compression_metrics.png`
- `distribution_change_harmonized_bands.png`

Remaining gaps:

- mark policy-relevant breakpoints directly in the time-series plots;
- add a short written summary linking observed shifts to institutional changes.

## Phase 3: strengthen the tail analysis

Priority: medium-high

Status:

- `Completed`
  add a tail-only Lognormal-versus-Pareto comparison
- `Partial`
  strengthen the upper-tail analysis beyond a descriptive Pareto index
- `Completed`
  add observed-versus-fitted top-bracket-share plots
- `Pending`
  test a true two-part splice model
- `Pending`
  add threshold sensitivity as a more systematic tail panel

Delivered outputs:

- `pareto_tail_diagnostics.csv`
- `tail_model_comparison.csv`
- `tail_model_winners.csv`
- `top_share_fit_comparison.csv`

Remaining gaps:

- implement a body-plus-tail splice model;
- test more than two tail thresholds and summarize stability.

## Phase 4: optional microdata bridge

Priority: medium

Status:

- `Completed`
  define a documented schema for optional local microdata input
- `Partial`
  keep the microdata branch runnable if a file exists
- `Pending`
  mirror the full grouped-model comparison on microdata
- `Pending`
  compare grouped versus microdata conclusions directly

Delivered outputs:

- `data/private/MICRODATA_SCHEMA.md`
- `optional_microdata_schema.csv`

Remaining gaps:

- extend the optional microdata branch from a placeholder fit into a full comparison layer;
- add a notebook section explicitly separating public evidence from private-extension evidence.

## Phase 5: notebook polish for research use

Priority: medium

Status:

- `Completed`
  add a short “main findings first” section near the top
- `Partial`
  add compact summary views, though one stronger consolidated table would still help
- `Completed`
  move repeated plotting helpers into `src/pt_salary_gamma_distribution/`
- `Completed`
  add a figure index to the README
- `Completed`
  add a glossary

Remaining gaps:

- add one single summary table combining winners, decile errors, sensitivity stability, and tail findings;
- reduce repeated notebook plotting code further if the notebook grows again.

## Highest-value next steps

If continuing immediately, the next best tasks are:

1. Add policy-break overlays to the main time-series plots.
2. Add a consolidated summary table with:
   winner by BIC,
   Gamma-versus-Lognormal gap,
   decile error,
   tail winner,
   sensitivity stability.
3. Implement a simple splice model:
   body distribution plus Pareto tail.
4. Expand the optional microdata branch into a real comparison workflow if a local file becomes available.

## Success condition

This notebook is in a very strong state when a careful reader can answer all of the following without guesswork:

- what public years are actually covered;
- which model wins and by how much;
- whether Gamma fails mainly in the minimum-wage mass, the body, or the tail;
- whether the conclusion survives reasonable grouped-data sensitivity checks;
- what the distributional changes mean in labour-market terms;
- how much the remaining uncertainty depends on the absence of public microdata.
