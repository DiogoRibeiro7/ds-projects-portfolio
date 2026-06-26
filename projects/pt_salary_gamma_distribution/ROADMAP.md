# Roadmap: `01_salary_gamma_distribution_portugal`

This roadmap is specific to:

```text
notebooks/01_salary_gamma_distribution_portugal.ipynb
```

Its purpose is to turn the notebook from a solid grouped-data analysis into a stronger distribution-study workflow with clearer empirical claims, better robustness, and cleaner extension paths.

## Current state

The notebook already does the core work well:

- downloads and validates the official GEP workbook sources;
- extracts grouped salary-bracket and decile-summary tables;
- restricts the reproducible grouped-data window to `2007–2024`;
- fits `Gamma`, `Lognormal`, `Weibull`, and `Generalized Gamma`;
- validates extraction against source cells;
- compares models with AIC, BIC, residuals, and decile-fit checks;
- includes grouped histograms and top-tail diagnostics.

The current empirical result is also already clear:

- `Lognormal` is the consistent grouped-data winner in the present setup;
- `Gamma` is still useful as a benchmark, but not as the best-supported full-distribution claim from the public grouped tables.

## Objective

Improve the notebook along three axes:

1. stronger statistical identification;
2. better economic interpretation;
3. cleaner extension into paper-quality analysis.

## Phase 1: tighten the current grouped-data model

Priority: high

Goal:
Make the grouped-data comparison harder to dismiss as a consequence of modeling shortcuts.

Tasks:

- Add bracket-definition metadata by year.
  Reason:
  Some periods use different grouped ranges, and the notebook should make those breaks explicit instead of treating them as a quiet background detail.

- Add sensitivity checks for the top open-ended bracket.
  Reason:
  Open-top handling is one of the main places where grouped fits can become assumption-sensitive.

- Add sensitivity checks for the exact minimum-wage bin.
  Reason:
  The notebook currently excludes it from the continuous likelihood. That is defensible, but it should be shown how much the final conclusions depend on that choice.

- Add likelihood and fit summaries with and without the top tail.
  Reason:
  This would separate “body fit” from “full-distribution fit”, which is central to the Gamma question.

- Add confidence intervals or bootstrap ranges for fitted parameters where computationally feasible.
  Reason:
  Right now the notebook compares point estimates only.

Deliverables:

- new processed output for sensitivity runs;
- one notebook section summarizing whether the model ranking is stable to these choices.

## Phase 2: improve the economic interpretation

Priority: high

Goal:
Make the notebook more useful to a reader who cares about Portuguese labour-market structure, not just abstract fit metrics.

Tasks:

- Add nominal-versus-real comparisons.
  Reason:
  The current notebook correctly warns about nominal drift, but the main plots are still nominal. A real-wage layer would make parameter trends easier to interpret.

- Add explicit interpretation of the minimum-wage regime.
  Reason:
  The minimum wage is not just a nuisance for fitting. It is a structural feature of the distribution.

- Add a wage-compression section.
  Candidate measures:
  decile ratios, median-to-top-decile comparisons, lower-tail versus upper-tail spread.

- Add a “distribution changed where?” section.
  Reason:
  The current notebook shows model errors, but not yet a compact answer to whether the biggest shift over time is in the lower tail, middle, or upper tail.

- Add a timeline overlay for policy-relevant breaks.
  Examples:
  minimum-wage increases, crisis period, recovery, inflation period.

Deliverables:

- 2 to 4 new context plots;
- one notebook section that translates model-fit findings into labour-market interpretation.

## Phase 3: strengthen the tail analysis

Priority: medium-high

Goal:
Handle the upper tail more explicitly, since this is one of the main reasons Gamma underperforms.

Tasks:

- Add a formal tail-only comparison:
  `Lognormal` tail vs `Pareto` tail vs simple splice model.

- Test a two-part model:
  body distribution plus separate tail component.

- Add a threshold sensitivity table for the tail diagnostic.
  Reason:
  A Pareto-style reading should not depend on one arbitrary bracket cutoff.

- Add direct plots of observed versus fitted top-bracket shares by model.

Deliverables:

- one tail-specific diagnostics table;
- one tail model comparison figure;
- one conclusion cell explaining whether the notebook’s main misspecification is primarily tail-driven.

## Phase 4: optional microdata bridge

Priority: medium

Goal:
Keep the notebook publicly reproducible while preparing a path to a stronger research version if anonymized microdata becomes available.

Tasks:

- Define a documented schema for optional local microdata input.
- Mirror the grouped-data model set on microdata.
- Compare grouped-fit conclusions against microdata-fit conclusions for overlapping years.
- Add one notebook section that clearly separates:
  public reproducible evidence vs private extension evidence.

Deliverables:

- microdata schema note;
- optional microdata comparison outputs;
- no dependency of the public notebook on restricted files.

## Phase 5: notebook polish for research use

Priority: medium

Goal:
Make the notebook easier to read, audit, and reuse.

Tasks:

- Add a short “main findings first” section near the top.
- Add a compact table of model winners, decile errors, and tail diagnostics in one place.
- Move repeated plotting helpers into `src/pt_salary_gamma_distribution/`.
- Add a figure index to the README.
- Add a small glossary for grouped likelihood, decile means, and open-top bins.

Deliverables:

- cleaner notebook navigation;
- lower maintenance burden;
- easier handoff to a reader or reviewer.

## Suggested execution order

1. Phase 1
2. Phase 2
3. Phase 3
4. Phase 5
5. Phase 4

Reason:
The strongest next gains come from making the grouped-data conclusions more robust and more interpretable before investing in optional restricted-data extensions.

## Concrete next sprint

If only one short sprint is available, do this:

1. Add minimum-wage-bin sensitivity analysis.
2. Add top-bracket sensitivity analysis.
3. Add real-wage context plots.
4. Add one compact findings table near the top of the notebook.

That would materially improve the notebook without changing its public reproducibility contract.

## Success condition

This notebook is in a strong state when a careful reader can answer all of the following without guesswork:

- what public years are actually covered;
- which model wins and by how much;
- whether Gamma fails mainly in the minimum-wage mass, the body, or the tail;
- whether the conclusion changes under reasonable grouped-data sensitivity choices;
- what the distributional changes mean in labour-market terms, not just statistical terms.
