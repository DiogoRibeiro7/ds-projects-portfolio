# Prospective protocol: inspection-driven orange sales

## Target decision

At each decision day, recommend one common sale day for the remaining
fresh-market-eligible inventory, or separate sale days when grades can be
physically segregated. The quality requirement applies at **delivery** and
is determined by market acceptance criteria independently of the model.

## Unit of analysis and sampling frame

- Harvest lot: cultivar, orchard block, harvest date, packing history and
  storage environment recorded before outcomes are observed.
- Predefined stratum: a maturity/condition category established at harvest.
  Colour is not an adequate substitute for internal quality assessment.
- Fruit inspection: sample identifiers, stratum, inspection day, random
  selection method, soluble solids, titratable acidity, firmness, defects,
  commercial pass/fail outcome and reason for failure.
- Cross-sectional design: draw fresh, representative random samples within
  each stratum at each inspection day. If fruit is inspected repeatedly,
  use a longitudinal survival model instead of the independent-binomial
  likelihood currently implemented.
- Destructive sampling: record and subtract removed fruit from sellable
  inventory; the prototype assumes inspection removals are negligible.
- Mass accounting: record fruit mass in inspection samples, because multiplying
  acceptable-fruit proportions by inventory kilograms assumes no relationship
  between fruit size and loss probability within a cohort.
- Storage: daily temperature/humidity traces, handling, handling damage and
  any removal/disposal events. Record both sale date and actual delivery date.

## Data contract

An aggregated inspection table uses:

| Field | Unit | Constraint |
|---|---|---|
| `cohort` | category | Preassigned at harvest; never future-defined |
| `day` | days since harvest | Positive whole day, unique per cohort/day |
| `n_checked` | fruit | Positive integer |
| `n_acceptable` | fruit | Integer from zero to `n_checked` |

The initial inventory records `cohort`, `mass_kg`, and
`initially_sellable`. The price scenario records a day-by-cohort gross price
in EUR/kg. Weibull `shape` and `scale` are positive and fitted from historical
inspections; placeholder values in the demonstrator are not real estimates.

## Fitting and evaluation without leakage

1. Define grading criteria and strata using information available at harvest.
2. Choose an inspection plan before observing subsequent deterioration.
3. At decision day \(d_0\), fit the survival model using inspections with
   `day <= d_0` only; never use subsequent losses to classify cohorts.
4. Calibrate/compare models with harvest-lot-level held-out validation. Random
   rows from the same lot do not constitute an independent test set.
5. Compare observed acceptable fractions with predicted survival using
   calibration plots, proper scores and uncertainty coverage across lots.
6. Assess the decision against prespecified baselines: sell immediately,
   fixed storage-duration policy, and pooled (unstratified) survival policy.
7. Evaluate realised net return including unsold inventory, storage,
   handling, delivery rejection, disposal and processing revenue, as
   applicable. Do not treat synthetic optimal expected value as realised
   economic improvement.

## Baseline model and sensitivity plan

- Primary prototype: stratum-specific Weibull survival with independent
  binomial inspections; fixed deterministic price path.
- First synthetic comparator implemented: mass-standardised pooled Weibull,
  paired economic decisions, held-out inspections at future days and
  harvest-lot bootstrap intervals. See [EVALUATION.md](EVALUATION.md).
- Further survival sensitivity: alternative piecewise/discrete hazards and
  misspecified-generating-process experiments.
- Sampling sensitivity: misclassification, inspection-frequency changes and
  between-fruit/within-lot dependence.
- Economic sensitivity: price-path errors, delivery delays, per-kg holding
  cost, grade-specific prices, transaction costs and buyer capacities.
- Decision uncertainty: bootstrap by *harvest lot* when empirical replicated
  lots exist, plus price/process scenario simulation. The prototype's
  parametric binomial bootstrap does not substitute for these tests.

## Release criteria for an empirical claim

Do not promote this study as validated until source provenance and acceptance
criteria are documented, tests pass, holdout predictions are calibrated and
observed decisions outperform prespecified operational benchmarks under a
transparent cost ledger. Negative findings must be preserved.
