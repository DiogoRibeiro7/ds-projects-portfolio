# Continuous feasibility frontiers for hypothetical orange storage

This follow-up inverts the [fixed, hypothetical break-even scenario
grid](QUOTE_BREAK_EVEN_SCENARIOS.md) for each **observed** 2026 GPP SIMA
wholesale orange quote pair. Instead of asking whether a small set of
assumed losses and costs breaks even, it answers two related conditional
questions:

1. *At a chosen hypothetical fraction of saleable fruit retained each week,
   what is the largest weekly cost consistent with the later wholesale quote?*
2. *At a chosen hypothetical storage cost per original kilogram, how much
   weekly fruit loss could the later quote theoretically accommodate?*

These are **algebraic quotation thresholds**. They are not empirical storage
losses, cost estimates, financial returns, observed farm decisions, or
evidence that the original lot could be sold at either quoted price.

## Equations and interpretation

Let \(P_t\) be the wholesale quote for a product, region and market panel
in an observed 2026 calendar week, and \(P_{t+h}\) the quote for the
**same exact panel** \(h\in\{1,2,3,4\}\) weeks later. Assume, strictly
for a sensitivity exercise:

- \(0<\rho\leq1\): *weekly* retained-saleable mass fraction;
- \(c\geq0\): *weekly* cost in EUR per **original** kilogram;
- a fixed product grade and the ability to execute at the two quotes.

The illustrative difference per original kilogram is

\[
A(\rho,c)=\rho^hP_{t+h}-hc-P_t.
\]

Holding either assumed parameter fixed gives the exact thresholds

\[
c_{\max}(\rho)=\frac{\rho^h P_{t+h}-P_t}{h},\qquad
\rho_{\min}(c)=\left(\frac{P_t+hc}{P_{t+h}}\right)^{1/h}.
\]

The raw *maximum tolerable weekly saleable-mass loss fraction* is

\[
\ell_{\max}(c)=1-\rho_{\min}(c).
\]

There is a crucial interpretation distinction:

- If \(c_{\max}<0\), **no nonnegative weekly cost** can break even at
  that assumed retention and observed future quote.
- If \(\rho_{\min}>1\), **no physically permissible weekly retention**
  (at or below 100%) can break even at that assumed cost and future quote.
- A threshold of zero cost or 100% retention is mere equality, not a
  positive margin for deterioration or expense.
- Negative ceilings and negative loss allowances are deliberately
  **not truncated to zero**; clipping would conceal infeasible quote pairs.

The implementation uses the same predeclared numerical equality tolerance
of \(10^{-10}\) EUR/original kg as the preceding grid. Every recovered
break-even frequency is checked against the direct grid computation.

## Frozen inputs and comparability

The source is the separately archived GPP SIMA wholesale quote extract:
`data/raw/sima_oranges_wholesale_2025-01-01_2026-10-31.csv.gz`, containing
3,412 recorded observations. It is not a lot-level storage dataset.

The pairing applies the [exact forward quotation-window
protocol](FORWARD_PRICE_WINDOWS.md):

| Horizon | Eligible observed anchors | Exactly quoted endpoint pairs |
| ---: | ---: | ---: |
| 1 calendar week | 453 | 433 |
| 2 calendar weeks | 442 | 403 |
| 3 calendar weeks | 431 | 373 |
| 4 calendar weeks | 420 | 343 |

Every balanced comparison uses the **same 343 starting panel-weeks** with
all four exact future quotations. The right-censored last source week is
excluded; unavailable targets remain missing. A separate all-available
report exposes how the larger sample changes with horizon.

To make the frontiers comparable to the existing exercise, fixed
illustrative parameter values are unchanged:

| Inverted threshold | Hypothetical parameter held fixed |
| --- | --- |
| Maximum EUR/original kg per week | Retained-saleable fraction \(\rho\): 1.00, 0.99, 0.98, 0.95 |
| Minimum weekly retention / maximum loss | Weekly cost \(c\): 0, 0.005, 0.010, 0.020 EUR/original kg |

Neither the parameter values nor their implied frontiers are independently
validated physical or accounting quantities. The future wholesale
quotation is an *ex post* observation, and common-anchor membership is
also only identifiable with hindsight.

## Outputs and validation

The offline, base-R script produces:

- `cost_ceiling_pairs.csv`: raw signed maximum costs for every observed
  pair and assumed retained mass fraction, including market, product,
  exact quote dates and common-cohort flag.
- `retention_floor_pairs.csv`: required weekly saleable retention and raw
  maximum weekly loss at each fixed hypothetical weekly cost.
- `cost_ceiling_balanced.csv`, `retention_floor_balanced.csv`: conditional
  fractions feasible at zero cost / perfect retention, positive cost/loss
  room, median boundaries and raw distribution quantiles.
- Two `*_all_available.csv` reports to reveal horizon-availability
  changes; two `*_market_balanced.csv` reports to reveal market mixture.
- `coverage.csv`, `assumed_grid.csv`, `settings.R`, `RESULTS.md` and the
  two-panel `quote_frontier.png` figure.

Tests cover a flat quoted-price path (zero room), a known rising path
(positive room), missing future quotations, invalid parameter inputs,
the frozen official archive and exact agreement with **all 16** prior
grid assumptions across **all four** horizons. The frontier and direct
break-even computations must agree on the 343 common starting quotes.

To regenerate from `projects/orange_storage_decision`:

```sh
Rscript tests/test_sima_quote_feasibility_frontier.R
Rscript scripts/run_sima_quote_feasibility_frontier.R \
  data/raw/sima_oranges_wholesale_2025-01-01_2026-10-31.csv.gz \
  results/replay_sima_quote_feasibility_frontier
```

The output directory must not already exist. CI retains a named artifact
`orange-sima-quote-feasibility-frontier` for independent inspection.

## Limits on conclusions

GPP wholesale quotes are not producer prices, observed sale proceeds or
actual storage profits. The archived data contain no lot identity,
original mass, maturity, measured defects, fruit grading transitions,
inventory loss, energy bill, handling charge, transport or commissions.
The quote date is also not verified as a publication timestamp.

Moreover, overlapping weekly quotes within products and markets are
statistically dependent; the balanced cohort is selected by future data
availability. These are descriptive conditional boundary distributions,
**not independent probability estimates or prospective decision rules**.

The original synthetic orange-storage model, conclusions and release
artifacts remain unchanged. Operational recommendations still require
the [prospective fruit and price observation protocol](PROTOCOL.md).
