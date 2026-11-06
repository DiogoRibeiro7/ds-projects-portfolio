# Missing future quotes: what can be concluded without imputation?

The [exact 1–4 week GPP SIMA wholesale quotation windows](FORWARD_PRICE_WINDOWS.md)
and the [hypothetical break-even grid](QUOTE_BREAK_EVEN_SCENARIOS.md)
have a deliberate observation limit: even after a calendar-week horizon
has elapsed, the same product/region/market may not be quoted that week.
Treating those missing future quotations as zero prices, unchanged prices,
or average price movements would fabricate data.

This R-only analysis instead reports **sharp logical bounds** for
hypothetical break-even frequencies, with no missing-outcome assumptions.
It also separates time maturity from the retrospective selection of
starting quotations that happened to have every future quote available.

## What the observed source establishes

The separately archived 2025–2026 GPP SIMA wholesale extract contains
3,412 recorded quotation rows. Among the 485 observed 2026 starting
product/region/market panel-weeks:

| Elapsed future horizon | Time-eligible anchors | Exact observed endpoints | Missing, despite eligibility | Unidentified share |
| ---: | ---: | ---: | ---: | ---: |
| 1 week | 453 | 433 | 20 | 4.4% |
| 2 weeks | 442 | 403 | 39 | 8.8% |
| 3 weeks | 431 | 373 | 58 | 13.5% |
| 4 weeks | 420 | 343 | 77 | 18.3% |

The *last quoted source week* remains conservatively censored as
potentially incomplete. An anchor with a target in a week that has not
fully elapsed is **ineligible**, not an eligible observation with an
unavailable future price. Conversely, an eligible window without an exact
matching quote remains genuinely missing. No carried-forward price is
permitted.

## Sharp bounds, not assumed distributions

For a specific elapsed horizon, fixed hypothetical weekly saleable
retention `rho` and cost `c`, the prior break-even calculation scores each
**observed** exact quote pair using

\[
A_{t,h}(\rho,c)=\rho^hP_{t+h}-hc-P_t.
\]

Suppose the comparison population contains `N` eligible anchors,
`O` observed exact future quotes, and `M=N-O` unavailable eligible
future quotes. If `S` observed pairs break even, every possible binary
completion of those missing outcomes implies

\[
\underbrace{\frac{S}{N}}_{\text{all missing fail}}
\leq
\text{eligible break-even share}
\leq
\underbrace{\frac{S+M}{N}}_{\text{all missing succeed}}.
\]

Exactly the same construction is applied to *strictly positive*
illustrative differences. The width of each identification interval
is `M/N`, regardless of the scenario's observed success fraction.
It is sharp for unrestricted binary outcomes: no narrower interval
is justified without additional information about the missing prices.

These are **not statistical confidence intervals**; they impose no
independence, missing-at-random model, smoothness restriction or causal
assumption. They do not quantify actual orchard storage profitability.

## Two populations: avoid conditioning on future matching

The code reports bounds for both:

1. `horizon_eligible` — a starting quotation is included if the particular
   future calendar-week horizon has elapsed. This population changes
   with the horizon (453, 442, 431, 420).
2. `common_maturity` — one fixed starting cohort for which **all four**
   horizons have elapsed, even if the archive lacks any of the endpoints.
   This is a time-based cohort, not a data-availability-matched cohort.

The earlier [continuous feasibility frontier](QUOTE_FEASIBILITY_FRONTIER.md)
uses the **343 fully observed** starting quotes to make all four observed
prices directly comparable. That complete-case cohort is useful for
conditional descriptions, but its membership depends on future
availability. It must not be confused with the broader fixed
`common_maturity` population.

Both new cohorts are still *retrospectively* determined by the frozen
archive's closing date. Neither is an implementable historical policy
or a claim that a grower had advance knowledge of quotation availability.

## Fixed hypothetical scenarios, transparent arithmetic

The 16 original illustration-only scenarios remain unchanged:

| Assumption | Fixed grid |
| --- | --- |
| Weekly retained-saleable fraction | 1, 0.99, 0.98, 0.95 |
| Cost per original kilogram per week (EUR) | 0, 0.005, 0.010, 0.020 |
| Exact observed horizons (weeks) | 1, 2, 3, 4 |

No actual loss, cost or lot-level sales observation exists in the
source. Bounds describe only whether the *hypothetical wholesale quote
difference* could have a nonnegative or strictly positive sign.

The report includes:

- `bounds.csv`: all scenario/cohort/horizon lower and upper frequencies,
  exact observed successes, missing endpoints and observed-only fractions.
- `market_bounds.csv`: market-specific numerators and denominators,
  with zero-denominator markets excluded from ratio calculations.
- `coverage.csv` and `market_coverage.csv`: source eligibility, observed
  and missing counts independent of hypothetical assumptions.
- `missing_eligible_windows.csv`: dates, products and markets of the
  observed starting quotations whose elapsed future quote is absent.
  No target price is invented.
- `scenarios.csv`, `settings.R`, `RESULTS.md` and
  `missing_quote_bounds.png`: a reproducible manifest, report and plot.

Tests compare the observed-only values to the direct prior break-even
grid, verify denominators and market partitions, test known flat and
missing-week synthetic paths, and assert interval/monotonicity identities.

## Reproduce in base R

Run from `projects/orange_storage_decision`:

```sh
Rscript tests/test_sima_missing_quote_bounds.R
Rscript scripts/run_sima_missing_quote_bounds.R \
  data/raw/sima_oranges_wholesale_2025-01-01_2026-10-31.csv.gz \
  results/replay_sima_missing_quote_bounds
```

The destination must not exist. CI uploads an evidence artifact named
`orange-sima-missing-quote-bounds`.

## Scientific limits

Wholesale quotations are not the realised prices of fruit from the
same stored lot. The archive provides no postharvest mass retention,
actual quality, storage cost, inventory flow, farmer sale execution,
commission or net revenue. Dates of quotations are not verified
first-publication timestamps. Overlapping panel-week records are
dependent, and the missing quotes could be missing systematically.

Consequently, *even the best-case upper bound is not evidence of
profitable storage*. The bounds make uncertainty about **missing
quotations** explicit; they cannot eliminate uncertainty about the
entirely unmeasured physical and economic storage process.

The completed synthetic orange-storage decision study is unchanged.
Any field decision still requires the [prospective observation and
validation protocol](PROTOCOL.md).
