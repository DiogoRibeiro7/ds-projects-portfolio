# How much does assumed quotation availability change orange-price forecasts?

This reproducible R study extends the [first GPP SIMA rolling-origin
benchmark](ROLLING_PRICE_BACKTEST.md). The previous study establishes what
could be predicted **if** all prices dated before each forecast origin were
already public. But the archived GPP export does not contain timestamps of
first publication, corrections or later revisions. Consequently, the earlier
date-based replay cannot establish what a real grower would have known.

This extension addresses a narrower, empirically testable sensitivity
question: *If the original dated quotations had become available after an
assumed delay of several calendar days, how would the coverage and price
errors of our two fixed benchmarks change?* The delays are scenarios, **not
estimates of GPP's real publication practice**.

## Evidence and immutable boundaries

- The source is the previously validated, [frozen 3,412-row GPP SIMA
  wholesale snapshot](MARKET_PRICES.md), covering observations between
  January 2025 and October 2026. It is not a transaction ledger.
- Exact `product_descriptor` × `region` × `market` panels are kept separate.
  Cultivars, grades, calibre, packing and origin are never blended.
- The target remains the **final dated** quotation recorded in an observed
  Monday–Sunday target week. Missing target weeks remain missing.
- Only the historical information set changes. The synthetic postharvest
  simulations, their decisions, results and biological assumptions are not
  modified or empirically validated by this exercise.

## Prespecified information-release scenarios

| Quantity | Fixed choice |
| --- | --- |
| First held-out week | Monday 5 January 2026 |
| Forecast origin for target week *t* | Sunday of week *t−1* |
| Assumed days from quotation date to availability | **0, 3, 7, 14, 21** |
| Information eligibility | `quote_date + lag_days <= origin_end` |
| Minimum training history | 8 observed historical calendar weeks |
| Most recent available quote age | No more than 3 weeks |
| First benchmark | Last available dated weekly quotation |
| Second benchmark | Median of preceding four calendar weeks' available weekly quotes |
| Median requirement | At least 2 available observed weeks within that window |
| Comparison | Identical scored targets within each lag; matched targets vs zero-lag |

For each target and hypothetical delay, the code returns to **the raw daily
quotation rows**, filters on the assumed as-of cutoff, and then retains the
latest *available* dated quotation per previous calendar week. This differs
from merely removing the final quote of a week when that quote is too recent:
the earlier quote of the same week may still have become available. It also
means that zero-day results must reproduce the earlier weekly benchmark
exactly; the source-level integration test enforces this invariant.

The test checks two kinds of look-ahead failure:

1. Future target prices and later observations must not change a forecast
   issued before the target week.
2. An unavailable Friday quotation cannot replace an eligible Monday
   quotation from the same previous week merely because Friday is the
   last dated observation in the final downloaded dataset.

## What is compared, and why denominators matter

`scenario_summary.csv` reports each scenario's number of observed targets,
number and fraction of **paired scored** targets, number of scored panels,
and two forms of MAE. A scored target requires **both** fixed benchmarks to
be computable. The target denominator is unchanged across release delays.

But marginal MAE can improve when an assumed lag simply removes difficult
to predict weeks or product/market pairs. We therefore additionally report
`matched_error_comparison.csv`: for each delay, compare forecast errors
with the **zero-delay baseline on exactly the target rows that both
scenarios can score**. If a severe lag leaves no paired targets, coverage is
reported as zero and MAE as **missing**, not as zero error.

This is a descriptive stress test with dependent time observations and
correlated product/market groups. The differences should not be interpreted
as statistically significant causal effects. More importantly, a measured
publication delay or full historical revision archive is still required to
replace any scenario with a genuine point-in-time replay.

## Reproduce with base R

Run from `projects/orange_storage_decision`:

```sh
Rscript tests/test_sima_market_prices.R
Rscript tests/test_sima_rolling_backtest.R
Rscript tests/test_sima_availability_stress.R

# Use a new destination; the runner refuses to overwrite earlier evidence.
Rscript scripts/run_sima_availability_stress.R \
  data/raw/sima_oranges_wholesale_2025-01-01_2026-10-31.csv.gz \
  results/replay_sima_availability_stress
```

No external R packages or network access are needed. The runner writes:

- `scenario_forecasts.csv`: as-of cutoffs, observed targets, histories,
  eligibility and both predictions/errors under every delay.
- `scenario_summary.csv`: full target denominators, paired coverage and
  per-scenario MAE and equal-panel MAE.
- `matched_error_comparison.csv`: apples-to-apples forecast errors against
  zero delay on the same target panel/weeks.
- `RESULTS.md` and `availability_coverage.png`: report and visual evidence.
- `settings.R`: frozen parameters, source checksum and R version.

The GitHub Actions workflow reruns the source-level tests and produces a
separate `orange-sima-availability-stress` evidence artifact. Nothing from
the completed synthetic storage studies is overwritten.

## What remains before any real storage recommendation

Even a perfect audit of quotation publication dates would **not** establish
whether storing oranges was profitable. SIMA wholesale quotations do not
record a grower's realised sale prices, contractual opportunities,
transport and market fees, quality losses, lot-level deterioration or
inspection histories. Those require [prospective lot-level data collection
and independent validation](PROTOCOL.md).
