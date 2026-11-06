# Are orange wholesale prices changing, or is the quoted product mix changing?

This R-only empirical audit extends the [frozen GPP SIMA wholesale quotation
snapshot](MARKET_PRICES.md) and the [rolling-origin price
study](ROLLING_PRICE_BACKTEST.md). The prior tests measure the performance of
simple forecasts. This exercise addresses an earlier descriptive question:
**Can an all-products monthly price movement disagree with the movement of
the same quoted orange products in the same market?**

The dataset contains 3,412 original dated market quotations from January 2025
through October 2026, not farm-gate transaction prices or quantities sold.
It has only three wholesale markets. This is not a national price index.

## First establish what constitutes a comparable orange

Each exact `product_descriptor` × `region` × `market` is a separate panel.
The full descriptor includes cultivar, commercial class, calibre, packaging
and origin. It is not valid to pool two distinct descriptors merely because
both are described as oranges.

As in the merged rolling-origin study, a calendar week runs Monday–Sunday and
contributes only its last dated observed quotation. Unquoted calendar weeks
are **not** filled. A weekly observation is assigned to a month/year by its
**actual final quote date**, not by that week's Monday. For example, a quote
dated 2 January 2026 belongs to January 2026 even if its Monday was 29
December 2025. This avoids a year-boundary classification error.

## Three distinct estimates, all explicitly labelled

For each calendar month `m` in January–October, compare 2026 with the
same calendar month in 2025:

1. **All-observed-week pooled mean change:** first take the arithmetic mean
   across *all observed panel-week prices* in each year/month, then form
   `mean_2026 / mean_2025 - 1`. This weights panels by their **observed
   quotation weeks**, not by sales. Product entry, exit and quotation
   frequency may all affect the result.
2. **Matched-panel observed-week mean change:** recompute the preceding
   mean using *only exact panels quoted in both years for that month*.
   Frequencies can still differ between the two years, but a completely
   different product/market can no longer enter one year's average.
3. **Matched equal-panel median-price change:** compute the median of
   available observed weekly prices within each matched panel/month/year.
   For matched panels `g`, take

   `exp(mean(log(median_2026[g] / median_2025[g]))) - 1`.

   Every matched panel has the same weight, regardless of the number of
   quoted weeks. This is a geometric mean **of panel-relative changes**,
   not the growth rate of a pooled national mean.

The three numbers are different *estimands*. Their difference is not a
causal decomposition or a pure composition effect. In particular, the
arithmetic weekly mean and geometric median ratio differ in weighting and
aggregation, even with the same set of panels.

## Why report coverage with every comparison?

The output records for every compared month:

- Distinct panels quoted in 2025 and 2026.
- Exactly matched panels, plus panels observed only in one year.
- Number of quoted panel-weeks in each year.
- Fraction of each year's quoted panel-weeks belonging to the matched set.
- The three nominal price movements, or `NA` when not estimable.

The separately supplied `panel_quotation_spans.csv` records each panel's
first/last quote dates, unique observed weeks, calendar span, and number of
unquoted weeks *inside that span*. It must not be read as a missing-data
quality score: gaps can reflect cultivar seasonality and valid market
non-quotations. No false weekly observations are invented.

Our tests also include a hand-worked reversal: two precisely matched
products become more expensive, but adding low-priced products in the
second year makes the pooled price decline. This is a **test fixture**,
not a claim that the observed GPP data demonstrate that particular
reversal. The actual result is calculated from the archived CSV by R.

## Reproduce with no network or non-base R dependencies

From `projects/orange_storage_decision`:

```sh
Rscript tests/test_sima_market_prices.R
Rscript tests/test_sima_rolling_backtest.R
Rscript tests/test_sima_composition_audit.R

Rscript scripts/run_sima_composition_audit.R \
  data/raw/sima_oranges_wholesale_2025-01-01_2026-10-31.csv.gz \
  results/replay_sima_composition_audit
```

Use a fresh destination; existing output directories are protected.
The runner produces `panel_monthly_quotes.csv`,
`panel_quotation_spans.csv`, `matched_panel_months.csv`,
`monthly_comparison.csv`, a `composition_audit.png` chart,
`RESULTS.md`, and a `settings.R` provenance file with the source MD5.
GitHub Actions independently reruns the R tests and publishes the report
as the `orange-sima-composition-audit` artifact.

## Limits and next evidence

This is a descriptive, nominal, ex-post 2025/2026 comparison. The available
sample cannot explain why the quoted prices moved or distinguish producer
returns from wholesale quotations. It neither reconstructs real historical
publication times (see the [hypothetical availability
stress test](AVAILABILITY_STRESS.md)) nor estimates the value of keeping
harvested fruit in storage. The completed synthetic biological/storage
model and its original results are unchanged.

For a real hold-or-sell evaluation, obtain lot-level grades, mass loss,
inspection histories, logistics and realised prices and costs under the
independent [prospective protocol](PROTOCOL.md).

An [exact 52-week historical-price benchmark](SEASONAL_52_BENCHMARK.md)
uses the same panel identities to evaluate the predictability of quoted
prior-year levels, still without imputation or profit claims.
