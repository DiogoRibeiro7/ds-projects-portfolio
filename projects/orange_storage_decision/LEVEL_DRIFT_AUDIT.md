# Why the 52-week orange price forecast differs from a recent quotation

This supplementary, R-only observational diagnostic follows the
[exact 52-week benchmark](SEASONAL_52_BENCHMARK.md), using the same
[official GPP SIMA archive](MARKET_PRICES.md), the same wholesale product,
region and market identity, and the same **2026 out-of-time targets**.
It does not revise the completed [synthetic storage study](README.md).

## A small, explicit question

For a fixed orange product, region and wholesale market, was last year's
*same-week* price already far from the most recent quotation before the
current forecast week began? If so, how much of the prior-year benchmark's
signed error can be accounted for by that **pre-existing level difference**?

This audit answers that *descriptive* question. It does **not** estimate a
stable seasonal effect, identify a cause of year-on-year price changes,
test a trading strategy or prove that storing oranges would pay off.

## Prespecified identity and chronology

For any observed target week \(t\), let:

- \(P_t\) be its final recorded weekly frequent wholesale quote;
- \(P_{t-52}\) be the quotation 364 days earlier, in the exact same panel;
- \(L_{t-1}\) be the latest prior quote accepted by the original forecast's
  eight-week history and maximum-three-week staleness rules.

Then, for every row with all three forecasts:

\[
P_t-P_{t-52}
=\underbrace{L_{t-1}-P_{t-52}}_{\text{pre-origin level gap}}
+\underbrace{P_t-L_{t-1}}_{\text{later price innovation}}.
\]

The decomposition is a mathematical identity for **signed** errors. It is
*not* additive for mean absolute errors, and its two components are not
statistically independent or causally identified. The last recorded quote
may be one, two or three weeks old, so “innovation” does not imply an
observed one-week return.

Every prior-year quote used was dated before the forecast origin. The
original source does not give first-publication and revision timestamps,
so **quotation date is not evidence of actual real-time availability**.

## Like-for-like denominators

The frozen GPP SIMA input contains **3,412** official quotation rows from
2025-01-02 to 2026-10-30. The existing evaluation considers **485**
observed product/market target weeks in January–October 2026.

| Comparison stage | Observed target panel-weeks |
| --- | ---: |
| All actually quoted 2026 targets | 485 |
| Exact 52-week earlier reference exists | 320 |
| Both existing recent-price baselines available | 408 |
| All three methods simultaneously available | **309** |

The audit retains all 485 target rows in its row-level output, marking
unavailable reference weeks and decomposition components as `NA`.
All displayed difference summaries use the **same 309 target rows**. They
are grouped by target quotation month, wholesale market and full original
panel identity. A mean across available rows is *not* a national mean:
sampled product mixes, quotation frequency, wholesale volumes and quoted
periods vary.

The audit reports:

1. Signed absolute year-on-year price difference (EUR/kg) and nominal ratio
   \(P_t/P_{t-52}-1\), before inflation adjustment.
2. Fraction of exact-paired 2026 observations above/below their 2025 quote.
3. Signed and mean absolute pre-origin level gaps.
4. Mean absolute recent-quote innovation, prior-year forecast error and
   four-week-median forecast error on **identical** rows.
5. Month, market and exact-panel support and coverage, separately from
   observed target denominators.

No p-values, IID confidence bounds, spurious inflation attribution or
causal storage-return conclusions are drawn from dependent weekly quotes.

## Reproduce in base R

From `projects/orange_storage_decision`:

```sh
Rscript tests/test_sima_level_drift_audit.R
Rscript scripts/run_sima_level_drift_audit.R \
  data/raw/sima_oranges_wholesale_2025-01-01_2026-10-31.csv.gz \
  results/replay_sima_level_drift_audit
```

The report refuses to overwrite an existing destination. It writes:

- `observed_rows.csv`: all target weeks, dated reference quotes, common
  availability and signed identity components, including `NA` rows.
- `overall.csv`, `coverage.csv`: aggregate metrics and honest denominators.
- `by_quote_month.csv`, `by_market.csv`, `by_exact_panel.csv`: conditional
  descriptive contrasts with each group's observed and common counts.
- `RESULTS.md`, `level_drift.png`, `settings.R`: human-readable evidence,
  labelled chart, exact source checksum and fixed analysis parameters.

The GitHub Actions R job repeats the fixed-source tests and generates a
downloadable `orange-sima-level-drift-audit` artifact. The independent
archived sample, baseline methods and original synthetic study remain intact.

## Interpretation limit

Even a large pre-existing quote gap is not a price gain a grower could
capture. There is no farm-gate conversion, transport, commissions, fruit
quality, recorded storage period, postharvest loss or realised sale amount.
An operational decision would require the separate
[prospective observation protocol](PROTOCOL.md).
