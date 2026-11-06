# Do wholesale orange quotations reward waiting 1–4 weeks?

This follow-up uses the same frozen [official GPP SIMA orange-price
archive](MARKET_PRICES.md), in base R. It moves from the
[forecast baseline](ROLLING_PRICE_BACKTEST.md) and
[52-week price-level audit](LEVEL_DRIFT_AUDIT.md) to **observed ex-post
price differences** across short calendar horizons.

It does *not* test whether storing an individual orange lot is profitable.
The original [synthetic selling-decision case study](README.md) remains
complete and unchanged.

## Research question

For a quote observed in a specific wholesale market, region and original
orange product descriptor in calendar week \(t\), what were the observed
changes to that exact panel's **frequent nominal EUR/kg quotation** in weeks
\(t+1, t+2, t+3, t+4\)?

For prespecified \(h\in\{1,2,3,4\}\), we calculate

\[
\Delta_{t,h} = P_{t+h} - P_t, \qquad
r_{t,h}^{\text{nominal}} = \frac{P_{t+h}}{P_t}-1.
\]

Here each \(P\) is the **last recorded quotation in its calendar week**
for that exact original product, region and wholesale market. A missing
endpoint remains missing: no seasonal imputation, nearest-week search,
substitution of another cultivar, wholesale market pooling or fictitious
sale execution occurs.

Unlike the previous forecast backtests, these are **realised future quote
differences**, which are unknown at the anchor date. They are not
predictions, tradeable returns or a counterfactual sell/hold experiment.

## Right-censoring and comparison populations

The source contains **3,412 wholesale quotation rows** over
2025-01-02–2026-10-30. We restrict anchor weeks to calendar weeks beginning
2026-01-05 or later. There are **485 observed 2026 anchor panel-weeks**.

The last quotation was recorded in the week beginning 2026-10-26. We
conservatively treat this final quoted week as *incomplete*, even though
it contains observations; a horizon is eligible only if its target week
starts by **2026-10-19**. That avoids crediting a right-edge partial week
as fully observed. After eligibility, an exact target-week quotation must
actually exist to form a pair.

| Horizon (weeks) | Observed anchors | Fully eligible endpoints | Exact paired endpoints |
| ---: | ---: | ---: | ---: |
| 1 | 485 | 453 | 433 |
| 2 | 485 | 442 | 403 |
| 3 | 485 | 431 | 373 |
| 4 | 485 | 420 | 343 |

Two complementary estimands prevent an unreported composition shift:

1. **All available pairs:** the observed endpoint sample at each horizon,
   which has a different set of anchor weeks.
2. **Balanced cohort:** the same **343 anchor panel-weeks** for which all
   four future quotes exist. Every horizon uses exactly those anchors.

The balanced cohort is selected by observing future quote availability.
It is suitable for an **ex-post descriptive contrast only**; an actual
grower could not know in advance which quotes would remain available.

Both outputs report mean and median signed price differences, mean
absolute differences, observed 10th/90th percentiles, median nominal
percentage changes, and proportions of higher/lower/unchanged quotes.
Balanced-cohort results are also split by original anchor quotation month,
wholesale market and full exact panel identity, with row counts.

The horizon counts Monday-to-Monday calendar weeks, not exactly \(7h\)
days between the underlying original quotation timestamps. Both dated
observations and their actual elapsed calendar days are exported.

## Reproduce in R

From `projects/orange_storage_decision`:

```sh
Rscript tests/test_sima_forward_quote_windows.R
Rscript scripts/run_sima_forward_quote_windows.R \
  data/raw/sima_oranges_wholesale_2025-01-01_2026-10-31.csv.gz \
  results/replay_sima_forward_quote_windows
```

The destination must not exist, and the report refuses overwrites. Outputs:

- `observed_windows.csv`: all 1,940 anchor/horizon rows, including censored
  and unmatched cases with NA endpoint dates/prices, plus balanced flags.
- `coverage.csv`: explicit horizon-eligibility, pair and common denominators.
- `all_available.csv`, `balanced.csv`: separate, transparently labelled
  conditional price-movement summaries for the two populations.
- `monthly_balanced.csv`, `market_balanced.csv`, `panel_balanced.csv`:
  descriptive summaries on the fixed 343-anchor cohort.
- `RESULTS.md`, `forward_windows.png`, `settings.R`: complete numerical
  explanation, standalone figure, source checksum and fixed parameters.

The existing R GitHub Actions job tests exact-week identity, missing
endpoints, right-edge censoring, two-market separation, invalid inputs
and frozen source regression counts. It uploads an
`orange-sima-forward-windows` evidence artifact.

## Interpret carefully

- Wholesale frequent quotations are not farm-gate revenue, retailer prices
  or any grower's actual selling prices; volumes are not provided.
- The same product descriptor does not identify a single harvested lot,
  preserved biological condition or the amount remaining sellable later.
- There is no observed storage cost, transport, depreciation, inspection,
  deterioration, quality loss, unsold quantity or sale execution.
- An apparent upside is not automatically a profitable decision to wait;
  downside and lost quality can dominate.
- Missing future quotations may reflect ordinary market and cultivar
  seasonality. Ex-post available-pair selection can bias the sample.
- Overlapping weekly observations are dependent; displayed empirical
  percentiles are descriptive, not independent-sample confidence limits.
- Source quote dates are not original publication/revision timestamps.
  No real-time availability guarantee is implied.

To evaluate *net* sell-or-hold decisions with actual fruit, follow the
[prospective observation and validation protocol](PROTOCOL.md).

The separate [illustrative break-even scenario grid](QUOTE_BREAK_EVEN_SCENARIOS.md)
applies explicitly hypothetical saleable-retention and weekly-cost assumptions
to these observed quote pairs. It must not be read as observed storage profit.
