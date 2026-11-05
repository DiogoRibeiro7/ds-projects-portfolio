# Observation-date-limited orange-price forecasting backtest

This is the first empirical **forecast evaluation** following the
[official GPP SIMA ingestion](MARKET_PRICES.md). It stays separate from the
completed 14-study synthetic postharvest decision analysis. The objective is
to measure what two transparent, low-complexity forecasts can predict from
historical market quotations. It is **not** a claim about realised producer
revenues or the optimal storage duration for any real batch.

## Why grade- and market-specific panels matter

SIMA's 3,412 orange quotations span multiple markets, cultivars, classes,
sizes, origins and packaging descriptions. A raw average would confound
price changes with changes in the composition of the quoted products.
Accordingly a panel is identified by the full `product_descriptor`, `region`
and `market`, restricted to the verified `wholesale`/`EUR/kg` source.

An observed calendar week runs Monday–Sunday. When a panel has several dated
quotations within a week, only the **last dated quotation** defines that
week's target. The number of original rows is retained as
`quotes_in_week`. Missing weeks are **not** interpolated, carried forward
as synthetic observations or assigned a zero price. The SIMA source week
is preserved at ingestion, while the backtest uses actual `quote_date` to
compute Monday boundaries consistently across years.

This rule sacrifices some sample size to keep comparisons like-for-like.
Some panels are active only during their cultivar's commercial season, so
missing-week coverage is itself an important result.

## Predeclared prospective-looking design

| Design field | Fixed rule |
| --- | --- |
| Target | Final observed frequent EUR/kg quotation for a panel in week *t* |
| Forecast origin | Sunday at the end of week *t−1* |
| First test week | Monday, **5 January 2026** |
| Training | Only observed weeks strictly before target week *t* |
| Minimum history | At least 8 previously observed weekly quotations |
| Staleness limit | Latest quotation no more than 3 calendar weeks old |
| Benchmark A | Last observed historical weekly quotation |
| Benchmark B | Median of observed weekly quotations from the prior 4 calendar weeks |
| Median coverage | At least 2 observations in those previous 4 weeks |
| Evaluation subset | Same target rows wherever both benchmarks exist |
| Excluded source rows | Explicitly provisional, if identified; unknown remains unknown |

Both benchmarks are fixed before the 2026 outcomes are examined. There is
no selection of panels using 2026 forecast errors or ranking by future
coverage. Every panel with a recorded target week enters the output, even
if it lacks enough history to forecast. The denominator for *paired coverage*
is **observed 2026 panel–weeks**, not all calendar panel–weeks; absent
quotation dates cannot be evaluated.

For each test week *t* and panel *g*, let `H(g,t)` consist exclusively of
observed weekly quotations before *t*. If `H(g,t)` contains enough observations
and its last price is recent enough, then

\[
\widehat p^{\mathrm{last}}_{g,t}=p_{g,s},\quad
s=\max\{u<t:\text{week }u\text{ is observed for panel }g\}.
\]

The alternative is the median of observed `H(g,t)` prices dated in the four
immediately preceding calendar weeks. It is unavailable when fewer than two
such weeks are observed. Crucially, neither rule learns from the target
week's last quotation, from later weeks or from a future-selected training
window. There is no interpolation of unobserved weeks.

## Error metrics, coverage and interpretation

On paired target rows, the R runner reports:

- Mean absolute error (MAE) in EUR/kg, weighted by observed target rows.
- Equal-panel MAE: calculate MAE within each scored panel, then give every
  panel equal weight regardless of how many target weeks it contributes.
- Weighted absolute percentage error (WAPE): total absolute error divided
  by the sum of observed target prices; this is a descriptive ratio, not a
  confidence interval.
- A panel-level table, number of scored targets and paired-coverage rate.

The runner intentionally does **not** attach conventional independent-row
95% intervals. Consecutive prices and multiple grades within the same
market are dependent; they are not independent replicated experiments.
No causal effect, algorithmic superiority or trading profit follows from
a difference in the reported errors.

### The remaining look-ahead limitation

The available metadata provide the *quoted observation date*, not a confirmed
first-publication timestamp or a revision log. The split prevents leakage
by quotation date, but cannot prove that the last historical quote was
actually accessible to a real grower by the forecast origin. The experiment
is therefore an **observation-date-limited pseudo-prospective backtest**, not
a point-in-time reconstructed trading strategy. Such reconstruction would
require a publication/revision audit of the SIMA source.

Also, even a perfectly forecast wholesale price is not the farm-gate price
for a particular shipment. There are no linked actual sale prices, storage
losses, quality inspections, shipping costs or contractual constraints.
The [original study's prospective protocol](PROTOCOL.md) remains necessary
before evaluating an actual sell-or-hold policy.

## Reproduce the empirical evidence with R

From `projects/orange_storage_decision`:

```sh
Rscript tests/test_sima_market_prices.R
Rscript tests/test_sima_rolling_backtest.R

# Generate the complete observation-based evaluation in a new directory.
Rscript scripts/run_sima_rolling_backtest.R \
  data/raw/sima_oranges_wholesale_2025-01-01_2026-10-31.csv.gz \
  results/replay_sima_rolling_backtest
```

The command refuses to overwrite an existing output directory. It writes:

- `weekly_panels.csv`: one observed quote per panel/week, with raw counts.
- `rolling_forecasts.csv`: dated origins, current targets, last historical
  quote dates, coverage checks and the two forecast errors, including NAs.
- `summary.csv`, `panel_summary.csv`, `coverage.csv`: paired descriptive errors.
- `RESULTS.md` and `rolling_price_errors.png`: a Medium-ready numerical
  summary and two-panel figure generated from the actual observed scores.
- `settings.R`: fixed evaluation window, estimators, source checksum and R
  version for reproducibility.

The GitHub Actions workflow independently runs both the real-data tests and
a disposable end-to-end report generation, uploading the evidence as the
`orange-sima-rolling-backtest` artifact for inspection. The original synthetic
results and source snapshot are never overwritten by the backtest runner.

Future work can compare a 52-week seasonal benchmark once enough matched
years are observed, test publication lags, and study product-level changes
in origin mix and missingness. Any decision-value comparison still requires
real lot-level observations and sales outcomes.
