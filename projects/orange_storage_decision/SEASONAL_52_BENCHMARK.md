# Can last year's orange price forecast this week's price?

This is a deliberately simple **negative-control benchmark** for the
[official GPP SIMA orange quotations](MARKET_PRICES.md). The previously
completed [rolling-origin study](ROLLING_PRICE_BACKTEST.md) compared recent
price persistence against a short-window median. The [product-mix
audit](COMPOSITION_AUDIT.md) showed why it is unsafe to pool cultivars,
grades, origins and markets. We now compare those recent-price benchmarks
with an exact **52-week prior-year price**.

The purpose is to ask whether last year's weekly prices offer a plausible
forecast for the same orange product in the same wholesale market. It is
not to discover a profitable storage rule, estimate a seasonal causal
effect or prove that one calendar year establishes seasonality.

## Fixed chronological rule

The frozen source contains 3,412 dated orange wholesale quotations between
January 2025 and October 2026. As in the earlier studies, we select each
exact product-descriptor × region × market panel's **final observed quote**
in a Monday–Sunday calendar week, without filling any unquoted weeks.

For every observed target panel/week *t* from 5 January 2026 onwards:

- Forecast origin: preceding Sunday.
- Exact prior-year reference: **week *t−52***, meaning Monday 364 days
  before the target week's Monday.
- Forecast: that prior year's observed frequent EUR/kg quotation.
- Missing reference week: mark the forecast `NA`. Do not borrow adjacent
  weeks, another market or another orange variety.
- Compare against the existing last-observed-quote and median-of-previous-
  four-calendar-weeks rules. Retain their original eight-week history
  minimum, two recent observations, and maximum three-week age.
- Score errors on **exactly the same target observations where all three
  forecasts are available**. Report coverage before discussing errors.

Because the 52-week reference is far before the origin, it cannot leak
the observed 2026 target price by quote date. However, the GPP export
still lacks the actual first-publication and revision timestamps, so this
is an *observation-date-limited* pseudo-prospective evaluation rather
than proof of information that was available to a real grower at the time.

## Coverage is an empirical result

The source-level regression check expects:

| Coverage description | Observed 2026 panel/weeks |
| --- | ---: |
| Original target quotations | 485 |
| Exact same-product, same-market previous-year quote | 320 |
| Both original short-horizon forecast rules | 408 |
| All three methods simultaneously available | 309 |

These counts are determined by the immutable archived source and fixed
rules, not selected according to 2026 forecast accuracy. The complete
R runner reports both observation-weighted and equal-panel MAE and WAPE
on the **309 three-way matched targets**, plus month-by-month coverage
and separate panel scores. An unquoted year-ago week is not a price of
zero or a failed trade.

This diagnostic should be interpreted cautiously. A twelve-month
quotation difference includes level changes in market prices, changes
in supply, quality and other economic conditions. One 2025–2026
comparison cannot identify a stable seasonal pattern. A large seasonal
naive error would be evidence against blindly reusing prior-year
quoted levels, not evidence that oranges should or should not be stored.

## Reproduce in base R

Run from `projects/orange_storage_decision`:

```sh
Rscript tests/test_sima_market_prices.R
Rscript tests/test_sima_rolling_backtest.R
Rscript tests/test_sima_seasonal_benchmark.R

Rscript scripts/run_sima_seasonal_benchmark.R \
  data/raw/sima_oranges_wholesale_2025-01-01_2026-10-31.csv.gz \
  results/replay_sima_seasonal_benchmark
```

The runner refuses to overwrite an output directory and produces:

- `seasonal_forecasts.csv`: every observed 2026 target, exact reference
  week/date, seasonal and recent forecasts, including unavailable rows.
- `summary.csv`: three-way paired errors with observation- and panel-
  weighted views.
- `monthly.csv` and `panel.csv`: concentration of matched coverage
  and errors across time and exact product/market panels.
- `coverage.csv`: honest baseline, seasonal and three-way denominators.
- `RESULTS.md`, `seasonal_mae.png` and `settings.R`: numerical report,
  evidence graphic, source checksum and frozen analysis parameters.

GitHub Actions runs the real-source regression and CLI smoke test,
publishing a separate `orange-sima-seasonal-52-backtest` artifact.
The existing synthetic storage study and its historical results remain
unchanged.

## Remaining research boundaries

The archive contains no actual lot-level maturity, quality inspections,
shelf-life losses, transport costs, commissions, quoted-to-realised
price bridge or producer revenues. Therefore this price forecast exercise
cannot validate any real sale-or-hold policy. Such a policy requires the
[prospective collection protocol](PROTOCOL.md), and a credible
full seasonal model would need more independently observed years.

The separate [level-drift audit](LEVEL_DRIFT_AUDIT.md) investigates the
signed price-gap identity behind the weak prior-year forecast, using exactly
the same common target observations without fitting a seasonal model.
