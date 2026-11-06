# Offline tests for exact-panel, exact-week observed future quotations.
source("R/sima_market_prices.R")
source("R/sima_rolling_backtest.R")
source("R/sima_composition_audit.R")
source("R/sima_forward_quote_windows.R")

expect_error <- function(expression) {
  stopifnot(tryCatch({ force(expression); FALSE }, error = function(e) TRUE))
}

# Twelve consecutive Monday quotations, with the final calendar week
# intentionally excluded as incomplete regardless of its existing quote.
dates <- as.Date("2026-01-05") + 7L * 0:11
prices <- as.numeric(seq_along(dates))
fixture <- data.frame(
  quote_date = dates, product_descriptor = "Laranja*Newhall*II*Caixa*Algarve*EUR/kg",
  region = "Beira Litoral", market = "MAC", market_stage = "wholesale",
  price_unit = "EUR/kg", min_eur_per_kg = prices - 0.1,
  max_eur_per_kg = prices + 0.1, frequent_eur_per_kg = prices,
  is_provisional = rep(NA, length(dates)), stringsAsFactors = FALSE
)
weekly <- sima_weekly_panels(fixture)
pairs <- pair_sima_forward_quotes(weekly, horizons = 1:3)
scores <- summarize_sima_forward_windows(pairs)
stopifnot(nrow(pairs) == 36L,
          identical(scores$coverage$observed_anchor_quotes, c(12L, 12L, 12L)),
          identical(scores$coverage$complete_horizon_eligible, c(10L, 9L, 8L)),
          identical(scores$coverage$exact_endpoint_observed, c(10L, 9L, 8L)),
          identical(scores$coverage$balanced_all_horizon_anchors,
                    c(8L, 8L, 8L)),
          all(scores$all_available$mean_signed_change_eur_per_kg == 1:3),
          all(scores$balanced$mean_signed_change_eur_per_kg == 1:3),
          all(scores$balanced$mean_absolute_change_eur_per_kg == 1:3),
          all(scores$balanced$share_price_higher == 1),
          all(scores$balanced$share_price_lower == 0),
          all(pairs$actual_quote_days_apart[pairs$endpoint_observed] %in%
                c(7L, 14L, 21L)),
          nrow(scores$market_balanced) == 3L,
          nrow(scores$panel_balanced) == 3L)

# An observed quote in the last source week must not be mistaken for a
# complete future observation. Each horizon keeps all 12 anchor rows.
last_anchor <- pairs[pairs$anchor_week_start == max(weekly$week_start), ]
stopifnot(nrow(last_anchor) == 3L,
          all(!last_anchor$horizon_eligible),
          all(!last_anchor$endpoint_observed),
          all(is.na(last_anchor$endpoint_eur_per_kg)))

# Removing one endpoint must retain its anchor and mark the exact match NA.
# No neighboring quote is carried forward or borrowed from another market.
missing <- fixture[fixture$quote_date != as.Date("2026-01-26"), , drop = FALSE]
gap <- pair_sima_forward_quotes(sima_weekly_panels(missing), horizons = 1:3)
affected <- gap[gap$anchor_week_start == as.Date("2026-01-19") &
                  gap$horizon_weeks == 1L, , drop = FALSE]
stopifnot(nrow(affected) == 1L, affected$horizon_eligible,
          !affected$endpoint_observed,
          is.na(affected$endpoint_quote_date),
          is.na(affected$signed_quote_change_eur_per_kg),
          sum(gap$balanced_all_horizons) < sum(pairs$balanced_all_horizons))

# The same grade in another market is never used as an endpoint substitute.
other <- fixture
other$market <- "MARL"
other$frequent_eur_per_kg <- prices + 100
other$min_eur_per_kg <- prices + 99.9
other$max_eur_per_kg <- prices + 100.1
two_markets <- summarize_sima_forward_windows(pair_sima_forward_quotes(
  sima_weekly_panels(rbind(fixture, other)), horizons = 1:3
))
stopifnot(nrow(two_markets$market_balanced) == 6L,
          nrow(two_markets$panel_balanced) == 6L,
          all(two_markets$market_balanced$mean_signed_change_eur_per_kg %in%
                1:3),
          two_markets$coverage$balanced_all_horizon_anchors[1L] == 16L)

expect_error(pair_sima_forward_quotes(weekly, horizons = c(1, 1)))
expect_error(pair_sima_forward_quotes(weekly, horizons = 0))
expect_error(pair_sima_forward_quotes(weekly, horizons = 53))
expect_error(pair_sima_forward_quotes(weekly,
                                      anchor_start = as.Date("2026-01-06")))
expect_error(pair_sima_forward_quotes(weekly,
                                      anchor_start = as.Date("2027-01-04")))
expect_error(pair_sima_forward_quotes(rbind(weekly, weekly[1L, ])))
expect_error(summarize_sima_forward_windows(pairs[FALSE, , drop = FALSE]))

# The official archive is immutable. Fixed counts verify right-censoring,
# unavailable exact matches and composition-balanced horizon denominators.
raw <- read_sima_orange_prices(
  "data/raw/sima_oranges_wholesale_2025-01-01_2026-10-31.csv.gz",
  market_stage = "wholesale"
)
official <- pair_sima_forward_quotes(sima_weekly_panels(raw))
evidence <- summarize_sima_forward_windows(official)
stopifnot(nrow(official) == 485L * 4L,
          identical(evidence$coverage$observed_anchor_quotes,
                    rep(485L, 4L)),
          identical(evidence$coverage$complete_horizon_eligible,
                    c(453L, 442L, 431L, 420L)),
          identical(evidence$coverage$exact_endpoint_observed,
                    c(433L, 403L, 373L, 343L)),
          identical(evidence$coverage$balanced_all_horizon_anchors,
                    rep(343L, 4L)),
          all(evidence$balanced$observed_pairs == 343L),
          sum(evidence$market_balanced$observed_pairs) == 343L * 4L,
          all(is.na(official$endpoint_eur_per_kg[!official$endpoint_observed])),
          all(official$endpoint_quote_date[official$endpoint_observed] >
                official$anchor_quote_date[official$endpoint_observed]),
          all(abs(evidence$balanced$share_price_higher +
                evidence$balanced$share_price_lower +
                evidence$balanced$share_price_unchanged - 1) < 1e-12))

cat("SIMA forward quote-window horizon tests passed.\n")
