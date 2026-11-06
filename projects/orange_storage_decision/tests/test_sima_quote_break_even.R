# Offline test of purely hypothetical retention/cost applied to SIMA quotes.
source("R/sima_market_prices.R")
source("R/sima_rolling_backtest.R")
source("R/sima_composition_audit.R")
source("R/sima_forward_quote_windows.R")
source("R/sima_quote_break_even.R")

expect_error <- function(expression) {
  stopifnot(tryCatch({ force(expression); FALSE }, error = function(e) TRUE))
}

# A flat quoted-price path makes the no-friction equality case visible.
# It is an accounting fixture, not a claim that oranges incur no decay.
dates <- as.Date("2026-01-05") + 7L * 0:11
fixture <- data.frame(
  quote_date = dates,
  product_descriptor = "Laranja*Newhall*II*Caixa*Algarve*EUR/kg",
  region = "Beira Litoral", market = "MAC", market_stage = "wholesale",
  price_unit = "EUR/kg", min_eur_per_kg = rep(0.9, 12),
  max_eur_per_kg = rep(1.1, 12), frequent_eur_per_kg = rep(1, 12),
  is_provisional = rep(NA, 12), stringsAsFactors = FALSE
)
windows <- pair_sima_forward_quotes(sima_weekly_panels(fixture),
                                     horizons = 1:3)
audit <- evaluate_sima_quote_break_even(
  windows, weekly_retention = c(1, 0.99),
  weekly_cost_eur_per_initial_kg = c(0, 0.01)
)
stopifnot(nrow(audit$scenarios) == 4L,
          nrow(audit$detail) == (10L + 9L + 8L) * 4L,
          all(audit$balanced$observed_pairs == 8L),
          nrow(audit$balanced) == 12L,
          all(audit$detail$required_forward_price_increase_eur_per_kg >= -1e-10))

frictionless <- audit$balanced[
  audit$balanced$weekly_retention == 1 &
    audit$balanced$weekly_cost_eur_per_initial_kg == 0,
  , drop = FALSE
]
stopifnot(nrow(frictionless) == 3L,
          all(frictionless$fraction_break_even_or_better == 1),
          all(frictionless$fraction_strictly_better == 0),
          all(frictionless$mean_illustrative_difference_eur_per_initial_kg == 0))

with_loss <- audit$balanced[
  audit$balanced$weekly_retention == 0.99 &
    audit$balanced$weekly_cost_eur_per_initial_kg == 0.01,
  , drop = FALSE
]
stopifnot(nrow(with_loss) == 3L,
          all(with_loss$fraction_break_even_or_better == 0),
          all(with_loss$fraction_strictly_better == 0),
          all(with_loss$mean_required_forward_price_increase_eur_per_kg > 0))

# The mathematical threshold must equal the direct net-quote comparison.
x <- audit$detail
stopifnot(all(abs(x$illustrative_difference_eur_per_initial_kg -
                  (x$retained_saleable_share * x$endpoint_eur_per_kg -
                     x$assumed_total_cost_eur_per_initial_kg -
                     x$anchor_eur_per_kg)) < 1e-12),
          all(abs(x$required_forward_quote_eur_per_saleable_kg -
                  (x$anchor_eur_per_kg +
                     x$assumed_total_cost_eur_per_initial_kg) /
                  x$retained_saleable_share) < 1e-12),
          all(!x$strictly_better | x$break_even_or_better))

# Missing future quotations never become zero sales or fabricated returns.
missing <- fixture[fixture$quote_date != as.Date("2026-01-26"), , drop = FALSE]
missing_windows <- pair_sima_forward_quotes(
  sima_weekly_panels(missing), horizons = 1:3
)
gap <- evaluate_sima_quote_break_even(missing_windows,
                                      weekly_retention = c(1, 0.99),
                                      weekly_cost_eur_per_initial_kg = c(0, 0.01))
stopifnot(nrow(gap$detail) < nrow(audit$detail),
          all(gap$detail$endpoint_observed),
          all(is.finite(gap$detail$illustrative_difference_eur_per_initial_kg)))

expect_error(evaluate_sima_quote_break_even(windows,
                                            weekly_retention = 0))
expect_error(evaluate_sima_quote_break_even(windows,
                                            weekly_retention = 1.01))
expect_error(evaluate_sima_quote_break_even(windows,
                                            weekly_retention = c(1, 1)))
expect_error(evaluate_sima_quote_break_even(windows,
                                            weekly_cost_eur_per_initial_kg = -0.01))
expect_error(evaluate_sima_quote_break_even(windows, tolerance = -1))

# Freeze the empirical denominators: 343 common anchors * four horizons,
# 16 *explicitly hypothetical* friction scenarios, no source rows invented.
raw <- read_sima_orange_prices(
  "data/raw/sima_oranges_wholesale_2025-01-01_2026-10-31.csv.gz",
  market_stage = "wholesale"
)
source_windows <- pair_sima_forward_quotes(sima_weekly_panels(raw))
empirical <- evaluate_sima_quote_break_even(source_windows)
stopifnot(nrow(empirical$scenarios) == 16L,
          nrow(empirical$detail) == sum(c(433L, 403L, 373L, 343L)) * 16L,
          nrow(empirical$balanced) == 16L * 4L,
          all(empirical$balanced$observed_pairs == 343L),
          all(empirical$balanced$fraction_strictly_better <=
                empirical$balanced$fraction_break_even_or_better),
          all(empirical$balanced$fraction_break_even_or_better >= 0),
          all(empirical$balanced$fraction_break_even_or_better <= 1))

reference <- summarize_sima_forward_windows(source_windows)$balanced
base <- empirical$balanced[
  empirical$balanced$weekly_retention == 1 &
    empirical$balanced$weekly_cost_eur_per_initial_kg == 0,
  , drop = FALSE
]
base <- base[order(base$horizon_weeks), , drop = FALSE]
stopifnot(all(abs(base$fraction_strictly_better -
                  reference$share_price_higher) < 1e-10),
          all(abs(base$fraction_break_even_or_better -
                    (1 - reference$share_price_lower)) < 1e-10),
          all(abs(base$mean_illustrative_difference_eur_per_initial_kg -
                    reference$mean_signed_change_eur_per_kg) < 1e-10))

# More cost or less retained fruit cannot improve break-even support when
# every compared scenario uses the same archived anchor/target prices.
for (h in 1:4) {
  x <- empirical$balanced[empirical$balanced$horizon_weeks == h, , drop = FALSE]
  for (rho in unique(x$weekly_retention)) {
    by_cost <- x[x$weekly_retention == rho, , drop = FALSE]
    by_cost <- by_cost[order(by_cost$weekly_cost_eur_per_initial_kg), ]
    stopifnot(all(diff(by_cost$fraction_break_even_or_better) <= 1e-12))
  }
  for (cost in unique(x$weekly_cost_eur_per_initial_kg)) {
    by_rho <- x[x$weekly_cost_eur_per_initial_kg == cost, , drop = FALSE]
    by_rho <- by_rho[order(by_rho$weekly_retention), ]
    stopifnot(all(diff(by_rho$fraction_break_even_or_better) >= -1e-12))
  }
}
cat("SIMA illustrative quote break-even stress tests passed.\n")
