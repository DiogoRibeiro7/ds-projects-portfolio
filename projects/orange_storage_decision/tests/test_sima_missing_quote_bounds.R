# Offline, assumption-free binary completion bounds for absent SIMA quotes.
source("R/sima_market_prices.R")
source("R/sima_rolling_backtest.R")
source("R/sima_composition_audit.R")
source("R/sima_forward_quote_windows.R")
source("R/sima_quote_break_even.R")
source("R/sima_missing_quote_bounds.R")

expect_error <- function(expression) {
  stopifnot(tryCatch({ force(expression); FALSE }, error = function(e) TRUE))
}

# A complete, constant observed price path fixes the binary outcome for
# each eligible quote: zero-friction equality, but no strict improvement.
fixture <- data.frame(
  quote_date = as.Date("2026-01-05") + 7L * 0:11,
  product_descriptor = "Laranja*Newhall*II*Caixa*Algarve*EUR/kg",
  region = "Beira Litoral", market = "MAC", market_stage = "wholesale",
  price_unit = "EUR/kg", min_eur_per_kg = rep(0.9, 12),
  max_eur_per_kg = rep(1.1, 12), frequent_eur_per_kg = rep(1, 12),
  is_provisional = rep(NA, 12), stringsAsFactors = FALSE
)
windows <- pair_sima_forward_quotes(sima_weekly_panels(fixture),
                                     horizons = 1:3)
result <- bound_sima_missing_forward_quotes(
  windows, weekly_retention = c(1, 0.99),
  weekly_cost_eur_per_initial_kg = c(0, 0.01)
)
stopifnot(nrow(result$scenarios) == 4L,
          nrow(result$coverage) == 6L,
          nrow(result$bounds) == 24L,
          nrow(result$missing_eligible_windows) == 0L,
          all(result$coverage$missing_eligible_endpoints == 0L),
          all(result$bounds$unidentified_fraction == 0),
          all(result$bounds$lower_break_even_fraction ==
                result$bounds$upper_break_even_fraction))
horizon_coverage <- result$coverage[
  result$coverage$cohort == "horizon_eligible", , drop = FALSE
]
common_coverage <- result$coverage[
  result$coverage$cohort == "common_maturity", , drop = FALSE
]
stopifnot(identical(horizon_coverage$eligible_anchors, c(10L, 9L, 8L)),
          all(common_coverage$eligible_anchors == 8L))

baseline <- result$bounds[
  result$bounds$weekly_retention == 1 &
    result$bounds$weekly_cost_eur_per_initial_kg == 0, , drop = FALSE
]
stopifnot(nrow(baseline) == 6L,
          all(baseline$lower_break_even_fraction == 1),
          all(baseline$upper_break_even_fraction == 1),
          all(baseline$lower_strict_improvement_fraction == 0),
          all(baseline$upper_strict_improvement_fraction == 0))

# Remove a future quotation, leaving its endpoint unknown rather than
# substituting zero, reusing the previous/latest price, or dropping the
# already-matured starting quote from the target denominator.
missing <- fixture[
  fixture$quote_date != as.Date("2026-01-26"), , drop = FALSE
]
missing_windows <- pair_sima_forward_quotes(
  sima_weekly_panels(missing), horizons = 1:3
)
gap <- bound_sima_missing_forward_quotes(
  missing_windows, weekly_retention = c(1, 0.99),
  weekly_cost_eur_per_initial_kg = c(0, 0.01)
)
gap_horizon <- gap$coverage[
  gap$coverage$cohort == "horizon_eligible", , drop = FALSE
]
gap_common <- gap$coverage[
  gap$coverage$cohort == "common_maturity", , drop = FALSE
]
stopifnot(identical(gap_horizon$eligible_anchors, c(9L, 8L, 7L)),
          all(gap_horizon$missing_eligible_endpoints == 1L),
          all(gap_common$eligible_anchors == 7L),
          all(gap_common$missing_eligible_endpoints == 1L),
          nrow(gap$missing_eligible_windows) == 3L,
          all(gap$missing_eligible_windows$common_maturity_anchor),
          all(abs(gap$bounds$upper_break_even_fraction -
                    gap$bounds$lower_break_even_fraction -
                    gap$bounds$unidentified_fraction) < 1e-12))

gap_flat <- gap$bounds[
  gap$bounds$cohort == "horizon_eligible" &
    gap$bounds$weekly_retention == 1 &
    gap$bounds$weekly_cost_eur_per_initial_kg == 0,
  , drop = FALSE
]
stopifnot(all(gap_flat$observed_break_even_count ==
                gap_flat$observed_endpoints),
          all(gap_flat$upper_break_even_fraction == 1),
          all(gap_flat$observed_strict_improvement_count == 0L),
          all(gap_flat$lower_strict_improvement_fraction == 0),
          all(abs(gap_flat$upper_strict_improvement_fraction -
                    gap_flat$missing_eligible_endpoints /
                    gap_flat$eligible_anchors) < 1e-12))

# Delegate malformed fixed-grid input validation to the scenario evaluator.
expect_error(bound_sima_missing_forward_quotes(windows,
                                                weekly_retention = 0))
expect_error(bound_sima_missing_forward_quotes(windows,
                                                weekly_retention = c(1, 1)))
expect_error(bound_sima_missing_forward_quotes(windows,
                                                weekly_cost_eur_per_initial_kg = -0.01))
expect_error(bound_sima_missing_forward_quotes(windows, tolerance = NA_real_))

# Frozen official source: missing rates are conditional on completed
# horizons, not censored final weeks or an invented price series.
raw <- read_sima_orange_prices(
  "data/raw/sima_oranges_wholesale_2025-01-01_2026-10-31.csv.gz",
  market_stage = "wholesale"
)
original_windows <- pair_sima_forward_quotes(sima_weekly_panels(raw))
empirical <- bound_sima_missing_forward_quotes(original_windows)
official_horizon <- empirical$coverage[
  empirical$coverage$cohort == "horizon_eligible", , drop = FALSE
]
official_common <- empirical$coverage[
  empirical$coverage$cohort == "common_maturity", , drop = FALSE
]
stopifnot(nrow(empirical$bounds) == 2L * 4L * 16L,
          all(official_horizon$eligible_anchors ==
                c(453L, 442L, 431L, 420L)),
          all(official_horizon$observed_endpoints ==
                c(433L, 403L, 373L, 343L)),
          all(official_horizon$missing_eligible_endpoints ==
                c(20L, 39L, 58L, 77L)),
          all(official_common$eligible_anchors == 420L),
          all(official_common$fully_matched_all_horizons == 343L),
          nrow(empirical$missing_eligible_windows) == 194L,
          all(!is.na(empirical$missing_eligible_windows$anchor_quote_date)),
          all(empirical$bounds$lower_break_even_fraction >= 0),
          all(empirical$bounds$upper_break_even_fraction <= 1),
          all(empirical$bounds$lower_strict_improvement_fraction <=
                empirical$bounds$lower_break_even_fraction),
          all(empirical$bounds$upper_strict_improvement_fraction <=
                empirical$bounds$upper_break_even_fraction))

# Market-specific numerators and denominators partition each horizon,
# cohort and fixed scenario without fabricating unmatched endpoints.
for (cohort in c("horizon_eligible", "common_maturity")) {
  for (h in 1:4) {
    for (id in empirical$scenarios$scenario_id) {
      x <- empirical$market_bounds[
        empirical$market_bounds$cohort == cohort &
          empirical$market_bounds$horizon_weeks == h &
          empirical$market_bounds$scenario_id == id,
        , drop = FALSE
      ]
      overall <- empirical$bounds[
        empirical$bounds$cohort == cohort &
          empirical$bounds$horizon_weeks == h &
          empirical$bounds$scenario_id == id,
        , drop = FALSE
      ]
      stopifnot(sum(x$eligible_anchors) == overall$eligible_anchors,
                sum(x$observed_endpoints) == overall$observed_endpoints,
                sum(x$missing_eligible_endpoints) ==
                  overall$missing_eligible_endpoints,
                sum(x$observed_break_even_count) ==
                  overall$observed_break_even_count)
    }
  }
}

# At one selected scenario, rising assumed cost cannot raise the lower or
# upper identification bound for the exact same matured anchor population.
for (cohort in c("horizon_eligible", "common_maturity")) {
  for (h in 1:4) {
    for (rho in c(1, 0.99, 0.98, 0.95)) {
      x <- empirical$bounds[
        empirical$bounds$cohort == cohort &
          empirical$bounds$horizon_weeks == h &
          empirical$bounds$weekly_retention == rho, , drop = FALSE
      ]
      x <- x[order(x$weekly_cost_eur_per_initial_kg), , drop = FALSE]
      stopifnot(all(diff(x$lower_break_even_fraction) <= 1e-12),
                all(diff(x$upper_break_even_fraction) <= 1e-12))
    }
  }
}
cat("SIMA missing eligible-quotation identification bound tests passed.\n")
