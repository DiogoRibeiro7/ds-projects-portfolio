# Run from projects/orange_storage_decision. Base R; no network calls.
source("R/sima_market_prices.R")
source("R/sima_rolling_backtest.R")
source("R/sima_availability_stress.R")

expect_error <- function(expression) {
  stopifnot(tryCatch({ force(expression); FALSE }, error = function(error) TRUE))
}

# A fully auditable quotation series, with a later quote in one prior week.
# On 4 January 2026, the 2 January quote would be known under a zero-day
# assumption, but not with a hypothetical three-day publication delay.
dates <- as.Date("2025-11-03") + 7L * 0:10
prices <- as.numeric(seq_along(dates))
fixture <- data.frame(
  quote_date = dates,
  product_descriptor = "Laranja*Newhall*II*Caixa*Algarve*EUR/kg",
  region = "Beira Litoral", market = "MAC", market_stage = "wholesale",
  price_unit = "EUR/kg", min_eur_per_kg = prices - 0.1,
  max_eur_per_kg = prices + 0.1, frequent_eur_per_kg = prices,
  is_provisional = rep(NA, length(dates)), stringsAsFactors = FALSE
)
late <- fixture[fixture$quote_date == as.Date("2025-12-29"), , drop = FALSE]
late$quote_date <- as.Date("2026-01-02")
late$min_eur_per_kg <- 98.9
late$max_eur_per_kg <- 99.1
late$frequent_eur_per_kg <- 99
fixture <- rbind(fixture, late)

lagged <- backtest_sima_availability_lags(
  fixture, lag_days = c(0L, 3L, 14L, 21L),
  min_history_weeks = 3L, max_age_weeks = 3L
)
stopifnot(nrow(lagged) == 8L,
          all(lagged$origin_end < lagged$target_week_start),
          all(is.na(lagged$last_training_assumed_available) |
                lagged$last_training_assumed_available <= lagged$origin_end))

# The zero-lag replay must exactly preserve the already merged benchmark.
reference <- backtest_sima_weekly_prices(
  sima_weekly_panels(fixture), min_history_weeks = 3L,
  max_age_weeks = 3L
)
zero <- lagged[lagged$assumed_release_lag_days == 0L, , drop = FALSE]
for (column in c("actual_eur_per_kg", "naive_eur_per_kg",
                 "median_4w_eur_per_kg", "last_training_quote_date",
                 "n_history_weeks", "n_recent_weeks",
                 "latest_quote_age_weeks")) {
  stopifnot(isTRUE(all.equal(zero[[column]], reference[[column]],
                             check.attributes = FALSE)))
}

# Crucial intra-week fallback: under three-day publication lag, the latest
# *available* quote is the Monday one (9), not the later Friday quote (99).
first_zero <- zero[zero$target_week_start == as.Date("2026-01-05"), ]
first_three <- lagged[
  lagged$target_week_start == as.Date("2026-01-05") &
    lagged$assumed_release_lag_days == 3L, , drop = FALSE
]
stopifnot(nrow(first_zero) == 1L, nrow(first_three) == 1L,
          first_zero$naive_eur_per_kg == 99,
          first_zero$last_training_quote_date == as.Date("2026-01-02"),
          first_three$naive_eur_per_kg == 9,
          first_three$last_training_quote_date == as.Date("2025-12-29"),
          first_three$last_training_assumed_available == as.Date("2026-01-01"),
          first_three$n_history_weeks == first_zero$n_history_weeks)

# An assumption that excludes nearly all recent quotes should reduce
# coverage. Zero paired rows must not become a fictitious zero error.
summary <- summarize_sima_availability_lags(lagged)
stopifnot(nrow(summary$overall) == 4L,
          nrow(summary$matched) == 4L,
          all(summary$overall$observed_target_panel_weeks == 2L),
          summary$overall$paired_panel_weeks[1L] == 2L,
          summary$overall$paired_panel_weeks[4L] == 0L,
          is.na(summary$overall$naive_mae_eur_per_kg[4L]),
          summary$matched$matched_panel_weeks[4L] == 0L,
          is.na(summary$matched$naive_mae_change_eur_per_kg[4L]),
          summary$matched$naive_mae_change_eur_per_kg[1L] == 0)

# Changes to the future target or later prices cannot alter the forecast
# issued before the first target week under *any* release-delay assumption.
future <- fixture
after_origin <- future$quote_date >= as.Date("2026-01-05")
future$frequent_eur_per_kg[after_origin] <- 100
future$min_eur_per_kg[after_origin] <- 99.9
future$max_eur_per_kg[after_origin] <- 100.1
changed <- backtest_sima_availability_lags(
  future, lag_days = c(0L, 3L, 14L, 21L),
  min_history_weeks = 3L, max_age_weeks = 3L
)
keep_original <- lagged$target_week_start == as.Date("2026-01-05")
keep_changed <- changed$target_week_start == as.Date("2026-01-05")
stopifnot(all(changed$actual_eur_per_kg[keep_changed] == 100),
          isTRUE(all.equal(
            lagged$naive_eur_per_kg[keep_original],
            changed$naive_eur_per_kg[keep_changed]
          )),
          isTRUE(all.equal(
            lagged$median_4w_eur_per_kg[keep_original],
            changed$median_4w_eur_per_kg[keep_changed]
          )))

# Markets and product descriptions must not borrow quotations across panels.
other <- fixture
other$market <- "MARL"
other$min_eur_per_kg <- other$min_eur_per_kg + 100
other$max_eur_per_kg <- other$max_eur_per_kg + 100
other$frequent_eur_per_kg <- other$frequent_eur_per_kg + 100
separate <- backtest_sima_availability_lags(
  rbind(fixture, other), lag_days = c(0L, 3L),
  min_history_weeks = 3L
)
stopifnot(length(unique(separate$panel_id)) == 2L,
          all(separate$naive_eur_per_kg[
            separate$market == "MARL" & !is.na(separate$naive_eur_per_kg)
          ] > 100))

expect_error(backtest_sima_availability_lags(fixture, lag_days = c(0, 0)))
expect_error(backtest_sima_availability_lags(fixture, lag_days = c(-1, 0)))
expect_error(backtest_sima_availability_lags(fixture, lag_days = c(0, 1.5)))
expect_error(backtest_sima_availability_lags(fixture, lag_days = c(3, 7)))
expect_error(backtest_sima_availability_lags(fixture,
                                            test_start = as.Date("2026-01-06")))
expect_error(backtest_sima_availability_lags(fixture, min_recent_weeks = 5L))
expect_error(backtest_sima_availability_lags(
  fixture, test_start = as.Date("2027-01-04")))
expect_error(summarize_sima_availability_lags(
  lagged[lagged$assumed_release_lag_days != 0L, ]))

# Real-data integration uses the frozen, original GPP 2025-2026 quotation
# snapshot. No release-delay parameter is estimated from the test outcomes.
quotes <- read_sima_orange_prices(
  "data/raw/sima_oranges_wholesale_2025-01-01_2026-10-31.csv.gz",
  market_stage = "wholesale"
)
real <- backtest_sima_availability_lags(quotes)
real_summary <- summarize_sima_availability_lags(real)
real_zero <- real[real$assumed_release_lag_days == 0L, , drop = FALSE]
rolling <- backtest_sima_weekly_prices(sima_weekly_panels(quotes))
stopifnot(nrow(real) == 5L * 485L,
          nrow(real_zero) == 485L,
          real_summary$overall$paired_panel_weeks[1L] == 408L,
          real_summary$overall$paired_panels[1L] == 27L,
          all(diff(real_summary$overall$paired_panel_weeks) <= 0L),
          all(real_summary$overall$observed_target_panel_weeks == 485L),
          all(is.na(real$last_training_assumed_available) |
                real$last_training_assumed_available <= real$origin_end),
          all(is.na(real$last_training_quote_date) |
                real$last_training_quote_date <= real$origin_end))
for (column in c("actual_eur_per_kg", "naive_eur_per_kg",
                 "median_4w_eur_per_kg", "last_training_quote_date")) {
  stopifnot(isTRUE(all.equal(real_zero[[column]], rolling[[column]],
                             check.attributes = FALSE)))
}

cat("SIMA hypothetical quotation-availability stress tests passed.\n")
