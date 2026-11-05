# Run from projects/orange_storage_decision. Base R only.
source("R/sima_market_prices.R")
source("R/sima_rolling_backtest.R")

expect_error <- function(expression) {
  stopifnot(tryCatch({ force(expression); FALSE }, error = function(error) TRUE))
}

# One observed Monday quote per week, all within a fixed product/market panel.
# These hand-worked observations are tests, not the empirical study evidence.
dates <- as.Date("2025-12-01") + 7L * 0:7
prices <- as.numeric(seq_along(dates))
fixture <- data.frame(
  quote_date = dates,
  product_descriptor = "Laranja*Newhall*II*Caixa*Algarve*EUR/kg",
  region = "Beira Litoral", market = "MAC", market_stage = "wholesale",
  price_unit = "EUR/kg", min_eur_per_kg = prices - 0.1,
  max_eur_per_kg = prices + 0.1, frequent_eur_per_kg = prices,
  is_provisional = rep(NA, length(dates)), stringsAsFactors = FALSE
)
weekly <- sima_weekly_panels(fixture)
stopifnot(nrow(weekly) == 8L, all(weekly$quotes_in_week == 1L),
          all(weekly$week_start == fixture$quote_date))

forecast <- backtest_sima_weekly_prices(
  weekly, test_start = as.Date("2026-01-05"),
  min_history_weeks = 3L, lookback_weeks = 4L,
  min_recent_weeks = 2L, max_age_weeks = 2L
)
stopifnot(
  nrow(forecast) == 3L,
  identical(forecast$actual_eur_per_kg, c(6, 7, 8)),
  identical(forecast$naive_eur_per_kg, c(5, 6, 7)),
  identical(forecast$median_4w_eur_per_kg, c(3.5, 4.5, 5.5)),
  all(forecast$last_training_quote_date <= forecast$origin_end),
  all(forecast$origin_end < forecast$target_week_start),
  all(forecast$n_recent_weeks == 4L),
  all(forecast$latest_quote_age_weeks == 1L)
)
score <- summarize_sima_rolling_prices(forecast)
stopifnot(nrow(score$overall) == 2L,
          all(score$overall$paired_quotes == 3L),
          identical(score$overall$observation_weighted_mae_eur_per_kg,
                    c(1, 2.5)),
          identical(score$overall$equal_panel_mae_eur_per_kg, c(1, 2.5)),
          score$coverage$paired_coverage_fraction == 1)

# Neither the current week's target quote nor later weeks may influence its
# forecast: the week-1 prediction survives an adversarial future-price change.
changed <- fixture
changed$frequent_eur_per_kg[changed$quote_date >= as.Date("2026-01-05")] <- 100
changed$max_eur_per_kg[changed$quote_date >= as.Date("2026-01-05")] <- 101
changed_weekly <- sima_weekly_panels(changed)
changed_forecast <- backtest_sima_weekly_prices(
  changed_weekly, test_start = as.Date("2026-01-05"),
  min_history_weeks = 3L, lookback_weeks = 4L,
  min_recent_weeks = 2L, max_age_weeks = 2L
)
stopifnot(changed_forecast$actual_eur_per_kg[1] == 100,
          changed_forecast$naive_eur_per_kg[1] == forecast$naive_eur_per_kg[1],
          changed_forecast$median_4w_eur_per_kg[1] ==
            forecast$median_4w_eur_per_kg[1])

# Different wholesale markets and product descriptions are not pooled.
other_market <- fixture
other_market$market <- "MARL"
other_market$frequent_eur_per_kg <- prices + 20
other_market$min_eur_per_kg <- prices + 19.9
other_market$max_eur_per_kg <- prices + 20.1
multi <- sima_weekly_panels(rbind(fixture, other_market))
stopifnot(nrow(multi) == 16L, length(unique(multi$panel_id)) == 2L)
multi_forecast <- backtest_sima_weekly_prices(
  multi, min_history_weeks = 3L, max_age_weeks = 2L
)
stopifnot(nrow(multi_forecast) == 6L,
          all(multi_forecast$naive_eur_per_kg[
            multi_forecast$market == "MARL"] > 20))

# Repeated observations within a week resolve to the latest dated quote;
# missing weeks remain absent and never become imputed training records.
extra <- fixture[4, , drop = FALSE]
extra$quote_date <- extra$quote_date + 2L
extra$frequent_eur_per_kg <- 4.5
extra$max_eur_per_kg <- 4.6
extra$min_eur_per_kg <- 4.4
extra_weekly <- sima_weekly_panels(rbind(fixture, extra))
stopifnot(nrow(extra_weekly) == 8L,
          extra_weekly$quotes_in_week[extra_weekly$week_start ==
                                        as.Date("2025-12-22")] == 2L,
          extra_weekly$quote_date[extra_weekly$week_start ==
                                   as.Date("2025-12-22")] == as.Date("2025-12-24"))
gap_fixture <- fixture[fixture$quote_date != as.Date("2025-12-15"), ]
gap_weekly <- sima_weekly_panels(gap_fixture)
stopifnot(nrow(gap_weekly) == 7L)
gap_forecast <- backtest_sima_weekly_prices(gap_weekly,
                                            min_history_weeks = 3L)
stopifnot(gap_forecast$n_recent_weeks[1] == 3L,
          gap_forecast$median_4w_eur_per_kg[1] == 4)

# Insufficient training, missing recent quotes or an unobserved test period
# must be represented honestly (NA or an explicit error), not extrapolated.
short <- backtest_sima_weekly_prices(weekly, min_history_weeks = 20L)
stopifnot(all(is.na(short$naive_eur_per_kg)),
          all(is.na(short$median_4w_eur_per_kg)))
expect_error(summarize_sima_rolling_prices(short))
stale <- backtest_sima_weekly_prices(
  gap_weekly[gap_weekly$week_start != as.Date("2025-12-29"), ],
  min_history_weeks = 3L, max_age_weeks = 1L
)
stopifnot(is.na(stale$naive_eur_per_kg[1]))
expect_error(backtest_sima_weekly_prices(weekly,
                                        test_start = as.Date("2026-01-06")))
expect_error(backtest_sima_weekly_prices(weekly, min_recent_weeks = 5L))
expect_error(backtest_sima_weekly_prices(weekly,
                                        test_start = as.Date("2027-01-04")))

bad <- fixture
bad$price_unit <- "EUR/unit"
expect_error(sima_weekly_panels(bad))
bad <- fixture
bad$frequent_eur_per_kg[1] <- -1
expect_error(sima_weekly_panels(bad))
expect_error(sima_weekly_panels(rbind(fixture, fixture[1, ])))

# The real-source integration check uses every archived quote. It makes no
# profitability claim and does not tune the parameters against test prices.
raw <- read_sima_orange_prices(
  "data/raw/sima_oranges_wholesale_2025-01-01_2026-10-31.csv.gz",
  market_stage = "wholesale"
)
real_weekly <- sima_weekly_panels(raw)
# Independently cross-checked against the source rows and Monday boundaries:
# 3,412 dated quotes collapse to 1,005 observed weeks in 39 exact panels.
stopifnot(nrow(real_weekly) == 1005L,
          length(unique(real_weekly$panel_id)) == 39L,
          !anyDuplicated(real_weekly[c("panel_id", "week_start")]),
          all(real_weekly$quote_date >= real_weekly$week_start),
          all(real_weekly$quote_date <= real_weekly$week_end),
          any(real_weekly$quotes_in_week > 1L))
real_forecast <- backtest_sima_weekly_prices(real_weekly)
real_score <- summarize_sima_rolling_prices(real_forecast)
stopifnot(nrow(real_forecast) == 485L,
          real_score$coverage$scored_paired_panel_weeks == 408L,
          real_score$coverage$scored_panels == 27L,
          real_score$coverage$paired_coverage_fraction > 0,
          all(is.finite(real_score$overall$observation_weighted_mae_eur_per_kg)),
          all(real_forecast$origin_end < real_forecast$target_week_start),
          all(is.na(real_forecast$last_training_quote_date) |
                real_forecast$last_training_quote_date <= real_forecast$origin_end))

cat("SIMA orange rolling-origin forecast tests passed.\n")
