# Offline, base-R validation of the exact 52-week seasonal negative control.
source("R/sima_market_prices.R")
source("R/sima_rolling_backtest.R")
source("R/sima_composition_audit.R")
source("R/sima_seasonal_benchmark.R")

expect_error <- function(expression) {
  stopifnot(tryCatch({ force(expression); FALSE }, error = function(error) TRUE))
}

# Continuous 55-week hand-worked data: exactly 52-week-old values are 1,2,3
# for the three January 2026 targets. No model fitting is involved.
dates <- as.Date("2025-01-06") + 7L * 0:54
prices <- as.numeric(seq_along(dates))
fixture <- data.frame(
  quote_date = dates,
  product_descriptor = "Laranja*Newhall*II*Caixa*Algarve*EUR/kg",
  region = "Beira Litoral", market = "MAC", market_stage = "wholesale",
  price_unit = "EUR/kg",
  min_eur_per_kg = prices - 0.1, max_eur_per_kg = prices + 0.1,
  frequent_eur_per_kg = prices,
  is_provisional = rep(NA, length(dates)),
  stringsAsFactors = FALSE
)
weekly <- sima_weekly_panels(fixture)
replay <- backtest_sima_seasonal_52(weekly)
stopifnot(nrow(replay) == 3L,
          identical(replay$actual_eur_per_kg, c(53, 54, 55)),
          identical(replay$naive_eur_per_kg, c(52, 53, 54)),
          identical(replay$seasonal_52_eur_per_kg, c(1, 2, 3)),
          all(replay$seasonal_reference_week_start ==
                replay$target_week_start - 364L),
          all(replay$seasonal_reference_quote_date < replay$origin_end))
score <- summarize_sima_seasonal_52(replay)
stopifnot(nrow(score$overall) == 3L,
          all(score$overall$common_scored_weeks == 3L),
          score$coverage$seasonal_52_available_panel_weeks == 3L,
          score$coverage$three_way_scored_panel_weeks == 3L,
          all(abs(score$overall$observation_weighted_mae_eur_per_kg -
                    c(1, 2.5, 52)) < 1e-10),
          nrow(score$monthly) == 1L,
          score$monthly$three_way_scored_weeks == 3L)

# Exact matching is mandatory. Removing an earlier-year quote must not
# substitute a neighboring week, nor invent a seasonal forecast of zero.
removed <- fixture[fixture$quote_date != as.Date("2025-01-06"), , drop = FALSE]
gap <- backtest_sima_seasonal_52(sima_weekly_panels(removed))
stopifnot(nrow(gap) == 3L,
          is.na(gap$seasonal_52_eur_per_kg[1L]),
          !is.na(gap$naive_eur_per_kg[1L]),
          sum(!is.na(gap$seasonal_52_eur_per_kg)) == 2L,
          summarize_sima_seasonal_52(gap)$coverage$three_way_scored_panel_weeks == 2L)

# Future perturbations may affect realised errors but cannot alter a
# forecast issued on the preceding Sunday for the first target week.
future <- fixture
future_rows <- future$quote_date >= as.Date("2026-01-05")
future$frequent_eur_per_kg[future_rows] <- 100
future$min_eur_per_kg[future_rows] <- 99.9
future$max_eur_per_kg[future_rows] <- 100.1
changed <- backtest_sima_seasonal_52(sima_weekly_panels(future))
stopifnot(changed$actual_eur_per_kg[1L] == 100,
          changed$naive_eur_per_kg[1L] == replay$naive_eur_per_kg[1L],
          changed$median_4w_eur_per_kg[1L] ==
            replay$median_4w_eur_per_kg[1L],
          changed$seasonal_52_eur_per_kg[1L] ==
            replay$seasonal_52_eur_per_kg[1L])

# Market identity cannot be relaxed when looking up the previous year.
other <- fixture
other$market <- "MARL"
other$frequent_eur_per_kg <- prices + 100
other$min_eur_per_kg <- prices + 99.9
other$max_eur_per_kg <- prices + 100.1
combined <- backtest_sima_seasonal_52(
  sima_weekly_panels(rbind(fixture, other))
)
stopifnot(nrow(combined) == 6L,
          all(combined$seasonal_52_eur_per_kg[
            combined$market == "MARL"] > 100),
          all(combined$seasonal_52_eur_per_kg[
            combined$market == "MAC"] < 100))

expect_error(backtest_sima_seasonal_52(weekly,
                                      test_start = as.Date("2026-01-06")))
expect_error(backtest_sima_seasonal_52(weekly,
                                      test_start = as.Date("2027-01-04")))
expect_error(backtest_sima_seasonal_52(rbind(weekly, weekly[1, ])))
expect_error(summarize_sima_seasonal_52(
  gap[is.na(gap$seasonal_52_eur_per_kg), , drop = FALSE]))

# Full-source checks use frozen official GPP dated rows. The three-way
# sample is fixed by a prior-year exact match AND both original gates.
raw <- read_sima_orange_prices(
  "data/raw/sima_oranges_wholesale_2025-01-01_2026-10-31.csv.gz",
  market_stage = "wholesale"
)
real <- backtest_sima_seasonal_52(sima_weekly_panels(raw))
real_score <- summarize_sima_seasonal_52(real)
stopifnot(nrow(real) == 485L,
          real_score$coverage$seasonal_52_available_panel_weeks == 320L,
          real_score$coverage$recent_baseline_paired_panel_weeks == 408L,
          real_score$coverage$three_way_scored_panel_weeks == 309L,
          all(real_score$overall$common_scored_weeks == 309L),
          nrow(real_score$monthly) == 10L,
          all(is.na(real$seasonal_reference_quote_date) |
                real$seasonal_reference_quote_date <= real$origin_end),
          all(real_score$overall$observation_weighted_mae_eur_per_kg > 0),
          real_score$overall$observation_weighted_mae_eur_per_kg[3L] >
            real_score$overall$observation_weighted_mae_eur_per_kg[1L])

cat("SIMA exact 52-week seasonal benchmark tests passed.\n")
