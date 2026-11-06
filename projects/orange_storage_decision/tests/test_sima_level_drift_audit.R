# Offline, base-R tests for the exact-week price-level-gap identity.
source("R/sima_market_prices.R")
source("R/sima_rolling_backtest.R")
source("R/sima_composition_audit.R")
source("R/sima_seasonal_benchmark.R")
source("R/sima_level_drift_audit.R")

expect_error <- function(expression) {
  stopifnot(tryCatch({ force(expression); FALSE }, error = function(e) TRUE))
}

# A hand-worked 55-week archive. For the 2026 weeks, prices 53:55 are
# compared with last year's 1:3 and the latest prior quotes 52:54.
dates <- as.Date("2025-01-06") + 7L * 0:54
price <- as.numeric(seq_along(dates))
fixture <- data.frame(
  quote_date = dates,
  product_descriptor = "Laranja*Newhall*II*Caixa*Algarve*EUR/kg",
  region = "Beira Litoral", market = "MAC", market_stage = "wholesale",
  price_unit = "EUR/kg", min_eur_per_kg = price - 0.1,
  max_eur_per_kg = price + 0.1, frequent_eur_per_kg = price,
  is_provisional = rep(NA, length(dates)), stringsAsFactors = FALSE
)
forecast <- backtest_sima_seasonal_52(sima_weekly_panels(fixture))
decomposition <- decompose_sima_seasonal_errors(forecast)
audit <- summarize_sima_level_drift(forecast)
stopifnot(nrow(decomposition) == 3L,
          all(decomposition$three_way_common),
          identical(decomposition$pre_origin_level_gap_eur_per_kg,
                    c(51, 51, 51)),
          identical(decomposition$recent_quote_innovation_eur_per_kg,
                    c(1, 1, 1)),
          identical(decomposition$seasonal_signed_error_eur_per_kg,
                    c(52, 52, 52)),
          all(abs(decomposition$decomposition_residual_eur_per_kg) < 1e-10),
          audit$coverage$three_way_common_panel_weeks == 3L,
          audit$overall$mean_signed_pre_origin_gap_eur_per_kg == 51,
          audit$overall$mean_absolute_recent_quote_innovation_eur_per_kg == 1,
          audit$overall$mean_absolute_seasonal_error_eur_per_kg == 52,
          audit$overall$mean_absolute_four_week_median_error_eur_per_kg == 2.5,
          audit$overall$share_yoy_quotes_higher == 1,
          nrow(audit$monthly) == 1L, nrow(audit$panel) == 1L)

# Removing an exact historical counterpart cannot be disguised as an
# unchanged prior-year price; the target remains in the exported rows.
missing <- fixture[fixture$quote_date != as.Date("2025-01-06"), , drop = FALSE]
gap <- decompose_sima_seasonal_errors(
  backtest_sima_seasonal_52(sima_weekly_panels(missing))
)
stopifnot(nrow(gap) == 3L,
          !gap$seasonal_available[1L],
          !gap$three_way_common[1L],
          is.na(gap$nominal_yoy_price_change_fraction[1L]),
          is.na(gap$pre_origin_level_gap_eur_per_kg[1L]),
          sum(gap$three_way_common) == 2L)

# A change to the first future target cannot change the pre-origin gap;
# it only changes the future innovation and realised seasonal error.
future <- fixture
future$frequent_eur_per_kg[future$quote_date >= as.Date("2026-01-05")] <- 100
future$min_eur_per_kg[future$quote_date >= as.Date("2026-01-05")] <- 99.9
future$max_eur_per_kg[future$quote_date >= as.Date("2026-01-05")] <- 100.1
future_audit <- decompose_sima_seasonal_errors(
  backtest_sima_seasonal_52(sima_weekly_panels(future))
)
stopifnot(future_audit$pre_origin_level_gap_eur_per_kg[1L] == 51,
          future_audit$recent_quote_innovation_eur_per_kg[1L] == 48,
          future_audit$seasonal_signed_error_eur_per_kg[1L] == 99)

# The same cultivar in another wholesale market is a separate exact panel.
other <- fixture
other$market <- "MARL"
other$frequent_eur_per_kg <- price + 100
other$min_eur_per_kg <- price + 99.9
other$max_eur_per_kg <- price + 100.1
separate <- summarize_sima_level_drift(backtest_sima_seasonal_52(
  sima_weekly_panels(rbind(fixture, other))
))
stopifnot(nrow(separate$market) == 2L,
          nrow(separate$panel) == 2L,
          separate$coverage$three_way_common_panel_weeks == 6L,
          all(separate$market$mean_absolute_pre_origin_gap_eur_per_kg == 51))

# Bad model inputs must fail, rather than produce a plausible-looking audit.
bad <- forecast
bad$origin_end[1L] <- bad$target_quote_date[1L]
expect_error(decompose_sima_seasonal_errors(bad))
bad <- forecast
bad$seasonal_reference_quote_date[1L] <- as.Date(NA)
expect_error(decompose_sima_seasonal_errors(bad))
bad <- forecast
bad$seasonal_52_eur_per_kg[1L] <- -1
expect_error(decompose_sima_seasonal_errors(bad))
expect_error(summarize_sima_level_drift(forecast[FALSE, , drop = FALSE]))

# Assertions against the frozen, independently archived official SIMA CSV.
# These denominators must not change when reporting signed price movements.
raw <- read_sima_orange_prices(
  "data/raw/sima_oranges_wholesale_2025-01-01_2026-10-31.csv.gz",
  market_stage = "wholesale"
)
real <- summarize_sima_level_drift(
  backtest_sima_seasonal_52(sima_weekly_panels(raw))
)
stopifnot(real$coverage$observed_target_panel_weeks == 485L,
          real$coverage$seasonal_52_available_panel_weeks == 320L,
          real$coverage$recent_baselines_available_panel_weeks == 408L,
          real$coverage$three_way_common_panel_weeks == 309L,
          nrow(real$monthly) == 10L,
          nrow(real$market) == 3L,
          sum(real$monthly$three_way_common_panel_weeks) == 309L,
          sum(real$market$three_way_common_panel_weeks) == 309L,
          sum(real$panel$three_way_common_panel_weeks) == 309L,
          all(abs(real$rows$decomposition_residual_eur_per_kg[
            real$rows$three_way_common]) < 1e-10),
          isTRUE(all.equal(
            real$overall$mean_absolute_seasonal_error_eur_per_kg,
            weighted.mean(real$market$mean_absolute_seasonal_error_eur_per_kg,
                          real$market$three_way_common_panel_weeks),
            tolerance = 1e-12
          )),
          real$overall$mean_absolute_seasonal_error_eur_per_kg >
            real$overall$mean_absolute_recent_quote_innovation_eur_per_kg)

cat("SIMA signed 52-week level-drift audit tests passed.\n")
