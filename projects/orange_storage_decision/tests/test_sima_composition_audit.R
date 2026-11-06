# Run from projects/orange_storage_decision; base R only, offline.
source("R/sima_market_prices.R")
source("R/sima_rolling_backtest.R")
source("R/sima_composition_audit.R")

expect_error <- function(expression) {
  stopifnot(tryCatch({ force(expression); FALSE }, error = function(error) TRUE))
}

quote <- function(date, label, price) {
  data.frame(
    quote_date = as.Date(date),
    product_descriptor = paste0("Laranja*", label, "*II*Caixa*Algarve*EUR/kg"),
    region = "Beira Litoral", market = "MAC",
    market_stage = "wholesale", price_unit = "EUR/kg",
    min_eur_per_kg = price - 0.1, max_eur_per_kg = price + 0.1,
    frequent_eur_per_kg = as.numeric(price), is_provisional = NA,
    stringsAsFactors = FALSE
  )
}

# A hand-worked composition reversal: both comparable orange grades rise
# from January 2025 to January 2026, but adding a low-priced quoted grade
# in 2026 makes the all-panel weekly arithmetic mean fall.
fixture <- do.call(rbind, list(
  quote("2025-01-06", "A", 1),
  quote("2025-01-06", "B", 10),
  quote("2026-01-05", "A", 2),
  quote("2026-01-05", "B", 11),
  quote("2026-01-05", "C", 0.5),
  quote("2026-01-12", "C", 0.5)
))
weekly <- sima_weekly_panels(fixture)
monthly <- sima_panel_monthly_prices(weekly)
spans <- sima_panel_quotation_spans(weekly)
comparison <- sima_compare_monthly_panel_mix(monthly, months = 1L)
x <- comparison$overview
stopifnot(nrow(weekly) == 6L, nrow(monthly) == 5L, nrow(spans) == 3L,
          nrow(x) == 1L, x$base_panels == 2L, x$comparison_panels == 3L,
          x$matched_panels == 2L,
          x$base_only_panels == 0L, x$comparison_only_panels == 1L,
          x$base_observed_panel_weeks == 2L,
          x$comparison_observed_panel_weeks == 4L,
          x$base_matched_week_share == 1,
          x$comparison_matched_week_share == 0.5,
          x$pooled_week_mean_change_fraction < 0,
          x$matched_week_mean_change_fraction > 0,
          x$matched_panel_geometric_change_fraction > 0,
          nrow(comparison$matched_pairs) == 2L,
          all(comparison$matched_pairs$median_price_change_fraction > 0),
          abs(x$pooled_week_mean_change_fraction - (3.5 / 5.5 - 1)) < 1e-12,
          abs(x$matched_week_mean_change_fraction - (6.5 / 5.5 - 1)) < 1e-12)

# A quote from 2 January can belong to a Monday-start week in December.
# The descriptive month must follow the actual quotation date, not Monday.
boundary <- sima_weekly_panels(quote("2026-01-02", "D", 1.2))
stopifnot(boundary$week_start == as.Date("2025-12-29"))
boundary_month <- sima_panel_monthly_prices(boundary)
stopifnot(boundary_month$year == 2026L, boundary_month$month == 1L)

# Within-span unquoted weeks are counted, not silently interpolated or
# interpreted as verified source failures during an orange off-season.
a_span <- spans[grepl("\\*A\\*", spans$product_descriptor), , drop = FALSE]
stopifnot(nrow(a_span) == 1L, a_span$observed_weeks == 2L,
          a_span$calendar_weeks_in_observed_span == 53L,
          a_span$unquoted_weeks_within_span == 51L,
          a_span$fraction_of_span_with_quote < 1)

# Sparse-overlap cases are explicit NA, not a fabricated zero price change.
absent <- sima_compare_monthly_panel_mix(monthly, months = 2L)
stopifnot(nrow(absent$overview) == 1L,
          absent$overview$matched_panels == 0L,
          is.na(absent$overview$pooled_week_mean_change_fraction),
          is.na(absent$overview$matched_panel_geometric_change_fraction),
          nrow(absent$matched_pairs) == 0L)

# Malformed or duplicate entries must be rejected before generating indices.
bad <- weekly
bad$quote_date[1L] <- bad$week_end[1L] + 1L
expect_error(sima_panel_monthly_prices(bad))
expect_error(sima_panel_monthly_prices(rbind(weekly, weekly[1, ])))
expect_error(sima_panel_quotation_spans(rbind(weekly, weekly[1, ])))
expect_error(sima_compare_monthly_panel_mix(monthly, months = c(1L, 1L)))
expect_error(sima_compare_monthly_panel_mix(monthly, months = 13L))
expect_error(sima_compare_monthly_panel_mix(
  monthly, base_year = 2025L, comparison_year = 2027L))
expect_error(sima_compare_monthly_panel_mix(rbind(monthly, monthly[1, ]),
                                           months = 1L))

# Complete integration against the immutable official 2025-2026 GPP export.
# These assertions are independently audited against the dated source rows.
real <- read_sima_orange_prices(
  "data/raw/sima_oranges_wholesale_2025-01-01_2026-10-31.csv.gz",
  market_stage = "wholesale"
)
real_weekly <- sima_weekly_panels(real)
real_monthly <- sima_panel_monthly_prices(real_weekly)
real_spans <- sima_panel_quotation_spans(real_weekly)
real_yoy <- sima_compare_monthly_panel_mix(real_monthly)
stopifnot(nrow(real_weekly) == 1005L,
          length(unique(real_weekly$panel_id)) == 39L,
          nrow(real_monthly) == 278L, nrow(real_spans) == 39L,
          nrow(real_yoy$overview) == 10L,
          identical(real_yoy$overview$month, 1:10),
          identical(real_yoy$overview$matched_panels,
                    c(6L, 9L, 14L, 8L, 8L, 13L, 7L, 4L, 8L, 13L)),
          nrow(real_yoy$matched_pairs) == 90L,
          all(real_yoy$overview$matched_panels <=
                pmin(real_yoy$overview$base_panels,
                     real_yoy$overview$comparison_panels)),
          all(is.finite(real_yoy$matched_pairs$log_median_price_ratio)),
          all(real_spans$observed_weeks <=
                real_spans$calendar_weeks_in_observed_span))

cat("SIMA monthly composition and coverage audit tests passed.\n")
