# Predeclared 52-week seasonal naive benchmark for observed GPP SIMA prices.
# Source sima_rolling_backtest.R and sima_composition_audit.R before this file.
# Base R only. No interpolation across missing quoted weeks.

#' Compare the existing rolling-origin forecasts with the exact prior-year
#' weekly quote in the **same product descriptor / region / wholesale market**.
#'
#' A target in week t uses week t-52, measured as precisely 364 days from
#' Monday to Monday. No nearest-week search, cross-panel borrowing, inflation
#' adjustment or 2026 outcome-based parameter selection is permitted.
#' The source quote was dated before the target forecast origin, but original
#' GPP first-publication and revision timestamps remain unknown.
#'
#' @param weekly Validated observed weekly SIMA panel from sima_weekly_panels().
#' @param test_start First Monday in the 2026 held-out evaluation.
#' @param min_history_weeks Baseline history gate, fixed at 8 by default.
#' @param lookback_weeks Existing four-calendar-week median window.
#' @param min_recent_weeks Existing minimum recent-observation requirement.
#' @param max_age_weeks Existing staleness gate for current-year forecasts.
#' @return All observed target weeks, including seasonal-NA targets, with
#'   exact dated prior-year lookup and three distinct forecast errors.
backtest_sima_seasonal_52 <- function(
    weekly, test_start = as.Date("2026-01-05"),
    min_history_weeks = 8L, lookback_weeks = 4L,
    min_recent_weeks = 2L, max_age_weeks = 3L) {
  validate_sima_composition_weekly(weekly)
  baseline <- backtest_sima_weekly_prices(
    weekly, test_start, min_history_weeks, lookback_weeks,
    min_recent_weeks, max_age_weeks
  )
  # Match all original dimensions, never the display-only panel label.
  make_key <- function(market, region, product, weeks) {
    paste(market, region, product, as.character(weeks), sep = "\u001f")
  }
  reference_key <- make_key(weekly$market, weekly$region,
                            weekly$product_descriptor, weekly$week_start)
  target_reference_week <- baseline$target_week_start - 52L * 7L
  sought_key <- make_key(baseline$market, baseline$region,
                         baseline$product_descriptor, target_reference_week)
  lookup <- match(sought_key, reference_key)
  # An unmatched lookup remains NA, not a carried-forward value or a zero.
  baseline$seasonal_reference_week_start <- target_reference_week
  baseline$seasonal_reference_quote_date <- weekly$quote_date[lookup]
  baseline$seasonal_52_eur_per_kg <- weekly$frequent_eur_per_kg[lookup]
  baseline$seasonal_52_abs_error_eur_per_kg <- abs(
    baseline$actual_eur_per_kg - baseline$seasonal_52_eur_per_kg
  )
  if (any(!is.na(baseline$seasonal_reference_quote_date) &
          baseline$seasonal_reference_quote_date > baseline$origin_end)) {
    stop("A seasonal reference lies after its forecast origin",
         call. = FALSE)
  }
  baseline
}

#' Descriptive three-way scores on **identical** observed target panel/weeks.
#'
#' Separate seasonal-only coverage from the subset on which seasonal naive,
#' last-observed and four-week median predictions all exist. No independent-
#' row uncertainty bounds or claims of a production-grade forecast are made.
#'
#' @param forecasts Output of backtest_sima_seasonal_52().
#' @return List with overall scores, monthly coverage/scores, panel scores
#'   and an honest forecast-availability summary.
summarize_sima_seasonal_52 <- function(forecasts) {
  fields <- c("panel_id", "market", "region", "product_descriptor",
              "target_week_start", "target_quote_date", "origin_end",
              "actual_eur_per_kg", "naive_eur_per_kg",
              "median_4w_eur_per_kg", "seasonal_52_eur_per_kg",
              "seasonal_reference_quote_date")
  if (!is.data.frame(forecasts) || nrow(forecasts) == 0L ||
      !all(fields %in% names(forecasts)) ||
      anyNA(forecasts$actual_eur_per_kg) ||
      any(!is.finite(forecasts$actual_eur_per_kg)) ||
      any(forecasts$actual_eur_per_kg <= 0) ||
      !inherits(forecasts$target_week_start, "Date") ||
      !inherits(forecasts$target_quote_date, "Date") ||
      anyNA(forecasts$target_week_start) ||
      anyNA(forecasts$target_quote_date)) {
    stop("Malformed observed seasonal-backtest rows", call. = FALSE)
  }
  key <- paste(forecasts$market, forecasts$region,
               forecasts$product_descriptor,
               forecasts$target_week_start, sep = "\u001f")
  if (anyDuplicated(key)) {
    stop("Duplicate exact product/market test panel-weeks", call. = FALSE)
  }
  seasonal_ok <- !is.na(forecasts$seasonal_52_eur_per_kg)
  recent_ok <- !is.na(forecasts$naive_eur_per_kg) &
    !is.na(forecasts$median_4w_eur_per_kg)
  common <- seasonal_ok & recent_ok
  if (!any(common)) {
    stop("No three-way common support for seasonal evaluation",
         call. = FALSE)
  }

  used <- forecasts[common, , drop = FALSE]
  methods <- c("last_observed_quote", "median_previous_4_weeks",
               "exact_52_week_seasonal_naive")
  predictions <- c("naive_eur_per_kg", "median_4w_eur_per_kg",
                   "seasonal_52_eur_per_kg")
  panel_key <- paste(used$market, used$region,
                     used$product_descriptor, sep = "\u001f")
  panels <- split(seq_len(nrow(used)), panel_key)
  panel_rows <- lapply(panels, function(indices) {
    x <- used[indices, , drop = FALSE]
    data.frame(
      panel_id = x$panel_id[1L], market = x$market[1L],
      region = x$region[1L], product_descriptor = x$product_descriptor[1L],
      n_three_way_weeks = nrow(x),
      last_quote_mae_eur_per_kg =
        mean(abs(x$actual_eur_per_kg - x$naive_eur_per_kg)),
      four_week_median_mae_eur_per_kg =
        mean(abs(x$actual_eur_per_kg - x$median_4w_eur_per_kg)),
      seasonal_52_mae_eur_per_kg =
        mean(abs(x$actual_eur_per_kg - x$seasonal_52_eur_per_kg)),
      stringsAsFactors = FALSE
    )
  })
  by_panel <- do.call(rbind, panel_rows)
  rownames(by_panel) <- NULL
  panel_columns <- c("last_quote_mae_eur_per_kg",
                     "four_week_median_mae_eur_per_kg",
                     "seasonal_52_mae_eur_per_kg")
  overall_rows <- lapply(seq_along(methods), function(i) {
    absolute_error <- abs(used$actual_eur_per_kg - used[[predictions[i]]])
    data.frame(
      method = methods[i], common_scored_weeks = nrow(used),
      scored_panels = nrow(by_panel),
      observation_weighted_mae_eur_per_kg = mean(absolute_error),
      equal_panel_mae_eur_per_kg = mean(by_panel[[panel_columns[i]]]),
      weighted_absolute_percentage_error =
        sum(absolute_error) / sum(used$actual_eur_per_kg),
      stringsAsFactors = FALSE
    )
  })
  overall <- do.call(rbind, overall_rows)

  # Month/year follow target quote_date; a week can straddle two months.
  target_month <- format(forecasts$target_quote_date, "%Y-%m")
  month_rows <- lapply(split(seq_len(nrow(forecasts)), target_month),
                       function(indices) {
    x <- forecasts[indices, , drop = FALSE]
    seasonal <- !is.na(x$seasonal_52_eur_per_kg)
    historical <- !is.na(x$naive_eur_per_kg) &
      !is.na(x$median_4w_eur_per_kg)
    paired <- seasonal & historical
    mae <- function(column) {
      if (!any(paired)) NA_real_ else
        mean(abs(x$actual_eur_per_kg[paired] - x[[column]][paired]))
    }
    data.frame(
      quote_month = format(x$target_quote_date[1L], "%Y-%m"),
      observed_target_weeks = nrow(x),
      seasonal_available_weeks = sum(seasonal),
      recent_baseline_paired_weeks = sum(historical),
      three_way_scored_weeks = sum(paired),
      three_way_coverage_fraction = mean(paired),
      three_way_panels = length(unique(x$panel_id[paired])),
      last_quote_mae_eur_per_kg = mae("naive_eur_per_kg"),
      four_week_median_mae_eur_per_kg = mae("median_4w_eur_per_kg"),
      seasonal_52_mae_eur_per_kg = mae("seasonal_52_eur_per_kg"),
      stringsAsFactors = FALSE
    )
  })
  monthly <- do.call(rbind, month_rows)
  rownames(monthly) <- NULL

  coverage <- data.frame(
    observed_target_panel_weeks = nrow(forecasts),
    seasonal_52_available_panel_weeks = sum(seasonal_ok),
    recent_baseline_paired_panel_weeks = sum(recent_ok),
    three_way_scored_panel_weeks = sum(common),
    seasonal_52_coverage_fraction = mean(seasonal_ok),
    three_way_coverage_fraction = mean(common),
    observed_target_panels = length(unique(forecasts$panel_id)),
    three_way_panels = nrow(by_panel),
    stringsAsFactors = FALSE
  )
  list(overall = overall, monthly = monthly, panel = by_panel,
       coverage = coverage)
}
