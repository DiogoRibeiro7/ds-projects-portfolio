# Rolling-origin evaluation of observed GPP SIMA orange quotations.
# Source sima_market_prices.R first. Uses base R only, never model simulations.

#' Validate the typed output from read_sima_orange_prices().
#' @param quotes Normalized original quotation rows, not an interpolated series.
#' @return TRUE invisibly; errors identify unsupported or invalid inputs.
validate_sima_backtest_quotes <- function(quotes) {
  required <- c("quote_date", "product_descriptor", "region", "market",
                "market_stage", "price_unit", "min_eur_per_kg",
                "max_eur_per_kg", "frequent_eur_per_kg", "is_provisional")
  if (!is.data.frame(quotes) || nrow(quotes) < 1L ||
      !all(required %in% names(quotes))) {
    stop("Expected nonempty, normalized SIMA quotation records", call. = FALSE)
  }
  if (!inherits(quotes$quote_date, "Date") || anyNA(quotes$quote_date)) {
    stop("quote_date must be a complete Date vector", call. = FALSE)
  }
  for (field in c("product_descriptor", "region", "market", "market_stage",
                  "price_unit")) {
    values <- quotes[[field]]
    if (!is.character(values) || anyNA(values) ||
        any(!nzchar(trimws(values)))) {
      stop(sprintf("%s must have nonempty character values", field),
           call. = FALSE)
    }
  }
  if (any(quotes$market_stage != "wholesale") ||
      any(quotes$price_unit != "EUR/kg")) {
    stop("Only like-for-like wholesale EUR/kg observations are supported",
         call. = FALSE)
  }
  for (field in c("min_eur_per_kg", "max_eur_per_kg",
                  "frequent_eur_per_kg")) {
    values <- quotes[[field]]
    if (!is.numeric(values) || anyNA(values) ||
        any(!is.finite(values)) || any(values <= 0)) {
      stop(sprintf("%s must contain positive, finite prices", field),
           call. = FALSE)
    }
  }
  if (any(quotes$min_eur_per_kg > quotes$frequent_eur_per_kg |
          quotes$frequent_eur_per_kg > quotes$max_eur_per_kg)) {
    stop("Frequent prices must lie within each reported SIMA range",
         call. = FALSE)
  }
  if (!is.logical(quotes$is_provisional)) {
    stop("is_provisional must be a logical column, with NA for unknown",
         call. = FALSE)
  }
  if (anyDuplicated(quotes[c("quote_date", "product_descriptor", "region", "market")])) {
    stop("Duplicate quotations for one product, market, region and date",
         call. = FALSE)
  }
  invisible(TRUE)
}

#' Find the Monday of the ISO-style calendar week for each Date.
#' @param dates One or more nonmissing Date values.
#' @return Date vector of Monday week starts (no string-based week parsing).
sima_monday <- function(dates) {
  if (!inherits(dates, "Date") || anyNA(dates)) {
    stop("dates must be nonmissing Date values", call. = FALSE)
  }
  weekday_sunday_zero <- as.POSIXlt(dates)$wday
  dates - as.integer((weekday_sunday_zero + 6L) %% 7L)
}

#' Reduce repeated observations to the final quoted day per calendar week.
#'
#' Each unique market/region/full-product descriptor is a separate panel.
#' This explicitly avoids mixing oranges of different cultivar, size, grade,
#' package or geographic origin. The within-week final quote is selected by
#' its observed date, without imputing missing calendar weeks.
#'
#' @param quotes Output of read_sima_orange_prices(). Explicitly provisional
#'   records are discarded; unknown status is preserved and not called final.
#' @return One sorted record per observed panel/calendar week.
sima_weekly_panels <- function(quotes) {
  validate_sima_backtest_quotes(quotes)
  retain <- is.na(quotes$is_provisional) | !quotes$is_provisional
  if (!any(retain)) {
    stop("No non-provisional or unknown quotations remain", call. = FALSE)
  }
  work <- quotes[retain, , drop = FALSE]
  work$week_start <- sima_monday(work$quote_date)
  # An internal separator avoids accidental collisions between field values.
  work$panel_key <- paste(work$market, work$region,
                          work$product_descriptor, sep = "\u001f")
  ord <- order(work$panel_key, work$week_start, work$quote_date,
               method = "radix")
  work <- work[ord, , drop = FALSE]
  weekly_key <- paste(work$panel_key, work$week_start, sep = "\u001e")
  work$quotes_in_week <- as.integer(ave(rep.int(1L, nrow(work)), weekly_key,
                                        FUN = length))
  work <- work[!duplicated(weekly_key, fromLast = TRUE), , drop = FALSE]
  work$week_end <- work$week_start + 6L
  work$panel_id <- paste(work$market, work$region,
                         work$product_descriptor, sep = " | ")
  fields <- c("panel_id", "market", "region", "product_descriptor",
              "week_start", "week_end", "quote_date", "quotes_in_week",
              "min_eur_per_kg", "max_eur_per_kg", "frequent_eur_per_kg",
              "is_provisional", "price_unit", "market_stage")
  result <- work[, fields, drop = FALSE]
  result <- result[order(result$week_start, result$panel_id,
                         method = "radix"), , drop = FALSE]
  rownames(result) <- NULL
  result
}

#' Check nonnegative integer-like parameters explicitly.
#' @param value Numeric scalar parameter.
#' @param label Error-message parameter name.
#' @param minimum Smallest allowed value.
#' @return TRUE invisibly, or an error.
assert_sima_integer <- function(value, label, minimum = 1L) {
  if (!is.numeric(value) || length(value) != 1L || is.na(value) ||
      !is.finite(value) || value < minimum || value != floor(value)) {
    stop(sprintf("%s must be a finite whole number >= %d", label, minimum),
         call. = FALSE)
  }
  invisible(TRUE)
}

#' Make one-week-ahead predictions at the end of each preceding week.
#'
#' A forecast for calendar week t may inspect only observations from weeks
#' strictly before t. Candidate methods are predeclared: the most recent
#' observed quote and the median of the preceding four *calendar* weeks.
#' Neither fills missing weeks; both have a maximum age for the last quote.
#'
#' @param weekly One observation per panel/week from sima_weekly_panels().
#' @param test_start First Monday to evaluate, fixed before scoring.
#' @param min_history_weeks Minimum unique prior observed weeks per panel.
#' @param lookback_weeks Number of previous calendar weeks used by the median.
#' @param min_recent_weeks Minimum observed weeks in that median window.
#' @param max_age_weeks Longest acceptable age of the most recent quote.
#' @return All observed target weeks in the test period, including targets
#'   with unavailable forecasts (NA). Price errors are in EUR/kg.
backtest_sima_weekly_prices <- function(weekly, test_start = as.Date("2026-01-05"),
                                        min_history_weeks = 8L,
                                        lookback_weeks = 4L,
                                        min_recent_weeks = 2L,
                                        max_age_weeks = 3L) {
  fields <- c("panel_id", "market", "region", "product_descriptor",
              "week_start", "quote_date", "frequent_eur_per_kg")
  if (!is.data.frame(weekly) || nrow(weekly) < 1L ||
      !all(fields %in% names(weekly)) ||
      !inherits(weekly$week_start, "Date") || anyNA(weekly$week_start) ||
      !inherits(weekly$quote_date, "Date") || anyNA(weekly$quote_date) ||
      !is.numeric(weekly$frequent_eur_per_kg) ||
      anyNA(weekly$frequent_eur_per_kg) ||
      any(!is.finite(weekly$frequent_eur_per_kg)) ||
      any(weekly$frequent_eur_per_kg <= 0) ||
      anyDuplicated(weekly[c("panel_id", "week_start")])) {
    stop("weekly needs valid, unique, positive panel/week quotations",
         call. = FALSE)
  }
  if (!inherits(test_start, "Date") || length(test_start) != 1L ||
      is.na(test_start) || sima_monday(test_start) != test_start) {
    stop("test_start must be exactly one Monday Date", call. = FALSE)
  }
  assert_sima_integer(min_history_weeks, "min_history_weeks")
  assert_sima_integer(lookback_weeks, "lookback_weeks")
  assert_sima_integer(min_recent_weeks, "min_recent_weeks")
  assert_sima_integer(max_age_weeks, "max_age_weeks")
  if (min_recent_weeks > lookback_weeks) {
    stop("min_recent_weeks cannot exceed lookback_weeks", call. = FALSE)
  }
  if (any(weekly$quote_date < weekly$week_start |
          weekly$quote_date > weekly$week_start + 6L)) {
    stop("Each quote must lie within its stated calendar week", call. = FALSE)
  }

  panel_rows <- split(seq_len(nrow(weekly)), weekly$panel_id)
  forecast_rows <- lapply(panel_rows, function(row_indices) {
    panel <- weekly[row_indices, , drop = FALSE]
    panel <- panel[order(panel$week_start), , drop = FALSE]
    target_rows <- which(panel$week_start >= test_start)
    if (length(target_rows) == 0L) return(NULL)
    rows <- lapply(target_rows, function(i) {
      target <- panel[i, , drop = FALSE]
      history <- panel[panel$week_start < target$week_start, , drop = FALSE]
      n_history <- nrow(history)
      last_quote <- if (n_history == 0L) NA_real_ else
        history$frequent_eur_per_kg[n_history]
      last_date <- if (n_history == 0L) as.Date(NA) else
        history$quote_date[n_history]
      age <- if (n_history == 0L) NA_integer_ else
        as.integer(target$week_start - history$week_start[n_history]) %/% 7L
      recent <- history[
        history$week_start >= target$week_start - 7L * lookback_weeks,
        , drop = FALSE
      ]
      allowed <- n_history >= min_history_weeks && !is.na(age) &&
        age <= max_age_weeks
      naive <- if (allowed) last_quote else NA_real_
      median_4w <- if (allowed && nrow(recent) >= min_recent_weeks) {
        stats::median(recent$frequent_eur_per_kg)
      } else NA_real_
      actual <- target$frequent_eur_per_kg
      data.frame(
        panel_id = target$panel_id, market = target$market,
        region = target$region,
        product_descriptor = target$product_descriptor,
        target_week_start = target$week_start,
        origin_end = target$week_start - 1L,
        target_quote_date = target$quote_date,
        last_training_quote_date = last_date,
        n_history_weeks = n_history, n_recent_weeks = nrow(recent),
        latest_quote_age_weeks = age, actual_eur_per_kg = actual,
        naive_eur_per_kg = naive, median_4w_eur_per_kg = median_4w,
        naive_abs_error_eur_per_kg = abs(actual - naive),
        median_4w_abs_error_eur_per_kg = abs(actual - median_4w),
        stringsAsFactors = FALSE
      )
    })
    do.call(rbind, rows)
  })
  forecast_rows <- Filter(Negate(is.null), forecast_rows)
  if (length(forecast_rows) == 0L) {
    stop("No observed target weeks occur on or after test_start",
         call. = FALSE)
  }
  result <- do.call(rbind, forecast_rows)
  result <- result[order(result$target_week_start, result$panel_id,
                         method = "radix"), , drop = FALSE]
  rownames(result) <- NULL
  result
}

#' Descriptive paired errors and coverage, with panels weighted equally.
#'
#' Only rows where both forecast methods were available enter comparisons.
#' Report both observation-weighted MAE and equal-panel MAE to show effects
#' of unbalanced panel coverage. No IID confidence interval is claimed.
#' @param forecasts Output of backtest_sima_weekly_prices().
#' @return List of overall summary, panel summary and coverage data frames.
summarize_sima_rolling_prices <- function(forecasts) {
  required <- c("panel_id", "market", "region", "product_descriptor",
                "actual_eur_per_kg", "naive_eur_per_kg",
                "median_4w_eur_per_kg", "target_week_start", "origin_end")
  if (!is.data.frame(forecasts) || nrow(forecasts) < 1L ||
      !all(required %in% names(forecasts)) ||
      anyNA(forecasts$actual_eur_per_kg) ||
      any(!is.finite(forecasts$actual_eur_per_kg)) ||
      any(forecasts$actual_eur_per_kg <= 0)) {
    stop("Malformed rolling-forecast evaluation records", call. = FALSE)
  }
  paired <- !is.na(forecasts$naive_eur_per_kg) &
    !is.na(forecasts$median_4w_eur_per_kg)
  if (!any(paired)) {
    stop("No paired forecasts satisfy the predeclared coverage criteria",
         call. = FALSE)
  }
  evaluated <- forecasts[paired, , drop = FALSE]
  by_panel <- split(seq_len(nrow(evaluated)), evaluated$panel_id)
  panel_rows <- lapply(by_panel, function(indices) {
    x <- evaluated[indices, , drop = FALSE]
    data.frame(
      panel_id = x$panel_id[1L], market = x$market[1L],
      region = x$region[1L], product_descriptor = x$product_descriptor[1L],
      n_paired = nrow(x),
      naive_mae_eur_per_kg = mean(abs(x$actual_eur_per_kg - x$naive_eur_per_kg)),
      median_4w_mae_eur_per_kg = mean(abs(x$actual_eur_per_kg -
                                               x$median_4w_eur_per_kg)),
      stringsAsFactors = FALSE
    )
  })
  panel <- do.call(rbind, panel_rows)
  rownames(panel) <- NULL
  naive_error <- abs(evaluated$actual_eur_per_kg - evaluated$naive_eur_per_kg)
  median_error <- abs(evaluated$actual_eur_per_kg - evaluated$median_4w_eur_per_kg)
  overall <- data.frame(
    method = c("last_observed_quote", "median_previous_4_weeks"),
    paired_quotes = nrow(evaluated), panels = nrow(panel),
    observation_weighted_mae_eur_per_kg = c(mean(naive_error), mean(median_error)),
    equal_panel_mae_eur_per_kg = c(mean(panel$naive_mae_eur_per_kg),
                                   mean(panel$median_4w_mae_eur_per_kg)),
    weighted_absolute_percentage_error = c(
      sum(naive_error) / sum(evaluated$actual_eur_per_kg),
      sum(median_error) / sum(evaluated$actual_eur_per_kg)
    ), stringsAsFactors = FALSE
  )
  coverage <- data.frame(
    observed_target_panel_weeks = nrow(forecasts),
    scored_paired_panel_weeks = nrow(evaluated),
    paired_coverage_fraction = nrow(evaluated) / nrow(forecasts),
    total_target_panels = length(unique(forecasts$panel_id)),
    scored_panels = nrow(panel), stringsAsFactors = FALSE
  )
  list(overall = overall, panel = panel, coverage = coverage)
}
