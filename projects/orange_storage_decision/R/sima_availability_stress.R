# Counterfactual information-availability tests for dated GPP SIMA quotations.
# Source sima_market_prices.R and sima_rolling_backtest.R before this file.
# Base R only. Calendar-day lags below are assumptions, not measured delays.

#' Check a set of hypothetical, nonnegative calendar-day release delays.
#' @param lag_days Unique numeric whole-day delays; must include zero.
#' @return The delays as an integer vector, in the supplied order.
validate_sima_lag_days <- function(lag_days) {
  if (!is.numeric(lag_days) || length(lag_days) < 1L ||
      anyNA(lag_days) || any(!is.finite(lag_days)) ||
      any(lag_days < 0) || any(lag_days != floor(lag_days)) ||
      anyDuplicated(lag_days) || !any(lag_days == 0) ||
      any(lag_days > .Machine$integer.max)) {
    stop("lag_days must be unique, nonnegative whole days including zero",
         call. = FALSE)
  }
  as.integer(lag_days)
}

#' Replay weekly price forecasts under assumed observation-to-release delays.
#'
#' Rebuild the **available** weekly history at each forecast origin from the
#' original dated rows. Re-aggregating is essential: when the final quote in a
#' historical week would not yet be available, an earlier dated quote from that
#' week may still be usable. The targets always remain the original final
#' observed quotation in each test week; they are never selected using a lag.
#'
#' This is an information-set stress test, not a reconstruction of the true
#' GPP publication chronology. SIMA does not provide publication timestamps
#' or revision histories in this frozen CSV.
#'
#' @param quotes Normalized GPP wholesale orange rows from the source importer.
#' @param lag_days Assumed calendar days from observation to availability.
#' @param test_start First Monday of the held-out test period.
#' @param min_history_weeks Minimum prior observed weeks needed to forecast.
#' @param lookback_weeks Previous calendar weeks used by the median benchmark.
#' @param min_recent_weeks Minimum observations for the median benchmark.
#' @param max_age_weeks Maximum age of the most recently available week.
#' @return One row per assumed delay and observed test panel/week. Unavailable
#'   forecasts are explicit NA, retaining the original target denominator.
backtest_sima_availability_lags <- function(
    quotes, lag_days = c(0L, 3L, 7L, 14L, 21L),
    test_start = as.Date("2026-01-05"), min_history_weeks = 8L,
    lookback_weeks = 4L, min_recent_weeks = 2L,
    max_age_weeks = 3L) {
  validate_sima_backtest_quotes(quotes)
  lag_days <- validate_sima_lag_days(lag_days)
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

  # Provisional records are never allowed into either target or history.
  keep <- is.na(quotes$is_provisional) | !quotes$is_provisional
  eligible_quotes <- quotes[keep, , drop = FALSE]
  targets <- sima_weekly_panels(eligible_quotes)
  targets <- targets[targets$week_start >= test_start, , drop = FALSE]
  if (nrow(targets) == 0L) {
    stop("No observed test targets remain", call. = FALSE)
  }

  # An internal, non-printing delimiter retains exact product identities.
  quote_key <- paste(eligible_quotes$market, eligible_quotes$region,
                     eligible_quotes$product_descriptor, sep = "\u001f")
  target_key <- paste(targets$market, targets$region,
                      targets$product_descriptor, sep = "\u001f")
  raw_indices <- split(seq_len(nrow(eligible_quotes)), quote_key)
  target_indices <- split(seq_len(nrow(targets)), target_key)

  rows <- vector("list", length(lag_days) * nrow(targets))
  position <- 0L
  for (key in names(target_indices)) {
    q <- eligible_quotes[raw_indices[[key]], , drop = FALSE]
    q <- q[order(q$quote_date, method = "radix"), , drop = FALSE]
    q$week_start <- sima_monday(q$quote_date)
    selected_targets <- targets[target_indices[[key]], , drop = FALSE]

    for (i in seq_len(nrow(selected_targets))) {
      target <- selected_targets[i, , drop = FALSE]
      origin <- target$week_start - 1L
      for (lag in lag_days) {
        # Assumed release day = quoted observation day + lag. There are no
        # source release timestamps with which to verify this assumption.
        available <- q[q$quote_date <= origin - lag, , drop = FALSE]
        available <- available[
          !duplicated(available$week_start, fromLast = TRUE), , drop = FALSE
        ]
        n_history <- nrow(available)
        last_price <- if (n_history == 0L) NA_real_ else
          available$frequent_eur_per_kg[n_history]
        last_date <- if (n_history == 0L) as.Date(NA) else
          available$quote_date[n_history]
        age <- if (n_history == 0L) NA_integer_ else
          as.integer(target$week_start - available$week_start[n_history]) %/% 7L
        recent <- available[
          available$week_start >= target$week_start - 7L * lookback_weeks,
          , drop = FALSE
        ]
        allowed <- n_history >= min_history_weeks && !is.na(age) &&
          age <= max_age_weeks
        naive <- if (allowed) last_price else NA_real_
        median_price <- if (allowed && nrow(recent) >= min_recent_weeks) {
          stats::median(recent$frequent_eur_per_kg)
        } else NA_real_
        actual <- target$frequent_eur_per_kg

        position <- position + 1L
        rows[[position]] <- data.frame(
          panel_id = target$panel_id, market = target$market,
          region = target$region,
          product_descriptor = target$product_descriptor,
          target_week_start = target$week_start, origin_end = origin,
          target_quote_date = target$quote_date,
          assumed_release_lag_days = lag,
          last_training_quote_date = last_date,
          last_training_assumed_available = last_date + lag,
          n_history_weeks = n_history, n_recent_weeks = nrow(recent),
          latest_quote_age_weeks = age, actual_eur_per_kg = actual,
          naive_eur_per_kg = naive, median_4w_eur_per_kg = median_price,
          naive_abs_error_eur_per_kg = abs(actual - naive),
          median_4w_abs_error_eur_per_kg = abs(actual - median_price),
          stringsAsFactors = FALSE
        )
      }
    }
  }
  result <- do.call(rbind, rows)
  result <- result[order(result$assumed_release_lag_days,
                         result$target_week_start, result$panel_id,
                         method = "radix"), , drop = FALSE]
  rownames(result) <- NULL
  result
}

#' Summarize scenario coverage and compare errors on identical target rows.
#'
#' The within-scenario summary scores only paired forecasts. Separate
#' matched-target comparisons against zero lag avoid misattributing changing
#' sample composition to forecast accuracy. No independent-row inference is
#' used: weeks, grades, regions and markets are correlated.
#' @param forecasts Output of backtest_sima_availability_lags().
#' @return List with coverage/MAE by lag and matched-target comparisons.
summarize_sima_availability_lags <- function(forecasts) {
  fields <- c("panel_id", "target_week_start", "assumed_release_lag_days",
              "actual_eur_per_kg", "naive_eur_per_kg",
              "median_4w_eur_per_kg", "market", "region",
              "product_descriptor")
  if (!is.data.frame(forecasts) || nrow(forecasts) == 0L ||
      !all(fields %in% names(forecasts)) ||
      anyNA(forecasts$assumed_release_lag_days)) {
    stop("Malformed availability-sensitivity forecasts", call. = FALSE)
  }
  lag_days <- sort(unique(forecasts$assumed_release_lag_days))
  if (!(0L %in% lag_days)) {
    stop("A zero-day scenario is required as the reference", call. = FALSE)
  }
  # The key contains a control character so market/descriptor punctuation
  # cannot accidentally collapse two distinct held-out targets.
  make_key <- function(x) {
    paste(x$market, x$region, x$product_descriptor,
          as.character(x$target_week_start), sep = "\u001f")
  }
  baseline <- forecasts[forecasts$assumed_release_lag_days == 0L, , drop = FALSE]
  baseline_key <- make_key(baseline)
  if (anyDuplicated(baseline_key)) {
    stop("Duplicate zero-lag target identities", call. = FALSE)
  }

  overall_rows <- vector("list", length(lag_days))
  matched_rows <- vector("list", length(lag_days))
  for (i in seq_along(lag_days)) {
    lag <- lag_days[i]
    scenario <- forecasts[
      forecasts$assumed_release_lag_days == lag, , drop = FALSE
    ]
    indices <- match(make_key(scenario), baseline_key)
    if (anyNA(indices) || anyDuplicated(indices) ||
        length(indices) != nrow(baseline) ||
        any(scenario$actual_eur_per_kg !=
              baseline$actual_eur_per_kg[indices])) {
      stop("Lag scenarios must use exactly the same observed targets",
           call. = FALSE)
    }
    base <- baseline[indices, , drop = FALSE]
    paired <- !is.na(scenario$naive_eur_per_kg) &
      !is.na(scenario$median_4w_eur_per_kg)
    base_paired <- !is.na(base$naive_eur_per_kg) &
      !is.na(base$median_4w_eur_per_kg)
    shared <- paired & base_paired
    if (any(paired)) {
      s <- summarize_sima_rolling_prices(scenario)
      naive_mae <- s$overall$observation_weighted_mae_eur_per_kg[1L]
      median_mae <- s$overall$observation_weighted_mae_eur_per_kg[2L]
      equal_naive <- s$overall$equal_panel_mae_eur_per_kg[1L]
      equal_median <- s$overall$equal_panel_mae_eur_per_kg[2L]
    } else {
      naive_mae <- median_mae <- equal_naive <- equal_median <- NA_real_
    }
    overall_rows[[i]] <- data.frame(
      assumed_release_lag_days = lag,
      observed_target_panel_weeks = nrow(scenario),
      paired_panel_weeks = sum(paired),
      paired_coverage_fraction = mean(paired),
      observed_target_panels = length(unique(scenario$panel_id)),
      paired_panels = length(unique(scenario$panel_id[paired])),
      naive_mae_eur_per_kg = naive_mae,
      median_4w_mae_eur_per_kg = median_mae,
      naive_equal_panel_mae_eur_per_kg = equal_naive,
      median_4w_equal_panel_mae_eur_per_kg = equal_median,
      stringsAsFactors = FALSE
    )
    # These columns are NA when no target has paired coverage under a lag.
    base_naive <- if (any(shared)) mean(abs(base$actual_eur_per_kg[shared] -
                                            base$naive_eur_per_kg[shared])) else NA_real_
    lag_naive <- if (any(shared)) mean(abs(scenario$actual_eur_per_kg[shared] -
                                           scenario$naive_eur_per_kg[shared])) else NA_real_
    base_median <- if (any(shared)) mean(abs(base$actual_eur_per_kg[shared] -
                                             base$median_4w_eur_per_kg[shared])) else NA_real_
    lag_median <- if (any(shared)) mean(abs(scenario$actual_eur_per_kg[shared] -
                                            scenario$median_4w_eur_per_kg[shared])) else NA_real_
    matched_rows[[i]] <- data.frame(
      assumed_release_lag_days = lag, matched_panel_weeks = sum(shared),
      baseline_naive_mae_eur_per_kg = base_naive,
      scenario_naive_mae_eur_per_kg = lag_naive,
      naive_mae_change_eur_per_kg = lag_naive - base_naive,
      baseline_median_4w_mae_eur_per_kg = base_median,
      scenario_median_4w_mae_eur_per_kg = lag_median,
      median_4w_mae_change_eur_per_kg = lag_median - base_median,
      stringsAsFactors = FALSE
    )
  }
  list(overall = do.call(rbind, overall_rows),
       matched = do.call(rbind, matched_rows))
}
