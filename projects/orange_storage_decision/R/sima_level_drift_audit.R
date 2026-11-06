# An observational decomposition of the exact 52-week SIMA price forecast.
# Source sima_market_prices.R, sima_rolling_backtest.R,
# sima_composition_audit.R and sima_seasonal_benchmark.R first.
# Base R only. No model fitting, imputation or inferred publication dates.

#' Decompose the prior-year price error into an already-observed level gap
#' and the subsequently observed change from the latest historical quote.
#'
#' On three-way common support the accounting identity is
#' actual - prior_year = (last_quote - prior_year) + (actual - last_quote).
#' This is a diagnostic identity, not a causal or independent contribution
#' decomposition. The latest quote can be up to three weeks old.
#'
#' @param forecasts Output of backtest_sima_seasonal_52().
#' @return Original observed rows with dated, signed error components,
#'   exact-year nominal ratios and an explicit common-support flag.
decompose_sima_seasonal_errors <- function(forecasts) {
  required <- c("panel_id", "market", "region", "product_descriptor",
                "target_week_start", "target_quote_date", "origin_end",
                "actual_eur_per_kg", "naive_eur_per_kg",
                "median_4w_eur_per_kg", "seasonal_52_eur_per_kg",
                "seasonal_reference_quote_date")
  if (!is.data.frame(forecasts) || nrow(forecasts) == 0L ||
      !all(required %in% names(forecasts))) {
    stop("Expected complete seasonal forecast columns", call. = FALSE)
  }
  for (field in c("target_week_start", "target_quote_date", "origin_end",
                  "seasonal_reference_quote_date")) {
    if (!inherits(forecasts[[field]], "Date")) {
      stop(sprintf("%s must be a Date vector", field), call. = FALSE)
    }
  }
  for (field in c("actual_eur_per_kg", "naive_eur_per_kg",
                  "median_4w_eur_per_kg", "seasonal_52_eur_per_kg")) {
    value <- forecasts[[field]]
    if (!is.numeric(value) ||
        any(!is.na(value) & (!is.finite(value) | value <= 0))) {
      stop(sprintf("%s must have positive finite available prices", field),
           call. = FALSE)
    }
  }
  if (anyNA(forecasts$actual_eur_per_kg) ||
      anyNA(forecasts$origin_end) ||
      anyNA(forecasts$target_quote_date) ||
      any(forecasts$origin_end >= forecasts$target_week_start) ||
      any(!is.na(forecasts$seasonal_reference_quote_date) &
            forecasts$seasonal_reference_quote_date > forecasts$origin_end) ||
      any(is.na(forecasts$seasonal_reference_quote_date) !=
            is.na(forecasts$seasonal_52_eur_per_kg))) {
    stop("Invalid observation chronology or missing reference dates",
         call. = FALSE)
  }
  # Validate all forecast keys and common availability against the frozen
  # benchmark, including duplicates across exact product/market/week keys.
  invisible(summarize_sima_seasonal_52(forecasts))

  result <- forecasts
  result$seasonal_available <- !is.na(result$seasonal_52_eur_per_kg)
  result$recent_baselines_available <- !is.na(result$naive_eur_per_kg) &
    !is.na(result$median_4w_eur_per_kg)
  result$three_way_common <- result$seasonal_available &
    result$recent_baselines_available

  # Ratios are calculated on exact 364-day reference pairs, including rows
  # that cannot be used for a full three-way forecast comparison.
  result$nominal_yoy_price_change_fraction <-
    result$actual_eur_per_kg / result$seasonal_52_eur_per_kg - 1
  result$nominal_yoy_signed_change_eur_per_kg <-
    result$actual_eur_per_kg - result$seasonal_52_eur_per_kg

  # Compute accounting components only where all three rules are available.
  result$pre_origin_level_gap_eur_per_kg <- NA_real_
  result$recent_quote_innovation_eur_per_kg <- NA_real_
  result$seasonal_signed_error_eur_per_kg <- NA_real_
  result$decomposition_residual_eur_per_kg <- NA_real_
  matched <- which(result$three_way_common)
  result$pre_origin_level_gap_eur_per_kg[matched] <-
    result$naive_eur_per_kg[matched] -
    result$seasonal_52_eur_per_kg[matched]
  result$recent_quote_innovation_eur_per_kg[matched] <-
    result$actual_eur_per_kg[matched] - result$naive_eur_per_kg[matched]
  result$seasonal_signed_error_eur_per_kg[matched] <-
    result$actual_eur_per_kg[matched] -
    result$seasonal_52_eur_per_kg[matched]
  result$decomposition_residual_eur_per_kg[matched] <-
    result$seasonal_signed_error_eur_per_kg[matched] -
    result$pre_origin_level_gap_eur_per_kg[matched] -
    result$recent_quote_innovation_eur_per_kg[matched]
  if (any(abs(result$decomposition_residual_eur_per_kg[matched]) > 1e-10)) {
    stop("Signed forecast-error identity is inconsistent", call. = FALSE)
  }
  result
}

#' Summarize observed nominal level differences on an identical support.
#'
#' Monthly, market and panel contrasts use only three-way common rows.
#' Report coverage separately: an absent prior-year quotation does not
#' imply a zero change. Nominal 52-week changes are not a price index.
#'
#' @param forecasts Output of backtest_sima_seasonal_52().
#' @return List of dated decomposition rows, overall metrics, coverage,
#'   monthly, wholesale-market and exact-panel descriptive comparisons.
summarize_sima_level_drift <- function(forecasts) {
  work <- decompose_sima_seasonal_errors(forecasts)

  # Each group's denominator is explicit; no aggregation silently changes
  # the population being compared across the three forecast methods.
  group_stats <- function(indices) {
    x <- work[indices, , drop = FALSE]
    common <- x[x$three_way_common, , drop = FALSE]
    n <- nrow(common)
    observed <- nrow(x)
    statistic <- function(f) if (n == 0L) NA_real_ else f(common)
    data.frame(
      observed_target_panel_weeks = observed,
      seasonal_52_available_panel_weeks = sum(x$seasonal_available),
      recent_baselines_available_panel_weeks =
        sum(x$recent_baselines_available),
      three_way_common_panel_weeks = n,
      common_fraction_of_observed = n / observed,
      mean_signed_yoy_change_eur_per_kg =
        statistic(function(z) mean(z$nominal_yoy_signed_change_eur_per_kg)),
      median_nominal_yoy_change_fraction =
        statistic(function(z) stats::median(z$nominal_yoy_price_change_fraction)),
      share_yoy_quotes_higher =
        statistic(function(z) mean(z$nominal_yoy_signed_change_eur_per_kg > 0)),
      share_yoy_quotes_lower =
        statistic(function(z) mean(z$nominal_yoy_signed_change_eur_per_kg < 0)),
      mean_signed_pre_origin_gap_eur_per_kg =
        statistic(function(z) mean(z$pre_origin_level_gap_eur_per_kg)),
      mean_absolute_pre_origin_gap_eur_per_kg =
        statistic(function(z) mean(abs(z$pre_origin_level_gap_eur_per_kg))),
      mean_absolute_recent_quote_innovation_eur_per_kg =
        statistic(function(z) mean(abs(z$recent_quote_innovation_eur_per_kg))),
      mean_absolute_seasonal_error_eur_per_kg =
        statistic(function(z) mean(abs(z$seasonal_signed_error_eur_per_kg))),
      mean_absolute_four_week_median_error_eur_per_kg =
        statistic(function(z) mean(abs(z$actual_eur_per_kg -
                                      z$median_4w_eur_per_kg))),
      stringsAsFactors = FALSE
    )
  }
  by_group <- function(keys, column) {
    groups <- split(seq_len(nrow(work)), keys)
    rows <- lapply(groups, function(index) {
      data.frame(group_label = keys[index[1L]], group_stats(index),
                 stringsAsFactors = FALSE)
    })
    result <- do.call(rbind, rows)
    names(result)[1L] <- column
    rownames(result) <- NULL
    result
  }

  overall <- group_stats(seq_len(nrow(work)))
  by_month <- by_group(format(work$target_quote_date, "%Y-%m"), "quote_month")
  by_month <- by_month[order(by_month$quote_month), , drop = FALSE]
  by_market <- by_group(work$market, "market")
  by_market <- by_market[order(by_market$market), , drop = FALSE]
  exact_panel_key <- paste(work$market, work$region,
                           work$product_descriptor, sep = "\u001f")
  by_panel <- by_group(exact_panel_key, "exact_panel_key")
  positions <- match(by_panel$exact_panel_key, exact_panel_key)
  by_panel$panel_id <- work$panel_id[positions]
  by_panel$market <- work$market[positions]
  by_panel$region <- work$region[positions]
  by_panel$product_descriptor <- work$product_descriptor[positions]
  by_panel$exact_panel_key <- NULL
  by_panel <- by_panel[, c("panel_id", "market", "region",
                           "product_descriptor", setdiff(names(by_panel),
                                                     c("panel_id", "market",
                                                       "region", "product_descriptor"))),
                       drop = FALSE]

  coverage <- data.frame(
    observed_target_panel_weeks = nrow(work),
    seasonal_52_available_panel_weeks = sum(work$seasonal_available),
    recent_baselines_available_panel_weeks =
      sum(work$recent_baselines_available),
    three_way_common_panel_weeks = sum(work$three_way_common),
    three_way_common_panels = sum(by_panel$three_way_common_panel_weeks > 0L),
    three_way_fraction_of_observed = mean(work$three_way_common),
    stringsAsFactors = FALSE
  )
  list(rows = work, overall = overall, coverage = coverage,
       monthly = by_month, market = by_market, panel = by_panel)
}
