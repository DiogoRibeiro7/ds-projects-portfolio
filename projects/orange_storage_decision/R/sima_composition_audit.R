# Coverage and like-for-like monthly prices in the frozen GPP SIMA archive.
# Source sima_market_prices.R and sima_rolling_backtest.R before this file.
# Uses base R; no interpolation, volume inference or publication-date claims.

#' Validate the observed, exact-product weekly GPP SIMA panel.
#' @param weekly Output from sima_weekly_panels().
#' @return TRUE invisibly, otherwise an informative error.
validate_sima_composition_weekly <- function(weekly) {
  required <- c("panel_id", "market", "region", "product_descriptor",
                "week_start", "week_end", "quote_date", "frequent_eur_per_kg")
  if (!is.data.frame(weekly) || nrow(weekly) < 1L ||
      !all(required %in% names(weekly))) {
    stop("Expected nonempty exact-product SIMA weekly quotations", call. = FALSE)
  }
  for (field in c("week_start", "week_end", "quote_date")) {
    if (!inherits(weekly[[field]], "Date") || anyNA(weekly[[field]])) {
      stop(sprintf("%s must be complete Date values", field), call. = FALSE)
    }
  }
  for (field in c("panel_id", "market", "region", "product_descriptor")) {
    x <- weekly[[field]]
    if (!is.character(x) || anyNA(x) || any(!nzchar(trimws(x)))) {
      stop(sprintf("%s must contain nonempty strings", field), call. = FALSE)
    }
  }
  x <- weekly$frequent_eur_per_kg
  if (!is.numeric(x) || anyNA(x) || any(!is.finite(x)) || any(x <= 0) ||
      any(weekly$week_end != weekly$week_start + 6L) ||
      any(weekly$quote_date < weekly$week_start |
            weekly$quote_date > weekly$week_end) ||
      anyDuplicated(weekly[c("market", "region",
                             "product_descriptor", "week_start")])) {
    stop("Invalid, duplicate or out-of-week SIMA observations",
         call. = FALSE)
  }
  invisible(TRUE)
}

#' One monthly price summary for each exact observed product/region/market.
#'
#' Month/year are defined by quote_date, **not** Monday week_start. The
#' latter may be in December for a January 2 quotation. Each observed
#' panel-week has equal weight in a panel/month; missing weeks stay absent.
#'
#' @param weekly Output of sima_weekly_panels().
#' @return Panel-month medians, means, first/last dates and observed counts.
sima_panel_monthly_prices <- function(weekly) {
  validate_sima_composition_weekly(weekly)
  work <- weekly
  work$year <- as.integer(format(work$quote_date, "%Y"))
  work$month <- as.integer(format(work$quote_date, "%m"))
  # Delimiters not present in the source's dimension labels prevent
  # accidental key collisions when a product descriptor contains punctuation.
  key <- paste(work$market, work$region, work$product_descriptor,
               work$year, work$month, sep = "\u001f")
  rows <- lapply(split(seq_len(nrow(work)), key), function(indices) {
    x <- work[indices, , drop = FALSE]
    data.frame(
      panel_id = x$panel_id[1L], market = x$market[1L],
      region = x$region[1L], product_descriptor = x$product_descriptor[1L],
      year = x$year[1L], month = x$month[1L],
      observed_weeks = nrow(x),
      median_eur_per_kg = stats::median(x$frequent_eur_per_kg),
      mean_eur_per_kg = mean(x$frequent_eur_per_kg),
      first_quote_date = min(x$quote_date),
      last_quote_date = max(x$quote_date), stringsAsFactors = FALSE
    )
  })
  result <- do.call(rbind, rows)
  result <- result[order(result$year, result$month, result$panel_id,
                         method = "radix"), , drop = FALSE]
  rownames(result) <- NULL
  result
}

#' Descriptive weekly quote density within each panel's observed span.
#'
#' The unquoted weeks are NOT labelled missing data: oranges can be seasonal,
#' and there may be no market quotation for a valid economic reason.
#' @param weekly Output of sima_weekly_panels().
#' @return Coverage and first/last quotation dates per exact panel.
sima_panel_quotation_spans <- function(weekly) {
  validate_sima_composition_weekly(weekly)
  key <- paste(weekly$market, weekly$region, weekly$product_descriptor,
               sep = "\u001f")
  rows <- lapply(split(seq_len(nrow(weekly)), key), function(indices) {
    x <- weekly[indices, , drop = FALSE]
    span <- as.integer(max(x$week_start) - min(x$week_start)) %/% 7L + 1L
    data.frame(
      panel_id = x$panel_id[1L], market = x$market[1L],
      region = x$region[1L], product_descriptor = x$product_descriptor[1L],
      first_quote_date = min(x$quote_date),
      last_quote_date = max(x$quote_date),
      observed_weeks = nrow(x), calendar_weeks_in_observed_span = span,
      unquoted_weeks_within_span = span - nrow(x),
      fraction_of_span_with_quote = nrow(x) / span,
      stringsAsFactors = FALSE
    )
  })
  result <- do.call(rbind, rows)
  result <- result[order(result$market, result$panel_id,
                         method = "radix"), , drop = FALSE]
  rownames(result) <- NULL
  result
}

#' Compare the same calendar month across two consecutive quoted years.
#'
#' The raw pooled **observed-week** mean is contrasted against both a
#' matched-panel observed-week mean and an equal-panel geometric ratio of
#' monthly median prices. These are distinct estimands; differences must not
#' be called a causal composition effect or a national price index.
#'
#' @param monthly Output of sima_panel_monthly_prices().
#' @param base_year First calendar year (e.g. 2025).
#' @param comparison_year The following calendar year (e.g. 2026).
#' @param months Unique calendar months to compare, in desired order.
#' @return List of the year/month overview and exact matched-panel pairs.
sima_compare_monthly_panel_mix <- function(
    monthly, base_year = 2025L, comparison_year = 2026L,
    months = 1:10) {
  fields <- c("panel_id", "market", "region", "product_descriptor",
              "year", "month", "observed_weeks",
              "median_eur_per_kg", "mean_eur_per_kg")
  if (!is.data.frame(monthly) || nrow(monthly) < 1L ||
      !all(fields %in% names(monthly))) {
    stop("Expected a valid monthly panel-price table", call. = FALSE)
  }
  assert_sima_integer(base_year, "base_year")
  assert_sima_integer(comparison_year, "comparison_year")
  if (comparison_year != base_year + 1L) {
    stop("Year-over-year comparisons require consecutive years",
         call. = FALSE)
  }
  if (!is.numeric(months) || length(months) < 1L || anyNA(months) ||
      any(!is.finite(months)) || any(months != floor(months)) ||
      any(months < 1L | months > 12L) || anyDuplicated(months)) {
    stop("months must be unique calendar month numbers from 1 to 12",
         call. = FALSE)
  }
  for (field in c("year", "month", "observed_weeks",
                  "median_eur_per_kg", "mean_eur_per_kg")) {
    x <- monthly[[field]]
    if (!is.numeric(x) || anyNA(x) || any(!is.finite(x))) {
      stop(sprintf("%s needs valid finite numeric values", field),
           call. = FALSE)
    }
  }
  if (any(monthly$observed_weeks < 1) ||
      any(monthly$observed_weeks != floor(monthly$observed_weeks)) ||
      any(monthly$median_eur_per_kg <= 0) ||
      any(monthly$mean_eur_per_kg <= 0) ||
      any(monthly$month < 1 | monthly$month > 12)) {
    stop("Invalid monthly coverage, calendar month or price",
         call. = FALSE)
  }
  make_key <- function(x) {
    paste(x$market, x$region, x$product_descriptor, sep = "\u001f")
  }
  monthly_key <- paste(make_key(monthly), monthly$year, monthly$month,
                       sep = "\u001f")
  if (anyDuplicated(monthly_key)) {
    stop("One exact product/market has duplicate monthly summaries",
         call. = FALSE)
  }
  pooled_mean <- function(x) {
    if (nrow(x) == 0L) return(NA_real_)
    sum(x$mean_eur_per_kg * x$observed_weeks) / sum(x$observed_weeks)
  }
  growth <- function(first, second) {
    if (is.na(first) || is.na(second)) NA_real_ else second / first - 1
  }

  overview <- vector("list", length(months))
  matched_rows <- list()
  for (i in seq_along(months)) {
    month <- as.integer(months[i])
    base <- monthly[monthly$year == base_year &
                      monthly$month == month, , drop = FALSE]
    later <- monthly[monthly$year == comparison_year &
                       monthly$month == month, , drop = FALSE]
    base_key <- make_key(base)
    later_key <- make_key(later)
    matched_key <- intersect(base_key, later_key)
    a <- base[match(matched_key, base_key), , drop = FALSE]
    b <- later[match(matched_key, later_key), , drop = FALSE]

    if (length(matched_key) > 0L) {
      ratios <- b$median_eur_per_kg / a$median_eur_per_kg
      pair <- data.frame(
        market = a$market, region = a$region,
        product_descriptor = a$product_descriptor, panel_id = a$panel_id,
        month = month, base_year = base_year,
        comparison_year = comparison_year,
        base_observed_weeks = a$observed_weeks,
        comparison_observed_weeks = b$observed_weeks,
        base_monthly_median_eur_per_kg = a$median_eur_per_kg,
        comparison_monthly_median_eur_per_kg = b$median_eur_per_kg,
        log_median_price_ratio = log(ratios),
        median_price_change_fraction = ratios - 1,
        stringsAsFactors = FALSE
      )
      matched_rows[[length(matched_rows) + 1L]] <- pair
      matched_geometric_growth <- exp(mean(log(ratios))) - 1
    } else {
      matched_geometric_growth <- NA_real_
    }

    base_n <- sum(base$observed_weeks)
    later_n <- sum(later$observed_weeks)
    overview[[i]] <- data.frame(
      month = month,
      base_year = base_year, comparison_year = comparison_year,
      base_panels = nrow(base), comparison_panels = nrow(later),
      matched_panels = length(matched_key),
      base_only_panels = length(setdiff(base_key, later_key)),
      comparison_only_panels = length(setdiff(later_key, base_key)),
      base_observed_panel_weeks = base_n,
      comparison_observed_panel_weeks = later_n,
      base_matched_panel_weeks = sum(a$observed_weeks),
      comparison_matched_panel_weeks = sum(b$observed_weeks),
      base_matched_week_share = if (base_n > 0) sum(a$observed_weeks) / base_n else NA_real_,
      comparison_matched_week_share = if (later_n > 0) sum(b$observed_weeks) / later_n else NA_real_,
      pooled_week_mean_change_fraction = growth(pooled_mean(base),
                                                 pooled_mean(later)),
      matched_week_mean_change_fraction = growth(pooled_mean(a),
                                                  pooled_mean(b)),
      matched_panel_geometric_change_fraction = matched_geometric_growth,
      stringsAsFactors = FALSE
    )
  }
  result_overview <- do.call(rbind, overview)
  rownames(result_overview) <- NULL
  result_pairs <- if (length(matched_rows) == 0L) {
    data.frame(
      market = character(), region = character(),
      product_descriptor = character(), panel_id = character(),
      month = integer(), base_year = integer(), comparison_year = integer(),
      base_observed_weeks = integer(), comparison_observed_weeks = integer(),
      base_monthly_median_eur_per_kg = double(),
      comparison_monthly_median_eur_per_kg = double(),
      log_median_price_ratio = double(),
      median_price_change_fraction = double(),
      stringsAsFactors = FALSE
    )
  } else do.call(rbind, matched_rows)
  rownames(result_pairs) <- NULL
  list(overview = result_overview, matched_pairs = result_pairs)
}
