# Observed forward wholesale quote differences at fixed calendar-week horizons.
# Source sima_market_prices.R, sima_rolling_backtest.R and
# sima_composition_audit.R first. Base R only; no fitted predictions.

#' Pair each observed 2026 orange quote with an exact same-panel later quote.
#'
#' A quote in week t is compared with the final weekly quote in week t+h,
#' where h is a prespecified number of calendar weeks. Missing endpoint
#' quotations remain NA. The last *observed* source week is conservatively
#' excluded as potentially incomplete; no right-censored horizon is scored.
#' Neither a future quotation nor a positive price move is a tradable return.
#'
#' @param weekly Output of sima_weekly_panels(), with full product identity.
#' @param anchor_start First Monday included as a potential 2026 anchor.
#' @param horizons Unique positive whole-week horizons, default 1 through 4.
#' @return One row per observed anchor/horizon, including unavailable rows,
#'   source quote dates, exact-week eligibility and balanced-cohort flags.
pair_sima_forward_quotes <- function(weekly,
                                     anchor_start = as.Date("2026-01-05"),
                                     horizons = 1:4) {
  validate_sima_composition_weekly(weekly)
  if (!inherits(anchor_start, "Date") || length(anchor_start) != 1L ||
      is.na(anchor_start) || sima_monday(anchor_start) != anchor_start) {
    stop("anchor_start must be one nonmissing Monday Date", call. = FALSE)
  }
  if (!is.numeric(horizons) || length(horizons) == 0L ||
      anyNA(horizons) || any(!is.finite(horizons)) ||
      any(horizons != floor(horizons)) || any(horizons < 1L) ||
      anyDuplicated(horizons)) {
    stop("horizons must contain distinct positive whole weeks",
         call. = FALSE)
  }
  # A bounded horizon also prevents integer-overflow dates and accidental
  # generation of an implausibly large projection window.
  if (any(horizons > 52L)) {
    stop("horizons greater than 52 weeks are unsupported", call. = FALSE)
  }
  anchors <- weekly[weekly$week_start >= anchor_start, , drop = FALSE]
  if (nrow(anchors) == 0L) {
    stop("No observed anchors at or after anchor_start", call. = FALSE)
  }
  final_observed_quote <- max(weekly$quote_date)
  # Conservative right-censoring: the week with the latest source quote
  # might not have fully elapsed, even if a price was quoted within it.
  final_complete_week <- sima_monday(final_observed_quote) - 7L
  make_key <- function(market, region, product, weeks) {
    paste(market, region, product, as.character(weeks), sep = "\u001f")
  }
  source_key <- make_key(weekly$market, weekly$region,
                         weekly$product_descriptor, weekly$week_start)
  rows <- lapply(as.integer(horizons), function(h) {
    later_week <- anchors$week_start + 7L * h
    counterpart <- match(make_key(anchors$market, anchors$region,
                                  anchors$product_descriptor, later_week),
                         source_key)
    eligible <- later_week <= final_complete_week
    observed <- eligible & !is.na(counterpart)
    later_quote <- weekly$quote_date[counterpart]
    later_price <- weekly$frequent_eur_per_kg[counterpart]
    later_quote[!observed] <- as.Date(NA)
    later_price[!observed] <- NA_real_
    change <- later_price - anchors$frequent_eur_per_kg
    data.frame(
      panel_id = anchors$panel_id,
      market = anchors$market, region = anchors$region,
      product_descriptor = anchors$product_descriptor,
      anchor_week_start = anchors$week_start,
      anchor_quote_date = anchors$quote_date,
      anchor_eur_per_kg = anchors$frequent_eur_per_kg,
      horizon_weeks = h,
      target_week_start = later_week,
      final_complete_source_week = final_complete_week,
      horizon_eligible = eligible,
      endpoint_observed = observed,
      endpoint_quote_date = later_quote,
      endpoint_eur_per_kg = later_price,
      actual_quote_days_apart = as.integer(later_quote - anchors$quote_date),
      signed_quote_change_eur_per_kg = change,
      nominal_quote_change_fraction = later_price /
        anchors$frequent_eur_per_kg - 1,
      stringsAsFactors = FALSE
    )
  })
  result <- do.call(rbind, rows)
  # Balanced analysis compares all horizons for the same starting quote.
  anchor_key <- make_key(result$market, result$region,
                         result$product_descriptor,
                         result$anchor_week_start)
  matched_by_anchor <- tapply(result$endpoint_observed, anchor_key, sum)
  result$balanced_all_horizons <- result$endpoint_observed &
    as.integer(matched_by_anchor[anchor_key]) == length(horizons)
  result <- result[order(result$anchor_week_start, result$market,
                         result$panel_id, result$horizon_weeks,
                         method = "radix"), , drop = FALSE]
  rownames(result) <- NULL
  if (any(result$endpoint_observed &
          (result$endpoint_quote_date <= result$anchor_quote_date |
             result$endpoint_quote_date < result$target_week_start |
             result$endpoint_quote_date > result$target_week_start + 6L))) {
    stop("A forward endpoint lies outside its target calendar week",
         call. = FALSE)
  }
  result
}

#' Descriptive, observed-only forward quote movements by fixed horizons.
#'
#' The balanced cohort is the intersection of all available endpoints for
#' every requested horizon. Its membership is defined ex post and must not
#' be interpreted as a real-time selection or tradeable holding policy.
#'
#' @param windows Output of pair_sima_forward_quotes().
#' @return Coverage; all-available and balanced horizon statistics; month,
#'   market and product-panel summaries within the balanced cohort.
summarize_sima_forward_windows <- function(windows) {
  needed <- c("panel_id", "market", "region", "product_descriptor",
              "anchor_week_start", "anchor_quote_date", "horizon_weeks",
              "horizon_eligible", "endpoint_observed",
              "balanced_all_horizons", "anchor_eur_per_kg",
              "endpoint_eur_per_kg", "signed_quote_change_eur_per_kg",
              "nominal_quote_change_fraction")
  if (!is.data.frame(windows) || nrow(windows) == 0L ||
      !all(needed %in% names(windows))) {
    stop("Expected nonempty dated forward-quote windows", call. = FALSE)
  }
  for (field in c("horizon_eligible", "endpoint_observed",
                  "balanced_all_horizons")) {
    if (!is.logical(windows[[field]]) || anyNA(windows[[field]])) {
      stop(sprintf("%s needs complete logical values", field), call. = FALSE)
    }
  }
  if (!is.numeric(windows$horizon_weeks) ||
      !is.numeric(windows$anchor_eur_per_kg) ||
      !is.numeric(windows$endpoint_eur_per_kg) ||
      !is.numeric(windows$signed_quote_change_eur_per_kg) ||
      !is.numeric(windows$nominal_quote_change_fraction) ||
      !inherits(windows$anchor_week_start, "Date") ||
      !inherits(windows$anchor_quote_date, "Date") ||
      anyNA(windows$anchor_week_start) ||
      anyNA(windows$anchor_quote_date) ||
      any(windows$endpoint_observed & !windows$horizon_eligible) ||
      any(windows$balanced_all_horizons & !windows$endpoint_observed) ||
      any(is.na(windows$endpoint_eur_per_kg) != !windows$endpoint_observed) ||
      any(!is.finite(windows$anchor_eur_per_kg)) ||
      any(windows$anchor_eur_per_kg <= 0) ||
      any(windows$endpoint_observed &
            (!is.finite(windows$endpoint_eur_per_kg) |
               windows$endpoint_eur_per_kg <= 0)) ||
      any(windows$endpoint_observed &
            (!is.finite(windows$signed_quote_change_eur_per_kg) |
               !is.finite(windows$nominal_quote_change_fraction)))) {
    stop("Inconsistent availability, dates or anchor prices",
         call. = FALSE)
  }
  key <- paste(windows$market, windows$region,
               windows$product_descriptor,
               windows$anchor_week_start, windows$horizon_weeks,
               sep = "\u001f")
  if (anyDuplicated(key)) {
    stop("Duplicate exact-panel anchor/horizon windows", call. = FALSE)
  }
  horizons <- sort(unique(windows$horizon_weeks))
  if (anyNA(horizons) || any(horizons < 1) ||
      any(horizons != floor(horizons))) {
    stop("Malformed horizon values", call. = FALSE)
  }
  summarize_change <- function(x) {
    n <- nrow(x)
    metric <- function(f) if (n == 0L) NA_real_ else f(x)
    data.frame(
      observed_pairs = n,
      mean_signed_change_eur_per_kg =
        metric(function(z) mean(z$signed_quote_change_eur_per_kg)),
      median_signed_change_eur_per_kg =
        metric(function(z) stats::median(z$signed_quote_change_eur_per_kg)),
      mean_absolute_change_eur_per_kg =
        metric(function(z) mean(abs(z$signed_quote_change_eur_per_kg))),
      tenth_percentile_change_eur_per_kg =
        metric(function(z) unname(stats::quantile(
          z$signed_quote_change_eur_per_kg, 0.10))),
      ninetieth_percentile_change_eur_per_kg =
        metric(function(z) unname(stats::quantile(
          z$signed_quote_change_eur_per_kg, 0.90))),
      median_nominal_change_fraction =
        metric(function(z) stats::median(z$nominal_quote_change_fraction)),
      share_price_higher =
        metric(function(z) mean(z$signed_quote_change_eur_per_kg > 0)),
      share_price_lower =
        metric(function(z) mean(z$signed_quote_change_eur_per_kg < 0)),
      share_price_unchanged =
        metric(function(z) mean(z$signed_quote_change_eur_per_kg == 0)),
      stringsAsFactors = FALSE
    )
  }
  available <- windows[windows$endpoint_observed, , drop = FALSE]
  balanced <- windows[windows$balanced_all_horizons, , drop = FALSE]
  group_horizon <- function(subset) {
    rows <- lapply(horizons, function(h) {
      data.frame(horizon_weeks = h,
                 summarize_change(subset[subset$horizon_weeks == h,
                                         , drop = FALSE]))
    })
    do.call(rbind, rows)
  }
  all_available <- group_horizon(available)
  balanced_summary <- group_horizon(balanced)
  coverage <- do.call(rbind, lapply(horizons, function(h) {
    x <- windows[windows$horizon_weeks == h, , drop = FALSE]
    data.frame(
      horizon_weeks = h, observed_anchor_quotes = nrow(x),
      complete_horizon_eligible = sum(x$horizon_eligible),
      exact_endpoint_observed = sum(x$endpoint_observed),
      balanced_all_horizon_anchors = sum(x$balanced_all_horizons),
      observed_share_of_eligible = if (any(x$horizon_eligible)) {
        sum(x$endpoint_observed) / sum(x$horizon_eligible)
      } else NA_real_, stringsAsFactors = FALSE
    )
  }))
  summarize_balanced_groups <- function(group_key, label) {
    if (nrow(balanced) == 0L) return(data.frame())
    groups <- split(seq_len(nrow(balanced)), group_key)
    result <- do.call(rbind, lapply(groups, function(index) {
      x <- balanced[index, , drop = FALSE]
      data.frame(label_value = group_key[index[1L]],
                 group_horizon(x), stringsAsFactors = FALSE)
    }))
    names(result)[1L] <- label
    rownames(result) <- NULL
    result
  }
  by_month <- summarize_balanced_groups(
    format(balanced$anchor_quote_date, "%Y-%m"), "anchor_quote_month")
  by_market <- summarize_balanced_groups(balanced$market, "market")
  panel_key <- paste(balanced$market, balanced$region,
                     balanced$product_descriptor, sep = "\u001f")
  by_panel <- summarize_balanced_groups(panel_key, "exact_panel_key")
  if (nrow(by_panel) > 0L) {
    position <- match(by_panel$exact_panel_key, panel_key)
    by_panel$panel_id <- balanced$panel_id[position]
    by_panel$market <- balanced$market[position]
    by_panel$region <- balanced$region[position]
    by_panel$product_descriptor <- balanced$product_descriptor[position]
    by_panel$exact_panel_key <- NULL
    by_panel <- by_panel[, c("panel_id", "market", "region",
                             "product_descriptor", "horizon_weeks",
                             setdiff(names(by_panel), c("panel_id", "market",
                                                        "region", "product_descriptor",
                                                        "horizon_weeks"))),
                         drop = FALSE]
  }
  list(windows = windows, coverage = coverage,
       all_available = all_available, balanced = balanced_summary,
       monthly_balanced = by_month, market_balanced = by_market,
       panel_balanced = by_panel)
}
