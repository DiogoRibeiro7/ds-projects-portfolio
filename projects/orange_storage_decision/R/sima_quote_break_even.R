# Illustrative storage friction applied to *observed wholesale quotations*.
# Source sima_market_prices.R, sima_rolling_backtest.R,
# sima_composition_audit.R and sima_forward_quote_windows.R first.
# Hypothetical retained-saleable fraction and cost are NOT empirical inputs.

#' Evaluate conditional quote-level break-even thresholds under a fixed grid.
#'
#' The proxy difference, expressed per original kg, is
#'     rho^h * P[t+h] - h * c - P[t]
#' where rho is an assumed weekly retained-saleable share, c is assumed
#' EUR per original kg per week, and h is an observed exact-week horizon.
#' The same original lot, sellability, realized producer price and costs
#' are NOT observed: this is a sensitivity calculation, not policy validation.
#'
#' @param windows Output of pair_sima_forward_quotes(); includes balanced
#'   all-horizon flags and all unavailable target observations.
#' @param weekly_retention Distinct hypothetical fractions in (0, 1].
#' @param weekly_cost_eur_per_initial_kg Distinct nonnegative assumed EUR/kg
#'   costs incurred for each whole week per original kg.
#' @param tolerance Numerical equality tolerance in EUR per original kg.
#' @return List with scenario-level rows, all-available and balanced
#'   summaries, balanced market breakdown and original coverage.
evaluate_sima_quote_break_even <- function(
    windows,
    weekly_retention = c(1, 0.99, 0.98, 0.95),
    weekly_cost_eur_per_initial_kg = c(0, 0.005, 0.01, 0.02),
    tolerance = 1e-10) {
  base <- summarize_sima_forward_windows(windows)
  check_grid <- function(values, name, minimum, maximum) {
    if (!is.numeric(values) || length(values) == 0L || anyNA(values) ||
        any(!is.finite(values)) || any(values < minimum) ||
        any(values > maximum) || anyDuplicated(values)) {
      stop(sprintf("%s must contain distinct finite values in [%.3f, %.3f]",
                   name, minimum, maximum), call. = FALSE)
    }
  }
  check_grid(weekly_retention, "weekly_retention", 0, 1)
  if (any(weekly_retention == 0)) {
    stop("weekly_retention must be strictly positive", call. = FALSE)
  }
  check_grid(weekly_cost_eur_per_initial_kg,
             "weekly_cost_eur_per_initial_kg", 0, Inf)
  if (!is.numeric(tolerance) || length(tolerance) != 1L ||
      is.na(tolerance) || !is.finite(tolerance) ||
      tolerance < 0 || tolerance > 1e-6) {
    stop("tolerance must be a finite nonnegative EUR/kg scalar <= 1e-6",
         call. = FALSE)
  }

  observed <- windows[windows$endpoint_observed, , drop = FALSE]
  if (nrow(observed) == 0L || !any(observed$balanced_all_horizons)) {
    stop("At least one balanced all-horizon observed quote is required",
         call. = FALSE)
  }
  grid <- expand.grid(
    weekly_retention = weekly_retention,
    weekly_cost_eur_per_initial_kg = weekly_cost_eur_per_initial_kg,
    KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE
  )
  grid$scenario_id <- seq_len(nrow(grid))

  # Replicate all *observed* quote pairs for each parameter combination;
  # ineligible/unquoted endpoint rows remain in the separate coverage table.
  observation_index <- rep(seq_len(nrow(observed)), times = nrow(grid))
  scenario_index <- rep(seq_len(nrow(grid)), each = nrow(observed))
  detail <- observed[observation_index, , drop = FALSE]
  rownames(detail) <- NULL
  detail$scenario_id <- grid$scenario_id[scenario_index]
  detail$weekly_retention <- grid$weekly_retention[scenario_index]
  detail$weekly_cost_eur_per_initial_kg <-
    grid$weekly_cost_eur_per_initial_kg[scenario_index]
  detail$retained_saleable_share <-
    detail$weekly_retention ^ detail$horizon_weeks
  detail$assumed_total_cost_eur_per_initial_kg <-
    detail$horizon_weeks * detail$weekly_cost_eur_per_initial_kg
  detail$required_forward_quote_eur_per_saleable_kg <-
    (detail$anchor_eur_per_kg +
       detail$assumed_total_cost_eur_per_initial_kg) /
    detail$retained_saleable_share
  detail$required_forward_price_increase_eur_per_kg <-
    detail$required_forward_quote_eur_per_saleable_kg -
    detail$anchor_eur_per_kg
  detail$illustrative_difference_eur_per_initial_kg <-
    detail$retained_saleable_share * detail$endpoint_eur_per_kg -
    detail$assumed_total_cost_eur_per_initial_kg - detail$anchor_eur_per_kg
  detail$break_even_or_better <-
    detail$illustrative_difference_eur_per_initial_kg >= -tolerance
  detail$strictly_better <-
    detail$illustrative_difference_eur_per_initial_kg > tolerance

  horizons <- sort(unique(observed$horizon_weeks))
  summarize_group <- function(x) {
    if (nrow(x) == 0L) {
      return(data.frame(
        observed_pairs = 0L, fraction_break_even_or_better = NA_real_,
        fraction_strictly_better = NA_real_,
        mean_illustrative_difference_eur_per_initial_kg = NA_real_,
        median_illustrative_difference_eur_per_initial_kg = NA_real_,
        tenth_percentile_difference_eur_per_initial_kg = NA_real_,
        mean_required_forward_price_increase_eur_per_kg = NA_real_
      ))
    }
    d <- x$illustrative_difference_eur_per_initial_kg
    data.frame(
      observed_pairs = nrow(x),
      fraction_break_even_or_better = mean(x$break_even_or_better),
      fraction_strictly_better = mean(x$strictly_better),
      mean_illustrative_difference_eur_per_initial_kg = mean(d),
      median_illustrative_difference_eur_per_initial_kg = stats::median(d),
      tenth_percentile_difference_eur_per_initial_kg =
        unname(stats::quantile(d, 0.10)),
      mean_required_forward_price_increase_eur_per_kg =
        mean(x$required_forward_price_increase_eur_per_kg),
      stringsAsFactors = FALSE
    )
  }
  tabulate <- function(rows, by_market = FALSE) {
    market_values <- if (by_market) sort(unique(rows$market)) else "all"
    output <- lapply(market_values, function(market) {
      sector <- if (by_market) {
        rows[rows$market == market, , drop = FALSE]
      } else {
        rows
      }
      do.call(rbind, lapply(seq_len(nrow(grid)), function(i) {
        scenario <- sector[sector$scenario_id == grid$scenario_id[i],
                           , drop = FALSE]
        do.call(rbind, lapply(horizons, function(h) {
          x <- scenario[scenario$horizon_weeks == h, , drop = FALSE]
          data.frame(
            market = market, scenario_id = grid$scenario_id[i],
            horizon_weeks = h, weekly_retention = grid$weekly_retention[i],
            weekly_cost_eur_per_initial_kg =
              grid$weekly_cost_eur_per_initial_kg[i],
            summarize_group(x), stringsAsFactors = FALSE
          )
        }))
      }))
    })
    result <- do.call(rbind, output)
    rownames(result) <- NULL
    if (!by_market) result$market <- NULL
    result
  }
  balanced <- detail[detail$balanced_all_horizons, , drop = FALSE]
  all_summary <- tabulate(detail)
  balanced_summary <- tabulate(balanced)
  by_market <- tabulate(balanced, by_market = TRUE)

  # On the frictionless scenario, strict improvement must reproduce the
  # original forward-quote higher-price frequency on the *same* anchors.
  zero <- balanced_summary[
    balanced_summary$weekly_retention == 1 &
      balanced_summary$weekly_cost_eur_per_initial_kg == 0,
    , drop = FALSE
  ]
  reference <- base$balanced[match(zero$horizon_weeks,
                                    base$balanced$horizon_weeks), , drop = FALSE]
  if (nrow(zero) > 0L &&
      any(abs(zero$fraction_strictly_better -
                reference$share_price_higher) > 1e-10)) {
    stop("Frictionless price movement does not match the source benchmark",
         call. = FALSE)
  }
  list(detail = detail, scenarios = grid, coverage = base$coverage,
       all_available = all_summary, balanced = balanced_summary,
       market_balanced = by_market)
}
