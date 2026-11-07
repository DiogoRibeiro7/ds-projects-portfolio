# Sharp logical bounds for unobserved *eligible* future SIMA quotations.
# Source wholesale, rolling, composition, forward-window and break-even
# modules first. Unknown endpoints are NEVER assigned fabricated prices.

#' Bound hypothetical quote feasibility when exact later quotes are absent.
#'
#' Two populations are evaluated:
#' - horizon_eligible: all starting quotations whose specified horizon has
#'   elapsed by the last complete source week;
#' - common_maturity: the same starting quotations for which *all* requested
#'   horizons have elapsed, independent of future quote availability.
#'
#' For N eligible anchors, O observed endpoints, M=N-O missing endpoints and
#' S observed hypothetical break-even successes, the sharp binary bounds
#' without missing-data assumptions are [S/N, (S+M)/N]. The same construction
#' is used for strict improvement. These are identification bounds, not
#' independent-sample confidence intervals, returns, or causal effects.
#'
#' @param windows Exact dated output of pair_sima_forward_quotes().
#' @param weekly_retention Distinct hypothetical weekly retained shares.
#' @param weekly_cost_eur_per_initial_kg Distinct hypothetical weekly costs.
#' @param tolerance Equality tolerance in EUR per original kilogram.
#' @return List of scenario bounds, market bounds, coverage, absent eligible
#'   windows and the fixed hypothetical assumption grid.
bound_sima_missing_forward_quotes <- function(
    windows,
    weekly_retention = c(1, 0.99, 0.98, 0.95),
    weekly_cost_eur_per_initial_kg = c(0, 0.005, 0.01, 0.02),
    tolerance = 1e-10) {
  # The direct scenario evaluator validates all observed values, flags,
  # parameters and the ex-post matched-pair reference population.
  direct <- evaluate_sima_quote_break_even(
    windows, weekly_retention, weekly_cost_eur_per_initial_kg, tolerance
  )
  horizons <- sort(unique(windows$horizon_weeks))

  # Establish common maturity by the time boundary, not by successful
  # future-quote matching. Every anchor has one row per horizon.
  anchor_key <- paste(
    windows$market, windows$region, windows$product_descriptor,
    windows$anchor_week_start, sep = "\u001f"
  )
  if (any(as.integer(table(anchor_key)) != length(horizons))) {
    stop("Every source anchor must appear once at each requested horizon",
         call. = FALSE)
  }
  maturity <- tapply(windows$horizon_eligible, anchor_key, all)
  windows$common_maturity_anchor <- as.logical(maturity[anchor_key])
  if (anyNA(windows$common_maturity_anchor)) {
    stop("Failed to identify the time-matured anchor population",
         call. = FALSE)
  }
  observed <- direct$detail
  observed_key <- paste(
    observed$market, observed$region, observed$product_descriptor,
    observed$anchor_week_start, sep = "\u001f"
  )
  observed$common_maturity_anchor <- as.logical(maturity[observed_key])
  if (anyNA(observed$common_maturity_anchor)) {
    stop("Scored quote pairs cannot be mapped to source anchors",
         call. = FALSE)
  }

  cohorts <- c("horizon_eligible", "common_maturity")
  select_cohort <- function(x, cohort) {
    if (cohort == "common_maturity") {
      x[x$common_maturity_anchor, , drop = FALSE]
    } else {
      x[x$horizon_eligible, , drop = FALSE]
    }
  }

  # Source denominators are computed *before* selecting observed prices.
  make_coverage <- function(by_market = FALSE) {
    markets <- if (by_market) sort(unique(windows$market)) else "all"
    records <- lapply(markets, function(market) {
      sub <- if (by_market) {
        windows[windows$market == market, , drop = FALSE]
      } else {
        windows
      }
      do.call(rbind, lapply(cohorts, function(cohort) {
        x <- select_cohort(sub, cohort)
        do.call(rbind, lapply(horizons, function(h) {
          z <- x[x$horizon_weeks == h, , drop = FALSE]
          n <- nrow(z)
          data.frame(
            cohort = cohort, market = market, horizon_weeks = h,
            eligible_anchors = n,
            observed_endpoints = sum(z$endpoint_observed),
            missing_eligible_endpoints = sum(!z$endpoint_observed),
            fully_matched_all_horizons = sum(z$balanced_all_horizons),
            missing_share = if (n > 0L) {
              mean(!z$endpoint_observed)
            } else NA_real_,
            stringsAsFactors = FALSE
          )
        }))
      }))
    })
    result <- do.call(rbind, records)
    rownames(result) <- NULL
    if (!by_market) result$market <- NULL
    result
  }
  coverage <- make_coverage()
  market_coverage <- make_coverage(TRUE)
  if (any(coverage$missing_eligible_endpoints +
            coverage$observed_endpoints != coverage$eligible_anchors)) {
    stop("Eligible missing/observed denominators do not add up",
         call. = FALSE)
  }
  if (any(coverage$eligible_anchors == 0L)) {
    stop("Each cohort/horizon needs at least one matured source anchor",
         call. = FALSE)
  }
  common <- coverage[coverage$cohort == "common_maturity", , drop = FALSE]
  if (length(unique(common$eligible_anchors)) != 1L) {
    stop("Matured comparison cohort changes across future horizons",
         call. = FALSE)
  }

  score <- function(cohort, market, scenario, h, n, o, m) {
    x <- observed[observed$scenario_id == scenario &
                    observed$horizon_weeks == h, , drop = FALSE]
    x <- select_cohort(x, cohort)
    if (market != "all") {
      x <- x[x$market == market, , drop = FALSE]
    }
    if (nrow(x) != o) {
      stop("Source observed endpoints and scenario scoring disagree",
           call. = FALSE)
    }
    if (n == 0L) return(NULL)
    success <- sum(x$break_even_or_better)
    strict <- sum(x$strictly_better)
    data.frame(
      cohort = cohort, market = market, horizon_weeks = h,
      scenario_id = scenario,
      eligible_anchors = n, observed_endpoints = o,
      missing_eligible_endpoints = m,
      observed_break_even_count = success,
      observed_strict_improvement_count = strict,
      observed_only_break_even_fraction =
        if (o > 0L) success / o else NA_real_,
      lower_break_even_fraction = success / n,
      upper_break_even_fraction = (success + m) / n,
      observed_only_strict_improvement_fraction =
        if (o > 0L) strict / o else NA_real_,
      lower_strict_improvement_fraction = strict / n,
      upper_strict_improvement_fraction = (strict + m) / n,
      unidentified_fraction = m / n,
      stringsAsFactors = FALSE
    )
  }

  tabulate <- function(source_coverage) {
    result <- do.call(rbind, lapply(
      seq_len(nrow(source_coverage)), function(i) {
        x <- source_coverage[i, , drop = FALSE]
        do.call(rbind, lapply(direct$scenarios$scenario_id, function(id) {
          score(x$cohort, if ("market" %in% names(x)) x$market else "all",
                id, x$horizon_weeks, x$eligible_anchors,
                x$observed_endpoints, x$missing_eligible_endpoints)
        }))
      }
    ))
    rownames(result) <- NULL
    if (!("market" %in% names(source_coverage))) result$market <- NULL
    match_index <- match(result$scenario_id, direct$scenarios$scenario_id)
    result$weekly_retention <- direct$scenarios$weekly_retention[match_index]
    result$weekly_cost_eur_per_initial_kg <-
      direct$scenarios$weekly_cost_eur_per_initial_kg[match_index]
    result
  }
  bounds <- tabulate(coverage)
  market_bounds <- tabulate(market_coverage)

  # The observed-only estimate must be inside both sharp completion
  # bounds. A missing outcome can be a success or a failure, never an
  # invented quoted price. This check is scenario-agnostic.
  validate_bounds <- function(x) {
    for (prefix in c("break_even", "strict_improvement")) {
      low <- x[[paste0("lower_", prefix, "_fraction")]]
      high <- x[[paste0("upper_", prefix, "_fraction")]]
      seen <- x[[paste0("observed_only_", prefix, "_fraction")]]
      if (any(low < -1e-12 | high > 1 + 1e-12 |
              low > high + 1e-12) ||
          any(abs(high - low - x$unidentified_fraction) > 1e-12) ||
          any(!is.na(seen) & (seen < low - 1e-12 |
                               seen > high + 1e-12))) {
        stop("Sharp missing-endpoint bounds violate logical constraints",
             call. = FALSE)
      }
    }
    if (any(x$lower_strict_improvement_fraction >
              x$lower_break_even_fraction + 1e-12) ||
        any(x$upper_strict_improvement_fraction >
              x$upper_break_even_fraction + 1e-12)) {
      stop("Strict improvement cannot exceed weak break-even",
           call. = FALSE)
    }
  }
  validate_bounds(bounds)
  validate_bounds(market_bounds)

  # Regression control: the observed-only horizon-specific frequencies
  # must agree with #692 before unknown target outcomes are considered.
  original <- direct$all_available
  horizon_only <- bounds[bounds$cohort == "horizon_eligible", , drop = FALSE]
  original_key <- paste(original$scenario_id, original$horizon_weeks)
  current_key <- paste(horizon_only$scenario_id, horizon_only$horizon_weeks)
  reference <- original[match(current_key, original_key), , drop = FALSE]
  if (anyNA(reference$observed_pairs) ||
      any(horizon_only$observed_endpoints != reference$observed_pairs) ||
      any(abs(horizon_only$observed_only_break_even_fraction -
                reference$fraction_break_even_or_better) > 1e-12) ||
      any(abs(horizon_only$observed_only_strict_improvement_fraction -
                reference$fraction_strictly_better) > 1e-12)) {
    stop("The missing-endpoint bounds alter observed scenario scores",
         call. = FALSE)
  }

  missing <- windows[windows$horizon_eligible &
                       !windows$endpoint_observed, , drop = FALSE]
  missing <- missing[, c(
    "panel_id", "market", "region", "product_descriptor",
    "anchor_week_start", "anchor_quote_date", "horizon_weeks",
    "target_week_start", "final_complete_source_week",
    "common_maturity_anchor"
  ), drop = FALSE]
  rownames(missing) <- NULL
  list(
    bounds = bounds, market_bounds = market_bounds,
    coverage = coverage, market_coverage = market_coverage,
    missing_eligible_windows = missing, scenarios = direct$scenarios
  )
}
