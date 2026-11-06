# Continuous hypothetical feasibility frontiers for archived SIMA quotations.
# Source the wholesale, rolling, composition, forward-window and break-even
# modules first. No physical fruit retention or storage costs are observed.

#' Invert the quote-level break-even inequality for cost and retention.
#'
#' Each observed exact-panel quote pair defines two *hypothetical* boundaries:
#'   c_max(rho) = (rho^h * P_future - P_start) / h
#'   rho_min(c) = ((P_start + h*c) / P_future)^(1/h)
#' where c is EUR per original kg per calendar week and rho is the weekly
#' retained-saleable mass share. Raw negative c_max and rho_min greater than
#' one deliberately remain visible: they indicate infeasibility for
#' nonnegative cost and a weekly retention share no greater than one.
#'
#' @param windows Output from pair_sima_forward_quotes().
#' @param weekly_retention Distinct hypothetical weekly shares in (0, 1].
#' @param weekly_cost_eur_per_initial_kg Distinct nonnegative hypothetical
#'   weekly costs, in EUR per originally stored kilogram.
#' @param tolerance Equality tolerance in EUR per original kilogram.
#' @return List with pair-level cost ceilings and retention requirements,
#'   all-available and common-anchor summaries, balanced-market summaries,
#'   source coverage, and the fixed assumption grid.
evaluate_sima_quote_feasibility_frontier <- function(
    windows,
    weekly_retention = c(1, 0.99, 0.98, 0.95),
    weekly_cost_eur_per_initial_kg = c(0, 0.005, 0.01, 0.02),
    tolerance = 1e-10) {
  # Reuse the validated exact-window population and existing scenario grid.
  benchmark <- evaluate_sima_quote_break_even(
    windows, weekly_retention, weekly_cost_eur_per_initial_kg, tolerance
  )
  required <- c("target_week_start", "endpoint_quote_date",
                "actual_quote_days_apart")
  if (!all(required %in% names(windows))) {
    stop("The exact-dated forward windows need target and quote dates",
         call. = FALSE)
  }
  observed <- windows[windows$endpoint_observed, , drop = FALSE]
  count <- nrow(observed)
  horizons <- sort(unique(observed$horizon_weeks))
  retentions <- sort(unique(benchmark$scenarios$weekly_retention),
                     decreasing = TRUE)
  costs <- sort(unique(
    benchmark$scenarios$weekly_cost_eur_per_initial_kg
  ))
  identity <- c(
    "panel_id", "market", "region", "product_descriptor",
    "anchor_week_start", "anchor_quote_date", "anchor_eur_per_kg",
    "horizon_weeks", "target_week_start", "endpoint_quote_date",
    "endpoint_eur_per_kg", "actual_quote_days_apart",
    "balanced_all_horizons"
  )

  expand_pairs <- function(parameters) {
    pair_index <- rep(seq_len(count), times = length(parameters))
    parameter_index <- rep(seq_along(parameters), each = count)
    x <- observed[pair_index, identity, drop = FALSE]
    rownames(x) <- NULL
    # This ID permits exact row-level cross-checking across both frontiers.
    x$observed_pair_id <- pair_index
    list(rows = x, parameter = parameters[parameter_index])
  }

  cost_expanded <- expand_pairs(retentions)
  cost_ceiling <- cost_expanded$rows
  cost_ceiling$assumed_weekly_retention <- cost_expanded$parameter
  cost_ceiling$retained_saleable_share <-
    cost_ceiling$assumed_weekly_retention ^ cost_ceiling$horizon_weeks
  cost_ceiling$raw_maximum_weekly_cost_eur_per_initial_kg <-
    (cost_ceiling$retained_saleable_share *
       cost_ceiling$endpoint_eur_per_kg - cost_ceiling$anchor_eur_per_kg) /
    cost_ceiling$horizon_weeks
  cost_ceiling$feasible_at_zero_cost <-
    cost_ceiling$raw_maximum_weekly_cost_eur_per_initial_kg >=
    -tolerance / cost_ceiling$horizon_weeks
  cost_ceiling$positive_cost_room <-
    cost_ceiling$raw_maximum_weekly_cost_eur_per_initial_kg >
    tolerance / cost_ceiling$horizon_weeks

  retention_expanded <- expand_pairs(costs)
  retention_floor <- retention_expanded$rows
  retention_floor$assumed_weekly_cost_eur_per_initial_kg <-
    retention_expanded$parameter
  retention_floor$required_weekly_retention <-
    ((retention_floor$anchor_eur_per_kg +
        retention_floor$horizon_weeks *
        retention_floor$assumed_weekly_cost_eur_per_initial_kg) /
       retention_floor$endpoint_eur_per_kg) ^
    (1 / retention_floor$horizon_weeks)
  retention_floor$raw_maximum_weekly_loss_fraction <-
    1 - retention_floor$required_weekly_retention
  retention_floor$feasible_at_full_retention <-
    (retention_floor$endpoint_eur_per_kg -
       retention_floor$horizon_weeks *
       retention_floor$assumed_weekly_cost_eur_per_initial_kg -
       retention_floor$anchor_eur_per_kg) >= -tolerance
  retention_floor$positive_loss_room <-
    (retention_floor$endpoint_eur_per_kg -
       retention_floor$horizon_weeks *
       retention_floor$assumed_weekly_cost_eur_per_initial_kg -
       retention_floor$anchor_eur_per_kg) > tolerance

  summarize_cost <- function(x) {
    if (nrow(x) == 0L) {
      return(data.frame(
        observed_pairs = 0L, fraction_feasible_at_zero_cost = NA_real_,
        fraction_with_positive_cost_room = NA_real_,
        mean_raw_ceiling_eur_per_initial_kg_week = NA_real_,
        median_raw_ceiling_eur_per_initial_kg_week = NA_real_,
        tenth_percentile_raw_ceiling_eur_per_initial_kg_week = NA_real_,
        ninetieth_percentile_raw_ceiling_eur_per_initial_kg_week = NA_real_
      ))
    }
    v <- x$raw_maximum_weekly_cost_eur_per_initial_kg
    data.frame(
      observed_pairs = nrow(x),
      fraction_feasible_at_zero_cost = mean(x$feasible_at_zero_cost),
      fraction_with_positive_cost_room = mean(x$positive_cost_room),
      mean_raw_ceiling_eur_per_initial_kg_week = mean(v),
      median_raw_ceiling_eur_per_initial_kg_week = stats::median(v),
      tenth_percentile_raw_ceiling_eur_per_initial_kg_week =
        unname(stats::quantile(v, 0.1)),
      ninetieth_percentile_raw_ceiling_eur_per_initial_kg_week =
        unname(stats::quantile(v, 0.9))
    )
  }

  summarize_retention <- function(x) {
    if (nrow(x) == 0L) {
      return(data.frame(
        observed_pairs = 0L, fraction_feasible_at_full_retention = NA_real_,
        fraction_with_positive_loss_room = NA_real_,
        median_required_weekly_retention = NA_real_,
        tenth_percentile_required_weekly_retention = NA_real_,
        ninetieth_percentile_required_weekly_retention = NA_real_,
        median_raw_maximum_weekly_loss_fraction = NA_real_
      ))
    }
    v <- x$required_weekly_retention
    data.frame(
      observed_pairs = nrow(x),
      fraction_feasible_at_full_retention =
        mean(x$feasible_at_full_retention),
      fraction_with_positive_loss_room = mean(x$positive_loss_room),
      median_required_weekly_retention = stats::median(v),
      tenth_percentile_required_weekly_retention =
        unname(stats::quantile(v, 0.1)),
      ninetieth_percentile_required_weekly_retention =
        unname(stats::quantile(v, 0.9)),
      median_raw_maximum_weekly_loss_fraction =
        stats::median(x$raw_maximum_weekly_loss_fraction)
    )
  }

  tabulate <- function(rows, parameter_name, parameter_values,
                       summarize, by_market = FALSE) {
    markets <- if (by_market) sort(unique(rows$market)) else "all"
    result <- do.call(rbind, lapply(markets, function(market) {
      sector <- if (by_market) {
        rows[rows$market == market, , drop = FALSE]
      } else {
        rows
      }
      do.call(rbind, lapply(parameter_values, function(parameter) {
        do.call(rbind, lapply(horizons, function(h) {
          x <- sector[sector[[parameter_name]] == parameter &
                        sector$horizon_weeks == h, , drop = FALSE]
          header <- data.frame(market = market, horizon_weeks = h,
                               stringsAsFactors = FALSE)
          header[[parameter_name]] <- parameter
          cbind(header, summarize(x))
        }))
      }))
    }))
    rownames(result) <- NULL
    if (!by_market) result$market <- NULL
    result
  }

  balanced_cost <- cost_ceiling[
    cost_ceiling$balanced_all_horizons, , drop = FALSE
  ]
  balanced_retention <- retention_floor[
    retention_floor$balanced_all_horizons, , drop = FALSE
  ]
  cost_summary <- function(x, by_market = FALSE) {
    tabulate(x, "assumed_weekly_retention", retentions,
             summarize_cost, by_market)
  }
  retention_summary <- function(x, by_market = FALSE) {
    tabulate(x, "assumed_weekly_cost_eur_per_initial_kg", costs,
             summarize_retention, by_market)
  }

  # Both inverted frontiers must reproduce every scenario's fraction from
  # the previously validated direct break-even calculation, on identical
  # observed pairs and under the same EUR/kg numerical tolerance.
  for (i in seq_len(nrow(benchmark$balanced))) {
    scenario <- benchmark$balanced[i, , drop = FALSE]
    h <- scenario$horizon_weeks
    rho <- scenario$weekly_retention
    cost <- scenario$weekly_cost_eur_per_initial_kg
    candidate_cost <- balanced_cost[
      balanced_cost$horizon_weeks == h &
        balanced_cost$assumed_weekly_retention == rho, , drop = FALSE
    ]
    candidate_retention <- balanced_retention[
      balanced_retention$horizon_weeks == h &
        balanced_retention$assumed_weekly_cost_eur_per_initial_kg == cost,
      , drop = FALSE
    ]
    cost_share <- mean(
      candidate_cost$raw_maximum_weekly_cost_eur_per_initial_kg >=
        cost - tolerance / h
    )
    # Express the tolerance as a price-ratio bound before taking roots.
    allowed_floor <- ((candidate_retention$anchor_eur_per_kg +
                         h * cost - tolerance) /
                        candidate_retention$endpoint_eur_per_kg) ^ (1 / h)
    retention_share <- mean(rho >= allowed_floor)
    reference <- scenario$fraction_break_even_or_better
    if (abs(cost_share - reference) > 1e-12 ||
        abs(retention_share - reference) > 1e-12) {
      stop("Continuous frontiers disagree with direct quote break-even",
           call. = FALSE)
    }
  }

  list(
    cost_ceiling_pairs = cost_ceiling,
    retention_floor_pairs = retention_floor,
    cost_ceiling_all_available = cost_summary(cost_ceiling),
    retention_floor_all_available = retention_summary(retention_floor),
    cost_ceiling_balanced = cost_summary(balanced_cost),
    retention_floor_balanced = retention_summary(balanced_retention),
    cost_ceiling_market_balanced = cost_summary(balanced_cost, TRUE),
    retention_floor_market_balanced =
      retention_summary(balanced_retention, TRUE),
    coverage = benchmark$coverage,
    assumed_grid = benchmark$scenarios
  )
}
