# Paired grid-resolution sensitivity for finite reference calibration.
# Source model.R, sampling.R, simulation.R, calibration.R, adaptive_policy.R,
# adaptive_evaluation.R, adaptive_stress.R, adaptive_label_error.R,
# adaptive_calibration_policy.R, adaptive_calibration_evaluation.R first.
# Base R only.

#' Require a bounded set of distinct quadrature sizes with a high-grid anchor.
#' The largest supported 9x9 rate grid is a comparison reference, NOT a
#' continuous-posterior or economically optimal benchmark.
validate_adaptive_grid_sizes <- function(grid_sizes) {
  if (!is.numeric(grid_sizes) || length(grid_sizes) < 2L ||
      anyNA(grid_sizes) || any(!is.finite(grid_sizes)) ||
      any(grid_sizes < 3 | grid_sizes > 9 | grid_sizes != floor(grid_sizes)) ||
      anyDuplicated(grid_sizes) || !(9L %in% grid_sizes)) {
    stop("grid_sizes need distinct integers in [3, 9], including 9",
         call. = FALSE)
  }
  invisible(TRUE)
}

#' One latent lot under shared grader audits, target labels and fruit outcomes.
#' Each budget/grid plan is fixed before ANY potential day-16 observations or
#' hidden fruit lifetimes are drawn. The high-resolution reference is a policy,
#' not an oracle and not a certified quadrature limit.
simulate_adaptive_grid_lot <- function(
    lot_id, budgets = c(scarce = 8L, ample = 80L),
    grid_sizes = c(3L, 5L, 7L, 9L),
    eur_per_check = 0.02, checks_per_cohort = 8L) {
  assert_nonnegative_scalar(lot_id, "lot_id", integer = TRUE)
  if (lot_id < 1L) stop("lot_id must be positive", call. = FALSE)
  validate_adaptive_reference_budgets(budgets)
  validate_adaptive_grid_sizes(grid_sizes)
  assert_nonnegative_scalar(eur_per_check, "eur_per_check")
  assert_nonnegative_scalar(checks_per_cohort, "checks_per_cohort", integer = TRUE)
  if (checks_per_cohort < 1L) {
    stop("checks_per_cohort must be positive", call. = FALSE)
  }

  scenarios <- adaptive_scenario_library(850 * stats::runif(1, 0.25, 0.60))
  prior <- normalize_scenario_weights(scenarios,
                                      c(short = 0.3, typical = 0.4, long = 0.3))
  latent_index <- sample.int(length(scenarios), 1L, prob = prior)
  truth <- scenarios[[latent_index]]
  prices <- simulate_price_scenario(12L, 28L)
  true_sensitivity <- stats::runif(1, 0.75, 0.95)
  true_specificity <- stats::runif(1, 0.75, 0.95)

  # The same independent reference stream underlies both audit budgets.
  positive_audit <- stats::runif(max(budgets))
  negative_audit <- stats::runif(max(budgets))
  eligible <- truth$cohort[truth$initially_sellable]
  pre_schedule <- expand.grid(cohort = eligible, day = c(6L, 12L),
                              KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
  pre_state <- matrix(stats::runif(nrow(pre_schedule) * 6L),
                      nrow = nrow(pre_schedule), ncol = 6L)
  pre_grading <- matrix(stats::runif(nrow(pre_schedule) * 6L),
                        nrow = nrow(pre_schedule), ncol = 6L)
  historical <- simulate_noisy_adaptive_counts(
    truth, pre_schedule, pre_state, pre_grading,
    true_sensitivity, true_specificity
  )

  # Every plan sees the same predecision target labels; within each budget it
  # also sees precisely the same reference confusion counts at every grid size.
  locked <- lapply(seq_along(budgets), function(b) {
    audit <- simulate_adaptive_reference_audit(
      positive_audit, negative_audit, budgets[b],
      true_sensitivity, true_specificity
    )
    lapply(grid_sizes, function(grid_size) {
      plan_adaptive_with_calibration_uncertainty(
        scenarios, prior, historical, audit, prices,
        storage_eur_per_kg_day = 0.001, decision_day = 12L,
        inspection_day = 16L, grid_size = grid_size,
        checks_per_cohort = checks_per_cohort,
        eur_per_check = eur_per_check, delivery_lag_days = 2L
      )
    })
  })

  future_schedule <- data.frame(cohort = eligible, day = 16L,
                                stringsAsFactors = FALSE)
  future_state <- matrix(stats::runif(length(eligible) * checks_per_cohort),
                         nrow = length(eligible), ncol = checks_per_cohort)
  future_grading <- matrix(stats::runif(length(eligible) * checks_per_cohort),
                           nrow = length(eligible), ncol = checks_per_cohort)
  future <- simulate_noisy_adaptive_counts(
    truth, future_schedule, future_state, future_grading,
    true_sensitivity, true_specificity
  )
  lifetimes <- simulate_fruit_lifetimes(truth)
  realised <- function(day) {
    realised_sale_value(day, truth, prices, lifetimes, 0.001, 12L, 2L)
  }

  do.call(rbind, lapply(seq_along(budgets), function(b) {
    do.call(rbind, lapply(seq_along(grid_sizes), function(g) {
      result <- locked[[b]][[g]]
      plan <- result$integrated
      followup_day <- sale_after_inspection(plan, future)$sale_day
      adaptive_day <- if (plan$action == "inspect") {
        followup_day
      } else plan$fixed_day
      labour <- plan$inspection_labour_eur
      data.frame(
        lot_id = lot_id, budget = names(budgets)[b],
        reference_per_class = unname(budgets[b]),
        generating_scenario = names(scenarios)[latent_index],
        true_sensitivity = true_sensitivity,
        true_specificity = true_specificity,
        grid_size = grid_sizes[g],
        supported_rate_nodes = nrow(result$reference_rate_grid$nodes),
        excluded_rate_nodes = result$reference_rate_grid$n_discarded,
        reference_mean_sensitivity = unname(result$point_rates["sensitivity"]),
        reference_mean_specificity = unname(result$point_rates["specificity"]),
        action = plan$action, fixed_day = plan$fixed_day,
        adaptive_day = adaptive_day,
        planned_option_value_eur = plan$net_information_value_eur,
        inspection_labour_eur = labour,
        fixed_eur = realised(plan$fixed_day), immediate_eur = realised(12L),
        adaptive_eur = realised(adaptive_day) -
          if (plan$action == "inspect") labour else 0,
        always_inspect_eur = realised(followup_day) - labour,
        stringsAsFactors = FALSE
      )
    }))
  }))
}

#' Independent lots, identical random streams across grid sizes within a lot.
run_adaptive_grid_experiment <- function(
    n_lots = 60L, seed = 20261109L,
    budgets = c(scarce = 8L, ample = 80L),
    grid_sizes = c(3L, 5L, 7L, 9L),
    eur_per_check = 0.02, checks_per_cohort = 8L) {
  assert_nonnegative_scalar(n_lots, "n_lots", integer = TRUE)
  assert_nonnegative_scalar(seed, "seed", integer = TRUE)
  validate_adaptive_reference_budgets(budgets)
  validate_adaptive_grid_sizes(grid_sizes)
  if (n_lots < 2L) stop("at least two independent lots required", call. = FALSE)
  set.seed(seed)
  do.call(rbind, lapply(seq_len(n_lots), function(i) {
    simulate_adaptive_grid_lot(i, budgets, grid_sizes,
                               eur_per_check, checks_per_cohort)
  }))
}

#' Require one complete grid/budget factorial for every simulated latent lot.
validate_adaptive_grid_records <- function(records) {
  required <- c("lot_id", "budget", "reference_per_class",
                "generating_scenario", "true_sensitivity", "true_specificity",
                "grid_size", "supported_rate_nodes", "excluded_rate_nodes",
                "reference_mean_sensitivity", "reference_mean_specificity",
                "action", "fixed_day", "adaptive_day", "planned_option_value_eur",
                "inspection_labour_eur", "fixed_eur", "immediate_eur",
                "adaptive_eur", "always_inspect_eur")
  character_fields <- c("budget", "generating_scenario", "action")
  numeric_fields <- setdiff(required, character_fields)
  if (!is.data.frame(records) || !all(required %in% names(records)) ||
      anyNA(records[required]) ||
      !all(vapply(records[character_fields], is.character, logical(1))) ||
      !all(vapply(records[numeric_fields], function(x) {
        is.numeric(x) && all(is.finite(x))
      }, logical(1))) ||
      any(!records$action %in% c("commit", "inspect")) ||
      anyDuplicated(records[c("lot_id", "budget", "grid_size")]) ||
      any(records$lot_id < 1 | records$lot_id != floor(records$lot_id)) ||
      any(records$inspection_labour_eur < 0) ||
      any(records$supported_rate_nodes < 1) ||
      any(records$excluded_rate_nodes < 0) ||
      any(records$supported_rate_nodes + records$excluded_rate_nodes !=
            records$grid_size^2)) {
    stop("grid records need unique complete finite decisions and node counts",
         call. = FALSE)
  }
  sizes <- sort(unique(records$grid_size))
  budgets <- sort(unique(records$budget))
  ids <- sort(unique(records$lot_id))
  if (length(ids) < 2L || length(budgets) != 2L ||
      length(unique(records$reference_per_class)) != 2L ||
      !isTRUE(tryCatch({ validate_adaptive_grid_sizes(sizes); TRUE },
                       error = function(e) FALSE)) ||
      any(!vapply(split(records, records$budget), function(part) {
        length(unique(part$reference_per_class)) == 1L
      }, logical(1))) ||
      any(!vapply(split(records, records$lot_id), function(part) {
        nrow(part) == length(budgets) * length(sizes) &&
          setequal(paste(part$budget, part$grid_size),
                   as.vector(outer(budgets, sizes, paste))) &&
          length(unique(part$generating_scenario)) == 1L &&
          length(unique(part$true_sensitivity)) == 1L &&
          length(unique(part$true_specificity)) == 1L
      }, logical(1)))) {
    stop("grid records must balance budgets and sizes on shared latent lots",
         call. = FALSE)
  }
  invisible(TRUE)
}

#' Paired comparisons with the largest supported grid, conditional on budget.
#' Resample whole latent lots jointly across all budgets and grid sizes. The
#' 9x9 reference is a sensitivity anchor; zero comparison is not zero bias.
summarize_adaptive_grid_experiment <- function(records, n_boot = 1000L) {
  validate_adaptive_grid_records(records)
  assert_nonnegative_scalar(n_boot, "n_boot", integer = TRUE)
  if (n_boot < 1L) stop("n_boot must be positive", call. = FALSE)
  ids <- sort(unique(records$lot_id))
  budgets <- sort(unique(records$budget))
  sizes <- sort(unique(records$grid_size))
  reference_size <- max(sizes)
  boot_indices <- matrix(replicate(n_boot, sample.int(length(ids), length(ids),
                                                      replace = TRUE)),
                         nrow = length(ids), ncol = n_boot)
  interval <- function(values) {
    means <- vapply(seq_len(n_boot), function(b) {
      mean(values[boot_indices[, b]])
    }, numeric(1))
    unname(stats::quantile(means, c(0.025, 0.975)))
  }
  do.call(rbind, lapply(budgets, function(budget) {
    base <- records[records$budget == budget &
                      records$grid_size == reference_size, , drop = FALSE]
    base <- base[match(ids, base$lot_id), , drop = FALSE]
    do.call(rbind, lapply(sizes, function(grid_size) {
      part <- records[records$budget == budget &
                        records$grid_size == grid_size, , drop = FALSE]
      part <- part[match(ids, part$lot_id), , drop = FALSE]
      difference <- part$adaptive_eur - base$adaptive_eur
      ci <- interval(difference)
      data.frame(
        budget = budget, reference_per_class = part$reference_per_class[1],
        grid_size = grid_size, reference_grid_size = reference_size,
        n_lots = length(ids),
        mean_supported_rate_nodes = mean(part$supported_rate_nodes),
        mean_excluded_rate_nodes = mean(part$excluded_rate_nodes),
        inspected_fraction = mean(part$action == "inspect"),
        fraction_action_changed = mean(part$action != base$action),
        fraction_sale_day_changed = mean(part$adaptive_day != base$adaptive_day),
        mean_abs_sensitivity_mean_difference = mean(abs(
          part$reference_mean_sensitivity - base$reference_mean_sensitivity)),
        mean_abs_specificity_mean_difference = mean(abs(
          part$reference_mean_specificity - base$reference_mean_specificity)),
        mean_abs_planned_value_difference_eur = mean(abs(
          part$planned_option_value_eur - base$planned_option_value_eur)),
        mean_return_difference_eur = mean(difference),
        lower_95_eur = ci[1], upper_95_eur = ci[2],
        fraction_below_reference = mean(difference < 0),
        row.names = NULL
      )
    }))
  }))
}
