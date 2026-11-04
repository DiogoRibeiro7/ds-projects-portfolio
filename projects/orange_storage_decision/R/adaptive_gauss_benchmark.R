# Decision-error benchmark against a higher-order numerical grader posterior.
# Source model.R, sampling.R, simulation.R, calibration.R, adaptive_policy.R,
# adaptive_evaluation.R, adaptive_stress.R, adaptive_label_error.R,
# adaptive_calibration_policy.R, adaptive_calibration_evaluation.R,
# adaptive_grid_resolution.R and adaptive_gauss_reference.R first.

#' Compare finite-grid decisions to two Gauss quadrature orders on one lot.
#'
#' Reference audits and historical target labels are drawn before all plans.
#' No day-16 counts or hidden fruit outcomes are generated. The true survival
#' scenario and grader rates are used only to simulate predecision evidence.
simulate_adaptive_gauss_snapshot <- function(
    lot_id, budgets = c(scarce = 8L, ample = 80L),
    grid_sizes = c(3L, 5L, 7L, 9L),
    reference_order = 64L, check_order = 32L,
    eur_per_check = 0.02, checks_per_cohort = 8L) {
  assert_nonnegative_scalar(lot_id, "lot_id", integer = TRUE)
  if (lot_id < 1L) stop("lot_id must be positive", call. = FALSE)
  validate_adaptive_reference_budgets(budgets)
  validate_adaptive_grid_sizes(grid_sizes)
  assert_nonnegative_scalar(reference_order, "reference_order", integer = TRUE)
  assert_nonnegative_scalar(check_order, "check_order", integer = TRUE)
  if (check_order < 2L || reference_order > 96L ||
      reference_order <= check_order) {
    stop("reference_order must exceed check_order within [2, 96]",
         call. = FALSE)
  }
  assert_nonnegative_scalar(eur_per_check, "eur_per_check")
  assert_nonnegative_scalar(checks_per_cohort, "checks_per_cohort", integer = TRUE)
  if (checks_per_cohort < 1L) stop("positive inspections required", call. = FALSE)

  scenarios <- adaptive_scenario_library(850 * stats::runif(1, 0.25, 0.60))
  prior <- normalize_scenario_weights(scenarios,
                                      c(short = 0.3, typical = 0.4, long = 0.3))
  latent_index <- sample.int(length(scenarios), 1L, prob = prior)
  truth <- scenarios[[latent_index]]
  prices <- simulate_price_scenario(12L, 28L)
  true_sensitivity <- stats::runif(1, 0.75, 0.95)
  true_specificity <- stats::runif(1, 0.75, 0.95)
  positive_audit <- stats::runif(max(budgets))
  negative_audit <- stats::runif(max(budgets))
  eligible <- truth$cohort[truth$initially_sellable]
  schedule <- expand.grid(cohort = eligible, day = c(6L, 12L),
                          KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
  state_uniforms <- matrix(stats::runif(nrow(schedule) * 6L),
                           nrow = nrow(schedule), ncol = 6L)
  grade_uniforms <- matrix(stats::runif(nrow(schedule) * 6L),
                           nrow = nrow(schedule), ncol = 6L)
  historical <- simulate_noisy_adaptive_counts(
    truth, schedule, state_uniforms, grade_uniforms,
    true_sensitivity, true_specificity
  )

  do.call(rbind, lapply(seq_along(budgets), function(b) {
    calibration <- simulate_adaptive_reference_audit(
      positive_audit, negative_audit, budgets[b],
      true_sensitivity, true_specificity
    )
    plan_for_order <- function(order) {
      plan_adaptive_gauss_reference(
        scenarios, prior, historical, calibration, prices,
        storage_eur_per_kg_day = 0.001, decision_day = 12L,
        inspection_day = 16L, order = order,
        checks_per_cohort = checks_per_cohort,
        eur_per_check = eur_per_check, delivery_lag_days = 2L
      )
    }
    reference <- plan_for_order(reference_order)
    check <- plan_for_order(check_order)
    stability_tv <- 0.5 * sum(abs(reference$posterior - check$posterior))
    stability_value <- abs(reference$net_information_value_eur -
                             check$net_information_value_eur)
    do.call(rbind, lapply(grid_sizes, function(grid_size) {
      finite <- plan_adaptive_with_calibration_uncertainty(
        scenarios, prior, historical, calibration, prices,
        storage_eur_per_kg_day = 0.001, decision_day = 12L,
        inspection_day = 16L, grid_size = grid_size,
        checks_per_cohort = checks_per_cohort,
        eur_per_check = eur_per_check, delivery_lag_days = 2L
      )
      scored <- score_finite_adaptive_under_reference(finite, reference)
      data.frame(
        lot_id = lot_id, budget = names(budgets)[b],
        reference_per_class = unname(budgets[b]),
        generating_scenario = names(scenarios)[latent_index],
        true_sensitivity = true_sensitivity,
        true_specificity = true_specificity,
        gauss_order = reference_order, control_order = check_order,
        orientation_probability = reference$reference$orientation_probability,
        gauss_posterior_stability_tv = stability_tv,
        gauss_information_stability_eur = stability_value,
        gauss_action_stability = as.integer(reference$action == check$action),
        scored, stringsAsFactors = FALSE
      )
    }))
  }))
}

#' Independent synthetic predecision snapshots, paired across all methods.
run_adaptive_gauss_benchmark <- function(
    n_lots = 20L, seed = 20261111L,
    budgets = c(scarce = 8L, ample = 80L),
    grid_sizes = c(3L, 5L, 7L, 9L),
    reference_order = 64L, check_order = 32L,
    eur_per_check = 0.02, checks_per_cohort = 8L) {
  assert_nonnegative_scalar(n_lots, "n_lots", integer = TRUE)
  assert_nonnegative_scalar(seed, "seed", integer = TRUE)
  if (n_lots < 2L) stop("at least two independent snapshots required", call. = FALSE)
  set.seed(seed)
  do.call(rbind, lapply(seq_len(n_lots), function(i) {
    simulate_adaptive_gauss_snapshot(
      i, budgets, grid_sizes, reference_order, check_order,
      eur_per_check, checks_per_cohort
    )
  }))
}

#' Validate balanced reference-budget and finite-grid benchmark records.
validate_adaptive_gauss_records <- function(records) {
  required <- c("lot_id", "budget", "reference_per_class",
                "generating_scenario", "true_sensitivity", "true_specificity",
                "gauss_order", "control_order", "orientation_probability",
                "gauss_posterior_stability_tv", "gauss_information_stability_eur",
                "gauss_action_stability", "grid_size", "finite_action",
                "reference_action", "finite_fixed_day", "reference_fixed_day",
                "scenario_posterior_total_variation",
                "conditional_day_disagreement_mass",
                "absolute_information_value_error_eur",
                "finite_expected_eur_under_reference",
                "reference_optimal_expected_eur", "expected_numerical_regret_eur")
  categorical <- c("budget", "generating_scenario", "finite_action",
                   "reference_action")
  numeric_fields <- setdiff(required, categorical)
  if (!is.data.frame(records) || !all(required %in% names(records)) ||
      nrow(records) < 4L || anyNA(records[required]) ||
      !all(vapply(records[categorical], is.character, logical(1))) ||
      !all(vapply(records[numeric_fields], function(x) {
        is.numeric(x) && all(is.finite(x))
      }, logical(1))) ||
      any(!records$finite_action %in% c("commit", "inspect")) ||
      any(!records$reference_action %in% c("commit", "inspect")) ||
      anyDuplicated(records[c("lot_id", "budget", "grid_size")]) ||
      any(records$lot_id < 1 | records$lot_id != floor(records$lot_id)) ||
      any(records$reference_per_class < 1) ||
      any(records$orientation_probability <= 0 |
            records$orientation_probability > 1 + 1e-8) ||
      any(records$scenario_posterior_total_variation < -1e-10 |
            records$scenario_posterior_total_variation > 1 + 1e-10) ||
      any(records$conditional_day_disagreement_mass < -1e-10 |
            records$conditional_day_disagreement_mass > 1 + 1e-10) ||
      any(records$absolute_information_value_error_eur < 0) ||
      any(records$expected_numerical_regret_eur < -1e-7)) {
    stop("benchmark requires valid paired numerical decision records",
         call. = FALSE)
  }
  budgets <- sort(unique(records$budget))
  sizes <- sort(unique(records$grid_size))
  ids <- sort(unique(records$lot_id))
  if (length(ids) < 2L || length(budgets) != 2L ||
      length(unique(records$reference_per_class)) != 2L ||
      !isTRUE(tryCatch({ validate_adaptive_grid_sizes(sizes); TRUE },
                       error = function(e) FALSE)) ||
      any(!vapply(split(records, records$lot_id), function(part) {
        nrow(part) == length(budgets) * length(sizes) &&
          setequal(paste(part$budget, part$grid_size),
                   as.vector(outer(budgets, sizes, paste))) &&
          length(unique(part$generating_scenario)) == 1L &&
          length(unique(part$true_sensitivity)) == 1L &&
          length(unique(part$true_specificity)) == 1L
      }, logical(1))) ||
      any(!vapply(split(records, interaction(records$lot_id, records$budget,
                                              drop = TRUE)), function(part) {
        length(unique(part$reference_per_class)) == 1L &&
          length(unique(part$gauss_order)) == 1L &&
          length(unique(part$control_order)) == 1L &&
          length(unique(part$gauss_information_stability_eur)) == 1L &&
          length(unique(part$gauss_posterior_stability_tv)) == 1L
      }, logical(1)))) {
    stop("benchmark must balance grids and audits within latent snapshots",
         call. = FALSE)
  }
  invisible(TRUE)
}

#' Descriptive numerical error distribution by audit budget and grid size.
#' These are NOT confidence intervals or field-performance estimates.
summarize_adaptive_gauss_benchmark <- function(records) {
  validate_adaptive_gauss_records(records)
  budgets <- sort(unique(records$budget))
  sizes <- sort(unique(records$grid_size))
  do.call(rbind, lapply(budgets, function(budget) {
    do.call(rbind, lapply(sizes, function(grid_size) {
      part <- records[records$budget == budget &
                        records$grid_size == grid_size, , drop = FALSE]
      data.frame(
        budget = budget, reference_per_class = part$reference_per_class[1],
        grid_size = grid_size, n_lots = nrow(part),
        gauss_order = part$gauss_order[1],
        mean_gauss_posterior_stability_tv = mean(part$gauss_posterior_stability_tv),
        max_gauss_information_stability_eur = max(part$gauss_information_stability_eur),
        fraction_gauss_action_changed = mean(part$gauss_action_stability == 0),
        fraction_finite_action_changed = mean(part$finite_action !=
                                                part$reference_action),
        fraction_finite_fixed_day_changed = mean(part$finite_fixed_day !=
                                                   part$reference_fixed_day),
        mean_scenario_posterior_tv = mean(part$scenario_posterior_total_variation),
        mean_contingent_disagreement_mass = mean(part$conditional_day_disagreement_mass),
        mean_abs_information_error_eur = mean(part$absolute_information_value_error_eur),
        max_abs_information_error_eur = max(part$absolute_information_value_error_eur),
        mean_expected_numerical_regret_eur = mean(part$expected_numerical_regret_eur),
        max_expected_numerical_regret_eur = max(part$expected_numerical_regret_eur),
        row.names = NULL
      )
    }))
  }))
}
