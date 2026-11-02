# One-step adaptive inspection with finite reference-calibration uncertainty.
# Source model.R, sampling.R, calibration.R and adaptive_policy.R first.
# Base R only; the joint discrete approximation is intentionally small.

#' Approximate the reference-only posterior for grader sensitivity/specificity.
#'
#' Each rate starts with an independent Beta(1, 1) prior. The audit updates
#' Se with TP/FN and Sp with TN/FP. A tensor grid of midpoint Beta quantiles
#' approximates that posterior, conditional on Se + Sp > 1. Conditioning fixes
#' the informative-grader orientation; removed nodes and the grid size are
#' returned explicitly. Reference labels are assumed correct and independent.
reference_rate_posterior_grid <- function(calibration, decision_day,
                                          grid_size = 5L) {
  counts <- inspection_calibration_counts(calibration, decision_day)
  assert_nonnegative_scalar(grid_size, "grid_size", integer = TRUE)
  if (grid_size < 3L || grid_size > 9L) {
    stop("grid_size must be between 3 and 9 midpoint quantiles", call. = FALSE)
  }
  probabilities <- (seq_len(grid_size) - 0.5) / grid_size
  sensitivity <- stats::qbeta(probabilities, 1 + counts$true_positive,
                              1 + counts$false_negative)
  specificity <- stats::qbeta(probabilities, 1 + counts$true_negative,
                              1 + counts$false_positive)
  all_nodes <- expand.grid(
    sensitivity = sensitivity, specificity = specificity,
    KEEP.OUT.ATTRS = FALSE
  )
  informative <- all_nodes$sensitivity + all_nodes$specificity > 1
  nodes <- all_nodes[informative, , drop = FALSE]
  if (!nrow(nodes)) {
    stop("audit and grid leave no informative grader-rate states",
         call. = FALSE)
  }
  rownames(nodes) <- NULL
  # The untruncated midpoint grid has equal probability at every node.
  # Reweight retained nodes to approximate the conditional Beta posterior.
  nodes$weight <- rep(1 / nrow(nodes), nrow(nodes))
  list(nodes = nodes, n_total = nrow(all_nodes),
       n_discarded = sum(!informative), reference = counts)
}

#' Expand survival scenarios by rate states, preserving true sales valuation.
#'
#' A rate-state duplicate has the same physical survival curve but a distinct
#' label likelihood. Posterior weights thus couple deterioration and grader
#' accuracy when ambiguous inspection results arrive.
joint_survival_grader_library <- function(scenarios, prior, rate_grid) {
  weights <- normalize_scenario_weights(scenarios, prior)
  if (!is.data.frame(rate_grid) ||
      !setequal(names(rate_grid), c("sensitivity", "specificity", "weight")) ||
      nrow(rate_grid) < 1L ||
      !all(vapply(rate_grid, function(x) is.numeric(x) &&
                    all(is.finite(x)), logical(1))) ||
      any(rate_grid$weight <= 0) ||
      abs(sum(rate_grid$weight) - 1) > 1e-10 ||
      any(rate_grid$sensitivity + rate_grid$specificity <= 1)) {
    stop("rate_grid must contain valid informative rates and weights",
         call. = FALSE)
  }
  if (length(scenarios) * nrow(rate_grid) > 250L) {
    stop("joint library limited to 250 survival-rate states", call. = FALSE)
  }
  n_rate <- nrow(rate_grid)
  library <- rep(scenarios, each = n_rate)
  state_names <- sprintf("joint_%03d", seq_along(library))
  names(library) <- state_names
  joint_prior <- stats::setNames(
    rep(unname(weights), each = n_rate) *
      rep(rate_grid$weight, times = length(scenarios)), state_names
  )
  sensitivity <- stats::setNames(rep(rate_grid$sensitivity,
                                       times = length(scenarios)), state_names)
  specificity <- stats::setNames(rep(rate_grid$specificity,
                                       times = length(scenarios)), state_names)
  invisible(adaptive_scenario_label_rates(library, sensitivity, specificity))
  list(scenarios = library, prior = joint_prior,
       sensitivity = sensitivity, specificity = specificity,
       survival_scenario = rep(names(scenarios), each = n_rate))
}

#' Compare a calibration-integrated adaptive plan with a posterior-mean plug-in.
#'
#' The audit updates the grading-rate prior ONCE. Already observed target-lot
#' labels then update the full joint survival/grader state, and the day-16
#' inspection option enumerates future reports under each joint state.
#' The plug-in alternative uses the same audit/labels but replaces the grader
#' posterior with its conditional mean before the day-12 survival update.
#' Every economic value uses true survival; no audit outcomes are future data.
plan_adaptive_with_calibration_uncertainty <- function(
    scenarios, prior, observations, calibration, prices,
    storage_eur_per_kg_day, decision_day, inspection_day,
    grid_size = 5L, checks_per_cohort = 8L,
    eur_per_check = 0.02, delivery_lag_days = 2L) {
  reference <- reference_rate_posterior_grid(calibration, decision_day,
                                             grid_size)
  joint <- joint_survival_grader_library(scenarios, prior, reference$nodes)
  posterior <- scenario_posterior(
    joint$scenarios, joint$prior, observations, decision_day,
    joint$sensitivity, joint$specificity
  )
  integrated <- plan_adaptive_inspection(
    joint$scenarios, posterior, prices, storage_eur_per_kg_day,
    decision_day, inspection_day, checks_per_cohort,
    eur_per_check, delivery_lag_days,
    joint$sensitivity, joint$specificity
  )

  mean_sensitivity <- sum(reference$nodes$sensitivity * reference$nodes$weight)
  mean_specificity <- sum(reference$nodes$specificity * reference$nodes$weight)
  point_posterior <- scenario_posterior(
    scenarios, prior, observations, decision_day,
    mean_sensitivity, mean_specificity
  )
  point <- plan_adaptive_inspection(
    scenarios, point_posterior, prices, storage_eur_per_kg_day,
    decision_day, inspection_day, checks_per_cohort,
    eur_per_check, delivery_lag_days,
    mean_sensitivity, mean_specificity
  )
  list(integrated = integrated, point = point,
       point_rates = c(sensitivity = mean_sensitivity,
                       specificity = mean_specificity),
       reference_rate_grid = reference)
}
