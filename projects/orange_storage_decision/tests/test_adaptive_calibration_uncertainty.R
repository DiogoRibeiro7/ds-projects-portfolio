# Run from projects/orange_storage_decision. Base R only.
source("R/model.R")
source("R/sampling.R")
source("R/simulation.R")
source("R/calibration.R")
source("R/adaptive_policy.R")
source("R/adaptive_evaluation.R")
source("R/adaptive_stress.R")
source("R/adaptive_label_error.R")
source("R/adaptive_calibration_policy.R")
source("R/adaptive_calibration_evaluation.R")
expect_error <- function(expression) {
  stopifnot(tryCatch({ force(expression); FALSE }, error = function(e) TRUE))
}

# Reference audit validation is independent of whether its *point* rates are
# informative. A weak or misleading audit still updates a truncated posterior.
reference <- data.frame(day = 10L, true_positive = 8L, false_negative = 2L,
                        true_negative = 9L, false_positive = 1L)
totals <- inspection_calibration_counts(reference, 12L)
stopifnot(totals$n_acceptable == 10L, totals$n_unacceptable == 10L,
          totals$true_positive == 8L, totals$true_negative == 9L)
weak <- data.frame(day = 10L, true_positive = 0L, false_negative = 2L,
                   true_negative = 1L, false_positive = 1L)
expect_error(inspection_calibration_rates(weak, 12L))
stopifnot(inspection_calibration_counts(weak, 12L)$n_acceptable == 2L)
expect_error(inspection_calibration_counts(transform(reference, day = 13), 12L))
expect_error(inspection_calibration_counts(transform(reference, false_positive = -1), 12L))

grid <- reference_rate_posterior_grid(reference, 12L, 5L)
stopifnot(grid$n_total == 25L,
          grid$n_total == nrow(grid$nodes) + grid$n_discarded,
          all(grid$nodes$sensitivity + grid$nodes$specificity > 1),
          all(grid$nodes$weight > 0),
          abs(sum(grid$nodes$weight) - 1) < 1e-12,
          any(abs(grid$nodes$sensitivity - qbeta(0.5, 9, 3)) < 1e-12),
          any(abs(grid$nodes$specificity - qbeta(0.5, 10, 2)) < 1e-12))
weak_grid <- reference_rate_posterior_grid(weak, 12L, 9L)
stopifnot(weak_grid$n_discarded > 0L, nrow(weak_grid$nodes) > 0L)
expect_error(reference_rate_posterior_grid(reference, 12L, 2L))
expect_error(reference_rate_posterior_grid(reference, 12L, 10L))
expect_error(reference_rate_posterior_grid(transform(reference, day = 13L),
                                           12L, 5L))

# Distinct grading rates for physically identical survival hypotheses change
# reported-label likelihoods, not expected commercial revenue.
fast <- data.frame(cohort = "a", mass_kg = 100,
                   initially_sellable = TRUE, shape = 1, scale = 3)
slow <- transform(fast, scale = 100)
scenarios <- list(fast = fast, slow = slow)
prior <- c(fast = 0.5, slow = 0.5)
rates <- data.frame(sensitivity = c(0.9, 0.6),
                    specificity = c(0.9, 0.8), weight = c(0.6, 0.4))
joint <- joint_survival_grader_library(scenarios, prior, rates)
stopifnot(length(joint$scenarios) == 4L,
          identical(names(joint$prior), names(joint$scenarios)),
          abs(sum(joint$prior) - 1) < 1e-12)
observed <- data.frame(cohort = "a", day = 2L,
                       n_checked = 1L, n_acceptable = 1L)
posterior <- scenario_posterior(joint$scenarios, joint$prior, observed, 2L,
                                joint$sensitivity, joint$specificity)
physical <- c(exp(-2 / 3), exp(-2 / 100))
q <- c(inspection_label_probability(physical[1], 0.9, 0.9),
       inspection_label_probability(physical[1], 0.6, 0.8),
       inspection_label_probability(physical[2], 0.9, 0.9),
       inspection_label_probability(physical[2], 0.6, 0.8))
expected <- unname(joint$prior) * q
expected <- expected / sum(expected)
stopifnot(isTRUE(all.equal(unname(posterior), expected, tolerance = 1e-12)))
expect_error(scenario_posterior(joint$scenarios, joint$prior, observed, 2L,
                                unname(joint$sensitivity), joint$specificity))
expect_error(adaptive_scenario_label_rates(joint$scenarios,
                                          joint$sensitivity[-1L],
                                          joint$specificity))
expect_error(joint_survival_grader_library(scenarios, prior,
                                          transform(rates, weight = 1)))

# A single true survival curve has zero information value for selling even if
# the grader-rate posterior changes after seeing additional labels.
single_joint <- joint_survival_grader_library(list(fast = fast), c(fast = 1), rates)
prices <- data.frame(cohort = "a", day = 1:3,
                     price_eur_per_kg = c(0, 1, 1.3))
single <- plan_adaptive_inspection(
  single_joint$scenarios, single_joint$prior, prices, 0, 1, 2,
  checks_per_cohort = 4L, eur_per_check = 0,
  sensitivity = single_joint$sensitivity,
  specificity = single_joint$specificity
)
stopifnot(single$action == "commit",
          abs(single$gross_information_value_eur) < 1e-9)

# Reference-only rate posterior precedes target-lot data. A full wrapper
# returns both integrated and posterior-mean point plans without future labels.
early_reference <- transform(reference, day = 2L)
comparison <- plan_adaptive_with_calibration_uncertainty(
  scenarios, prior, observed, early_reference, prices, 0, 2, 3,
  grid_size = 3L, checks_per_cohort = 4L, eur_per_check = 0
)
stopifnot(length(comparison$integrated$scenarios) ==
            length(scenarios) * nrow(comparison$reference_rate_grid$nodes),
          length(comparison$point$scenarios) == length(scenarios),
          comparison$point_rates["sensitivity"] > 0.5,
          comparison$point_rates["specificity"] > 0.5)
expect_error(plan_adaptive_with_calibration_uncertainty(
  scenarios, prior, transform(observed, day = 3L), early_reference,
  prices, 0, 2, 3, grid_size = 3L, checks_per_cohort = 4L
))

# Independent audits share the first n reference draws, and do not reuse
# target-lot inspection labels as known reference-standard outcomes.
positive <- c(0.1, 0.8, 0.9, 0.2)
negative <- c(0.1, 0.95, 0.3, 0.2)
small_audit <- simulate_adaptive_reference_audit(positive, negative, 2L,
                                                 0.75, 0.8)
large_audit <- simulate_adaptive_reference_audit(positive, negative, 4L,
                                                 0.75, 0.8)
stopifnot(small_audit$true_positive == 1L,
          small_audit$true_negative == 1L,
          large_audit$true_positive == 2L,
          large_audit$true_negative == 3L)
expect_error(simulate_adaptive_reference_audit(positive, negative, 5L,
                                               0.75, 0.8))
expect_error(validate_adaptive_reference_budgets(c(a = 8L, b = 8L)))

# A temporal spy forbids potential day-16 counts and hidden outcomes until
# both reference-budget plans have been computed.
budgets <- c(scarce = 8L, ample = 80L)
original_plan <- plan_adaptive_with_calibration_uncertainty
original_counts <- simulate_noisy_adaptive_counts
original_lifetimes <- simulate_fruit_lifetimes
plan_calls <- 0L
plan_adaptive_with_calibration_uncertainty <- function(...) {
  plan_calls <<- plan_calls + 1L
  original_plan(...)
}
simulate_noisy_adaptive_counts <- function(truth, schedule, state_uniforms,
                                           grading_uniforms, sensitivity,
                                           specificity) {
  if (any(schedule$day > 12L)) stopifnot(plan_calls == length(budgets))
  original_counts(truth, schedule, state_uniforms, grading_uniforms,
                  sensitivity, specificity)
}
simulate_fruit_lifetimes <- function(truth, mean_fruit_kg = 0.20) {
  stopifnot(plan_calls == length(budgets))
  original_lifetimes(truth, mean_fruit_kg)
}
set.seed(491)
spy <- simulate_adaptive_calibration_lot(1L, budgets, grid_size = 3L)
plan_adaptive_with_calibration_uncertainty <- original_plan
simulate_noisy_adaptive_counts <- original_counts
simulate_fruit_lifetimes <- original_lifetimes
stopifnot(plan_calls == length(budgets), nrow(spy) == 2L)

# Paired outcomes, one reference-size row per latent lot. The integrated
# policy's realised return equals its always-inspect potential outcome only
# when it actually elected to inspect.
records <- run_adaptive_calibration_experiment(6L, seed = 808L, grid_size = 3L)
again <- run_adaptive_calibration_experiment(6L, seed = 808L, grid_size = 3L)
stopifnot(identical(records, again), nrow(records) == 12L,
          all(records$integrated_eur == ifelse(
            records$integrated_action == "inspect",
            records$always_inspect_eur, records$integrated_fixed_eur)),
          all(records$integrated_day[records$integrated_action == "commit"] ==
                records$integrated_fixed_day[
                  records$integrated_action == "commit"]),
          all(records$point_day[records$point_action == "commit"] ==
                records$point_fixed_day[records$point_action == "commit"]),
          all(records$integrated_planned_gain_eur >= 0))
set.seed(424)
summary <- summarize_adaptive_calibration_experiment(records, 100L)
stopifnot(nrow(summary$by_budget) == 2L, nrow(summary$contrast) == 1L,
          all(is.finite(summary$by_budget$mean_gain_vs_point_eur)),
          all(summary$by_budget$lower_95_point_eur <=
                summary$by_budget$upper_95_point_eur))
expect_error(run_adaptive_calibration_experiment(1L))
expect_error(summarize_adaptive_calibration_experiment(records, 0L))
expect_error(summarize_adaptive_calibration_experiment(records[-1, ], 10L))
expect_error(summarize_adaptive_calibration_experiment(
  rbind(records, records[1, ]), 10L
))
cat("Orange adaptive reference-calibration uncertainty tests passed.\n")
