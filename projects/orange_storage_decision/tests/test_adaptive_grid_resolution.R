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
source("R/adaptive_grid_resolution.R")
expect_error <- function(expression) {
  stopifnot(tryCatch({ force(expression); FALSE }, error = function(e) TRUE))
}

# Only the supported grid family is compared. The high-resolution grid is a
# policy benchmark rather than a quadrature convergence certificate.
validate_adaptive_grid_sizes(c(3L, 5L, 7L, 9L))
expect_error(validate_adaptive_grid_sizes(c(3L, 5L, 7L)))
expect_error(validate_adaptive_grid_sizes(c(3L, 3L, 9L)))
expect_error(validate_adaptive_grid_sizes(c(2L, 9L)))
expect_error(validate_adaptive_grid_sizes(c(3L, 10L)))
expect_error(validate_adaptive_grid_sizes(c(3, 9.5)))

# Same reference audit yields grids with different node counts and posterior
# moments. Their excluded-node counts must be fully accounted for.
reference <- data.frame(day = 10L, true_positive = 5L, false_negative = 3L,
                        true_negative = 6L, false_positive = 2L)
for (n in c(3L, 5L, 7L, 9L)) {
  grid <- reference_rate_posterior_grid(reference, 12L, n)
  stopifnot(grid$n_total == n^2L,
            nrow(grid$nodes) + grid$n_discarded == n^2L,
            all(grid$nodes$sensitivity + grid$nodes$specificity > 1),
            abs(sum(grid$nodes$weight) - 1) < 1e-12)
}

# All reference-grid decisions must be frozen BEFORE potential future labels
# or latent fruit outcomes are generated. The spy counts the exact factorial.
budgets <- c(scarce = 8L, ample = 80L)
sizes <- c(3L, 9L)
original_plan <- plan_adaptive_with_calibration_uncertainty
original_counts <- simulate_noisy_adaptive_counts
original_lifetimes <- simulate_fruit_lifetimes
plan_calls <- 0L
expected_calls <- length(budgets) * length(sizes)
plan_adaptive_with_calibration_uncertainty <- function(...) {
  plan_calls <<- plan_calls + 1L
  original_plan(...)
}
simulate_noisy_adaptive_counts <- function(truth, schedule, state_uniforms,
                                           grading_uniforms, sensitivity,
                                           specificity) {
  if (any(schedule$day > 12L)) stopifnot(plan_calls == expected_calls)
  original_counts(truth, schedule, state_uniforms, grading_uniforms,
                  sensitivity, specificity)
}
simulate_fruit_lifetimes <- function(truth, mean_fruit_kg = 0.20) {
  stopifnot(plan_calls == expected_calls)
  original_lifetimes(truth, mean_fruit_kg)
}
set.seed(990)
spy <- simulate_adaptive_grid_lot(1L, budgets, sizes)
plan_adaptive_with_calibration_uncertainty <- original_plan
simulate_noisy_adaptive_counts <- original_counts
simulate_fruit_lifetimes <- original_lifetimes
stopifnot(plan_calls == expected_calls, nrow(spy) == expected_calls)

# Reproducible, balanced paired lots. The policy never sees the generating
# scenario or true grader rates. Returns use common fruit potential outcomes.
records <- run_adaptive_grid_experiment(3L, seed = 991L,
                                         grid_sizes = c(3L, 5L, 9L))
again <- run_adaptive_grid_experiment(3L, seed = 991L,
                                       grid_sizes = c(3L, 5L, 9L))
stopifnot(identical(records, again), nrow(records) == 3L * 2L * 3L,
          all(records$supported_rate_nodes + records$excluded_rate_nodes ==
                records$grid_size^2),
          all(records$adaptive_day >= 12L),
          all(records$adaptive_eur == ifelse(
            records$action == "inspect",
            records$always_inspect_eur, records$fixed_eur)),
          all(records$adaptive_day[records$action == "commit"] ==
                records$fixed_day[records$action == "commit"]),
          all(vapply(split(records, records$lot_id), function(part) {
            length(unique(part$immediate_eur)) == 1L &&
              length(unique(part$true_sensitivity)) == 1L &&
              length(unique(part$true_specificity)) == 1L &&
              length(unique(part$generating_scenario)) == 1L
          }, logical(1))))

# A paired lot bootstrap uses the 9x9 policy as a numerical sensitivity
# reference; self-comparisons must be exactly zero by construction.
set.seed(992)
report <- summarize_adaptive_grid_experiment(records, 100L)
stopifnot(nrow(report) == 2L * 3L,
          all(is.finite(report$mean_return_difference_eur)),
          all(report$lower_95_eur <= report$upper_95_eur),
          all(report$fraction_action_changed >= 0 &
                report$fraction_action_changed <= 1),
          all(report$fraction_sale_day_changed >= 0 &
                report$fraction_sale_day_changed <= 1))
reference_rows <- report$grid_size == 9L
stopifnot(all(report$mean_return_difference_eur[reference_rows] == 0),
          all(report$lower_95_eur[reference_rows] == 0),
          all(report$upper_95_eur[reference_rows] == 0),
          all(report$fraction_action_changed[reference_rows] == 0),
          all(report$fraction_sale_day_changed[reference_rows] == 0),
          all(report$mean_abs_planned_value_difference_eur[reference_rows] == 0))
expect_error(run_adaptive_grid_experiment(1L))
expect_error(summarize_adaptive_grid_experiment(records, 0L))
expect_error(summarize_adaptive_grid_experiment(records[-1L, ], 10L))
expect_error(summarize_adaptive_grid_experiment(rbind(records, records[1L, ]),
                                               10L))
expect_error(summarize_adaptive_grid_experiment(
  transform(records, excluded_rate_nodes = -1L), 10L
))
cat("Orange adaptive calibration grid-resolution tests passed.\n")
