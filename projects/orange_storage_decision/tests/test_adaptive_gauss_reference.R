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
source("R/adaptive_gauss_reference.R")
source("R/adaptive_gauss_benchmark.R")
expect_error <- function(expression) {
  stopifnot(tryCatch({ force(expression); FALSE }, error = function(e) TRUE))
}

# Gauss-Legendre integration over [0,1] is exact on low-degree polynomials.
rule <- unit_gauss_legendre_rule(8L)
stopifnot(nrow(rule) == 8L, all(diff(rule$node) > 0),
          all(rule$weight > 0), abs(sum(rule$weight) - 1) < 1e-14,
          abs(sum(rule$node * rule$weight) - 0.5) < 1e-14,
          abs(sum(rule$node^4 * rule$weight) - 0.2) < 1e-14)
expect_error(unit_gauss_legendre_rule(1L))
expect_error(unit_gauss_legendre_rule(97L))
expect_error(unit_gauss_legendre_rule(8.5))

# With an independent Beta(2,1) reference posterior for both rates,
# P(Se+Sp>1)=5/6, and E[Se | informative]=E[Sp | informative]=18/25.
analytic_audit <- data.frame(day = 10L, true_positive = 1L,
                             false_negative = 0L, true_negative = 1L,
                             false_positive = 0L)
analytic <- gauss_grader_rate_reference(analytic_audit, 12L, 8L)
stopifnot(abs(analytic$orientation_probability - 5 / 6) < 1e-12,
          abs(sum(analytic$nodes$weight) - 1) < 1e-12,
          abs(sum(analytic$nodes$weight * analytic$nodes$sensitivity) -
                18 / 25) < 1e-12,
          abs(sum(analytic$nodes$weight * analytic$nodes$specificity) -
                18 / 25) < 1e-12,
          all(analytic$nodes$sensitivity + analytic$nodes$specificity > 1))
expect_error(gauss_grader_rate_reference(transform(analytic_audit, day = 13L),
                                         12L, 8L))

# One survival hypothesis cannot obtain selling information solely from
# uncertain grader accuracy. The inspection-on-day-2 option has no gross gain.
fast <- data.frame(cohort = "a", mass_kg = 100,
                   initially_sellable = TRUE, shape = 1, scale = 3)
prices <- data.frame(cohort = "a", day = 1:3,
                     price_eur_per_kg = c(0, 1, 1.3))
early_audit <- transform(analytic_audit, day = 1L)
single <- plan_adaptive_gauss_reference(
  list(fast = fast), c(fast = 1), NULL, early_audit, prices,
  0, 1L, 2L, order = 16L, checks_per_cohort = 4L, eur_per_check = 0
)
stopifnot(single$action == "commit", single$fixed_day == 2L,
          abs(single$net_information_value_eur) < 1e-9,
          abs(sum(single$outcomes$probability) - 1) < 1e-12)
single_finite <- plan_adaptive_with_calibration_uncertainty(
  list(fast = fast), c(fast = 1), NULL, early_audit, prices,
  0, 1L, 2L, grid_size = 3L, checks_per_cohort = 4L,
  eur_per_check = 0
)
single_score <- score_finite_adaptive_under_reference(single_finite, single)
stopifnot(abs(single_score$expected_numerical_regret_eur) < 1e-9,
          abs(single_score$scenario_posterior_total_variation) < 1e-12)

# Realistic two-grade case: independent audits and predecision labels only.
set.seed(604)
scenarios <- adaptive_scenario_library(300)
prior <- c(short = 0.3, typical = 0.4, long = 0.3)
truth <- scenarios$typical
schedule <- expand.grid(cohort = c("advanced", "intermediate"),
                        day = c(6L, 12L), KEEP.OUT.ATTRS = FALSE,
                        stringsAsFactors = FALSE)
state <- matrix(stats::runif(nrow(schedule) * 6L), nrow = nrow(schedule))
grading <- matrix(stats::runif(nrow(schedule) * 6L), nrow = nrow(schedule))
historical <- simulate_noisy_adaptive_counts(
  truth, schedule, state, grading, 0.8, 0.9
)
market <- simulate_price_scenario(12L, 28L)
audit <- data.frame(day = 10L, true_positive = 7L,
                    false_negative = 1L, true_negative = 7L,
                    false_positive = 1L)
run_reference <- function(order = 64L, labels = historical, cal = audit) {
  plan_adaptive_gauss_reference(
    scenarios, prior, labels, cal, market,
    storage_eur_per_kg_day = 0.001,
    decision_day = 12L, inspection_day = 16L,
    order = order, checks_per_cohort = 8L,
    eur_per_check = 0.02, delivery_lag_days = 2L
  )
}
high <- run_reference(64L)
control <- run_reference(32L)
stopifnot(abs(sum(high$posterior) - 1) < 1e-12,
          abs(sum(high$outcomes$probability) - 1) < 1e-12,
          all(high$followup_days >= 16),
          max(abs(high$posterior - control$posterior)) < 1e-6,
          abs(high$net_information_value_eur -
                control$net_information_value_eur) < 1e-5)
expect_error(run_reference(32L, transform(historical, day = day + 5L)))
expect_error(run_reference(32L, cal = transform(audit, day = 13L)))

finite <- plan_adaptive_with_calibration_uncertainty(
  scenarios, prior, historical, audit, market, 0.001, 12L, 16L,
  grid_size = 5L, checks_per_cohort = 8L, eur_per_check = 0.02,
  delivery_lag_days = 2L
)
score <- score_finite_adaptive_under_reference(finite, high)
stopifnot(score$grid_size == 5L, score$expected_numerical_regret_eur >= -1e-7,
          score$scenario_posterior_total_variation >= 0,
          score$scenario_posterior_total_variation <= 1,
          score$conditional_day_disagreement_mass >= 0,
          score$conditional_day_disagreement_mass <= 1)

# Verify the finite follow-up day from the marginalised outcome weights
# against the original policy's Bayes update on several exact count outcomes.
n_rate <- nrow(finite$reference_rate_grid$nodes)
state_weights <- matrix(unname(finite$integrated$posterior),
                        nrow = n_rate, ncol = length(scenarios))
outcomes <- adaptive_joint_outcome_weights(
  scenarios, state_weights, finite$reference_rate_grid$nodes, 16L, 8L
)
future_scores <- outcomes$weights %*% high$future_values
days <- high$future_days[max.col(future_scores, ties.method = "first")]
for (i in c(1L, 41L, nrow(outcomes$counts))) {
  observed <- data.frame(cohort = colnames(outcomes$counts), day = 16L,
                         n_checked = 8L,
                         n_acceptable = as.integer(outcomes$counts[i, ]),
                         stringsAsFactors = FALSE)
  stopifnot(sale_after_inspection(finite$integrated, observed)$sale_day == days[i])
}

# The benchmark never draws day-16 counts or physical fruit outcomes. All
# policies see only day-10 reference and day-6/12 target information.
original_labels <- simulate_noisy_adaptive_counts
original_lifetimes <- simulate_fruit_lifetimes
simulate_noisy_adaptive_counts <- function(truth, schedule, state_uniforms,
                                           grading_uniforms, sensitivity,
                                           specificity) {
  stopifnot(all(schedule$day <= 12L))
  original_labels(truth, schedule, state_uniforms, grading_uniforms,
                  sensitivity, specificity)
}
simulate_fruit_lifetimes <- function(...) {
  stop("held-out fruit outcomes must never be generated by benchmark")
}
records <- run_adaptive_gauss_benchmark(
  2L, seed = 20261112L, grid_sizes = c(3L, 9L),
  reference_order = 32L, check_order = 16L
)
simulate_noisy_adaptive_counts <- original_labels
simulate_fruit_lifetimes <- original_lifetimes
again <- run_adaptive_gauss_benchmark(
  2L, seed = 20261112L, grid_sizes = c(3L, 9L),
  reference_order = 32L, check_order = 16L
)
stopifnot(identical(records, again), nrow(records) == 8L,
          all(records$expected_numerical_regret_eur >= -1e-7))
summary <- summarize_adaptive_gauss_benchmark(records)
stopifnot(nrow(summary) == 4L,
          all(is.finite(summary$mean_expected_numerical_regret_eur)),
          all(summary$mean_scenario_posterior_tv >= 0))
expect_error(run_adaptive_gauss_benchmark(1L))
expect_error(summarize_adaptive_gauss_benchmark(records[-1L, ]))
expect_error(summarize_adaptive_gauss_benchmark(rbind(records, records[1L, ])))
expect_error(summarize_adaptive_gauss_benchmark(
  transform(records, expected_numerical_regret_eur = -1)
))
cat("Orange adaptive Gaussian reference tests passed.\n")
