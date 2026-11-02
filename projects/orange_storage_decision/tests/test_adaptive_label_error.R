# Run from projects/orange_storage_decision. Base R only.
source("R/model.R")
source("R/sampling.R")
source("R/simulation.R")
source("R/adaptive_policy.R")
source("R/adaptive_evaluation.R")
source("R/adaptive_stress.R")
source("R/adaptive_label_error.R")
expect_error <- function(expression) {
  stopifnot(tryCatch({ force(expression); FALSE }, error = function(e) TRUE))
}

# Hand-check the noisy-label binomial likelihood for one observation. The
# scenario prior concerns true survival, not the probability of a label.
fast <- data.frame(cohort = "a", mass_kg = 100,
                   initially_sellable = TRUE, shape = 1, scale = 3)
slow <- transform(fast, scale = 100)
scenarios <- list(fast = fast, slow = slow)
prior <- c(fast = 0.5, slow = 0.5)
observation <- data.frame(cohort = "a", day = 2L,
                          n_checked = 1L, n_acceptable = 1L)
true_probability <- c(fast = exp(-2 / 3), slow = exp(-2 / 100))
label_probability <- 0.8 * true_probability + 0.1 * (1 - true_probability)
actual <- scenario_posterior(scenarios, prior, observation, 2L, 0.8, 0.9)
stopifnot(abs(actual["slow"] - label_probability["slow"] /
                sum(label_probability)) < 1e-12)
observation$n_acceptable <- 0L
actual_failure <- scenario_posterior(scenarios, prior, observation, 2L, 0.8, 0.9)
stopifnot(abs(actual_failure["slow"] - (1 - label_probability["slow"]) /
                sum(1 - label_probability)) < 1e-12)
expect_error(scenario_posterior(scenarios, prior, observation, 1L, 0.8, 0.9))
expect_error(scenario_posterior(scenarios, prior, sensitivity = 0.4,
                                specificity = 0.5))
expect_error(scenario_posterior(scenarios, prior, sensitivity = 1.1))

# Perfect grading is backwards compatible. Lower information quality cannot
# improve the pre-cost value of the optional experiment in this toy example.
prices <- data.frame(cohort = "a", day = 1:3,
                     price_eur_per_kg = c(0, 1, 1.3))
perfect <- plan_adaptive_inspection(scenarios, prior, prices, 0, 1, 2,
                                    checks_per_cohort = 10L,
                                    eur_per_check = 0)
explicit_perfect <- plan_adaptive_inspection(scenarios, prior, prices, 0, 1, 2,
                                             checks_per_cohort = 10L,
                                             eur_per_check = 0,
                                             sensitivity = 1, specificity = 1)
noisy <- plan_adaptive_inspection(scenarios, prior, prices, 0, 1, 2,
                                  checks_per_cohort = 10L, eur_per_check = 0,
                                  sensitivity = 0.75, specificity = 0.80)
stopifnot(identical(perfect, explicit_perfect),
          noisy$gross_information_value_eur <=
            perfect$gross_information_value_eur + 1e-9,
          noisy$gross_information_value_eur >= -1e-9,
          identical(noisy$sensitivity, 0.75),
          identical(noisy$specificity, 0.80))
observation <- data.frame(cohort = "a", day = 2L,
                          n_checked = 10L, n_acceptable = 3L)
expected_after <- scenario_posterior(scenarios, prior, observation, 2,
                                     0.75, 0.80)
actual_after <- sale_after_inspection(noisy, observation)$posterior
stopifnot(isTRUE(all.equal(actual_after, expected_after, tolerance = 1e-12)))
expect_error(plan_adaptive_inspection(scenarios, prior, prices, 0, 1, 2,
                                      sensitivity = 0.3, specificity = 0.3))

# Conditional grade draws must implement both false accepts and false rejects,
# with exactly the same perfect-label sample under Se=Sp=1.
schedule <- data.frame(cohort = "a", day = 2L)
state_uniforms <- matrix(c(0.1, 0.8, 0.9, 0.2), nrow = 1L)
grading_uniforms <- matrix(c(0.9, 0.05, 0.3, 0.9), nrow = 1L)
raw_counts <- stress_inspection_counts(fast, schedule, state_uniforms)
perfect_counts <- simulate_noisy_adaptive_counts(
  fast, schedule, state_uniforms, grading_uniforms, 1, 1
)
noisy_counts <- simulate_noisy_adaptive_counts(
  fast, schedule, state_uniforms, grading_uniforms, 0.75, 0.8
)
stopifnot(identical(raw_counts, perfect_counts),
          perfect_counts$n_acceptable == 2L,
          noisy_counts$n_acceptable == 1L)
expect_error(simulate_noisy_adaptive_counts(
  fast, schedule, state_uniforms, matrix(0.5, 2, 2), 0.75, 0.8
))
expect_error(simulate_noisy_adaptive_counts(
  fast, schedule, state_uniforms, grading_uniforms, 0.2, 0.4
))
regimes <- default_adaptive_label_regimes()
expect_error(validate_adaptive_label_regimes(regimes[-1, ]))
expect_error(validate_adaptive_label_regimes(transform(regimes, specificity = 0)))
expect_error(validate_adaptive_label_regimes(rbind(regimes, regimes[2, ])))

# A spy prevents day-16 labels and fruit lifetimes being generated before
# both day-12 policies have been frozen for *every* label regime.
original_plan <- plan_adaptive_inspection
original_labels <- simulate_noisy_adaptive_counts
original_lifetimes <- simulate_fruit_lifetimes
n_plans <- 0L
expected_plans <- 2L * nrow(regimes)
plan_adaptive_inspection <- function(...) {
  n_plans <<- n_plans + 1L
  original_plan(...)
}
simulate_noisy_adaptive_counts <- function(truth, schedule, state_uniforms,
                                           grading_uniforms, sensitivity,
                                           specificity) {
  if (any(schedule$day > 12L)) stopifnot(n_plans == expected_plans)
  original_labels(truth, schedule, state_uniforms, grading_uniforms,
                  sensitivity, specificity)
}
simulate_fruit_lifetimes <- function(truth, mean_fruit_kg = 0.20) {
  stopifnot(n_plans == expected_plans)
  original_lifetimes(truth, mean_fruit_kg)
}
set.seed(345)
spy <- simulate_adaptive_label_lot(1L)
plan_adaptive_inspection <- original_plan
simulate_noisy_adaptive_counts <- original_labels
simulate_fruit_lifetimes <- original_lifetimes
stopifnot(n_plans == expected_plans, nrow(spy) == nrow(regimes))

# Every regime shares a latent lot, fruit outcomes and prices. Reported labels
# alone differ. An adaptive commitment cannot benefit from future labels.
records <- run_adaptive_label_experiment(8L, seed = 646L)
again <- run_adaptive_label_experiment(8L, seed = 646L)
stopifnot(identical(records, again), nrow(records) == 8L * nrow(regimes),
          all(records$calibrated_day >= 12L),
          all(records$naive_day >= 12L),
          all(records$calibrated_eur == ifelse(
            records$calibrated_action == "inspect",
            records$always_inspect_eur, records$fixed_eur)),
          all(records$calibrated_day[records$calibrated_action == "commit"] ==
                records$calibrated_fixed_day[
                  records$calibrated_action == "commit"]),
          all(records$naive_day[records$naive_action == "commit"] ==
                records$naive_fixed_day[records$naive_action == "commit"]),
          all(records$calibrated_planned_gain_eur >= 0))
perfect_rows <- records[records$regime == "perfect", ]
stopifnot(identical(perfect_rows$calibrated_action, perfect_rows$naive_action),
          identical(perfect_rows$calibrated_day, perfect_rows$naive_day),
          identical(perfect_rows$calibrated_eur, perfect_rows$naive_eur))
set.seed(848)
report <- summarize_adaptive_label_experiment(records, 100L)
stopifnot(nrow(report$by_regime) == nrow(regimes),
          nrow(report$contrasts) == nrow(regimes) - 1L,
          all(is.finite(report$by_regime$mean_gain_vs_naive_eur)),
          all(report$by_regime$lower_95_naive_eur <=
                report$by_regime$upper_95_naive_eur),
          report$by_regime$mean_gain_vs_naive_eur[
            report$by_regime$regime == "perfect"] == 0)
expect_error(run_adaptive_label_experiment(1L))
expect_error(summarize_adaptive_label_experiment(records, 0L))
expect_error(summarize_adaptive_label_experiment(records[-1, ], 10L))
expect_error(summarize_adaptive_label_experiment(rbind(records, records[1, ]),
                                                10L))
cat("Orange adaptive inspection label-error tests passed.\n")
