# Dependency-free prospective evaluation tests, run from project directory:
# Rscript tests/test_evaluation.R

source("R/model.R")
source("R/sampling.R")
source("R/evaluation.R")
source("R/simulation.R")

expect_error <- function(expression) {
  observed <- tryCatch({
    force(expression)
    FALSE
  }, error = function(error) TRUE)
  stopifnot(observed)
}

truth <- data.frame(
  cohort = c("advanced", "intermediate", "initially_immature"),
  mass_kg = c(300, 550, 150),
  initially_sellable = c(TRUE, TRUE, FALSE),
  shape = c(2.1, 1.7, 1),
  scale = c(16, 40, 1),
  stringsAsFactors = FALSE
)

# Deterministic counts remove sampling noise from the model comparison.
make_counts <- function(days, sample_size) {
  inspections <- expand.grid(
    cohort = c("advanced", "intermediate"), day = days,
    KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE
  )
  inspections$n_checked <- as.integer(sample_size)
  position <- match(inspections$cohort, truth$cohort)
  inspections$n_acceptable <- as.integer(round(
    sample_size * weibull_survival(
      inspections$day, truth$shape[position], truth$scale[position]
    )
  ))
  inspections
}
train <- make_counts(c(3, 6, 9, 12), 2500L)
holdout <- make_counts(c(16, 20, 24), 2500L)

# The weighted pooled baseline respects mass fractions, not oversampling.
weights <- inventory_sampling_weights(train, truth)
for (day in unique(train$day)) {
  rows <- train$day == day
  stopifnot(isTRUE(all.equal(
    sum(train$n_checked[rows] * weights[rows]),
    sum(train$n_checked[rows]), tolerance = 1e-10
  )))
}
expect_error(inventory_sampling_weights(train[1, , drop = FALSE],
                                        transform(truth, mass_kg = -1)))

stratified <- with_fitted_survival(truth, fit_weibull_by_cohort(train, truth))
pooled_fit <- fit_pooled_weibull(train, truth)
stopifnot(is.finite(pooled_fit$objective), pooled_fit$shape > 0,
          pooled_fit$scale > 0)
pooled <- with_pooled_survival(truth, pooled_fit)
stopifnot(length(unique(pooled$shape[pooled$initially_sellable])) == 1L)
stopifnot(length(unique(pooled$scale[pooled$initially_sellable])) == 1L)
expect_error(with_pooled_survival(truth, list(shape = 0, scale = 2)))

stratified_score <- score_future_inspections(holdout, stratified, 12L)
pooled_score <- score_future_inspections(holdout, pooled, 12L)
stopifnot(stratified_score$brier >= 0, stratified_score$brier <= 1,
          stratified_score$log_loss > 0,
          stratified_score$brier < pooled_score$brier,
          stratified_score$log_loss < pooled_score$log_loss)
expect_error(score_future_inspections(train, stratified, 12L))
expect_error(score_future_inspections(holdout, stratified, 16L))

# Reuse one fixed set of fruit-level potential outcomes for all policies.
set.seed(123L)
lifetimes <- simulate_fruit_lifetimes(truth)
prices <- simulate_price_scenario(decision_day = 12L, horizon = 28L)
sale_value <- function(day) {
  realised_sale_value(day, truth, prices, lifetimes, 0.001, 12L, 2L)
}
stopifnot(identical(sale_value(12L), sale_value(12L)))
stopifnot(is.finite(sale_value(20L)))
expect_error(sale_value(29L))
expect_error(realised_sale_value(12L, truth, prices, list(), 0.001, 12L))

# Every lot is independent; runs with the same seed must be bit-for-bit stable.
experiment_a <- run_policy_experiment(n_lots = 6L, seed = 400L)
experiment_b <- run_policy_experiment(n_lots = 6L, seed = 400L)
stopifnot(identical(experiment_a, experiment_b))
stopifnot(nrow(experiment_a) == 6L)
stopifnot(all(is.finite(as.matrix(experiment_a))))
stopifnot(all(experiment_a$sale_day_stratified >= 12L))
stopifnot(all(experiment_a$sale_day_pooled <= 28L))
stopifnot(all(experiment_a$brier_stratified >= 0))
expect_error(run_policy_experiment(n_lots = 1L))

set.seed(20L)
interval <- paired_lot_interval(c(-1, 0, 1, 2), n_boot = 200L)
stopifnot(identical(names(interval), c("mean", "lower_95", "upper_95")))
stopifnot(interval[["mean"]] == 0.5)
stopifnot(interval[["lower_95"]] <= interval[["mean"]])
stopifnot(interval[["upper_95"]] >= interval[["mean"]])
expect_error(paired_lot_interval(1))
expect_error(paired_lot_interval(c(1, NA_real_)))

cat("Orange storage out-of-time evaluation tests passed.\n")
