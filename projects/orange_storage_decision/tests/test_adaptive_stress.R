# Run from projects/orange_storage_decision. Base R only.
source("R/model.R")
source("R/sampling.R")
source("R/simulation.R")
source("R/adaptive_policy.R")
source("R/adaptive_evaluation.R")
source("R/adaptive_stress.R")
expect_error <- function(expression) {
  stopifnot(tryCatch({ force(expression); FALSE }, error = function(e) TRUE))
}

# Common uniforms implement the intended independent-binomial likelihood and
# make acceptability pathwise monotone when true shelf life increases.
stock <- data.frame(cohort = c("a", "b"), mass_kg = c(100, 50),
                    initially_sellable = c(TRUE, TRUE),
                    shape = c(1, 1), scale = c(10, 20))
schedule <- expand.grid(cohort = stock$cohort, day = c(6, 12),
                        KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
u <- matrix(c(0.1, 0.4, 0.7, 0.2, 0.5, 0.9,
              0.1, 0.2, 0.3, 0.4, 0.7, 0.9), nrow = 4L, ncol = 3L)
counts <- stress_inspection_counts(stock, schedule, u)
index <- match(schedule$cohort, stock$cohort)
prob <- weibull_survival(schedule$day, stock$shape[index], stock$scale[index])
stopifnot(identical(counts$n_acceptable, as.integer(rowSums(u < prob))),
          all(counts$n_checked == 3), all(counts$n_acceptable <= 3),
          all(stress_inspection_counts(transform(stock, scale = scale * 0.75),
                                       schedule, u)$n_acceptable <=
                stress_inspection_counts(transform(stock, scale = scale * 1.25),
                                          schedule, u)$n_acceptable))
expect_error(stress_inspection_counts(stock, schedule, u[-1, ]))
expect_error(stress_inspection_counts(stock, schedule, replace(u, 1, NA_real_)))
expect_error(stress_inspection_counts(stock, schedule, u + 1))
expect_error(stress_inspection_counts(stock, transform(schedule, day = 0), u))
expect_error(stress_inspection_counts(stock, transform(schedule, shape = 1), u))
expect_error(validate_adaptive_shifts(c(shorter = 0.75, longer = 1.25)))
expect_error(validate_adaptive_shifts(c(shorter = 0, unchanged = 1)))
expect_error(validate_adaptive_shifts(c(shorter = 1, unchanged = 1)))
expect_error(validate_adaptive_shifts(c(shorter = 0.75, unchanged = 1,
                                      unchanged = 1.25)))

# Prospective ordering: all three day-12 policies must be frozen before any
# day-16 inspection labels or individual future fruit lifetimes are generated.
original_plan <- plan_adaptive_inspection
original_counts <- stress_inspection_counts
original_draw <- draw_stress_unit_lifetimes
plans <- 0L
plan_adaptive_inspection <- function(...) {
  plans <<- plans + 1L
  original_plan(...)
}
stress_inspection_counts <- function(truth, schedule, uniforms) {
  if (any(schedule$day > 12)) stopifnot(plans == 3L)
  original_counts(truth, schedule, uniforms)
}
draw_stress_unit_lifetimes <- function(...) {
  stopifnot(plans == 3L)
  original_draw(...)
}
set.seed(125)
spy <- simulate_adaptive_stress_lot(1)
plan_adaptive_inspection <- original_plan
stress_inspection_counts <- original_counts
draw_stress_unit_lifetimes <- original_draw
stopifnot(plans == 3L, nrow(spy) == 3L)

# Paired scenarios: a shared latent scenario, one price path, the same initial
# fruit mass and common unit Weibull potential outcomes across true scale shifts.
records <- run_adaptive_stress_experiment(6, seed = 407)
again <- run_adaptive_stress_experiment(6, seed = 407)
stopifnot(identical(records, again), nrow(records) == 18L,
          all(records$action %in% c("commit", "inspect")),
          all(records$adaptive_eur == ifelse(records$action == "inspect",
                                             records$always_inspect_eur,
                                             records$fixed_eur)),
          all(records$planned_gain_eur >= 0),
          all(records$always_inspect_day >= 16L),
          all(records$adaptive_day >= 12L))
validate_adaptive_stress_records(records)
for (lot in split(records, records$lot_id)) {
  ordered <- lot[match(c("shorter", "unchanged", "longer"), lot$shift), ]
  stopifnot(length(unique(ordered$generating_scenario)) == 1L,
            ordered$immediate_eur[1] <= ordered$immediate_eur[2],
            ordered$immediate_eur[2] <= ordered$immediate_eur[3])
}

# Reordering the true-world regimes cannot change the common random numbers or
# the corresponding decisions and outcomes; only report-row order changes.
reversed <- run_adaptive_stress_experiment(6, seed = 407,
  shifts = c(longer = 1.25, unchanged = 1, shorter = 0.75))
sort_records <- function(x) {
  x <- x[order(x$lot_id, x$shift), , drop = FALSE]
  rownames(x) <- NULL
  x
}
stopifnot(identical(sort_records(records), sort_records(reversed)))

# Substantial check charges must select fixed commitment in every world.
expensive <- run_adaptive_stress_experiment(2, seed = 73,
                                            eur_per_check = 100000)
stopifnot(all(expensive$action == "commit"),
          all(expensive$adaptive_eur == expensive$fixed_eur))
expect_error(run_adaptive_stress_experiment(1))
expect_error(run_adaptive_stress_experiment(2, checks_per_cohort = 0))
expect_error(run_adaptive_stress_experiment(2, shifts = c(shorter = 0.75,
                                                          longer = 1.25)))

# All confidence intervals resample *lots*, with the same draw indices used
# for every shift. Row reordering does not alter the reported contrasts.
set.seed(22)
summary <- summarize_adaptive_stress(records, 200)
set.seed(22)
shuffled_summary <- summarize_adaptive_stress(records[nrow(records):1, ], 200)
stopifnot(identical(summary, shuffled_summary),
          nrow(summary$by_shift) == 3L, nrow(summary$contrasts) == 2L,
          all(summary$by_shift$lower_95_eur <= summary$by_shift$upper_95_eur),
          all(summary$contrasts$lower_95_eur <= summary$contrasts$upper_95_eur),
          all(is.finite(summary$by_shift$mean_realised_gain_eur)))
expect_error(summarize_adaptive_stress(records, 0))
expect_error(summarize_adaptive_stress(records[-1, ]))
expect_error(summarize_adaptive_stress(rbind(records, records[1, ])))
bad <- records
bad$generating_scenario[1] <- "invented"
expect_error(summarize_adaptive_stress(bad))
cat("Orange adaptive shelf-life stress tests passed.\n")
