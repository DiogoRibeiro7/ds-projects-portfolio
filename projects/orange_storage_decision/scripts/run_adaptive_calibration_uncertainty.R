# Run from projects/orange_storage_decision:
# Rscript scripts/run_adaptive_calibration_uncertainty.R [number_of_lots] [output_directory]
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

args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 2L) {
  stop("usage: run_adaptive_calibration_uncertainty.R [number_of_lots] [output_directory]")
}
n_lots <- if (length(args)) suppressWarnings(as.numeric(args[1])) else 200L
destination <- if (length(args) == 2L) args[2] else
  "results/adaptive_calibration_uncertainty"
settings <- list(
  n_lots = n_lots, seed = 20261107L,
  bootstrap_seed = 20261108L, bootstrap_replicates = 1000L,
  reference_per_class = c(scarce = 8L, ample = 80L),
  beta_prior_shape = c(alpha = 1, beta = 1), midpoint_grid_size = 5L,
  grader_orientation = "sensitivity + specificity > 1",
  true_rate_range = c(lower = 0.75, upper = 0.95),
  policy_prior = c(short = 0.3, typical = 0.4, long = 0.3),
  policy_scale_multipliers = c(short = 0.70, typical = 1, long = 1.35),
  historical_days = c(6L, 12L), historical_checks_per_cohort = 6L,
  reference_day = 10L, decision_day = 12L, inspection_day = 16L,
  checks_per_cohort = 8L, eur_per_check = 0.02,
  delivery_lag_days = 2L, last_sale_day = 28L,
  storage_eur_per_kg_day = 0.001, r_version = R.version.string
)
records <- run_adaptive_calibration_experiment(
  n_lots, settings$seed, settings$reference_per_class,
  settings$midpoint_grid_size, settings$eur_per_check,
  settings$checks_per_cohort
)
set.seed(settings$bootstrap_seed)
summary <- summarize_adaptive_calibration_experiment(records,
                                                     settings$bootstrap_replicates)

dir.create(destination, recursive = TRUE, showWarnings = FALSE)
utils::write.csv(records, file.path(destination, "records.csv"), row.names = FALSE)
utils::write.csv(summary$by_budget, file.path(destination, "by_budget.csv"),
                 row.names = FALSE)
utils::write.csv(summary$contrast, file.path(destination, "contrast.csv"),
                 row.names = FALSE)
writeLines(trimws(capture.output(dput(settings)), which = "right"),
           file.path(destination, "settings.R"), useBytes = TRUE)

lines <- c(
  "# Adaptive inspection with finite reference-grader calibration", "",
  sprintf("%d independent synthetic lots, each paired across two reference budgets.",
          n_lots),
  "Both policies receive identical independent reference audits and target-lot",
  "reported labels. One integrates a conditional Beta posterior for grading",
  "rates over a finite midpoint grid; the other plugs in the grid's mean rates.",
  "Both decide on day 12 whether to inspect on day 16 or commit to a sale.", "",
  "## Integrated versus posterior-mean plug-in and fixed commitment", "",
  "| Reference per true class | Inspect integrated / plug-in | Gain vs plug-in EUR [95% paired CI] | Below plug-in | Gain vs fixed EUR [95% paired CI] | Mean abs Se/Sp error |",
  "|---|---:|---:|---:|---:|---:|"
)
for (i in seq_len(nrow(summary$by_budget))) {
  x <- summary$by_budget[i, ]
  lines <- c(lines, sprintf(
    "| %s (%d) | %.1f%% / %.1f%% | %+.2f [%+.2f, %+.2f] | %.1f%% | %+.2f [%+.2f, %+.2f] | %.3f / %.3f |",
    x$budget, x$reference_per_class,
    100 * x$integrated_inspected_fraction, 100 * x$point_inspected_fraction,
    x$mean_gain_vs_point_eur, x$lower_95_point_eur, x$upper_95_point_eur,
    100 * x$fraction_below_point,
    x$mean_gain_vs_fixed_eur, x$lower_95_fixed_eur, x$upper_95_fixed_eur,
    x$mean_abs_sensitivity_error, x$mean_abs_specificity_error
  ))
}
x <- summary$contrast[1, ]
lines <- c(lines, "", "## Change in the integrated-minus-plug-in premium", "",
  sprintf("%s: %+.2f EUR [%+.2f, %+.2f] (paired whole-lot 95%% interval).",
          gsub("_", " ", x$contrast), x$mean_gain_difference_eur,
          x$lower_95_eur, x$upper_95_eur), "",
  "## Assumptions and reproducibility", "",
  "Sensitivity and specificity are independently Beta(1,1) before an audited",
  "confusion table and are conditioned on Se+Sp>1. Five midpoint-quantile",
  "values per rate approximate the conditional posterior. Excluded nodes are",
  "reported for every lot; the approximation is not exact continuous Bayes.",
  "Synthetic true rates vary between 0.75 and 0.95 independently per lot.",
  "Reference tests are independent of target inspections; the larger budget",
  "contains the smaller budget's audit draws. All sale decisions precede day-16",
  "potential reports and fruit outcomes. Market prices and survival scenario",
  "library are synthetic and known to both policies. Inspections are",
  "nondestructive and incur labour only when ordered. Negative gains remain.",
  "Intervals cover between-lot simulation variability, not uncertainty in",
  "the assumption of perfect reference labels or real-world effectiveness.", "",
  sprintf("Reproduce: `Rscript scripts/run_adaptive_calibration_uncertainty.R %d`.",
          n_lots),
  "[Design](../../ADAPTIVE_CALIBRATION_UNCERTAINTY.md), [records](records.csv),",
  "[budget summary](by_budget.csv), [contrast](contrast.csv),",
  "[settings](settings.R).")
writeLines(lines, file.path(destination, "RESULTS.md"), useBytes = TRUE)
cat(sprintf("Wrote %d paired calibration-uncertainty records to %s\n",
            nrow(records), destination))
print(summary$by_budget, row.names = FALSE)
print(summary$contrast, row.names = FALSE)
