# Run from projects/orange_storage_decision:
# Rscript scripts/run_adaptive_label_error.R [number_of_lots] [output_directory]
source("R/model.R")
source("R/sampling.R")
source("R/simulation.R")
source("R/adaptive_policy.R")
source("R/adaptive_evaluation.R")
source("R/adaptive_stress.R")
source("R/adaptive_label_error.R")

args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 2L) {
  stop("usage: run_adaptive_label_error.R [number_of_lots] [output_directory]")
}
n_lots <- if (length(args)) suppressWarnings(as.numeric(args[1])) else 200L
destination <- if (length(args) == 2L) args[2] else "results/adaptive_label_error"
settings <- list(n_lots = n_lots, seed = 20261105L,
                 bootstrap_seed = 20261106L, bootstrap_replicates = 1000L,
                 label_regimes = default_adaptive_label_regimes(),
                 policy_prior = c(short = 0.3, typical = 0.4, long = 0.3),
                 policy_scale_multipliers = c(short = 0.70, typical = 1,
                                              long = 1.35),
                 historical_days = c(6L, 12L),
                 historical_checks_per_cohort = 6L,
                 decision_day = 12L, inspection_day = 16L,
                 checks_per_cohort = 8L, eur_per_check = 0.02,
                 delivery_lag_days = 2L, last_sale_day = 28L,
                 storage_eur_per_kg_day = 0.001, r_version = R.version.string)

records <- run_adaptive_label_experiment(
  n_lots, settings$seed, settings$label_regimes,
  settings$eur_per_check, settings$checks_per_cohort
)
set.seed(settings$bootstrap_seed)
summary <- summarize_adaptive_label_experiment(records,
                                               settings$bootstrap_replicates)

dir.create(destination, recursive = TRUE, showWarnings = FALSE)
utils::write.csv(records, file.path(destination, "records.csv"), row.names = FALSE)
utils::write.csv(summary$by_regime, file.path(destination, "by_regime.csv"),
                 row.names = FALSE)
utils::write.csv(summary$contrasts, file.path(destination, "contrasts.csv"),
                 row.names = FALSE)
writeLines(trimws(capture.output(dput(settings)), which = "right"),
           file.path(destination, "settings.R"), useBytes = TRUE)

lines <- c(
  "# Adaptive inspection with imperfect quality labels", "",
  sprintf("%d independent latent lots, each evaluated under three paired grading regimes.",
          n_lots),
  "The calibrated and naive policies see exactly the same reported counts.",
  "Only the calibrated policy uses externally assumed grading sensitivity",
  "and specificity. Both use the same predeclared survival library and prices.",
  "The perfect-label control should give identical calibrated and naive decisions.", "",
  "## Calibrated adaptive policy versus naive adaptive and calibrated fixed sale", "",
  "| Labels (Se/Sp) | Calibrated/naive inspect | Gain vs naive EUR [95% paired CI] | Below naive | Gain vs fixed EUR [95% paired CI] |",
  "|---|---:|---:|---:|---:|"
)
for (i in seq_len(nrow(summary$by_regime))) {
  x <- summary$by_regime[i, ]
  lines <- c(lines, sprintf(
    "| %s (%.2f/%.2f) | %.1f%% / %.1f%% | %+.2f [%+.2f, %+.2f] | %.1f%% | %+.2f [%+.2f, %+.2f] |",
    x$regime, x$sensitivity, x$specificity,
    100 * x$calibrated_inspected_fraction,
    100 * x$naive_inspected_fraction,
    x$mean_gain_vs_naive_eur, x$lower_95_naive_eur, x$upper_95_naive_eur,
    100 * x$fraction_below_naive,
    x$mean_gain_vs_fixed_eur, x$lower_95_fixed_eur, x$upper_95_fixed_eur
  ))
}
lines <- c(lines, "", "## Change in the optional inspection premium", "",
  "Each contrast subtracts the same lot's calibrated adaptive-minus-fixed",
  "return under perfect labels. Bootstrap draws resample shared whole lots.", "",
  "| Comparison | Difference EUR [95% paired CI] |", "|---|---:|")
for (i in seq_len(nrow(summary$contrasts))) {
  x <- summary$contrasts[i, ]
  lines <- c(lines, sprintf("| %s | %+.2f [%+.2f, %+.2f] |",
                           gsub("_", " ", x$contrast),
                           x$mean_inspection_premium_difference_eur,
                           x$lower_95_eur, x$upper_95_eur))
}
lines <- c(lines, "", "## Limits and reproducibility", "",
  "True quality is sampled first, then each label is generated independently",
  "conditional on its true state and the prespecified sensitivity/specificity.",
  "All day-12 plans are frozen before potential day-16 observations and fruit",
  "lifetimes are drawn. Commit policies cannot revise their selling day.",
  "Common draws couple label regimes and both policies within each lot.",
  "Inspections are independent, nondestructive and charged only when used.",
  "True survival follows the finite candidate library; prices are known",
  "synthetic forecasts; calibration rates are treated as known, not estimated.",
  "Negative gains are preserved. Percentile intervals cover between-lot",
  "simulation uncertainty, not field effectiveness or calibration uncertainty.", "",
  sprintf("Reproduce: `Rscript scripts/run_adaptive_label_error.R %d`.", n_lots),
  "[Design](../../ADAPTIVE_LABEL_ERROR.md), [records](records.csv),",
  "[regime summary](by_regime.csv), [contrasts](contrasts.csv),",
  "[settings](settings.R).")
writeLines(lines, file.path(destination, "RESULTS.md"), useBytes = TRUE)
cat(sprintf("Wrote %d paired label-error records to %s\n",
            nrow(records), destination))
print(summary$by_regime, row.names = FALSE)
print(summary$contrasts, row.names = FALSE)
