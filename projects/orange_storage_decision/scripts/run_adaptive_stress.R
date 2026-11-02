# Run from projects/orange_storage_decision:
# Rscript scripts/run_adaptive_stress.R [number_of_lots] [output_directory]
source("R/model.R")
source("R/sampling.R")
source("R/simulation.R")
source("R/adaptive_policy.R")
source("R/adaptive_evaluation.R")
source("R/adaptive_stress.R")

args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 2L) {
  stop("usage: run_adaptive_stress.R [number_of_lots] [output_directory]")
}
n_lots <- if (length(args)) suppressWarnings(as.numeric(args[1])) else 200L
destination <- if (length(args) == 2L) args[2] else "results/adaptive_stress"
settings <- list(n_lots = n_lots, seed = 20261103L, bootstrap_seed = 20261104L,
                 bootstrap_replicates = 1000L,
                 truth_scale_shifts = c(shorter = 0.75, unchanged = 1,
                                        longer = 1.25),
                 policy_prior = c(short = 0.3, typical = 0.4, long = 0.3),
                 policy_scale_multipliers = c(short = 0.70, typical = 1,
                                              long = 1.35),
                 training_days = c(6L, 12L), training_checks_per_cohort = 6L,
                 decision_day = 12L, inspection_day = 16L,
                 checks_per_cohort = 8L, eur_per_check = 0.02,
                 delivery_lag_days = 2L, last_sale_day = 28L,
                 storage_eur_per_kg_day = 0.001, r_version = R.version.string)
records <- run_adaptive_stress_experiment(
  n_lots, settings$seed, settings$truth_scale_shifts,
  settings$eur_per_check, settings$checks_per_cohort
)
set.seed(settings$bootstrap_seed)
summary <- summarize_adaptive_stress(records, settings$bootstrap_replicates)

dir.create(destination, recursive = TRUE, showWarnings = FALSE)
utils::write.csv(records, file.path(destination, "records.csv"), row.names = FALSE)
utils::write.csv(summary$by_shift, file.path(destination, "by_shift.csv"),
                 row.names = FALSE)
utils::write.csv(summary$contrasts, file.path(destination, "contrasts.csv"),
                 row.names = FALSE)
writeLines(trimws(capture.output(dput(settings)), which = "right"),
           file.path(destination, "settings.R"), useBytes = TRUE)

lines <- c("# Adaptive inspection under true shelf-life shifts", "",
  sprintf("%d independent synthetic lots evaluated under three paired truth regimes.",
          n_lots),
  "Policies retain the unchanged three-scenario survival library, prior,",
  "hypothetical price path and inspection budget. True survival scales are",
  "multiplied by 0.75, 1.00 or 1.25. All shifts use common inspection uniforms",
  "and the same unit-Weibull fruit lifetimes per lot; a real shelf-life forecast",
  "does not know the generating scenario or the true multiplier.", "",
  "## Adaptive gain versus best fixed day-12 commitment", "",
  "| True scale | Inspected | Planned gain EUR | Realised gain EUR [95% paired CI] | Below fixed | Gain vs immediate EUR |",
  "|---|---:|---:|---:|---:|---:|")
for (i in seq_len(nrow(summary$by_shift))) {
  x <- summary$by_shift[i, ]
  lines <- c(lines, sprintf("| %s (%.2f) | %.1f%% | %+.2f | %+.2f [%+.2f, %+.2f] | %.1f%% | %+.2f |",
    x$shift, x$scale_multiplier, 100 * x$inspected_fraction,
    x$mean_planned_gain_eur, x$mean_realised_gain_eur,
    x$lower_95_eur, x$upper_95_eur,
    100 * x$fraction_below_fixed, x$mean_gain_vs_immediate_eur))
}
lines <- c(lines, "", "## Difference in the policy premium versus unchanged truth", "",
  "Each contrast subtracts the same lot's adaptive-minus-fixed gain in the",
  "unchanged world. Intervals resample the *same* whole-lot indices across shifts.", "",
  "| Contrast | Mean EUR [95% paired CI] |", "|---|---:|")
for (i in seq_len(nrow(summary$contrasts))) {
  x <- summary$contrasts[i, ]
  lines <- c(lines, sprintf("| %s | %+.2f [%+.2f, %+.2f] |",
    gsub("_", " ", x$contrast), x$mean_gain_difference_eur,
    x$lower_95_eur, x$upper_95_eur))
}
lines <- c(lines, "", "## Interpretation and reproducibility", "",
  "The day-12 plan is frozen before day-16 potential inspection labels and",
  "latent fruit lifetimes are drawn. Fixed commitment does not see day-16 labels.",
  "The true-curve oracle day is a scoring reference only; its *realised* return",
  "need not dominate every other policy on a finite lot. The planned gain is",
  "max(0, fitted inspection-option advantage), not realised profit.",
  "Future prices are assumed known, labels are perfect, inspections do not",
  "destroy fruit, and strata have independent binomial observations. The true",
  "hazard change is a synthetic stress, not an empirically measured effect.",
  "Negative gains and negative contrasts are retained. Percentile 95% intervals",
  "do not adjust for multiple comparisons or prove commercial value.", "",
  sprintf("Reproduce: `Rscript scripts/run_adaptive_stress.R %d`.", n_lots),
  "[Design](../../ADAPTIVE_STRESS.md), [records](records.csv),",
  "[per-shift summary](by_shift.csv), [contrasts](contrasts.csv),",
  "[settings](settings.R).")
writeLines(lines, file.path(destination, "RESULTS.md"), useBytes = TRUE)
cat(sprintf("Wrote %d paired stress records to %s\n", nrow(records), destination))
print(summary$by_shift, row.names = FALSE)
print(summary$contrasts, row.names = FALSE)
