# Run from projects/orange_storage_decision:
# Rscript scripts/run_adaptive_grid_resolution.R [number_of_lots] [output_directory]
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

args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 2L) {
  stop("usage: run_adaptive_grid_resolution.R [number_of_lots] [output_directory]")
}
n_lots <- if (length(args)) suppressWarnings(as.numeric(args[1])) else 60L
destination <- if (length(args) == 2L) args[2] else
  "results/adaptive_grid_resolution"
settings <- list(
  n_lots = n_lots, seed = 20261109L,
  bootstrap_seed = 20261110L, bootstrap_replicates = 1000L,
  reference_per_class = c(scarce = 8L, ample = 80L),
  grid_sizes = c(3L, 5L, 7L, 9L), reference_grid_size = 9L,
  beta_prior_shape = c(alpha = 1, beta = 1),
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
records <- run_adaptive_grid_experiment(
  n_lots, settings$seed, settings$reference_per_class,
  settings$grid_sizes, settings$eur_per_check, settings$checks_per_cohort
)
set.seed(settings$bootstrap_seed)
summary <- summarize_adaptive_grid_experiment(records,
                                              settings$bootstrap_replicates)

dir.create(destination, recursive = TRUE, showWarnings = FALSE)
utils::write.csv(records, file.path(destination, "records.csv"), row.names = FALSE)
utils::write.csv(summary, file.path(destination, "by_grid.csv"), row.names = FALSE)
writeLines(trimws(capture.output(dput(settings)), which = "right"),
           file.path(destination, "settings.R"), useBytes = TRUE)

lines <- c(
  "# Numerical grid sensitivity of the adaptive inspection policy", "",
  sprintf("%d independent synthetic latent lots, each paired across two audit budgets and four grid sizes.",
          n_lots),
  "Every choice is made from the same day-12 observations and reference audit",
  "for its budget. All grid choices share actual fruit outcomes and prices.",
  "The 9x9 midpoint grid is a *comparison anchor*, not a continuous posterior",
  "benchmark. Zero difference against that anchor does not prove convergence.", "",
  "## Decisions and returns versus the 9x9 policy", "",
  "| Audit per class | Grid | Mean retained/excluded nodes | Inspection | Different action/day | Mean abs option-value difference EUR | Return difference EUR [95% paired CI] | Below anchor |",
  "|---|---:|---:|---:|---:|---:|---:|---:|"
)
for (i in seq_len(nrow(summary))) {
  x <- summary[i, ]
  lines <- c(lines, sprintf(
    "| %s (%d) | %dx%d | %.1f / %.1f | %.1f%% | %.1f%% / %.1f%% | %.3f | %+.2f [%+.2f, %+.2f] | %.1f%% |",
    x$budget, x$reference_per_class, x$grid_size, x$grid_size,
    x$mean_supported_rate_nodes, x$mean_excluded_rate_nodes,
    100 * x$inspected_fraction, 100 * x$fraction_action_changed,
    100 * x$fraction_sale_day_changed,
    x$mean_abs_planned_value_difference_eur,
    x$mean_return_difference_eur, x$lower_95_eur, x$upper_95_eur,
    100 * x$fraction_below_reference
  ))
}
lines <- c(lines, "", "## Interpretation and reproducibility", "",
  "Reference posteriors are approximated by independent Beta(1,1) priors",
  "updated with independent confusion counts, then conditioned on Se+Sp>1.",
  "Grids use deterministic midpoint marginal quantiles of size 3, 5, 7 or 9",
  "and renormalise the informative nodes. This approximation is discontinuous",
  "near the orientation boundary and may change non-monotonically with grid",
  "resolution; these rows cannot certify integration error or convergence.",
  "The 9x9 rate grid can create up to 243 joint states. The study does not",
  "include a grid above this size, an exact posterior integral, or real data.",
  "Within each lot, both audit budgets share the same independent reference",
  "draw stream; all grids in one budget share the same audited counts.",
  "Every plan is frozen before future labels and fruit lifetimes are drawn.",
  "Negative paired returns remain in the records. Intervals resample whole",
  "lots across budgets and grids, not individual fruit or policy records.",
  "The simulation does not establish calibration accuracy or commercial gain.", "",
  sprintf("Reproduce: `Rscript scripts/run_adaptive_grid_resolution.R %d`.",
          n_lots),
  "[Design](../../ADAPTIVE_GRID_RESOLUTION.md), [paired records](records.csv),",
  "[grid comparison](by_grid.csv), [settings](settings.R).")
writeLines(lines, file.path(destination, "RESULTS.md"), useBytes = TRUE)
cat(sprintf("Wrote %d paired resolution records to %s\n",
            nrow(records), destination))
print(summary, row.names = FALSE)
