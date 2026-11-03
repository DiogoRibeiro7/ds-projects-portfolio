# Run from projects/orange_storage_decision:
# Rscript scripts/run_adaptive_gauss_reference.R [number_of_lots] [output_directory]
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

args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 2L) {
  stop("usage: run_adaptive_gauss_reference.R [number_of_lots] [output_directory]")
}
n_lots <- if (length(args)) suppressWarnings(as.numeric(args[1])) else 20L
destination <- if (length(args) == 2L) args[2] else
  "results/adaptive_gauss_reference"
settings <- list(
  n_lots = n_lots, seed = 20261111L,
  reference_per_class = c(scarce = 8L, ample = 80L),
  grid_sizes = c(3L, 5L, 7L, 9L),
  gauss_reference_order = 64L, gauss_stability_order = 32L,
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
records <- run_adaptive_gauss_benchmark(
  n_lots, settings$seed, settings$reference_per_class,
  settings$grid_sizes, settings$gauss_reference_order,
  settings$gauss_stability_order,
  settings$eur_per_check, settings$checks_per_cohort
)
summary <- summarize_adaptive_gauss_benchmark(records)

dir.create(destination, recursive = TRUE, showWarnings = FALSE)
utils::write.csv(records, file.path(destination, "records.csv"), row.names = FALSE)
utils::write.csv(summary, file.path(destination, "by_grid.csv"), row.names = FALSE)
writeLines(trimws(capture.output(dput(settings)), which = "right"),
           file.path(destination, "settings.R"), useBytes = TRUE)

lines <- c(
  "# Continuous-rate quadrature reference for adaptive inspection", "",
  sprintf("%d independent predecision synthetic snapshots; no future quality outcomes are generated.",
          n_lots),
  "The reference integrates an audit-updated Beta rate posterior over the",
  "informative Se+Sp>1 triangle using order-64 Gauss-Legendre quadrature.",
  "Order 32 supplies a separate numerical stability diagnostic, not a formal",
  "integration error bound. Each finite grid sees the same audits, labels,",
  "price assumptions and scenario prior as the reference.", "",
  "## Finite-grid policies evaluated under the higher-order posterior", "",
  "| Reference audit per class | Midpoint grid | Different day-12 action | Mean contingent-day disagreement mass | Mean/max option-value error EUR | Mean/max expected numerical regret EUR | Max order-32/64 value gap EUR |",
  "|---|---:|---:|---:|---:|---:|---:|"
)
for (i in seq_len(nrow(summary))) {
  x <- summary[i, ]
  lines <- c(lines, sprintf(
    "| %s (%d) | %dx%d | %.1f%% | %.1f%% | %.3f / %.3f | %+.3f / %+.3f | %.4g |",
    x$budget, x$reference_per_class, x$grid_size, x$grid_size,
    100 * x$fraction_finite_action_changed,
    100 * x$mean_contingent_disagreement_mass,
    x$mean_abs_information_error_eur, x$max_abs_information_error_eur,
    x$mean_expected_numerical_regret_eur,
    x$max_expected_numerical_regret_eur,
    x$max_gauss_information_stability_eur
  ))
}
lines <- c(lines, "", "## Interpretation limits", "",
  "Expected numerical regret means the reference-optimal conditional expected",
  "EUR minus the expected EUR of the finite policy's frozen sale mapping,",
  "both measured under the reference posterior and the same candidate actions.",
  "It is NOT a realised fruit return, out-of-lot prediction score, or bound on",
  "commercial deployment losses. Tiny negative values from rounding are kept.",
  "The contingent disagreement mass weights day-16 possible count outcomes",
  "by their reference predictive probabilities; it remains hypothetical if",
  "a policy commits instead of inspecting.",
  "Both quadrature orders are finite; agreement does not constitute a certified",
  "error bound. No continuous truth, grader drift, reference-label error or",
  "real market prices are measured. No bootstrap confidence intervals are",
  "assigned to the numerical approximation errors reported here.", "",
  sprintf("Reproduce: `Rscript scripts/run_adaptive_gauss_reference.R %d`.",
          n_lots),
  "[Protocol](../../ADAPTIVE_GAUSS_REFERENCE.md), [paired records](records.csv),",
  "[summary](by_grid.csv), [settings](settings.R).")
writeLines(lines, file.path(destination, "RESULTS.md"), useBytes = TRUE)
cat(sprintf("Wrote %d numerical benchmark records to %s\n",
            nrow(records), destination))
print(summary, row.names = FALSE)
