# Reproducible base-R missing-endpoint bounds. Run from orange_storage_decision.
# No unavailable future price is imputed, interpolated or inferred.
source("R/sima_market_prices.R")
source("R/sima_rolling_backtest.R")
source("R/sima_composition_audit.R")
source("R/sima_forward_quote_windows.R")
source("R/sima_quote_break_even.R")
source("R/sima_missing_quote_bounds.R")

args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 2L) {
  stop("Usage: Rscript scripts/run_sima_missing_quote_bounds.R [source.csv.gz] [new_dir]",
       call. = FALSE)
}
input <- if (length(args) >= 1L) args[[1L]] else
  "data/raw/sima_oranges_wholesale_2025-01-01_2026-10-31.csv.gz"
destination <- if (length(args) == 2L) args[[2L]] else
  "results/replay_sima_missing_quote_bounds"
assert_sima_string(input, "input")
assert_sima_string(destination, "destination")
if (!file.exists(input)) {
  stop("The dated wholesale GPP SIMA source is missing", call. = FALSE)
}
if (file.exists(destination) || dir.exists(destination)) {
  stop("Refusing to overwrite an existing missing-quote bounds report",
       call. = FALSE)
}
settings <- list(
  observed_source = "GPP SIMA Mercados Abastecedores (wholesale)",
  source_file = basename(input),
  source_md5 = unname(tools::md5sum(input)),
  anchor_start = as.Date("2026-01-05"),
  exact_calendar_week_horizons = 1:4,
  hypothetical_weekly_retained_saleable_share = c(1, 0.99, 0.98, 0.95),
  hypothetical_weekly_cost_eur_per_initial_kg = c(0, 0.005, 0.01, 0.02),
  tolerance_eur_per_initial_kg = 1e-10,
  last_quoted_week_conservatively_censored = TRUE,
  no_assumptions_about_unobserved_eligible_outcomes = TRUE,
  missing_endpoints_are_not_imputed = TRUE,
  binary_outcome_bounds_not_confidence_intervals = TRUE,
  r_version = R.version.string
)
quotes <- read_sima_orange_prices(input, market_stage = "wholesale",
                                   exclude_provisional = TRUE)
windows <- pair_sima_forward_quotes(
  sima_weekly_panels(quotes), settings$anchor_start,
  settings$exact_calendar_week_horizons
)
result <- bound_sima_missing_forward_quotes(
  windows, settings$hypothetical_weekly_retained_saleable_share,
  settings$hypothetical_weekly_cost_eur_per_initial_kg,
  settings$tolerance_eur_per_initial_kg
)
dir.create(destination, recursive = TRUE, showWarnings = FALSE)
tables <- list(
  bounds = result$bounds, market_bounds = result$market_bounds,
  coverage = result$coverage, market_coverage = result$market_coverage,
  missing_eligible_windows = result$missing_eligible_windows,
  scenarios = result$scenarios
)
for (name in names(tables)) {
  utils::write.csv(tables[[name]],
                   file.path(destination, paste0(name, ".csv")),
                   row.names = FALSE, na = "")
}
writeLines(capture.output(dput(settings)), file.path(destination, "settings.R"),
           useBytes = TRUE)

get_example <- function(cohort, rho, cost) {
  x <- result$bounds[
    result$bounds$cohort == cohort &
      result$bounds$weekly_retention == rho &
      result$bounds$weekly_cost_eur_per_initial_kg == cost,
    , drop = FALSE
  ]
  x[order(x$horizon_weeks), , drop = FALSE]
}
examples <- data.frame(
  label = c("Frictionless", "1% loss + 0.005 EUR",
            "2% loss + 0.010 EUR", "5% loss + 0.020 EUR"),
  rho = c(1, 0.99, 0.98, 0.95),
  cost = c(0, 0.005, 0.01, 0.02),
  stringsAsFactors = FALSE
)
report <- c(
  "# What if the future wholesale orange quote is missing?", "",
  "**No-assumption bounds on hypothetical quotation feasibility, not farm returns.**",
  "Only GPP SIMA wholesale quotes and exact week positions are observed.",
  "Fruit retention and storage cost remain explicitly hypothetical.", "",
  sprintf("Archived source: `%s` (MD5 `%s`), %d wholesale observations.",
          settings$source_file, settings$source_md5, nrow(quotes)), "",
  "## Bounds without fabricating a later quotation", "",
  "For each fully elapsed calendar-week horizon, let N be the number of",
  "time-eligible starting quotes, O the number of exact observed endpoints,",
  "M=N-O the absent eligible endpoints, and S the observed quote-based",
  "break-even successes at a fixed hypothetical retention/cost scenario.", "",
  "The logical identification interval for the eligible population is",
  "`[S/N, (S+M)/N]`. Strict improvement has the analogous interval.",
  "Every absent endpoint can be a binary success or failure, with **no**",
  "imputed price. These are sharp worst/best completions, **not confidence",
  "intervals** and not evidence of actual stored-fruit returns.", "",
  "## The two different denominators", "",
  "- `horizon_eligible`: the specific forward horizon elapsed by the last",
  "  complete quoted source week. Its denominator changes across horizons.",
  "- `common_maturity`: the **same** starting quotes whose *all four*",
  "  horizons have elapsed. Matching the future quotes is not required.",
  "- The 343 fully matched anchors used previously form a **different**,",
  "  ex-post complete-case subset. Common maturity does not condition",
  "  on those eventual future observations.", "",
  "| Cohort | Horizon | Eligible | Observed | Missing | Missing share | Fully matched |",
  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"
)
for (i in seq_len(nrow(result$coverage))) {
  x <- result$coverage[i, , drop = FALSE]
  report <- c(report, sprintf(
    "| %s | %d weeks | %d | %d | %d | %.1f%% | %d |",
    x$cohort, x$horizon_weeks, x$eligible_anchors,
    x$observed_endpoints, x$missing_eligible_endpoints,
    100 * x$missing_share, x$fully_matched_all_horizons
  ))
}
report <- c(
  report, "", "## Example hypothetical break-even bounds", "",
  "The observed-only fraction excludes absent target quotes. Lower and",
  "upper bounds allow all elapsed eligible endpoints to be failures or",
  "successes respectively; no unknown price is estimated.", "",
  "| Cohort | Hypothetical scenario | Horizon | Observed-only | Lower | Upper | Strict lower | Strict upper |",
  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"
)
for (cohort in c("horizon_eligible", "common_maturity")) {
  for (j in seq_len(nrow(examples))) {
    x <- get_example(cohort, examples$rho[j], examples$cost[j])
    for (i in seq_len(nrow(x))) {
      report <- c(report, sprintf(
        "| %s | %s | %d weeks | %.1f%% | %.1f%% | %.1f%% | %.1f%% | %.1f%% |",
        cohort, examples$label[j], x$horizon_weeks[i],
        100 * x$observed_only_break_even_fraction[i],
        100 * x$lower_break_even_fraction[i],
        100 * x$upper_break_even_fraction[i],
        100 * x$lower_strict_improvement_fraction[i],
        100 * x$upper_strict_improvement_fraction[i]
      ))
    }
  }
}
report <- c(
  report, "",
  "![Unknown eligible quotation outcomes](missing_quote_bounds.png)",
  "", "## Interpretation boundaries", "",
  "- Right-censored future horizons are excluded; absence is counted",
  "  only for quote endpoints whose target week has fully elapsed.",
  "- No missing price is set to zero, filled with an adjacent quote,",
  "  interpolated, forecast, or treated as an observed producer return.",
  "- The results bound a *hypothetical quote-based sign*, not actual",
  "  storage costs, fruit losses, sales, prices received or profit.",
  "- The fixed common-maturity cohort removes selection based on future",
  "  quote *matching*, but still depends on the retrospective archive cutoff.",
  "- Outcomes may be missing systematically. Overlapping market weeks",
  "  are dependent. Bounds are not IID uncertainty or causal estimates.",
  "- Quote dates are not verified first-publication timestamps.",
  "- The completed synthetic orange-storage model remains unchanged.",
  "", "## Reproduction", "", "From `projects/orange_storage_decision`:", "",
  "```sh", "Rscript tests/test_sima_missing_quote_bounds.R",
  "Rscript scripts/run_sima_missing_quote_bounds.R \\",
  sprintf("  %s \\", input),
  "  results/replay_sima_missing_quote_bounds_new", "```", "",
  "The output directory must be new. The artifact contains per-scenario",
  "cohort bounds, by-market partitions, coverage, absent eligible windows,",
  "hypothetical settings, the report and the PNG figure."
)
writeLines(report, file.path(destination, "RESULTS.md"), useBytes = TRUE)

# Vertical segments are sharp identification intervals. Solid points
# represent observed-only frequencies; they are not imputed midpoints.
colors <- c("#4A7898", "#B6716D")
plot_cohort <- function(cohort) {
  graphics::plot(
    NA_real_, xlim = c(0.7, 4.3), ylim = c(0, 100),
    xlab = "Elapsed calendar-week horizon",
    ylab = "At-least-break-even quote fraction (%)",
    main = if (cohort == "horizon_eligible") {
      "Each horizon's time-eligible anchors"
    } else {
      "Fixed all-horizons-matured anchors"
    }, xaxt = "n"
  )
  graphics::axis(1, at = 1:4)
  for (j in 1:2) {
    x <- get_example(cohort, examples$rho[j], examples$cost[j])
    position <- x$horizon_weeks + (j - 1.5) * 0.16
    graphics::segments(
      position, 100 * x$lower_break_even_fraction,
      position, 100 * x$upper_break_even_fraction,
      col = colors[j], lwd = 3
    )
    graphics::points(
      position, 100 * x$observed_only_break_even_fraction,
      col = colors[j], pch = 19, cex = 0.9
    )
  }
  graphics::legend("topright", legend = examples$label[1:2],
                   col = colors, pch = 19, lwd = 3, cex = 0.75, bty = "n")
}
grDevices::png(file.path(destination, "missing_quote_bounds.png"),
                width = 1480, height = 650, res = 135)
graphics::par(mfrow = c(1L, 2L), mar = c(5, 5, 3, 1))
plot_cohort("horizon_eligible")
plot_cohort("common_maturity")
invisible(grDevices::dev.off())

print(result$coverage, row.names = FALSE)
cat(sprintf("Wrote exact SIMA missing-quote bounds to %s\n", destination))
