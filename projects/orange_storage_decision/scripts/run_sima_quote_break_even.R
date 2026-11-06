# Base-R scenario stress. Run from projects/orange_storage_decision.
# Rscript scripts/run_sima_quote_break_even.R [source.csv.gz] [new_dir]
# All costs and retention factors are HYPOTHETICAL, not GPP measurements.
source("R/sima_market_prices.R")
source("R/sima_rolling_backtest.R")
source("R/sima_composition_audit.R")
source("R/sima_forward_quote_windows.R")
source("R/sima_quote_break_even.R")

args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 2L) {
  stop("Usage: Rscript scripts/run_sima_quote_break_even.R [source.csv.gz] [new_dir]",
       call. = FALSE)
}
input <- if (length(args) >= 1L) args[[1L]] else
  "data/raw/sima_oranges_wholesale_2025-01-01_2026-10-31.csv.gz"
destination <- if (length(args) == 2L) args[[2L]] else
  "results/replay_sima_quote_break_even"
assert_sima_string(input, "input")
assert_sima_string(destination, "destination")
if (!file.exists(input)) {
  stop("The dated official GPP SIMA source is missing", call. = FALSE)
}
if (file.exists(destination) || dir.exists(destination)) {
  stop("Refusing to overwrite an existing SIMA break-even report directory",
       call. = FALSE)
}

settings <- list(
  observed_source = "GPP SIMA Mercados Abastecedores (wholesale)",
  source_file = basename(input),
  source_md5 = unname(tools::md5sum(input)),
  anchor_start = as.Date("2026-01-05"),
  observed_horizons_weeks = 1:4,
  assumed_weekly_retention = c(1, 0.99, 0.98, 0.95),
  assumed_weekly_cost_eur_per_initial_kg = c(0, 0.005, 0.01, 0.02),
  equality_tolerance_eur_per_initial_kg = 1e-10,
  retention_and_cost_are_not_observed = TRUE,
  last_quoted_week_censored = TRUE,
  conditional_population = "343 anchors with all four exact-market endpoint quotes",
  r_version = R.version.string
)
quotes <- read_sima_orange_prices(input, market_stage = "wholesale",
                                   exclude_provisional = TRUE)
weekly <- sima_weekly_panels(quotes)
windows <- pair_sima_forward_quotes(
  weekly, settings$anchor_start, settings$observed_horizons_weeks
)
result <- evaluate_sima_quote_break_even(
  windows, settings$assumed_weekly_retention,
  settings$assumed_weekly_cost_eur_per_initial_kg,
  settings$equality_tolerance_eur_per_initial_kg
)

dir.create(destination, recursive = TRUE, showWarnings = FALSE)
tables <- list(
  scenario_detail = result$detail,
  scenario_grid = result$scenarios,
  coverage = result$coverage,
  all_available = result$all_available,
  balanced = result$balanced,
  market_balanced = result$market_balanced
)
for (name in names(tables)) {
  utils::write.csv(tables[[name]],
                   file.path(destination, paste0(name, ".csv")),
                   row.names = FALSE, na = "")
}
writeLines(capture.output(dput(settings)), file.path(destination, "settings.R"),
           useBytes = TRUE)

at_scenario <- function(retention, cost) {
  x <- result$balanced[
    result$balanced$weekly_retention == retention &
      result$balanced$weekly_cost_eur_per_initial_kg == cost,
    , drop = FALSE
  ]
  x[order(x$horizon_weeks), , drop = FALSE]
}
example <- data.frame(
  scenario = c("No friction", "1% loss + 0.005 EUR", "2% loss + 0.010 EUR",
               "5% loss + 0.020 EUR"),
  retention = c(1, 0.99, 0.98, 0.95),
  cost = c(0, 0.005, 0.01, 0.02),
  stringsAsFactors = FALSE
)
coverage <- result$coverage
report <- c(
  "# How much wholesale price increase would justify waiting?", "",
  "**An illustrative threshold stress, not an observed storage experiment.**",
  "Only the GPP SIMA wholesale quotations and their calendar positions",
  "come from real data. Retention and storage costs are explicit assumptions.",
  "", sprintf("Archived GPP SIMA source: `%s`, MD5 `%s`.",
              settings$source_file, settings$source_md5),
  sprintf("%d wholesale source quotations, with %d observed 2026 anchors.",
          nrow(quotes), coverage$observed_anchor_quotes[1L]), "",
  "## Hypothetical break-even threshold", "",
  "For an assumed retained-saleable mass share `rho^h`, a weekly cost",
  "`c` per original kg and observed calendar-week horizon `h`:", "",
  "`illustrative_advantage = rho^h * P_future - h*c - P_start`", "",
  "Break-even requires an observed future wholesale quote of at least:",
  "", "`P_future >= (P_start + h*c) / rho^h`", "",
  "A **strict** positive illustrative advantage uses `>`, rather than",
  "counting unchanged prices as improvements. These expressions assume",
  "that a grower could transact at both quoted prices and that surviving",
  "fruit has unchanged market class; neither assumption is established.", "",
  "## Fixed comparison population", "",
  "All scenarios use the same **343 observed 2026 anchor panel-weeks**",
  "with exact same-product/region/market quotes at horizons 1, 2, 3 and 4.",
  "The larger available-pair population is also exported separately.",
  "The incomplete final quoted source week is conservatively censored.", "",
  "| Future weeks | Original anchors | Eligible | Available endpoints | Balanced anchors |",
  "| ---: | ---: | ---: | ---: | ---: |"
)
for (i in seq_len(nrow(coverage))) {
  x <- coverage[i, , drop = FALSE]
  report <- c(report, sprintf("| %d | %d | %d | %d | %d |",
                              x$horizon_weeks, x$observed_anchor_quotes,
                              x$complete_horizon_eligible,
                              x$exact_endpoint_observed,
                              x$balanced_all_horizon_anchors))
}
report <- c(report, "", "## How the assumed friction changes the answer", "",
            "Each scenario uses a **constant assumed weekly retained share**",
            "and **additive weekly EUR per original kg cost**. Neither was",
            "estimated from GPP. The following four scenarios are examples;",
            "the complete 4 × 4 parameter grid is in the CSV outputs.", "",
            "| Assumed weekly retained saleable share | Assumed cost/week (EUR/kg) | Horizon | At least break-even | Strictly better |",
            "| ---: | ---: | ---: | ---: | ---: |")
for (j in seq_len(nrow(example))) {
  x <- at_scenario(example$retention[j], example$cost[j])
  for (i in seq_len(nrow(x))) {
    report <- c(report, sprintf("| %.0f%% | %.3f | %d weeks | %.1f%% | %.1f%% |",
                                100 * example$retention[j], example$cost[j],
                                x$horizon_weeks[i],
                                100 * x$fraction_break_even_or_better[i],
                                100 * x$fraction_strictly_better[i]))
  }
}
report <- c(
  report, "", "![Sensitivity of quote-level break-even frequency](quote_break_even.png)",
  "", "## Non-negotiable interpretation limits", "",
  "- The **only empirical inputs** are dated nominal wholesale orange",
  "  quotations, not sell-through quantities or farmer returns.",
  "- Retention factors (100%, 99%, 98%, 95% per week) and assumed weekly",
  "  costs (0, 0.005, 0.010, 0.020 EUR per original kg) are **hypothetical**.",
  "  They are not measured GPP storage losses, cost estimates or advice.",
  "- The formula assumes multiplicative retained saleable mass, additive",
  "  incurred costs, unchanged grading and sales eligibility. Real fruit",
  "  quality, hazards, maturation, losses and sales constraints are unknown.",
  "- Positive quote-based scenarios are **not net storage returns**. There",
  "  are no observed lot-level outcomes, transport, producer prices, actual",
  "  inventory transitions, commissions or sale executions.",
  "- The 343-anchor population depends on future quotation coverage and",
  "  cannot be selected by a grower in advance. A missed target quote is NA.",
  "- Week-over-week observations overlap and depend on the market and",
  "  product mix. These are descriptive conditional frequencies, not",
  "  independently sampled estimates or counterfactual treatment effects.",
  "- Quotation dates are not proof of real-time source publication. No",
  "  live backtest or implementation claim is made.",
  "- The completed synthetic orange-storage model and its published",
  "  conclusions remain unchanged.", "",
  "## Reproduction", "",
  "From `projects/orange_storage_decision`:", "",
  "```sh", "Rscript tests/test_sima_quote_break_even.R",
  "Rscript scripts/run_sima_quote_break_even.R \\",
  sprintf("  %s \\", input),
  "  results/replay_sima_quote_break_even_new", "```", "",
  "Outputs: `scenario_detail.csv`, `scenario_grid.csv`, `coverage.csv`,",
  "`all_available.csv`, `balanced.csv`, `market_balanced.csv`,",
  "`quote_break_even.png`, `RESULTS.md` and `settings.R`.",
  "The destination must be new; no report is overwritten."
)
writeLines(report, file.path(destination, "RESULTS.md"), useBytes = TRUE)

grDevices::png(file.path(destination, "quote_break_even.png"),
                width = 1440, height = 630, res = 135)
graphics::par(mfrow = c(1L, 2L), mar = c(5, 5, 3, 1))
colors <- c("#4A7898", "#69A08B", "#C79A5F", "#B6716D")
curves <- vapply(seq_len(nrow(example)), function(i) {
  100 * at_scenario(example$retention[i], example$cost[i])$
    fraction_break_even_or_better
}, numeric(4L))
graphics::matplot(
  1:4, curves, type = "b", pch = c(16, 17, 15, 18), lty = 1,
  col = colors, ylim = c(0, 100),
  xlab = "Calendar-week horizon", ylab = "Balanced pairs at break-even (%)",
  main = "Assumed retention and cost sensitivity"
)
graphics::legend("topright", legend = example$scenario,
                 col = colors, pch = c(16, 17, 15, 18), lty = 1,
                 cex = 0.7, bty = "n")
four <- result$balanced[result$balanced$horizon_weeks == 4L,
                        , drop = FALSE]
retention <- sort(unique(four$weekly_retention), decreasing = TRUE)
cost <- sort(unique(four$weekly_cost_eur_per_initial_kg))
grid <- vapply(retention, function(rho) {
  vapply(cost, function(c) {
    x <- four[four$weekly_retention == rho &
                four$weekly_cost_eur_per_initial_kg == c, , drop = FALSE]
    100 * x$fraction_break_even_or_better
  }, numeric(1L))
}, numeric(length(cost)))
graphics::barplot(
  grid, beside = TRUE,
  names.arg = sprintf("%.0f%%", 100 * retention),
  col = colors, ylim = c(0, 100),
  xlab = "Assumed weekly retained saleable fraction",
  ylab = "4-week balanced pairs at break-even (%)",
  main = "4-week threshold across full 4 x 4 grid"
)
graphics::legend("topright", legend = sprintf("%.3f EUR/week", cost),
                 fill = colors, cex = 0.75, bty = "n")
invisible(grDevices::dev.off())

print(result$coverage, row.names = FALSE)
print(result$balanced, row.names = FALSE)
cat(sprintf("Wrote SIMA quote break-even stress to %s\n", destination))
