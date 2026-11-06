# Base-R reproducible conditional quote frontier report.
# Run from projects/orange_storage_decision. The thresholds describe
# hypothetical cost and saleable mass, never observed lot-level economics.
source("R/sima_market_prices.R")
source("R/sima_rolling_backtest.R")
source("R/sima_composition_audit.R")
source("R/sima_forward_quote_windows.R")
source("R/sima_quote_break_even.R")
source("R/sima_quote_feasibility_frontier.R")

args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 2L) {
  stop(paste(
    "Usage: Rscript scripts/run_sima_quote_feasibility_frontier.R",
    "[source.csv.gz] [new_dir]"
  ), call. = FALSE)
}
input <- if (length(args) >= 1L) args[[1L]] else
  "data/raw/sima_oranges_wholesale_2025-01-01_2026-10-31.csv.gz"
destination <- if (length(args) == 2L) args[[2L]] else
  "results/replay_sima_quote_feasibility_frontier"
assert_sima_string(input, "input")
assert_sima_string(destination, "destination")
if (!file.exists(input)) {
  stop("The dated GPP SIMA wholesale source is missing", call. = FALSE)
}
if (file.exists(destination) || dir.exists(destination)) {
  stop("Refusing to overwrite an existing feasibility-frontier report",
       call. = FALSE)
}

settings <- list(
  observed_source = "GPP SIMA Mercados Abastecedores (wholesale)",
  source_file = basename(input),
  source_md5 = unname(tools::md5sum(input)),
  anchor_start = as.Date("2026-01-05"),
  observed_horizons_weeks = 1:4,
  hypothetical_weekly_retention_for_cost_ceiling = c(1, 0.99, 0.98, 0.95),
  hypothetical_weekly_cost_for_retention_floor_eur_per_initial_kg =
    c(0, 0.005, 0.01, 0.02),
  equality_tolerance_eur_per_initial_kg = 1e-10,
  selection = "343 anchors with all four exact future weekly quotes",
  retention_and_cost_are_not_observed = TRUE,
  last_quoted_week_censored = TRUE,
  r_version = R.version.string
)
quotes <- read_sima_orange_prices(input, market_stage = "wholesale",
                                   exclude_provisional = TRUE)
windows <- pair_sima_forward_quotes(
  sima_weekly_panels(quotes), settings$anchor_start,
  settings$observed_horizons_weeks
)
result <- evaluate_sima_quote_feasibility_frontier(
  windows, settings$hypothetical_weekly_retention_for_cost_ceiling,
  settings$hypothetical_weekly_cost_for_retention_floor_eur_per_initial_kg,
  settings$equality_tolerance_eur_per_initial_kg
)

dir.create(destination, recursive = TRUE, showWarnings = FALSE)
tables <- list(
  cost_ceiling_pairs = result$cost_ceiling_pairs,
  retention_floor_pairs = result$retention_floor_pairs,
  cost_ceiling_balanced = result$cost_ceiling_balanced,
  retention_floor_balanced = result$retention_floor_balanced,
  cost_ceiling_all_available = result$cost_ceiling_all_available,
  retention_floor_all_available = result$retention_floor_all_available,
  cost_ceiling_market_balanced = result$cost_ceiling_market_balanced,
  retention_floor_market_balanced = result$retention_floor_market_balanced,
  coverage = result$coverage,
  assumed_grid = result$assumed_grid
)
for (name in names(tables)) {
  utils::write.csv(tables[[name]],
                   file.path(destination, paste0(name, ".csv")),
                   row.names = FALSE, na = "")
}
writeLines(capture.output(dput(settings)), file.path(destination, "settings.R"),
           useBytes = TRUE)

at_retention <- function(rho) {
  x <- result$cost_ceiling_balanced[
    result$cost_ceiling_balanced$assumed_weekly_retention == rho,
    , drop = FALSE
  ]
  x[order(x$horizon_weeks), , drop = FALSE]
}
at_cost <- function(cost) {
  x <- result$retention_floor_balanced[
    result$retention_floor_balanced$
      assumed_weekly_cost_eur_per_initial_kg == cost,
    , drop = FALSE
  ]
  x[order(x$horizon_weeks), , drop = FALSE]
}

coverage <- result$coverage
report <- c(
  "# Maximum hypothetical storage friction supported by recorded orange quotes",
  "", "**Continuous conditional quotation thresholds; not farm economics.**",
  "The source supplies wholesale prices and exact calendar weeks only.",
  "No orchard inventory, cost, quality, saleable fraction or actual sale is recorded.",
  "", sprintf("Source: `%s` (MD5 `%s`); %d original wholesale quotes.",
              settings$source_file, settings$source_md5, nrow(quotes)),
  "", "## Inverted break-even equations", "",
  "For initial wholesale quotation `P_start`, quoted future price",
  "`P_future`, exact calendar-week horizon `h`, hypothetical retained",
  "saleable weekly fraction `rho`, and weekly cost `c` per original kg:",
  "", "`illustrative_difference = rho^h * P_future - h*c - P_start`",
  "", "`maximum_weekly_cost = (rho^h * P_future - P_start) / h`",
  "", "`minimum_weekly_retention = ((P_start + h*c) / P_future)^(1/h)`",
  "", "`maximum_weekly_loss = 1 - minimum_weekly_retention`", "",
  "**Negative maximum costs or losses are intentionally untruncated.**",
  "They mean even perfect cost or retention assumptions cannot satisfy",
  "break-even under that specific observed quote pair and the remaining",
  "fixed hypothetical parameter. No parameter is fitted from the archive.",
  "", "## Fixed comparison population", "",
  "All summary comparisons use the identical **343 2026 starting quotes**",
  "with exact 1-, 2-, 3-, and 4-week future quotations for the same",
  "product/region/market panel. The final quoted source week is censored.",
  "Additional all-available summaries expose horizon-dependent coverage.",
  "", "| Horizon | Anchors | Complete horizon eligible | Quoted endpoint | Balanced |",
  "| ---: | ---: | ---: | ---: | ---: |"
)
for (i in seq_len(nrow(coverage))) {
  x <- coverage[i, , drop = FALSE]
  report <- c(report, sprintf("| %d weeks | %d | %d | %d | %d |",
                              x$horizon_weeks, x$observed_anchor_quotes,
                              x$complete_horizon_eligible,
                              x$exact_endpoint_observed,
                              x$balanced_all_horizon_anchors))
}
report <- c(
  report, "", "## Continuous cost ceiling by assumed retained mass", "",
  "The reported *raw median ceiling* can be negative. The feasibility",
  "share means the observed future quote would cover **zero** weekly",
  "storage cost at that hypothetical weekly retention; it does not say",
  "any storage cost was zero in practice.", "",
  "| Retained saleable share/week | Horizon | Median max cost (EUR/original kg/week) | Zero-cost feasible | Strictly positive cost room |",
  "| ---: | ---: | ---: | ---: | ---: |"
)
for (rho in settings$hypothetical_weekly_retention_for_cost_ceiling) {
  x <- at_retention(rho)
  for (i in seq_len(nrow(x))) {
    report <- c(report, sprintf(
      "| %.0f%% | %d weeks | %+.5f | %.1f%% | %.1f%% |",
      100 * rho, x$horizon_weeks[i],
      x$median_raw_ceiling_eur_per_initial_kg_week[i],
      100 * x$fraction_feasible_at_zero_cost[i],
      100 * x$fraction_with_positive_cost_room[i]
    ))
  }
}
report <- c(
  report, "", "## Continuous retention requirement by assumed weekly cost", "",
  "A retention requirement above **100%** means not feasible even",
  "with zero mass loss for that specific price pair and assumed cost.",
  "The raw median loss allowance may therefore be negative.", "",
  "| Cost (EUR/original kg/week) | Horizon | Median required weekly retention | Perfect-retention feasible | Positive loss room |",
  "| ---: | ---: | ---: | ---: | ---: |"
)
for (cost in settings$hypothetical_weekly_cost_for_retention_floor_eur_per_initial_kg) {
  x <- at_cost(cost)
  for (i in seq_len(nrow(x))) {
    report <- c(report, sprintf(
      "| %.3f | %d weeks | %.2f%% | %.1f%% | %.1f%% |",
      cost, x$horizon_weeks[i],
      100 * x$median_required_weekly_retention[i],
      100 * x$fraction_feasible_at_full_retention[i],
      100 * x$fraction_with_positive_loss_room[i]
    ))
  }
}
report <- c(
  report, "", "![Continuous hypothetical cost and retention frontiers](quote_frontier.png)",
  "", "## Scientific constraints", "",
  "- The source is GPP SIMA quoted wholesale prices, not realised farm",
  "  selling prices or returns for the *same physical fruit lot*.",
  "- Retention and cost parameters are chosen hypothetically and are not",
  "  measured, estimated, or calibrated from GPP or postharvest data.",
  "- Every threshold assumes full tradability at the quoted prices,",
  "  unchanged product class and constant weekly loss/cost mechanisms.",
  "- The common 343-anchor sample is selected by future quote availability;",
  "  it is not a prospective trading population or selling policy.",
  "- Overlapping weekly records within product/market panels are dependent.",
  "  Quantiles and percentages are descriptive, not IID uncertainty intervals.",
  "- Source quote dates are not verified first-publication timestamps.",
  "- This algebra identifies conditional boundary values, **not** actual",
  "  cost-effective storage, a causal treatment effect or an optimal horizon.",
  "- The synthetic postharvest model and its published conclusions are intact.",
  "", "## Reproduction", "",
  "Run from `projects/orange_storage_decision`:", "",
  "```sh", "Rscript tests/test_sima_quote_feasibility_frontier.R",
  "Rscript scripts/run_sima_quote_feasibility_frontier.R \\",
  sprintf("  %s \\", input),
  "  results/replay_sima_quote_feasibility_frontier_new", "```", "",
  "The output directory must be new. All pair-level thresholds, balanced",
  "and all-available horizon summaries, market breakdowns, an assumption",
  "grid, source coverage, settings, this report and a PNG are included."
)
writeLines(report, file.path(destination, "RESULTS.md"), useBytes = TRUE)

colors <- c("#4A7898", "#69A08B", "#C79A5F", "#B6716D")
grDevices::png(file.path(destination, "quote_frontier.png"),
                width = 1480, height = 650, res = 135)
graphics::par(mfrow = c(1L, 2L), mar = c(5, 5, 3, 1))
cost_curves <- vapply(
  settings$hypothetical_weekly_retention_for_cost_ceiling,
  function(rho) at_retention(rho)$median_raw_ceiling_eur_per_initial_kg_week,
  numeric(4L)
)
graphics::matplot(
  1:4, cost_curves, type = "b", pch = 16:19, lty = 1,
  col = colors, xlab = "Observed calendar-week horizon",
  ylab = "Median raw cost ceiling (EUR/initial kg/week)",
  main = "Assumed retention: maximum weekly cost"
)
graphics::abline(h = 0, lty = 2, col = "grey40")
graphics::legend(
  "topright",
  legend = sprintf("%.0f%% retained", 100 *
                     settings$hypothetical_weekly_retention_for_cost_ceiling),
  col = colors, pch = 16:19, lty = 1, cex = 0.75, bty = "n"
)
retention_curves <- vapply(
  settings$hypothetical_weekly_cost_for_retention_floor_eur_per_initial_kg,
  function(c) 100 * at_cost(c)$median_required_weekly_retention,
  numeric(4L)
)
graphics::matplot(
  1:4, retention_curves, type = "b", pch = 16:19, lty = 1,
  col = colors, xlab = "Observed calendar-week horizon",
  ylab = "Median required weekly retention (%)",
  main = "Assumed cost: minimum saleable retention"
)
graphics::abline(h = 100, lty = 2, col = "grey40")
graphics::legend(
  "topright",
  legend = sprintf("%.3f EUR/week",
                   settings$hypothetical_weekly_cost_for_retention_floor_eur_per_initial_kg),
  col = colors, pch = 16:19, lty = 1, cex = 0.75, bty = "n"
)
invisible(grDevices::dev.off())
print(result$coverage, row.names = FALSE)
print(result$cost_ceiling_balanced, row.names = FALSE)
print(result$retention_floor_balanced, row.names = FALSE)
cat(sprintf("Wrote quote feasibility frontiers to %s\n", destination))
