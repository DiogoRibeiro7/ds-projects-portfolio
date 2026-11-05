# Run from projects/orange_storage_decision, using base R only.
# Rscript scripts/run_sima_rolling_backtest.R [source.csv.gz] [new_output_dir]
# The fixed test window and estimator settings are deliberately not tuned
# against the observed 2026 outcomes.
source("R/sima_market_prices.R")
source("R/sima_rolling_backtest.R")

args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 2L) {
  stop(paste("Usage: Rscript scripts/run_sima_rolling_backtest.R",
             "[source.csv.gz] [new_output_dir]"), call. = FALSE)
}
snapshot <- "data/raw/sima_oranges_wholesale_2025-01-01_2026-10-31.csv.gz"
input <- if (length(args) >= 1L) args[[1L]] else snapshot
destination <- if (length(args) >= 2L) args[[2L]] else
  "results/replay_sima_rolling_backtest"
assert_sima_string(destination, "destination")
if (file.exists(destination) || dir.exists(destination)) {
  stop("Refusing to overwrite an existing backtest output directory",
       call. = FALSE)
}

settings <- list(
  data_source = "GPP SIMA Mercados Abastecedores (wholesale)",
  input_file = basename(input),
  input_md5 = unname(tools::md5sum(input)),
  test_start = as.Date("2026-01-05"),
  min_history_weeks = 8L,
  lookback_weeks = 4L,
  min_recent_weeks = 2L,
  max_age_weeks = 3L,
  weekly_quotation = "last dated observation for exact product/region/market",
  r_version = R.version.string
)
prices <- read_sima_orange_prices(input, market_stage = "wholesale",
                                  exclude_provisional = TRUE)
weekly <- sima_weekly_panels(prices)
forecasts <- backtest_sima_weekly_prices(
  weekly, settings$test_start, settings$min_history_weeks,
  settings$lookback_weeks, settings$min_recent_weeks,
  settings$max_age_weeks
)
summary <- summarize_sima_rolling_prices(forecasts)

dir.create(destination, recursive = TRUE, showWarnings = FALSE)
utils::write.csv(weekly, file.path(destination, "weekly_panels.csv"), row.names = FALSE)
utils::write.csv(forecasts, file.path(destination, "rolling_forecasts.csv"),
                 row.names = FALSE, na = "")
utils::write.csv(summary$overall, file.path(destination, "summary.csv"),
                 row.names = FALSE)
utils::write.csv(summary$panel, file.path(destination, "panel_summary.csv"),
                 row.names = FALSE)
utils::write.csv(summary$coverage, file.path(destination, "coverage.csv"),
                 row.names = FALSE)
writeLines(capture.output(dput(settings)), file.path(destination, "settings.R"),
           useBytes = TRUE)

report <- c(
  "# GPP SIMA orange quotations: first rolling-origin backtest", "",
  "**Observed wholesale prices, not realised sales or lot outcomes.**", "",
  sprintf("Source: %s; MD5: `%s`.", settings$input_file, settings$input_md5),
  sprintf("One-week-ahead target dates: %s onwards.", settings$test_start),
  sprintf("Data: %d source quotation rows reduced to %d observed panel/weeks.",
          nrow(prices), nrow(weekly)),
  "Each panel is an exact product descriptor, region and wholesale market;",
  "each observed week uses its latest dated quotation. Missing weeks remain",
  "missing. Both predictors use only earlier completed weeks.", "",
  "## Data coverage", "",
  sprintf("Observed 2026 target panel/weeks: **%d**.",
          summary$coverage$observed_target_panel_weeks),
  sprintf("Paired scored target panel/weeks: **%d** (%.1f%% of observed targets).",
          summary$coverage$scored_paired_panel_weeks,
          100 * summary$coverage$paired_coverage_fraction),
  sprintf("Distinct observed target panels: %d; scored panels: %d.",
          summary$coverage$total_target_panels, summary$coverage$scored_panels),
  "These are *observed-target* coverage figures, not the proportion of all",
  "calendar weeks or of all orange commerce. Seasonally missing quotes are",
  "not imputed or treated as zero prices.", "",
  "## Paired one-week-ahead price forecast errors", "",
  "All errors are descriptive; MAE units are EUR per kg. Both methods are",
  "evaluated on exactly the same target rows (no selective score panels).", "",
  "| Method | Paired quotes | Panels | Quote-weighted MAE | Equal-panel MAE | WAPE |",
  "| --- | ---: | ---: | ---: | ---: | ---: |"
)
for (i in seq_len(nrow(summary$overall))) {
  x <- summary$overall[i, , drop = FALSE]
  report <- c(report, sprintf("| %s | %d | %d | %.4f | %.4f | %.2f%% |",
                              x$method, x$paired_quotes, x$panels,
                              x$observation_weighted_mae_eur_per_kg,
                              x$equal_panel_mae_eur_per_kg,
                              100 * x$weighted_absolute_percentage_error))
}
report <- c(
  report, "", "![Paired price-error evidence](rolling_price_errors.png)", "",
  "## Interpretation boundaries", "",
  "- Naive predicts the most recent available quotation; median_4w predicts",
  "  the median of observed quotations within the prior four calendar weeks.",
  sprintf("- At least %d historical observed weeks, %d recent weeks and a",
          settings$min_history_weeks, settings$min_recent_weeks),
  sprintf("  latest quote no older than %d weeks are required for paired scoring.",
          settings$max_age_weeks),
  "- A target is the final dated quotation observed in its calendar week,",
  "  not a trader's realised price. Its availability is established ex post.",
  "- Quote dates are known, but the exact GPP publication/revision timestamps",
  "  are not. This is an observation-date-limited backtest, **not** a certified",
  "  real-time forecast replay. No inference is made about trade execution.",
  "- Reporting covers two candidate baselines predeclared before inspection",
  "  of the held-out outcomes; no hyperparameters or panels were selected on",
  "  2026 forecast errors. As-of availability applies at the weekly boundary.",
  "- Rows within a market/grade and consecutive weeks are dependent. The",
  "  summary deliberately omits misleading independent-row confidence bounds.",
  "- There is still no lot-level shelf-life, shipment, cost or realised-sale",
  "  data. Therefore no hold-or-sell profitability claim is possible.", "",
  "## Output and reproduction", "",
  "- `weekly_panels.csv`: validated weekly quotes and intra-week counts.",
  "- `rolling_forecasts.csv`: every observed test target, including uncovered",
  "  rows, forecast origins, latest training date, counts and paired errors.",
  "- `summary.csv`, `panel_summary.csv`, `coverage.csv`: descriptive summaries.",
  "- `settings.R`: frozen rules, source checksum and R version.", "",
  "From the project root, rerun with a new output directory:", "",
  "```sh", "Rscript scripts/run_sima_rolling_backtest.R \\",
  sprintf("  %s \\", input),
  "  results/replay_sima_rolling_backtest_new", "```", "",
  "See the prospective protocol in the project's PROTOCOL.md for what is",
  "needed to validate a storage decision rather than a quotation forecast."
)
writeLines(report, file.path(destination, "RESULTS.md"), useBytes = TRUE)

paired <- forecasts[
  !is.na(forecasts$naive_eur_per_kg) & !is.na(forecasts$median_4w_eur_per_kg),
  , drop = FALSE
]
grDevices::png(file.path(destination, "rolling_price_errors.png"),
                width = 1450, height = 680, res = 140)
graphics::par(mfrow = c(1, 2), mar = c(5, 5, 3, 1))
graphics::boxplot(
  list(`Last quote` = paired$naive_abs_error_eur_per_kg,
       `Median 4 weeks` = paired$median_4w_abs_error_eur_per_kg),
  ylab = "Absolute forecast error (EUR/kg)",
  main = "Paired observed target weeks", col = c("#A8CBE3", "#DFC49B")
)
plot_max <- max(c(summary$panel$naive_mae_eur_per_kg,
                  summary$panel$median_4w_mae_eur_per_kg, 0.001)) * 1.05
graphics::plot(
  summary$panel$naive_mae_eur_per_kg,
  summary$panel$median_4w_mae_eur_per_kg,
  xlim = c(0, plot_max), ylim = c(0, plot_max), pch = 19,
  col = "#376A8A", cex = 0.8,
  xlab = "Last quote MAE (EUR/kg)",
  ylab = "Four-week median MAE (EUR/kg)",
  main = "Equal-granularity panels"
)
graphics::abline(a = 0, b = 1, lty = 2, col = "grey45")
invisible(grDevices::dev.off())

print(summary$overall, row.names = FALSE)
print(summary$coverage, row.names = FALSE)
cat(sprintf("Wrote observed rolling-origin evidence to %s\n", destination))
