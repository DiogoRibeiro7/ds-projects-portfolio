# Run from projects/orange_storage_decision, using only base R.
# Rscript scripts/run_sima_availability_stress.R [source.csv.gz] [new_output_dir]
# Calendar-day release delays are hypothetical, not GPP publication records.
source("R/sima_market_prices.R")
source("R/sima_rolling_backtest.R")
source("R/sima_availability_stress.R")

args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 2L) {
  stop(paste("Usage: Rscript scripts/run_sima_availability_stress.R",
             "[source.csv.gz] [new_output_dir]"), call. = FALSE)
}
snapshot <- "data/raw/sima_oranges_wholesale_2025-01-01_2026-10-31.csv.gz"
input <- if (length(args) >= 1L) args[[1L]] else snapshot
destination <- if (length(args) >= 2L) args[[2L]] else
  "results/replay_sima_availability_stress"
assert_sima_string(destination, "destination")
if (file.exists(destination) || dir.exists(destination)) {
  stop("Refusing to overwrite an existing availability report directory",
       call. = FALSE)
}

settings <- list(
  data_source = "GPP SIMA Mercados Abastecedores (wholesale)",
  input_file = basename(input),
  input_md5 = unname(tools::md5sum(input)),
  assumed_release_lag_days = c(0L, 3L, 7L, 14L, 21L),
  test_start = as.Date("2026-01-05"),
  min_history_weeks = 8L,
  lookback_weeks = 4L,
  min_recent_weeks = 2L,
  max_age_weeks = 3L,
  target = "last dated observed quote for each exact weekly product/market",
  history = "all dated quotes assumed released by preceding Sunday",
  r_version = R.version.string
)
quotes <- read_sima_orange_prices(input, market_stage = "wholesale",
                                  exclude_provisional = TRUE)
scenarios <- backtest_sima_availability_lags(
  quotes, settings$assumed_release_lag_days, settings$test_start,
  settings$min_history_weeks, settings$lookback_weeks,
  settings$min_recent_weeks, settings$max_age_weeks
)
evidence <- summarize_sima_availability_lags(scenarios)

dir.create(destination, recursive = TRUE, showWarnings = FALSE)
utils::write.csv(scenarios, file.path(destination, "scenario_forecasts.csv"),
                 row.names = FALSE, na = "")
utils::write.csv(evidence$overall,
                 file.path(destination, "scenario_summary.csv"),
                 row.names = FALSE, na = "")
utils::write.csv(evidence$matched,
                 file.path(destination, "matched_error_comparison.csv"),
                 row.names = FALSE, na = "")
writeLines(capture.output(dput(settings)), file.path(destination, "settings.R"),
           useBytes = TRUE)

format_error <- function(value) {
  if (is.na(value)) "unavailable" else sprintf("%.4f", value)
}
format_change <- function(value) {
  if (is.na(value)) "unavailable" else sprintf("%+.4f", value)
}

report <- c(
  "# Orange wholesale price forecasts under hypothetical release delays", "",
  "**Observed GPP SIMA quotations; release delays are assumptions, not measured facts.**", "",
  sprintf("Snapshot: `%s` (MD5 `%s`).", settings$input_file,
          settings$input_md5),
  sprintf("Held-out target weeks begin %s; %d source quotations.",
          settings$test_start, nrow(quotes)),
  "The target is the last dated quotation actually observed for each",
  "product/region/market in a target week. The price is a wholesale",
  "quotation, not a verified transaction or a grower's realised return.", "",
  "## Assumed delay and forecast coverage", "",
  "The forecast is issued on the preceding Sunday. In a lag-*d* scenario,",
  "a historical quotation dated *q* is available only when `q + d <= origin`.",
  "Training weeks are rebuilt from all eligible dated quotes at each origin:",
  "if a week's final quotation is too recent, an earlier dated quote from",
  "that week can still enter the forecast. No unobserved price is imputed.", "",
  "| Assumed lag (days) | Observed targets | Paired forecasts | Coverage | Last quote MAE (EUR/kg) | Four-week median MAE (EUR/kg) |",
  "| ---: | ---: | ---: | ---: | ---: | ---: |"
)
for (i in seq_len(nrow(evidence$overall))) {
  x <- evidence$overall[i, , drop = FALSE]
  report <- c(report, sprintf("| %d | %d | %d | %.1f%% | %s | %s |",
                              x$assumed_release_lag_days,
                              x$observed_target_panel_weeks,
                              x$paired_panel_weeks,
                              100 * x$paired_coverage_fraction,
                              format_error(x$naive_mae_eur_per_kg),
                              format_error(x$median_4w_mae_eur_per_kg)))
}
report <- c(report, "", "The MAE values above are calculated separately on",
            "each lag's available target rows. Since the covered target mix",
            "changes with the assumed delay, **do not infer a performance",
            "effect from those marginal MAE columns alone**.", "",
            "## Matched-target comparison with zero delay", "",
            "The comparison below restricts both scenarios to the exact",
            "same product/market/target-week rows. Positive error changes",
            "mean the lagged information set did worse on that matched set.", "",
            "| Lag (days) | Shared targets | Change in last-quote MAE | Change in four-week median MAE |",
            "| ---: | ---: | ---: | ---: |")
for (i in seq_len(nrow(evidence$matched))) {
  x <- evidence$matched[i, , drop = FALSE]
  report <- c(report, sprintf("| %d | %d | %s | %s |",
                              x$assumed_release_lag_days,
                              x$matched_panel_weeks,
                              format_change(x$naive_mae_change_eur_per_kg),
                              format_change(x$median_4w_mae_change_eur_per_kg)))
}
report <- c(
  report, "", "![Observed target coverage by assumed delay](availability_coverage.png)", "",
  "## Interpretation and evidence boundary", "",
  "- The source CSV has quotation dates but neither first-publication",
  "  timestamps nor a history of subsequent price revisions. The 0-, 3-,",
  "  7-, 14- and 21-day delays are **counterfactual scenarios**; none is",
  "  estimated from GPP's actual reporting practice.",
  "- The zero-day scenario reproduces the earlier observation-date-limited",
  "  rolling-origin evaluation; it does **not** certify point-in-time access.",
  "- Source product descriptors, market and region are kept separate.",
  "  The two fixed baselines are last quote and previous-four-calendar-week",
  "  median; neither is re-tuned against held-out 2026 price outcomes.",
  "- Coverage is conditional on a recorded target quotation. Market/grade",
  "  combinations without target quotations are not counted as successes",
  "  or forecastable calendar weeks.",
  "- Some lag assumptions may leave no paired predictions. They are reported",
  "  with zero coverage and unavailable MAE, never as zero forecast error.",
  "- This is descriptive sensitivity analysis of a forecast-information set,",
  "  not an estimate of publication delays, prediction uncertainty or a",
  "  causal effect. Correlated panels and weeks preclude naive IID inference.",
  "- No lot-level perishability, transport, fees, farm-gate prices or actual",
  "  sales are available. The synthetic storage-policy conclusions remain",
  "  completely separate from this empirical price exercise.", "",
  "## Reproduction", "",
  "From `projects/orange_storage_decision`, run:", "",
  "```sh", "Rscript tests/test_sima_availability_stress.R",
  "Rscript scripts/run_sima_availability_stress.R \\",
  sprintf("  %s \\", input),
  "  results/replay_sima_availability_stress_new", "```", "",
  "The output directory must be new. The runner writes dated scenario rows,",
  "coverage/MAE summaries, matched-error comparisons, the plotted figure,",
  "and exact source checksum, parameters and R version in `settings.R`."
)
writeLines(report, file.path(destination, "RESULTS.md"), useBytes = TRUE)

graphics_file <- file.path(destination, "availability_coverage.png")
grDevices::png(graphics_file, width = 1450, height = 620, res = 140)
graphics::par(mfrow = c(1L, 2L), mar = c(5, 5, 3, 1))
graphics::barplot(
  100 * evidence$overall$paired_coverage_fraction,
  names.arg = evidence$overall$assumed_release_lag_days,
  ylim = c(0, 100), col = "#7AA5C8",
  xlab = "Assumed release delay (calendar days)",
  ylab = "Paired target coverage (%)",
  main = "Coverage of observed target weeks"
)
mae <- c(evidence$overall$naive_mae_eur_per_kg,
         evidence$overall$median_4w_mae_eur_per_kg)
maximum <- if (any(is.finite(mae))) max(mae[is.finite(mae)]) * 1.2 else 0
graphics::plot(
  evidence$overall$assumed_release_lag_days,
  evidence$overall$naive_mae_eur_per_kg,
  type = "b", pch = 19, col = "#276484", ylim = c(0, max(maximum, 0.001)),
  xlab = "Assumed release delay (calendar days)",
  ylab = "MAE among paired targets (EUR/kg)",
  main = "Unmatched-sample MAE is descriptive"
)
graphics::lines(
  evidence$overall$assumed_release_lag_days,
  evidence$overall$median_4w_mae_eur_per_kg,
  type = "b", pch = 17, col = "#A56639"
)
graphics::legend("topright", c("Last quote", "Four-week median"),
                 col = c("#276484", "#A56639"), pch = c(19, 17),
                 lty = 1, bty = "n")
invisible(grDevices::dev.off())

print(evidence$overall, row.names = FALSE)
print(evidence$matched, row.names = FALSE)
cat(sprintf("Wrote hypothetical SIMA release-lag evidence to %s\n", destination))
