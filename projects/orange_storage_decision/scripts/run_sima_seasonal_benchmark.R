# Run from projects/orange_storage_decision, using only base R.
# Rscript scripts/run_sima_seasonal_benchmark.R [source.csv.gz] [new_output_dir]
# Exact 52-week comparison is fixed before inspecting 2026 target errors.
source("R/sima_market_prices.R")
source("R/sima_rolling_backtest.R")
source("R/sima_composition_audit.R")
source("R/sima_seasonal_benchmark.R")

args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 2L) {
  stop(paste("Usage: Rscript scripts/run_sima_seasonal_benchmark.R",
             "[source.csv.gz] [new_output_dir]"), call. = FALSE)
}
snapshot <- "data/raw/sima_oranges_wholesale_2025-01-01_2026-10-31.csv.gz"
input <- if (length(args) >= 1L) args[[1L]] else snapshot
destination <- if (length(args) >= 2L) args[[2L]] else
  "results/replay_sima_seasonal_benchmark"
assert_sima_string(destination, "destination")
if (file.exists(destination) || dir.exists(destination)) {
  stop("Refusing to overwrite an existing seasonal benchmark directory",
       call. = FALSE)
}

settings <- list(
  source = "GPP SIMA Mercados Abastecedores (wholesale)",
  source_file = basename(input),
  source_md5 = unname(tools::md5sum(input)),
  test_start = as.Date("2026-01-05"),
  seasonal_reference_weeks = 52L,
  exact_previous_week_only = TRUE,
  min_history_weeks = 8L,
  lookback_weeks = 4L,
  min_recent_weeks = 2L,
  max_age_weeks = 3L,
  panel_identity = "full product descriptor / region / wholesale market",
  r_version = R.version.string
)
prices <- read_sima_orange_prices(input, market_stage = "wholesale",
                                  exclude_provisional = TRUE)
weekly <- sima_weekly_panels(prices)
forecasts <- backtest_sima_seasonal_52(
  weekly, settings$test_start, settings$min_history_weeks,
  settings$lookback_weeks, settings$min_recent_weeks,
  settings$max_age_weeks
)
evidence <- summarize_sima_seasonal_52(forecasts)

dir.create(destination, recursive = TRUE, showWarnings = FALSE)
utils::write.csv(forecasts, file.path(destination, "seasonal_forecasts.csv"),
                 row.names = FALSE, na = "")
utils::write.csv(evidence$overall, file.path(destination, "summary.csv"),
                 row.names = FALSE, na = "")
utils::write.csv(evidence$monthly, file.path(destination, "monthly.csv"),
                 row.names = FALSE, na = "")
utils::write.csv(evidence$panel, file.path(destination, "panel.csv"),
                 row.names = FALSE, na = "")
utils::write.csv(evidence$coverage, file.path(destination, "coverage.csv"),
                 row.names = FALSE, na = "")
writeLines(capture.output(dput(settings)), file.path(destination, "settings.R"),
           useBytes = TRUE)

cov <- evidence$coverage
report <- c(
  "# A 52-week seasonal naive price forecast for Portuguese oranges", "",
  "**Exact previous-year price, not a fitted model or realised sale return.**", "",
  sprintf("Official source: `%s` (MD5 `%s`).",
          settings$source_file, settings$source_md5),
  sprintf("%d original wholesale quotation rows; %d observed panel/weeks.",
          nrow(prices), nrow(weekly)),
  "The product descriptor, region and wholesale market are never pooled.", "",
  "## Fixed 2026 out-of-time evaluation", "",
  sprintf("Observed 2026 target panel/weeks: **%d**.",
          cov$observed_target_panel_weeks),
  sprintf("Exact same-panel 52-week reference available: **%d** (%.1f%%).",
          cov$seasonal_52_available_panel_weeks,
          100 * cov$seasonal_52_coverage_fraction),
  sprintf("Existing recent baselines both available: **%d**.",
          cov$recent_baseline_paired_panel_weeks),
  sprintf("All three methods available on the **same** target rows: **%d** (%.1f%%).",
          cov$three_way_scored_panel_weeks,
          100 * cov$three_way_coverage_fraction),
  "The denominator is observed target quotations, not all calendar weeks.",
  "An absent exact prior-year week remains NA and is never imputed.", "",
  "## Three-way like-for-like errors", "",
  "The only scored comparison uses identical product/market/target weeks",
  "for every method. MAE is in nominal EUR/kg. Equal-panel MAE gives",
  "each scored product/region/market panel equal weight; WAPE is total",
  "absolute error divided by the sum of observed target prices.", "",
  "| Historical forecast rule | Common target rows | Panels | Observed-week MAE | Equal-panel MAE | WAPE |",
  "| --- | ---: | ---: | ---: | ---: | ---: |"
)
for (i in seq_len(nrow(evidence$overall))) {
  x <- evidence$overall[i, , drop = FALSE]
  report <- c(report, sprintf("| %s | %d | %d | %.4f | %.4f | %.2f%% |",
                              x$method, x$common_scored_weeks, x$scored_panels,
                              x$observation_weighted_mae_eur_per_kg,
                              x$equal_panel_mae_eur_per_kg,
                              100 * x$weighted_absolute_percentage_error))
}
report <- c(
  report, "", "![Exact 52-week coverage and three-way errors](seasonal_mae.png)", "",
  "## Why this is a negative control, not a storage policy", "",
  "- The seasonal rule predicts week *t* using the quotation observed in",
  "  exactly the same panel in week *t−52* (364 calendar days earlier).",
  "  It does not find a nearest available neighboring week or borrow prices",
  "  from a different cultivar, grade, origin or wholesale market.",
  "- All observed reference quotation dates predate the target forecast",
  "  origin. As elsewhere, missing GPP publication/revision timestamps mean",
  "  this is **observation-date-limited**; not certified real-time availability.",
  "- The seasonal method's coverage is a property of this short archived",
  "  source, not evidence that all Portuguese orange markets have",
  "  the same year-to-year support.",
  "- Two calendar years cannot identify a stable seasonal effect or",
  "  establish statistical superiority. Price levels can move between years.",
  "- Errors have temporal and cross-product dependence. No unjustified",
  "  independent-row significance tests or causal interpretations are used.",
  "- None of these wholesale quoted prices are realised producer revenues;",
  "  no costs, shipping, quality loss or actual stored lots are in the data.",
  "  The complete synthetic orange-storage study remains unchanged.", "",
  "## Reproduce and inspect", "",
  "From `projects/orange_storage_decision`:", "",
  "```sh", "Rscript tests/test_sima_seasonal_benchmark.R",
  "Rscript scripts/run_sima_seasonal_benchmark.R \\",
  sprintf("  %s \\", input),
  "  results/replay_sima_seasonal_benchmark_new", "```", "",
  "Outputs: `seasonal_forecasts.csv` (including uncovered targets),",
  "`summary.csv`, `monthly.csv`, `panel.csv`, `coverage.csv`,",
  "`seasonal_mae.png` and `settings.R`. The destination must be new."
)
writeLines(report, file.path(destination, "RESULTS.md"), useBytes = TRUE)

grDevices::png(file.path(destination, "seasonal_mae.png"),
                width = 1490, height = 630, res = 140)
graphics::par(mfrow = c(1L, 2L), mar = c(5, 5, 3, 1))
maes <- evidence$overall$observation_weighted_mae_eur_per_kg
graphics::barplot(
  maes, names.arg = c("Last quote", "4-week median", "52-week seasonal"),
  col = c("#3C7697", "#74A8A2", "#C68D59"),
  ylim = c(0, max(maes) * 1.15),
  ylab = "MAE on identical target rows (EUR/kg)",
  main = "Three-way forecast benchmark"
)
monthly <- evidence$monthly
graphics::barplot(
  100 * monthly$three_way_coverage_fraction,
  names.arg = substr(monthly$quote_month, 6L, 7L),
  col = "#7AA5C8", ylim = c(0, 100),
  xlab = "2026 quotation month",
  ylab = "Observed target coverage (%)",
  main = "Exact prior-year and recent forecasts"
)
invisible(grDevices::dev.off())

print(evidence$overall, row.names = FALSE)
print(evidence$coverage, row.names = FALSE)
cat(sprintf("Wrote exact 52-week SIMA evidence to %s\n", destination))
