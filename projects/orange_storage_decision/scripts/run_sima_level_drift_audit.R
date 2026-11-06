# Run from projects/orange_storage_decision. Base R only.
# Rscript scripts/run_sima_level_drift_audit.R [official_source.csv.gz] [new_dir]
# No sampling, fitted parameters, interpolation or overwrite.
source("R/sima_market_prices.R")
source("R/sima_rolling_backtest.R")
source("R/sima_composition_audit.R")
source("R/sima_seasonal_benchmark.R")
source("R/sima_level_drift_audit.R")

args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 2L) {
  stop("Usage: Rscript scripts/run_sima_level_drift_audit.R [source.csv.gz] [new_dir]",
       call. = FALSE)
}
input <- if (length(args) > 0L) args[[1L]] else
  "data/raw/sima_oranges_wholesale_2025-01-01_2026-10-31.csv.gz"
destination <- if (length(args) > 1L) args[[2L]] else
  "results/replay_sima_level_drift_audit"
assert_sima_string(input, "input")
assert_sima_string(destination, "destination")
if (!file.exists(input)) {
  stop("The dated official SIMA archive cannot be found", call. = FALSE)
}
if (file.exists(destination) || dir.exists(destination)) {
  stop("Refusing to overwrite an existing level-drift audit directory",
       call. = FALSE)
}

settings <- list(
  source = "GPP SIMA Mercados Abastecedores (wholesale)",
  input_file = basename(input),
  source_md5 = unname(tools::md5sum(input)),
  test_start = as.Date("2026-01-05"),
  seasonal_lag_in_weeks = 52L,
  min_history_weeks = 8L,
  lookback_weeks = 4L,
  min_recent_weeks = 2L,
  max_age_weeks = 3L,
  exact_identity = "market/region/full product descriptor/week_start",
  r_version = R.version.string
)

quotes <- read_sima_orange_prices(input, market_stage = "wholesale",
                                   exclude_provisional = TRUE)
forecasts <- backtest_sima_seasonal_52(
  sima_weekly_panels(quotes), settings$test_start,
  settings$min_history_weeks, settings$lookback_weeks,
  settings$min_recent_weeks, settings$max_age_weeks
)
evidence <- summarize_sima_level_drift(forecasts)
stopifnot(all(abs(evidence$rows$decomposition_residual_eur_per_kg[
  evidence$rows$three_way_common]) < 1e-10))

dir.create(destination, recursive = TRUE, showWarnings = FALSE)
files <- list(
  observed_rows = evidence$rows,
  overall = evidence$overall,
  coverage = evidence$coverage,
  by_quote_month = evidence$monthly,
  by_market = evidence$market,
  by_exact_panel = evidence$panel
)
for (name in names(files)) {
  utils::write.csv(files[[name]], file.path(destination, paste0(name, ".csv")),
                   row.names = FALSE, na = "")
}
writeLines(capture.output(dput(settings)), file.path(destination, "settings.R"),
           useBytes = TRUE)

cv <- evidence$coverage
ov <- evidence$overall
report <- c(
  "# Why last year's orange price misses the next week's price", "",
  "An **exact-week, exact-product and exact-wholesale-market** observational",
  "audit of nominal quotation movements; not an estimated seasonal effect.", "",
  sprintf("Source: official GPP SIMA archive `%s` (MD5 `%s`).",
          settings$input_file, settings$source_md5),
  sprintf("%d original quotations, not converted to hypothetical sale prices.",
          nrow(quotes)), "",
  "## What can be compared", "",
  sprintf("- **%d** observed 2026 target panel-weeks.",
          cv$observed_target_panel_weeks),
  sprintf("- **%d** have a quotation in the same panel exactly 364 days earlier.",
          cv$seasonal_52_available_panel_weeks),
  sprintf("- **%d** have both existing recent-price baselines.",
          cv$recent_baselines_available_panel_weeks),
  sprintf("- **%d** (%.1f%%) permit a strict three-way error decomposition,",
          cv$three_way_common_panel_weeks,
          100 * cv$three_way_fraction_of_observed),
  "  across the same observed panel-weeks for all reported methods.", "",
  "## An exact accounting identity, not a causal finding", "",
  "For each comparable target quotation:", "",
  "`actual_2026 − quoted_2025 = (latest_pre-origin_quote − quoted_2025) +",
  "(actual_2026 − latest_pre-origin_quote)`", "",
  "The first term is a **pre-origin price-level gap**. The second is the",
  "**subsequent price innovation** relative to the most recent observable",
  "quote, which may be up to three weeks old. The sum is an identity,",
  "but *absolute* errors do not add in general.", "",
  "## Comparable descriptive magnitudes", "",
  "| Measure (three-way common support only) | EUR/kg or fraction |",
  "| --- | ---: |",
  sprintf("| Average signed prior-year change | %.4f EUR/kg |",
          ov$mean_signed_yoy_change_eur_per_kg),
  sprintf("| Median nominal prior-year change | %.1f%% |",
          100 * ov$median_nominal_yoy_change_fraction),
  sprintf("| Share of higher 2026 comparable quotes | %.1f%% |",
          100 * ov$share_yoy_quotes_higher),
  sprintf("| Mean signed pre-origin gap | %.4f EUR/kg |",
          ov$mean_signed_pre_origin_gap_eur_per_kg),
  sprintf("| Mean absolute pre-origin gap | %.4f EUR/kg |",
          ov$mean_absolute_pre_origin_gap_eur_per_kg),
  sprintf("| Mean absolute recent-quote innovation | %.4f EUR/kg |",
          ov$mean_absolute_recent_quote_innovation_eur_per_kg),
  sprintf("| Mean absolute seasonal forecast error | %.4f EUR/kg |",
          ov$mean_absolute_seasonal_error_eur_per_kg),
  sprintf("| Mean absolute four-week median forecast error | %.4f EUR/kg |",
          ov$mean_absolute_four_week_median_error_eur_per_kg), "",
  "![Observed quotation-level gap and signed monthly contrasts](level_drift.png)",
  "", "## Where the comparison can mislead", "",
  "- Fewer than **two complete calendar years** of quotations are available.",
  "  This does not identify a repeating seasonal effect or a long-run trend.",
  "- Results are conditional on a quotation being recorded and on exact",
  "  matched product/market weeks. Unquoted weeks are not fabricated.",
  "- Ratios use **nominal wholesale EUR/kg quotes**, without inflation",
  "  adjustment, sales volumes, quality changes or retail transformations.",
  "  No pooled monthly price index or Portuguese average is inferred.",
  "- Week t−52 is **364 days** earlier, not necessarily the same calendar",
  "  date. Regional markets, grades, packaging and origins stay separate.",
  "- Quotation timestamps are not first-publication/revision timestamps:",
  "  source dates alone do not prove a production-time data feed.",
  "- Repeated quotations are dependent. No independent-row confidence",
  "  intervals, invented causal interpretations, or selected best model.",
  "- These are **not realised storage returns**, retail prices or producer",
  "  revenues; no physical storage lot, spoilage, transport or cost data.",
  "- The original synthetic orange-storage decision study is unchanged.", "",
  "## Reproduce", "",
  "From `projects/orange_storage_decision`:", "",
  "```sh", "Rscript tests/test_sima_level_drift_audit.R",
  "Rscript scripts/run_sima_level_drift_audit.R \\",
  sprintf("  %s \\", input),
  "  results/replay_sima_level_drift_new", "```", "",
  "CSV files: `observed_rows`, `overall`, `coverage`, `by_quote_month`,",
  "`by_market`, `by_exact_panel` (each `.csv`), plus `settings.R`,",
  "`RESULTS.md` and `level_drift.png`. The output directory must be new."
)
writeLines(report, file.path(destination, "RESULTS.md"), useBytes = TRUE)

grDevices::png(file.path(destination, "level_drift.png"),
                width = 1450, height = 610, res = 135)
graphics::par(mfrow = c(1L, 2L), mar = c(5, 5, 3, 1))
absolute_metrics <- c(
  ov$mean_absolute_pre_origin_gap_eur_per_kg,
  ov$mean_absolute_recent_quote_innovation_eur_per_kg,
  ov$mean_absolute_seasonal_error_eur_per_kg
)
graphics::barplot(
  absolute_metrics, names.arg = c("Level gap", "Recent change", "Year-old error"),
  col = c("#4A7796", "#77A8A0", "#CB9360"),
  ylim = c(0, max(absolute_metrics) * 1.2),
  ylab = "Mean absolute magnitude (EUR/kg)",
  main = "Components are not additive in absolute value"
)
month_change <- evidence$monthly$mean_signed_yoy_change_eur_per_kg
plot_limits <- range(c(0, month_change), na.rm = TRUE)
plot_limits <- if (diff(plot_limits) <= 0) c(-0.1, 0.1) else
  plot_limits * 1.2
graphics::barplot(
  month_change,
  names.arg = substr(evidence$monthly$quote_month, 6L, 7L),
  col = ifelse(month_change < 0, "#77A8A0", "#CB9360"),
  ylim = plot_limits, xlab = "2026 quotation month",
  ylab = "Mean signed year-over-year difference (EUR/kg)",
  main = "Exact matched quotes, not a national price index"
)
graphics::abline(h = 0, col = "#404040")
invisible(grDevices::dev.off())

print(evidence$overall, row.names = FALSE)
print(evidence$coverage, row.names = FALSE)
cat(sprintf("Wrote SIMA level-drift audit to %s\n", destination))
