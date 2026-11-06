# Run from projects/orange_storage_decision with base R only.
# Rscript scripts/run_sima_forward_quote_windows.R [official.csv.gz] [new_dir]
# All results are descriptive ex-post wholesale quotations, not profits.
source("R/sima_market_prices.R")
source("R/sima_rolling_backtest.R")
source("R/sima_composition_audit.R")
source("R/sima_forward_quote_windows.R")

args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 2L) {
  stop("Usage: Rscript scripts/run_sima_forward_quote_windows.R [source.csv.gz] [new_dir]",
       call. = FALSE)
}
input <- if (length(args) >= 1L) args[[1L]] else
  "data/raw/sima_oranges_wholesale_2025-01-01_2026-10-31.csv.gz"
destination <- if (length(args) == 2L) args[[2L]] else
  "results/replay_sima_forward_quote_windows"
assert_sima_string(input, "input")
assert_sima_string(destination, "destination")
if (!file.exists(input)) {
  stop("The dated official GPP SIMA CSV is missing", call. = FALSE)
}
if (file.exists(destination) || dir.exists(destination)) {
  stop("Refusing to overwrite an existing forward-price report directory",
       call. = FALSE)
}

settings <- list(
  source = "GPP SIMA Mercados Abastecedores (wholesale)",
  input_file = basename(input), source_md5 = unname(tools::md5sum(input)),
  anchor_start = as.Date("2026-01-05"),
  horizons_weeks = 1:4,
  last_observed_calendar_week_excluded = TRUE,
  exact_panel_dimensions = "market/region/full product descriptor",
  no_missing_week_imputation = TRUE,
  same_anchors_for_balanced_comparison = TRUE,
  r_version = R.version.string
)
quotes <- read_sima_orange_prices(input, market_stage = "wholesale",
                                   exclude_provisional = TRUE)
weekly <- sima_weekly_panels(quotes)
windows <- pair_sima_forward_quotes(
  weekly, anchor_start = settings$anchor_start,
  horizons = settings$horizons_weeks
)
evidence <- summarize_sima_forward_windows(windows)

dir.create(destination, recursive = TRUE, showWarnings = FALSE)
datasets <- list(
  observed_windows = evidence$windows,
  coverage = evidence$coverage,
  all_available = evidence$all_available,
  balanced = evidence$balanced,
  monthly_balanced = evidence$monthly_balanced,
  market_balanced = evidence$market_balanced,
  panel_balanced = evidence$panel_balanced
)
for (name in names(datasets)) {
  utils::write.csv(datasets[[name]],
                   file.path(destination, paste0(name, ".csv")),
                   row.names = FALSE, na = "")
}
writeLines(capture.output(dput(settings)), file.path(destination, "settings.R"),
           useBytes = TRUE)

coverage <- evidence$coverage
balanced <- evidence$balanced
report <- c(
  "# What happens to wholesale orange quotations after 1–4 weeks?", "",
  "**Observed future quote differences, not a realised storage-policy trial.**",
  "All original product/region/market identities are retained.", "",
  sprintf("Frozen official source: `%s` (MD5 `%s`).",
          settings$input_file, settings$source_md5),
  sprintf("%d original GPP SIMA quotations; %d observed 2026 anchor panel-weeks.",
          nrow(quotes), coverage$observed_anchor_quotes[1L]),
  sprintf("Last source quotation: %s; conservatively last complete week: %s.",
          max(weekly$quote_date),
          unique(evidence$windows$final_complete_source_week)), "",
  "## Horizon-eligible and matched endpoints", "",
  "Each observed 2026 anchor is retained for every horizon. Eligibility",
  "requires the target week to finish before the final quoted source week;",
  "the target must also have an **observed exact-panel quotation**.",
  "Unavailable endpoints stay NA, never a zero or carried-forward value.", "",
  "| Future calendar weeks | Observed anchors | Eligible | Exact matched endpoints | Balanced cohort |",
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
report <- c(report, "", "## Changes on the same starting observations", "",
            "The **balanced cohort** requires all four future endpoints.",
            "It is an ex-post completeness sample, not an actionable",
            "holding or selection policy. Every horizon below uses the",
            "same starting product/market/week observations.", "",
            "| Horizon (weeks) | Mean Δ (EUR/kg) | Median Δ | 10th percentile | 90th percentile | Higher | Lower |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
for (i in seq_len(nrow(balanced))) {
  x <- balanced[i, , drop = FALSE]
  report <- c(report, sprintf("| %d | %+.4f | %+.4f | %+.4f | %+.4f | %.1f%% | %.1f%% |",
                              x$horizon_weeks,
                              x$mean_signed_change_eur_per_kg,
                              x$median_signed_change_eur_per_kg,
                              x$tenth_percentile_change_eur_per_kg,
                              x$ninetieth_percentile_change_eur_per_kg,
                              100 * x$share_price_higher,
                              100 * x$share_price_lower))
}
report <- c(
  report, "", "![Observed same-panel future quotation distributions](forward_windows.png)",
  "", "## Interpretation limits", "",
  "- The horizon is an integer number of **calendar weeks** between each",
  "  weekly Monday; actual original quote dates can differ by other day counts.",
  "- Changes are in nominal EUR/kg *frequent wholesale quotations* for the",
  "  same descriptor, region and market. Nothing is volume-weighted or",
  "  a statistically representative national orange price index.",
  "- The dataset cannot tell whether a specific orange lot could still be",
  "  sold after waiting. Spoilage, maturity, transport, storage costs,",
  "  commissions, delivered quality and realised prices are unobserved.",
  "- These are realised **ex-post** quote differences. Future quotes are",
  "  not known to the grower on the anchor date; this is not forecasting.",
  "- No intervention randomly assigned a lot to wait; neither positive",
  "  movements nor the differences between horizons estimate causal gains.",
  "- Source publication/revision timestamps are not available. Matching",
  "  is by recorded quotation date, not certified live publication time.",
  "- Missing future quotes may reflect seasonal market absence rather",
  "  than source failure; observed-endpoint conditioning may select cases.",
  "- Product/market windows overlap across weeks. Tail percentiles and",
  "  means are descriptive, with no IID uncertainty or p-value claims.",
  "- No part of the original synthetic orange-storage model is modified.", "",
  "## Reproduce in base R", "",
  "From `projects/orange_storage_decision`:", "",
  "```sh", "Rscript tests/test_sima_forward_quote_windows.R",
  "Rscript scripts/run_sima_forward_quote_windows.R \\",
  sprintf("  %s \\", input),
  "  results/replay_sima_forward_quote_windows_new", "```", "",
  "Evidence: `observed_windows.csv`, `coverage.csv`, `all_available.csv`,",
  "`balanced.csv`, `monthly_balanced.csv`, `market_balanced.csv`,",
  "`panel_balanced.csv`, `settings.R`, `RESULTS.md`, `forward_windows.png`.",
  "An existing output directory is never overwritten."
)
writeLines(report, file.path(destination, "RESULTS.md"), useBytes = TRUE)

grDevices::png(file.path(destination, "forward_windows.png"),
                width = 1450, height = 620, res = 135)
graphics::par(mfrow = c(1L, 2L), mar = c(5, 5, 3, 1))
proportions <- rbind(
  100 * balanced$share_price_higher,
  100 * balanced$share_price_lower,
  100 * balanced$share_price_unchanged
)
graphics::barplot(
  proportions, names.arg = balanced$horizon_weeks,
  col = c("#4D9478", "#C37968", "#A1B2BF"),
  ylim = c(0, 100), xlab = "Future calendar weeks",
  ylab = "Balanced observed anchors (%)",
  main = "Quote higher, lower or unchanged"
)
graphics::legend("topright", legend = c("Higher", "Lower", "Unchanged"),
                 fill = c("#4D9478", "#C37968", "#A1B2BF"), bty = "n")
percentiles <- as.matrix(balanced[, c(
  "tenth_percentile_change_eur_per_kg",
  "median_signed_change_eur_per_kg",
  "ninetieth_percentile_change_eur_per_kg"
)])
graphics::matplot(
  balanced$horizon_weeks, percentiles, type = "b", pch = c(16, 17, 15),
  lty = 1, col = c("#C37968", "#4D7193", "#4D9478"),
  xlab = "Future calendar weeks", ylab = "Signed quote difference (EUR/kg)",
  ylim = range(c(0, percentiles)),
  main = "Observed 10th, median and 90th percentiles"
)
graphics::abline(h = 0, col = "#4B4B4B", lty = 2)
graphics::legend("topleft", legend = c("10th", "Median", "90th"),
                 col = c("#C37968", "#4D7193", "#4D9478"),
                 pch = c(16, 17, 15), lty = 1, bty = "n")
invisible(grDevices::dev.off())

print(evidence$coverage, row.names = FALSE)
print(evidence$balanced, row.names = FALSE)
cat(sprintf("Wrote observed forward quotation windows to %s\n", destination))
