# Base R empirical composition/coverage audit. Run from project root.
# Rscript scripts/run_sima_composition_audit.R [source.csv.gz] [new_output_dir]
source("R/sima_market_prices.R")
source("R/sima_rolling_backtest.R")
source("R/sima_composition_audit.R")

args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 2L) {
  stop(paste("Usage: Rscript scripts/run_sima_composition_audit.R",
             "[source.csv.gz] [new_output_dir]"), call. = FALSE)
}
snapshot <- "data/raw/sima_oranges_wholesale_2025-01-01_2026-10-31.csv.gz"
input <- if (length(args) >= 1L) args[[1L]] else snapshot
destination <- if (length(args) >= 2L) args[[2L]] else
  "results/replay_sima_composition_audit"
assert_sima_string(destination, "destination")
if (file.exists(destination) || dir.exists(destination)) {
  stop("Refusing to overwrite an existing SIMA composition-audit directory",
       call. = FALSE)
}

settings <- list(
  source = "GPP SIMA Mercados Abastecedores (wholesale)",
  source_file = basename(input),
  source_md5 = unname(tools::md5sum(input)),
  base_year = 2025L, comparison_year = 2026L, months = 1:10,
  quote_unit = "EUR/kg",
  week_rule = "latest dated original quote per exact product/market/week",
  month_rule = "calendar month of the final dated quote, not Monday start",
  matched_index_rule = "equal-panel geometric ratio of monthly weekly medians",
  pooled_rule = "weekly-quotation-weighted arithmetic mean across observed panels",
  r_version = R.version.string
)
quotes <- read_sima_orange_prices(input, market_stage = "wholesale",
                                  exclude_provisional = TRUE)
weekly <- sima_weekly_panels(quotes)
monthly <- sima_panel_monthly_prices(weekly)
spans <- sima_panel_quotation_spans(weekly)
comparison <- sima_compare_monthly_panel_mix(
  monthly, settings$base_year, settings$comparison_year, settings$months
)

dir.create(destination, recursive = TRUE, showWarnings = FALSE)
utils::write.csv(monthly, file.path(destination, "panel_monthly_quotes.csv"),
                 row.names = FALSE, na = "")
utils::write.csv(spans, file.path(destination, "panel_quotation_spans.csv"),
                 row.names = FALSE, na = "")
utils::write.csv(comparison$overview,
                 file.path(destination, "monthly_comparison.csv"),
                 row.names = FALSE, na = "")
utils::write.csv(comparison$matched_pairs,
                 file.path(destination, "matched_panel_months.csv"),
                 row.names = FALSE, na = "")
writeLines(capture.output(dput(settings)), file.path(destination, "settings.R"),
           useBytes = TRUE)

percent <- function(x) {
  if (is.na(x)) "unavailable" else sprintf("%+.1f%%", 100 * x)
}
share <- function(x) {
  if (is.na(x)) "unavailable" else sprintf("%.1f%%", 100 * x)
}
report <- c(
  "# GPP SIMA oranges: product-mix and quotation-coverage audit", "",
  "**Official dated wholesale quotations, not verified sales transactions.**", "",
  sprintf("Frozen snapshot: `%s`; MD5: `%s`.",
          settings$source_file, settings$source_md5),
  sprintf("%d archived quotations; %d exact observed panel/weeks; %d panels;",
          nrow(quotes), nrow(weekly), nrow(spans)),
  sprintf("%d observed panel/months.", nrow(monthly)),
  "Product descriptor (including cultivar, grade, calibre, packaging and",
  "origin), region and wholesale market remain separate.", "",
  "## What changed between comparable months?", "",
  "The *pooled mean* weights each observed panel-week and may move when the",
  "product/market mix or quotation frequency changes. The *matched weekly",
  "mean* uses only panels present in both years for that month but still",
  "weights by their observed weeks. The *matched equal-panel* comparison",
  "is the geometric mean of each shared panel's median-price ratio:",
  "no market volume weights or unobserved prices are invented.", "",
  "| Month | Panels 2025 / 2026 | Matched | Matched-week shares 2025 / 2026 | Pooled weekly mean | Matched weekly mean | Equal-panel median ratio |",
  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"
)
month_names <- month.abb[settings$months]
for (i in seq_len(nrow(comparison$overview))) {
  x <- comparison$overview[i, , drop = FALSE]
  report <- c(report, sprintf(
    "| %s | %d / %d | %d | %s / %s | %s | %s | %s |",
    month_names[i], x$base_panels, x$comparison_panels,
    x$matched_panels, share(x$base_matched_week_share),
    share(x$comparison_matched_week_share),
    percent(x$pooled_week_mean_change_fraction),
    percent(x$matched_week_mean_change_fraction),
    percent(x$matched_panel_geometric_change_fraction)
  ))
}
report <- c(
  report, "", "![Pooled versus matched monthly quotations](composition_audit.png)", "",
  "## Essential limitations", "",
  "- These are **nominal quoted EUR/kg prices**. The source is neither a",
  "  population-weighted Portuguese market price index nor a record of",
  "  executed transactions. There are no sales or shipment quantities.",
  "- Changing the product, market or origin mix can change a pooled mean",
  "  even if prices within shared panels move differently. The three columns",
  "  are deliberately different estimands; their gap is **not** an identified",
  "  causal composition effect or a formal decomposition.",
  "- A month with only a few shared panels is a weak basis for generalisation.",
  "  Matched counts and both years' observed-week shares are disclosed.",
  "- A panel's unquoted weeks between its first and last observation may",
  "  reflect seasonality, trading patterns or reporting practices. They",
  "  are **not** automatically classified as data-quality failures.",
  "- Each observed Monday–Sunday week contributes only its last dated quote.",
  "  Its calendar month is assigned by that **quote date**, not by the",
  "  preceding Monday (which can fall in another month or year).",
  "- This is an ex-post descriptive audit of two archived calendar years,",
  "  not a point-in-time forecast, causal estimate, inflation adjustment",
  "  or evidence that storing a harvest was profitable.",
  "- The original completed synthetic storage-policy study remains",
  "  separate and unchanged. Lot-level sales, deterioration, inspections",
  "  and costs are still necessary for any operational recommendation.", "",
  "## Reproduction", "",
  "From `projects/orange_storage_decision`:", "",
  "```sh", "Rscript tests/test_sima_composition_audit.R",
  "Rscript scripts/run_sima_composition_audit.R \\",
  sprintf("  %s \\", input),
  "  results/replay_sima_composition_audit_new",
  "```", "",
  "The destination must not exist. All monthly/panel tables, the two-panel",
  "figure, the source checksum and frozen R settings are reproduced offline."
)
writeLines(report, file.path(destination, "RESULTS.md"), useBytes = TRUE)

overview <- comparison$overview
growth_values <- 100 * c(overview$pooled_week_mean_change_fraction,
                         overview$matched_week_mean_change_fraction,
                         overview$matched_panel_geometric_change_fraction)
extents <- range(c(0, growth_values[is.finite(growth_values)]))
padding <- max(3, diff(extents) * 0.12)
grDevices::png(file.path(destination, "composition_audit.png"),
                width = 1560, height = 680, res = 140)
graphics::par(mfrow = c(1L, 2L), mar = c(6, 5, 3, 1))
graphics::plot(
  overview$month, 100 * overview$pooled_week_mean_change_fraction,
  type = "b", pch = 19, col = "#8B4E40",
  xlim = c(1, 10), ylim = extents + c(-padding, padding),
  xaxt = "n", xlab = "Calendar month",
  ylab = "Nominal 2026 vs 2025 price change (%)",
  main = "Pooled and matched price movements"
)
graphics::axis(1, at = overview$month, labels = month.abb[overview$month])
graphics::abline(h = 0, col = "grey70", lty = 2)
graphics::lines(
  overview$month, 100 * overview$matched_week_mean_change_fraction,
  type = "b", pch = 17, col = "#39718B"
)
graphics::lines(
  overview$month, 100 * overview$matched_panel_geometric_change_fraction,
  type = "b", pch = 15, col = "#4E7A54"
)
graphics::legend(
  "bottomleft",
  c("All observed weeks", "Matched weeks", "Equal-panel medians"),
  col = c("#8B4E40", "#39718B", "#4E7A54"),
  pch = c(19, 17, 15), lty = 1, bty = "n", cex = 0.85
)
counts <- rbind(
  "2025 panels" = overview$base_panels,
  "2026 panels" = overview$comparison_panels,
  "Shared panels" = overview$matched_panels
)
graphics::barplot(
  counts, beside = TRUE, col = c("#8B4E40", "#39718B", "#4E7A54"),
  names.arg = month.abb[overview$month], las = 2,
  ylim = c(0, max(counts) * 1.15), ylab = "Distinct quoted product-market panels",
  main = "Common support by calendar month"
)
graphics::legend("topleft", legend = rownames(counts),
                 fill = c("#8B4E40", "#39718B", "#4E7A54"),
                 bty = "n", cex = 0.9)
invisible(grDevices::dev.off())

print(overview, row.names = FALSE)
cat(sprintf("Wrote GPP SIMA composition evidence to %s\n", destination))
