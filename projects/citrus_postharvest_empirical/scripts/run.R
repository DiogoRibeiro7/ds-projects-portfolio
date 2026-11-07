#!/usr/bin/env Rscript
# Reproduce descriptive orange outcomes and three publication-ready PNG figures.
# Usage: Rscript scripts/run.R [raw_dir] [output_dir]

args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 2L) {
  stop("Usage: Rscript scripts/run.R [raw_dir] [output_dir]")
}
raw_dir <- if (length(args) >= 1L) args[1L] else "data/raw"
output_dir <- if (length(args) >= 2L) args[2L] else "results/generated"

source("R/source.R", local = TRUE)
source("R/analysis.R", local = TRUE)
verify_upct_sources(raw_dir)
quality <- read_upct_orange_quality(file.path(raw_dir, "Quality Data.csv"))
in_vivo <- read_upct_orange_in_vivo(file.path(raw_dir, "in vivo data.csv"))

firmness_summary <- summarise_upct_quality(quality$firmness, "firmness")
weight_summary <- summarise_upct_quality(quality$weight_loss, "weight_loss")
quality_summary <- rbind(firmness_summary, weight_summary)
in_vivo_summary <- summarise_upct_in_vivo(in_vivo)
contrasts <- summarise_upct_contrasts(quality_summary, in_vivo_summary)

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(quality_summary, file.path(output_dir, "orange_quality_summary.csv"), row.names = FALSE)
write.csv(in_vivo_summary, file.path(output_dir, "orange_in_vivo_summary.csv"), row.names = FALSE)
write.csv(contrasts, file.path(output_dir, "endpoint_contrasts.csv"), row.names = FALSE)

plot_upct_series(
  firmness_summary, "mean", "Orange firmness by coating", "Mean firmness (N)",
  file.path(output_dir, "orange_firmness.png")
)
plot_upct_series(
  weight_summary, "mean", "Orange weight loss by coating", "Mean weight loss (%)",
  file.path(output_dir, "orange_weight_loss.png"), y_min = 0
)
plot_upct_series(
  in_vivo_summary[in_vivo_summary$outcome == "decay", ], "percent",
  "Observed decay in inoculated oranges", "Decayed oranges (%)",
  file.path(output_dir, "orange_decay.png"), y_min = 0, y_max = 100
)

cat(sprintf("Verified three upstream files; wrote three CSV reports and three figures to %s\n", output_dir))
