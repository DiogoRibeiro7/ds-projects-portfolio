#!/usr/bin/env Rscript
# Dependency-free regression tests against the exact published source bytes.

source("R/source.R", local = TRUE)
source("R/analysis.R", local = TRUE)

verify_upct_sources("data/raw")
quality <- read_upct_orange_quality("data/raw/Quality Data.csv")
in_vivo <- read_upct_orange_in_vivo("data/raw/in vivo data.csv")

stopifnot(nrow(quality$firmness) == 200L)
stopifnot(nrow(quality$weight_loss) == 60L)
stopifnot(nrow(in_vivo) == 90L)
stopifnot(all(quality$firmness$table_id == "orange_firmness"))
stopifnot(all(quality$weight_loss$table_id == "orange_weight_loss"))
stopifnot(!any(quality$firmness$treatment == "LEMON"))
stopifnot(!any(quality$weight_loss$treatment == "LEMON"))
stopifnot(all(quality$firmness$source_line >= 3L))
stopifnot(all(quality$weight_loss$source_line >= 3L))
stopifnot(all(in_vivo$total == 20L))

firmness <- summarise_upct_quality(quality$firmness, "firmness")
weight <- summarise_upct_quality(quality$weight_loss, "weight_loss")
decay <- summarise_upct_in_vivo(in_vivo)
contrasts <- summarise_upct_contrasts(rbind(firmness, weight), decay)

stopifnot(nrow(firmness) == 20L)
stopifnot(nrow(weight) == 20L)
stopifnot(nrow(decay) == 30L)
stopifnot(all(firmness$n_observations == 10L))
stopifnot(all(weight$n_observations == 3L))
stopifnot(all(decay$n_boxes == 3L))
stopifnot(all(decay$n_exposed == 60L))
stopifnot(nrow(contrasts) == 28L)

group_value <- function(data, treatment, day, value_column, outcome = NULL) {
  subset <- data[data$treatment == treatment & data$day == day, ]
  if (!is.null(outcome)) {
    subset <- subset[subset$outcome == outcome, ]
  }
  stopifnot(nrow(subset) == 1L)
  as.numeric(subset[[value_column]])
}

# Independent hand calculations from source orange rows, not published model fits.
stopifnot(abs(group_value(firmness, "CTRL", 15L, "mean") - 13.7340) < 0.0001)
stopifnot(abs(group_value(firmness, "CTRL-W", 15L, "mean") - 16.0200) < 0.0001)
stopifnot(abs(group_value(weight, "CTRL", 15L, "mean") - 6.9859) < 0.0002)
stopifnot(abs(group_value(weight, "GRAPEFRUIT", 15L, "mean") - 4.2489) < 0.0002)
stopifnot(group_value(decay, "CTRL", 10L, "percent", "decay") == 55)
stopifnot(group_value(decay, "CTRL-W", 10L, "percent", "decay") == 30)
stopifnot(group_value(decay, "AVOCADO", 10L, "percent", "decay") == 25)
stopifnot(group_value(decay, "GRAPEFRUIT", 10L, "percent", "decay") == 50)

contrast_row <- contrasts[
  contrasts$endpoint == "decay_day10" &
    contrasts$treatment == "AVOCADO" & contrasts$comparator == "CTRL-W", ]
stopifnot(nrow(contrast_row) == 1L)
stopifnot(contrast_row$difference == -5)

# A missing or replaced source is a hard failure, never a silent fallback.
missing_source_error <- tryCatch(
  { verify_upct_sources(file.path(tempdir(), "missing-upct-data")); FALSE },
  error = function(e) TRUE
)
stopifnot(isTRUE(missing_source_error))

cat("PASS: source checksums, orange-only partitions, sample sizes and observed contrasts\n")
