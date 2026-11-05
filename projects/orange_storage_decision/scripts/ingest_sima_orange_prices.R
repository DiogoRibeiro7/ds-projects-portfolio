# Run from projects/orange_storage_decision. Base R only.
# Usage: Rscript scripts/ingest_sima_orange_prices.R INPUT.csv[.gz] OUTPUT.csv STAGE
# STAGE must be wholesale for the currently verified source; no overwriting.
source("R/sima_market_prices.R")

arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) != 3L) {
  stop(paste(
    "Usage: Rscript scripts/ingest_sima_orange_prices.R",
    "INPUT.csv[.gz] OUTPUT.csv wholesale"
  ), call. = FALSE)
}

source_path <- arguments[[1L]]
destination <- arguments[[2L]]
market_stage <- arguments[[3L]]
assert_sima_string(destination, "destination")
if (!grepl("\\.csv$", destination, ignore.case = TRUE)) {
  stop("Destination must be a .csv file", call. = FALSE)
}
if (!dir.exists(dirname(destination))) {
  stop("Output directory must already exist", call. = FALSE)
}
if (file.exists(destination) || dir.exists(destination)) {
  stop("Refusing to overwrite an existing output", call. = FALSE)
}
if (identical(normalizePath(source_path, mustWork = FALSE),
              normalizePath(destination, mustWork = FALSE))) {
  stop("Input and output paths must differ", call. = FALSE)
}

quotes <- read_sima_orange_prices(source_path, market_stage = market_stage)
utils::write.csv(quotes, file = destination, row.names = FALSE, na = "")
cat(sprintf("Wrote %d validated %s orange quotations to %s\n",
            nrow(quotes), market_stage, destination))
cat(sprintf("Observation dates: %s to %s; markets: %s\n",
            min(quotes$quote_date), max(quotes$quote_date),
            paste(sort(unique(quotes$market)), collapse = ", ")))
if (anyNA(quotes$is_provisional)) {
  cat("Provisional status was not supplied for some/all rows; unknown is retained.\n")
}
