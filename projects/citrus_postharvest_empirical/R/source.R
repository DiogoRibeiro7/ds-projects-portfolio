# Read and validate the published UPCT orange postharvest dataset.
#
# All parsing uses base R. Orange firmness and weight loss occupy independent
# side-by-side tables in the upstream CSV; their row positions are NOT fruit IDs.

upct_treatments <- c("CTRL", "CTRL-W", "AVOCADO", "ORANGE", "GRAPEFRUIT")

upct_source_manifest <- function() {
  data.frame(
    filename = c("Quality Data.csv", "in vivo data.csv", "README.txt"),
    md5 = c(
      "70edda22637ac7c9eddc3afe83b5ac52",
      "38d991b12f99f0cbac2a01ba07b3d16b",
      "c1068954440a7fffb1d858690bcb0194"
    ),
    stringsAsFactors = FALSE
  )
}

verify_upct_sources <- function(raw_dir) {
  if (!is.character(raw_dir) || length(raw_dir) != 1L || is.na(raw_dir)) {
    stop("raw_dir must be one non-missing character path")
  }
  manifest <- upct_source_manifest()
  paths <- file.path(raw_dir, manifest$filename)
  if (!all(file.exists(paths))) {
    stop(sprintf("Missing upstream file(s): %s", paste(manifest$filename[!file.exists(paths)], collapse = ", ")))
  }
  actual <- unname(tools::md5sum(paths))
  if (anyNA(actual) || !identical(tolower(actual), manifest$md5)) {
    stop("UPCT source checksum mismatch; do not analyse modified or partial files")
  }
  invisible(manifest)
}

assert_group_design <- function(data, days, expected_per_group, table_name) {
  if (!all(data$treatment %in% upct_treatments) || anyNA(data$treatment)) {
    stop(sprintf("Unexpected treatment in %s", table_name))
  }
  if (!identical(sort(unique(data$day)), as.integer(days))) {
    stop(sprintf("Unexpected collection days in %s", table_name))
  }
  counts <- table(
    factor(data$treatment, levels = upct_treatments),
    factor(data$day, levels = days)
  )
  if (!all(counts == expected_per_group)) {
    stop(sprintf("Unexpected per-treatment/day sample sizes in %s", table_name))
  }
  invisible(TRUE)
}

read_upct_orange_quality <- function(path) {
  if (!is.character(path) || length(path) != 1L || !file.exists(path)) {
    stop("Quality CSV path must refer to an existing file")
  }
  # Skip the species and duplicated field-header rows. Columns 8--13 are
  # LEMON measurements; they are deliberately never used in this study.
  raw <- utils::read.csv2(
    file = path,
    header = FALSE,
    skip = 2L,
    col.names = sprintf("V%d", seq_len(13L)),
    na.strings = "",
    fileEncoding = "UTF-8-BOM",
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  if (nrow(raw) != 200L || ncol(raw) != 13L || any(!is.na(raw$V7))) {
    stop("Unexpected source layout for the parallel quality tables")
  }

  firmness <- data.frame(
    table_id = "orange_firmness",
    source_line = seq_len(nrow(raw)) + 2L,
    treatment = as.character(raw$V1),
    day = as.integer(raw$V2),
    value = as.numeric(raw$V3),
    unit = "N",
    stringsAsFactors = FALSE
  )
  keep_weight <- which(!is.na(raw$V4) & nzchar(as.character(raw$V4)))
  weight_loss <- data.frame(
    table_id = "orange_weight_loss",
    source_line = keep_weight + 2L,
    treatment = as.character(raw$V4[keep_weight]),
    day = as.integer(raw$V5[keep_weight]),
    value = as.numeric(raw$V6[keep_weight]),
    unit = "percent",
    stringsAsFactors = FALSE
  )

  if (anyNA(firmness) || anyNA(weight_loss)) {
    stop("Missing or non-numeric orange quality observations")
  }
  if (any(firmness$value <= 0) || any(weight_loss$value < 0 | weight_loss$value > 100)) {
    stop("Orange quality observations outside physical bounds")
  }
  assert_group_design(firmness, c(0L, 5L, 10L, 15L), 10L, "firmness")
  assert_group_design(weight_loss, c(5L, 8L, 10L, 15L), 3L, "weight loss")

  # There is no defensible fruit-level link between these distinct tables.
  list(firmness = firmness, weight_loss = weight_loss)
}

read_upct_orange_in_vivo <- function(path) {
  if (!is.character(path) || length(path) != 1L || !file.exists(path)) {
    stop("In-vivo CSV path must refer to an existing file")
  }
  lines <- readLines(path, encoding = "UTF-8", warn = FALSE)
  orange_start <- which(startsWith(lines, "ORANGES-FBPs;"))
  if (length(orange_start) != 1L || orange_start + 2L >= length(lines)) {
    stop("Cannot identify the orange-only in-vivo section")
  }
  orange_lines <- lines[seq.int(orange_start + 2L, length(lines))]
  orange_lines <- orange_lines[nzchar(trimws(orange_lines))]
  raw <- utils::read.table(
    text = paste(orange_lines, collapse = "\n"),
    sep = ";",
    dec = ".",
    header = FALSE,
    col.names = sprintf("V%d", seq_len(14L)),
    na.strings = "",
    quote = "",
    comment.char = "",
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  if (nrow(raw) != 15L || ncol(raw) != 14L || anyNA(raw)) {
    stop("Unexpected orange inoculation table: expected 15 complete boxes")
  }

  days <- c(5L, 7L, 10L)
  get_int <- function(index) as.integer(raw[[paste0("V", index)]])
  total <- get_int(8L)
  if (anyNA(total) || !all(total == 20L)) {
    stop("In-vivo orange denominator must be 20 per source box")
  }

  results <- lapply(seq_along(days), function(j) {
    decay <- get_int(1L + j)
    sporulation <- get_int(4L + j)
    reported_decay <- as.numeric(raw[[paste0("V", 8L + j)]])
    reported_sporulation <- as.numeric(raw[[paste0("V", 11L + j)]])
    if (anyNA(decay) || anyNA(sporulation) ||
        any(decay < 0L | decay > total) ||
        any(sporulation < 0L | sporulation > decay) ||
        any(abs(reported_decay - 100 * decay / total) > 1e-8) ||
        any(abs(reported_sporulation - 100 * sporulation / total) > 1e-8)) {
      stop(sprintf("Inconsistent orange count, denominator or percentage on day %d", days[j]))
    }
    rbind(
      data.frame(
        table_id = "orange_in_vivo", source_line = orange_start + 1L + seq_len(nrow(raw)),
        treatment = as.character(raw$V1), day = days[j],
        outcome = "decay", count = decay, total = total, stringsAsFactors = FALSE
      ),
      data.frame(
        table_id = "orange_in_vivo", source_line = orange_start + 1L + seq_len(nrow(raw)),
        treatment = as.character(raw$V1), day = days[j],
        outcome = "sporulation", count = sporulation, total = total, stringsAsFactors = FALSE
      )
    )
  })
  observations <- do.call(rbind, results)
  rownames(observations) <- NULL
  if (!all(observations$treatment %in% upct_treatments) || anyNA(observations)) {
    stop("Unexpected or missing orange in-vivo observations")
  }
  for (outcome in c("decay", "sporulation")) {
    assert_group_design(observations[observations$outcome == outcome, ], days, 3L, outcome)
  }
  observations
}
