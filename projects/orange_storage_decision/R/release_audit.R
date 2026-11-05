# Base-R integrity checks for the completed synthetic case study.
# These checks validate committed evidence; they do not rerun simulations or
# establish empirical validity for the selling policy.

orange_archived_studies <- function() {
  c(
    "archive_uncertainty", "calibration_uncertainty", "historical_baseline",
    "historical_shift", "inspection_budget", "inspection_error",
    "inspection_plan_value", "inspection_removals", "price_sensitivity",
    "robust_policy", "shape_ensemble", "shared_shape", "survival_regret",
    "survival_stress"
  )
}

validate_orange_release_root <- function(project_root) {
  if (!is.character(project_root) || length(project_root) != 1L ||
      is.na(project_root) || !nzchar(trimws(project_root)) ||
      !dir.exists(project_root)) {
    stop("project_root must name an existing directory", call. = FALSE)
  }
  normalizePath(project_root, mustWork = TRUE)
}

# Read inline Markdown links and images, skipping URI schemes and page anchors.
# Local targets must exist relative to the document that references them.
check_orange_markdown_links <- function(document) {
  if (!is.character(document) || length(document) != 1L ||
      is.na(document) || !file.exists(document) || dir.exists(document)) {
    stop("document must name an existing Markdown file", call. = FALSE)
  }
  lines <- readLines(document, warn = FALSE, encoding = "UTF-8")
  matched <- regmatches(lines, gregexpr("\\]\\([^)]+\\)", lines, perl = TRUE))
  tokens <- unlist(matched, use.names = FALSE)
  if (length(tokens) == 0L) return(0L)

  # Remove the leading `](` and trailing `)` from each Markdown target.
  targets <- substring(tokens, 3L, nchar(tokens) - 1L)
  external <- grepl("^([[:alpha:]][[:alnum:]+.-]*:|#)", targets,
                    perl = TRUE)
  local <- sub("[?#].*$", "", targets[!external])
  local <- local[nzchar(local)]
  missing <- local[!file.exists(file.path(dirname(document), local))]
  if (length(missing) != 0L) {
    stop(sprintf("Broken local links in %s: %s", document,
                 paste(unique(missing), collapse = ", ")), call. = FALSE)
  }
  as.integer(length(local))
}

# Require every archived study to keep its explanation, settings, machine-
# readable result data and figure. Parse CSV headers/first rows to catch
# empty, malformed or accidentally replaced evidence files.
check_orange_archived_studies <- function(project_root) {
  total_csv <- 0L
  for (study in orange_archived_studies()) {
    directory <- file.path(project_root, "results", study)
    required <- file.path(directory, c("RESULTS.md", "settings.R"))
    if (!dir.exists(directory) || !all(file.exists(required))) {
      stop(sprintf("Missing archived report/settings: %s", study),
           call. = FALSE)
    }
    csv <- list.files(directory, pattern = "\\.csv$", full.names = TRUE)
    figures <- list.files(directory, pattern = "\\.png$", full.names = TRUE)
    if (length(csv) == 0L || length(figures) == 0L) {
      stop(sprintf("Missing archived CSV/figure: %s", study), call. = FALSE)
    }
    evidence_files <- c(required, csv, figures)
    if (anyNA(file.info(evidence_files)$size) ||
        any(file.info(evidence_files)$size == 0)) {
      stop(sprintf("Empty or unreadable archived file: %s", study),
           call. = FALSE)
    }
    for (path in csv) {
      first <- tryCatch(
        utils::read.csv(path, nrows = 1L, check.names = FALSE,
                        stringsAsFactors = FALSE),
        error = function(error) {
          stop(sprintf("Unreadable CSV %s: %s", path, conditionMessage(error)),
               call. = FALSE)
        }
      )
      if (nrow(first) != 1L || ncol(first) == 0L ||
          anyNA(names(first)) || anyDuplicated(names(first)) > 0L) {
        stop(sprintf("Empty data or duplicated CSV columns: %s", path),
             call. = FALSE)
      }
    }
    total_csv <- total_csv + as.integer(length(csv))
  }
  total_csv
}

# Check selected two-decimal figures from the synthesis against the committed
# CSV summaries. Exact source rows are identified by their declared scenarios,
# not by their position in a file. This is a transcription audit, not a test
# that the simulated policy produces the same outcome in the real world.
check_orange_headline_metric <- function(project_root, study, filename,
                                         selector, metric, expected) {
  if (!is.character(study) || length(study) != 1L || is.na(study) ||
      !(study %in% orange_archived_studies()) ||
      !is.character(filename) || length(filename) != 1L ||
      is.na(filename) || !grepl("^[[:alnum:]_]+\\.csv$", filename) ||
      !is.list(selector) || length(selector) == 0L ||
      is.null(names(selector)) || any(!nzchar(names(selector))) ||
      !is.character(metric) || length(metric) != 1L || is.na(metric) ||
      !is.numeric(expected) || length(expected) != 1L ||
      !is.finite(expected)) {
    stop("Invalid archived headline metric specification", call. = FALSE)
  }

  path <- file.path(project_root, "results", study, filename)
  evidence <- utils::read.csv(path, check.names = FALSE,
                              stringsAsFactors = FALSE)
  required <- c(names(selector), metric)
  if (!all(required %in% names(evidence))) {
    stop(sprintf("Missing expected columns in %s", path), call. = FALSE)
  }
  matches <- rep(TRUE, nrow(evidence))
  for (field in names(selector)) {
    value <- selector[[field]]
    if (length(value) != 1L || is.na(value) ||
        !(is.numeric(value) || is.character(value))) {
      stop("Each selector value must be one number or string", call. = FALSE)
    }
    matches <- matches & !is.na(evidence[[field]]) &
      evidence[[field]] == value
  }
  if (sum(matches) != 1L) {
    stop(sprintf("Expected exactly one source row for %s:%s", study, metric),
         call. = FALSE)
  }
  observed <- evidence[[metric]][matches]
  if (!is.numeric(observed) || !is.finite(observed) ||
      !isTRUE(all.equal(round(observed, 2L), expected, tolerance = 0))) {
    stop(sprintf("Headline figure differs from %s:%s (expected %.2f)",
                 study, metric, expected), call. = FALSE)
  }
  invisible(TRUE)
}

orange_synthesis_anchors <- function() {
  # Each record is the source, one uniquely keyed row, a column and the
  # expected value as printed to two decimal places in the synthesis.
  list(
    list("price_sensitivity", "summary.csv",
         list(scenario = "perfect_forecast"), "stratified_minus_immediate_eur", 13.95),
    list("price_sensitivity", "summary.csv",
         list(scenario = "mean_40pct_lower"), "stratified_minus_immediate_eur", -29.15),
    list("robust_policy", "summary.csv",
         list(scenario = "mean_40pct_lower"), "robust_minus_stratified_mean_eur", 17.96),
    list("robust_policy", "summary.csv",
         list(scenario = "mean_40pct_lower"), "robust_minus_immediate_mean_eur", -8.75),
    list("survival_stress", "summary.csv",
         list(scenario = "double_hazard"), "stratified_minus_immediate_mean_eur", -29.85),
    list("historical_shift", "summary.csv",
         list(lifetime_multiplier = 0.75, metric = "gain_day12_eur"), "mean", -4.86),
    list("historical_shift", "summary.csv",
         list(lifetime_multiplier = 0.75, metric = "bias_day16_kg"), "mean", 137.44),
    list("inspection_error", "summary.csv",
         list(scenario = "balanced_errors"), "corrected_minus_naive_eur_mean", 48.32),
    list("inspection_error", "summary.csv",
         list(scenario = "balanced_errors"), "corrected_minus_immediate_eur_mean", 0.12),
    list("inspection_plan_value", "summary.csv",
         list(per_group = 30, labour_eur_per_check = 0.02), "gain_vs_day12_eur_mean", -41.09),
    list("inspection_plan_value", "summary.csv",
         list(per_group = 90, labour_eur_per_check = 0.02), "gain_vs_day12_eur_mean", -95.07),
    list("inspection_plan_value", "summary.csv",
         list(per_group = 180, labour_eur_per_check = 0.02), "gain_vs_day12_eur_mean", -184.30),
    list("historical_baseline", "return_summary.csv",
         list(comparator = "day12"), "gain_mean_eur", 20.31),
    list("archive_uncertainty", "comparisons.csv",
         list(metric = "return_eur"), "mean", 1.00),
    list("shape_ensemble", "summary.csv",
         list(scenario = "common_shape"), "ensemble_minus_shared_eur_mean", -3.93),
    list("survival_regret", "summary.csv",
         list(lifetime_multiplier = 0.75, metric = "robust_minus_nominal_eur"),
         "mean", -0.82),
    list("survival_regret", "summary.csv",
         list(lifetime_multiplier = 1.25, metric = "robust_minus_nominal_eur"),
         "mean", 0.77)
  )
}

# Public entry point, run from the project directory in CI or by readers.
audit_orange_release <- function(project_root = ".") {
  root <- validate_orange_release_root(project_root)
  required_docs <- file.path(root, c("README.md", "PROTOCOL.md",
                                     "SCIENTIFIC_SYNTHESIS.md", "REPRODUCIBILITY.md"))
  if (!all(file.exists(required_docs))) {
    stop("Missing project release documentation", call. = FALSE)
  }
  csv_count <- check_orange_archived_studies(root)
  docs <- list.files(root, pattern = "\\.md$", recursive = TRUE,
                     full.names = TRUE)
  links <- sum(vapply(docs, check_orange_markdown_links, integer(1)))
  anchors <- orange_synthesis_anchors()
  for (anchor in anchors) {
    check_orange_headline_metric(root, anchor[[1]], anchor[[2]], anchor[[3]],
                                 anchor[[4]], anchor[[5]])
  }
  list(archived_studies = length(orange_archived_studies()),
       csv_files = csv_count, markdown_documents = length(docs),
       verified_local_links = links, verified_headlines = length(anchors))
}
