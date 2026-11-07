# Descriptive, orange-only analyses of the independently measured endpoints.
# No statistical test, fruit-level pairing, causal identification or economic
# optimisation is attempted on these source files.

summarise_upct_quality <- function(observations, metric) {
  required <- c("table_id", "source_line", "treatment", "day", "value", "unit")
  if (!is.data.frame(observations) || !all(required %in% names(observations)) ||
      !is.character(metric) || length(metric) != 1L) {
    stop("Invalid quality observations or metric")
  }
  if (anyNA(observations$value) || any(!is.finite(observations$value))) {
    stop("Quality observations must be finite")
  }

  groups <- split(observations, interaction(observations$treatment, observations$day, drop = TRUE))
  rows <- lapply(groups, function(group) {
    data.frame(
      metric = metric,
      treatment = group$treatment[1L],
      day = group$day[1L],
      n_observations = nrow(group),
      mean = mean(group$value),
      sd = stats::sd(group$value),
      min = min(group$value),
      max = max(group$value),
      unit = group$unit[1L],
      stringsAsFactors = FALSE
    )
  })
  result <- do.call(rbind, rows)
  result <- result[order(match(result$treatment, upct_treatments), result$day), ]
  rownames(result) <- NULL
  result
}

summarise_upct_in_vivo <- function(observations) {
  required <- c("treatment", "day", "outcome", "count", "total")
  if (!is.data.frame(observations) || !all(required %in% names(observations))) {
    stop("Invalid in-vivo orange observations")
  }
  groups <- split(
    observations,
    interaction(observations$treatment, observations$day, observations$outcome, drop = TRUE)
  )
  rows <- lapply(groups, function(group) {
    n_affected <- sum(group$count)
    n_exposed <- sum(group$total)
    data.frame(
      outcome = group$outcome[1L],
      treatment = group$treatment[1L],
      day = group$day[1L],
      n_boxes = nrow(group),
      n_affected = n_affected,
      n_exposed = n_exposed,
      percent = 100 * n_affected / n_exposed,
      stringsAsFactors = FALSE
    )
  })
  result <- do.call(rbind, rows)
  result <- result[order(result$outcome, match(result$treatment, upct_treatments), result$day), ]
  rownames(result) <- NULL
  result
}

# Within-endpoint contrasts against both distinct control conditions. These
# are differences in observed GROUP MEANS, not matched individual changes.
summarise_upct_contrasts <- function(quality_summary, in_vivo_summary) {
  if (!is.data.frame(quality_summary) || !is.data.frame(in_vivo_summary)) {
    stop("Expected two independent summary data frames")
  }
  endpoints <- list(
    list(data = quality_summary[quality_summary$metric == "firmness", ], day = 15L,
         value = "mean", name = "firmness_day15", unit = "N"),
    list(data = quality_summary[quality_summary$metric == "weight_loss", ], day = 15L,
         value = "mean", name = "weight_loss_day15", unit = "percentage_points"),
    list(data = in_vivo_summary[in_vivo_summary$outcome == "decay", ], day = 10L,
         value = "percent", name = "decay_day10", unit = "percentage_points"),
    list(data = in_vivo_summary[in_vivo_summary$outcome == "sporulation", ], day = 10L,
         value = "percent", name = "sporulation_day10", unit = "percentage_points")
  )
  rows <- lapply(endpoints, function(endpoint) {
    data <- endpoint$data[endpoint$data$day == endpoint$day, ]
    if (!setequal(data$treatment, upct_treatments) || nrow(data) != length(upct_treatments)) {
      stop(sprintf("Missing treatment means for %s", endpoint$name))
    }
    pairs <- expand.grid(
      treatment = setdiff(upct_treatments, "CTRL"),
      comparator = c("CTRL", "CTRL-W"),
      stringsAsFactors = FALSE
    )
    # The waxed control cannot meaningfully be contrasted with itself.
    pairs <- pairs[pairs$treatment != pairs$comparator, ]
    value_of <- function(treatment) data[[endpoint$value]][match(treatment, data$treatment)]
    data.frame(
      endpoint = endpoint$name,
      day = endpoint$day,
      treatment = pairs$treatment,
      comparator = pairs$comparator,
      treatment_mean = value_of(pairs$treatment),
      comparator_mean = value_of(pairs$comparator),
      difference = value_of(pairs$treatment) - value_of(pairs$comparator),
      unit = endpoint$unit,
      stringsAsFactors = FALSE
    )
  })
  result <- do.call(rbind, rows)
  rownames(result) <- NULL
  result
}

plot_upct_series <- function(summary, value_column, title, y_label, path, y_min = NULL, y_max = NULL) {
  if (!is.data.frame(summary) || !value_column %in% names(summary) ||
      !is.character(path) || length(path) != 1L) {
    stop("Invalid inputs to plot_upct_series")
  }
  colors <- c("#1F2937", "#2563EB", "#16A34A", "#EA580C", "#9333EA")
  values <- summary[[value_column]]
  lower <- if (is.null(y_min)) min(values) * 0.9 else y_min
  upper <- if (is.null(y_max)) max(values) * 1.1 else y_max
  if (!is.finite(lower) || !is.finite(upper) || lower >= upper) {
    stop("Invalid chart range")
  }

  # The pinned Linux CI R image provides Cairo for a headless PNG device.
  if (!capabilities("cairo")) {
    stop("Cairo PNG support is required for publication figures")
  }
  grDevices::png(filename = path, width = 1600L, height = 950L, res = 160L, type = "cairo")
  on.exit(grDevices::dev.off(), add = TRUE)
  graphics::par(mar = c(6, 5.3, 4, 2) + 0.1)
  days <- sort(unique(summary$day))
  graphics::plot(
    NA_real_, NA_real_, xlim = range(days), ylim = c(lower, upper),
    xlab = "Day of the experiment", ylab = y_label, main = title, xaxt = "n"
  )
  graphics::axis(side = 1L, at = days)
  graphics::grid(col = "#E5E7EB", lty = "dotted")
  for (i in seq_along(upct_treatments)) {
    treatment <- upct_treatments[i]
    subset <- summary[summary$treatment == treatment, ]
    subset <- subset[order(subset$day), ]
    graphics::lines(subset$day, subset[[value_column]], type = "b", lwd = 2L, pch = 16L,
                    col = colors[i])
  }
  graphics::legend(
    "topright", legend = upct_treatments, col = colors, lwd = 2L,
    pch = 16L, cex = 0.78, bty = "n", ncol = 2L
  )
  graphics::mtext("Independent endpoint groups; plotted lines do not identify individual fruit trajectories.",
                  side = 1L, line = 4.5, cex = 0.73, col = "#4B5563")
  invisible(path)
}
