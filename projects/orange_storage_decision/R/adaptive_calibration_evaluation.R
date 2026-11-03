# Paired synthetic evaluation of finite grader-reference calibration.
# Source model.R, sampling.R, simulation.R, calibration.R,
# adaptive_policy.R, adaptive_evaluation.R, adaptive_stress.R,
# adaptive_label_error.R and adaptive_calibration_policy.R first.

#' Validate two independent-audit budgets with unique names and sample sizes.
validate_adaptive_reference_budgets <- function(budgets) {
  labels <- names(budgets)
  if (!is.numeric(budgets) || length(budgets) != 2L || anyNA(budgets) ||
      any(!is.finite(budgets)) || any(budgets < 1 | budgets != floor(budgets)) ||
      anyDuplicated(unname(budgets)) || is.null(labels) || anyNA(labels) ||
      any(!nzchar(trimws(labels))) || anyDuplicated(labels)) {
    stop("budgets need two distinct named positive whole sample sizes",
         call. = FALSE)
  }
  invisible(TRUE)
}

#' Create independent reference-standard confusion counts using common draws.
#' Each budget audits the first n reference-positive and reference-negative
#' samples from the same prespecified sample stream. Grading truth is known
#' only for this independent audit, never for target-lot inspection records.
simulate_adaptive_reference_audit <- function(positive_uniforms,
                                               negative_uniforms, n_reference,
                                               sensitivity, specificity) {
  assert_nonnegative_scalar(n_reference, "n_reference", integer = TRUE)
  invisible(inspection_label_probability(0.5, sensitivity, specificity))
  for (uniforms in list(positive_uniforms, negative_uniforms)) {
    if (!is.numeric(uniforms) || anyNA(uniforms) ||
        any(!is.finite(uniforms)) || any(uniforms < 0 | uniforms >= 1) ||
        length(uniforms) < n_reference) {
      stop("audit uniforms must cover n_reference in [0, 1)", call. = FALSE)
    }
  }
  if (n_reference < 1L) stop("n_reference must be positive", call. = FALSE)
  tp <- sum(positive_uniforms[seq_len(n_reference)] < sensitivity)
  tn <- sum(negative_uniforms[seq_len(n_reference)] < specificity)
  data.frame(day = 10L, true_positive = as.integer(tp),
             false_negative = as.integer(n_reference - tp),
             true_negative = as.integer(tn),
             false_positive = as.integer(n_reference - tn))
}

#' One latent lot, shared across two finite reference-audit budgets.
#' Audit and historical data exist by day 12. Both plans for BOTH budgets are
#' frozen before generating day-16 labels or fruit failure times.
simulate_adaptive_calibration_lot <- function(
    lot_id, budgets = c(scarce = 8L, ample = 80L), grid_size = 5L,
    eur_per_check = 0.02, checks_per_cohort = 8L) {
  assert_nonnegative_scalar(lot_id, "lot_id", integer = TRUE)
  if (lot_id < 1L) stop("lot_id must be positive", call. = FALSE)
  validate_adaptive_reference_budgets(budgets)
  assert_nonnegative_scalar(eur_per_check, "eur_per_check")
  assert_nonnegative_scalar(checks_per_cohort, "checks_per_cohort", integer = TRUE)
  if (checks_per_cohort < 1L) {
    stop("checks_per_cohort must be positive", call. = FALSE)
  }

  scenarios <- adaptive_scenario_library(850 * stats::runif(1, 0.25, 0.60))
  prior <- normalize_scenario_weights(scenarios,
                                      c(short = 0.3, typical = 0.4, long = 0.3))
  latent_index <- sample.int(length(scenarios), 1L, prob = prior)
  truth <- scenarios[[latent_index]]
  prices <- simulate_price_scenario(12L, 28L)
  true_sensitivity <- stats::runif(1, 0.75, 0.95)
  true_specificity <- stats::runif(1, 0.75, 0.95)

  max_reference <- max(budgets)
  positive_audit <- stats::runif(max_reference)
  negative_audit <- stats::runif(max_reference)
  eligible <- truth$cohort[truth$initially_sellable]
  pre_schedule <- expand.grid(cohort = eligible, day = c(6L, 12L),
                              KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
  pre_state <- matrix(stats::runif(nrow(pre_schedule) * 6L),
                      nrow = nrow(pre_schedule), ncol = 6L)
  pre_grading <- matrix(stats::runif(nrow(pre_schedule) * 6L),
                        nrow = nrow(pre_schedule), ncol = 6L)
  historical <- simulate_noisy_adaptive_counts(
    truth, pre_schedule, pre_state, pre_grading,
    true_sensitivity, true_specificity
  )

  locked <- lapply(seq_along(budgets), function(i) {
    audit <- simulate_adaptive_reference_audit(
      positive_audit, negative_audit, budgets[i],
      true_sensitivity, true_specificity
    )
    plan_adaptive_with_calibration_uncertainty(
      scenarios, prior, historical, audit, prices,
      storage_eur_per_kg_day = 0.001, decision_day = 12L,
      inspection_day = 16L, grid_size = grid_size,
      checks_per_cohort = checks_per_cohort,
      eur_per_check = eur_per_check, delivery_lag_days = 2L
    )
  })

  # No day-16 labels or physical fruit outcomes exist during day-12 planning.
  future_schedule <- data.frame(cohort = eligible, day = 16L,
                                stringsAsFactors = FALSE)
  future_state <- matrix(stats::runif(length(eligible) * checks_per_cohort),
                         nrow = length(eligible), ncol = checks_per_cohort)
  future_grading <- matrix(stats::runif(length(eligible) * checks_per_cohort),
                           nrow = length(eligible), ncol = checks_per_cohort)
  future <- simulate_noisy_adaptive_counts(
    truth, future_schedule, future_state, future_grading,
    true_sensitivity, true_specificity
  )
  lifetimes <- simulate_fruit_lifetimes(truth)
  realised <- function(day) {
    realised_sale_value(day, truth, prices, lifetimes, 0.001, 12L, 2L)
  }

  do.call(rbind, lapply(seq_along(budgets), function(i) {
    result <- locked[[i]]
    integrated <- result$integrated
    point <- result$point
    inspected_day <- sale_after_inspection(integrated, future)$sale_day
    point_inspected_day <- sale_after_inspection(point, future)$sale_day
    integrated_day <- if (integrated$action == "inspect") {
      inspected_day
    } else integrated$fixed_day
    point_day <- if (point$action == "inspect") {
      point_inspected_day
    } else point$fixed_day
    labour <- integrated$inspection_labour_eur
    data.frame(
      lot_id = lot_id, budget = names(budgets)[i],
      reference_per_class = unname(budgets[i]),
      generating_scenario = names(scenarios)[latent_index],
      true_sensitivity = true_sensitivity, true_specificity = true_specificity,
      point_sensitivity = unname(result$point_rates["sensitivity"]),
      point_specificity = unname(result$point_rates["specificity"]),
      supported_rate_nodes = nrow(result$reference_rate_grid$nodes),
      excluded_rate_nodes = result$reference_rate_grid$n_discarded,
      integrated_action = integrated$action, point_action = point$action,
      integrated_fixed_day = integrated$fixed_day,
      point_fixed_day = point$fixed_day,
      integrated_day = integrated_day, point_day = point_day,
      integrated_planned_gain_eur = max(0, integrated$net_information_value_eur),
      inspection_labour_eur = labour,
      integrated_fixed_eur = realised(integrated$fixed_day),
      immediate_eur = realised(12L),
      integrated_eur = realised(integrated_day) -
        if (integrated$action == "inspect") labour else 0,
      point_eur = realised(point_day) -
        if (point$action == "inspect") labour else 0,
      always_inspect_eur = realised(inspected_day) - labour,
      stringsAsFactors = FALSE
    )
  }))
}

#' Independent lots; audit budgets stay paired within each lot.
run_adaptive_calibration_experiment <- function(
    n_lots = 200L, seed = 20261107L,
    budgets = c(scarce = 8L, ample = 80L), grid_size = 5L,
    eur_per_check = 0.02, checks_per_cohort = 8L) {
  assert_nonnegative_scalar(n_lots, "n_lots", integer = TRUE)
  assert_nonnegative_scalar(seed, "seed", integer = TRUE)
  validate_adaptive_reference_budgets(budgets)
  if (n_lots < 2L) stop("at least two independent lots required", call. = FALSE)
  set.seed(seed)
  do.call(rbind, lapply(seq_len(n_lots), function(i) {
    simulate_adaptive_calibration_lot(i, budgets, grid_size,
                                      eur_per_check, checks_per_cohort)
  }))
}

#' Check complete paired reference budgets and valid realised returns.
validate_adaptive_calibration_records <- function(records) {
  required <- c("lot_id", "budget", "reference_per_class",
                "generating_scenario", "true_sensitivity", "true_specificity",
                "point_sensitivity", "point_specificity", "supported_rate_nodes",
                "excluded_rate_nodes", "integrated_action", "point_action",
                "integrated_fixed_day", "point_fixed_day", "integrated_day",
                "point_day", "integrated_planned_gain_eur", "inspection_labour_eur",
                "integrated_fixed_eur", "immediate_eur", "integrated_eur",
                "point_eur", "always_inspect_eur")
  character_fields <- c("budget", "generating_scenario",
                        "integrated_action", "point_action")
  numeric_fields <- setdiff(required, character_fields)
  if (!is.data.frame(records) || !all(required %in% names(records)) ||
      nrow(records) < 4L || anyNA(records[required]) ||
      !all(vapply(records[character_fields], is.character, logical(1))) ||
      !all(vapply(records[numeric_fields], function(x) {
        is.numeric(x) && all(is.finite(x))
      }, logical(1))) ||
      any(!records$integrated_action %in% c("commit", "inspect")) ||
      any(!records$point_action %in% c("commit", "inspect")) ||
      anyDuplicated(records[c("lot_id", "budget")]) ||
      any(records$lot_id < 1 | records$lot_id != floor(records$lot_id)) ||
      any(records$reference_per_class < 1 |
            records$reference_per_class != floor(records$reference_per_class)) ||
      any(records$true_sensitivity < 0 | records$true_sensitivity > 1) ||
      any(records$true_specificity < 0 | records$true_specificity > 1) ||
      any(records$point_sensitivity < 0 | records$point_sensitivity > 1) ||
      any(records$point_specificity < 0 | records$point_specificity > 1) ||
      any(records$point_sensitivity + records$point_specificity <= 1) ||
      any(records$supported_rate_nodes < 1 |
            records$supported_rate_nodes != floor(records$supported_rate_nodes)) ||
      any(records$excluded_rate_nodes < 0 |
            records$excluded_rate_nodes != floor(records$excluded_rate_nodes)) ||
      any(records$inspection_labour_eur < 0) ||
      any(records$integrated_planned_gain_eur < 0)) {
    stop("records need unique budgets, valid actions and finite paired returns",
         call. = FALSE)
  }
  labels <- sort(unique(records$budget))
  ids <- unique(records$lot_id)
  if (length(labels) != 2L || length(ids) < 2L ||
      length(unique(records$reference_per_class)) != 2L ||
      any(!vapply(split(records, records$budget), function(part) {
        length(unique(part$reference_per_class)) == 1L
      }, logical(1))) ||
      any(!vapply(split(records, records$lot_id), function(part) {
        setequal(part$budget, labels) &&
          length(unique(part$generating_scenario)) == 1L &&
          length(unique(part$true_sensitivity)) == 1L &&
          length(unique(part$true_specificity)) == 1L
      }, logical(1)))) {
    stop("records must pair two audit budgets on each latent lot",
         call. = FALSE)
  }
  invisible(TRUE)
}

#' Whole-lot bootstrap of integrated versus plug-in and fixed commitments.
#' All uncertainty regimes and policy contrasts reuse the same lot indices.
summarize_adaptive_calibration_experiment <- function(records, n_boot = 1000L) {
  validate_adaptive_calibration_records(records)
  assert_nonnegative_scalar(n_boot, "n_boot", integer = TRUE)
  if (n_boot < 1L) stop("n_boot must be positive", call. = FALSE)
  labels <- sort(unique(records$budget))
  ids <- sort(unique(records$lot_id))
  ordered <- lapply(labels, function(label) {
    part <- records[records$budget == label, , drop = FALSE]
    part[match(ids, part$lot_id), , drop = FALSE]
  })
  paired_matrix <- function(denominator) {
    matrix(vapply(ordered, function(part) part$integrated_eur - part[[denominator]],
                  numeric(length(ids))), nrow = length(ids),
           dimnames = list(NULL, labels))
  }
  versus_point <- paired_matrix("point_eur")
  versus_fixed <- paired_matrix("integrated_fixed_eur")
  boot_indices <- matrix(replicate(n_boot, sample.int(length(ids), length(ids),
                                                      replace = TRUE)),
                         nrow = length(ids), ncol = n_boot)
  interval <- function(values) {
    means <- vapply(seq_len(n_boot), function(b) {
      mean(values[boot_indices[, b]])
    }, numeric(1))
    unname(stats::quantile(means, c(0.025, 0.975)))
  }

  by_budget <- do.call(rbind, lapply(seq_along(labels), function(i) {
    part <- ordered[[i]]
    point_gain <- versus_point[, i]
    fixed_gain <- versus_fixed[, i]
    point_ci <- interval(point_gain)
    fixed_ci <- interval(fixed_gain)
    data.frame(
      budget = labels[i], reference_per_class = part$reference_per_class[1],
      n_lots = length(ids),
      integrated_inspected_fraction = mean(part$integrated_action == "inspect"),
      point_inspected_fraction = mean(part$point_action == "inspect"),
      mean_gain_vs_point_eur = mean(point_gain),
      lower_95_point_eur = point_ci[1], upper_95_point_eur = point_ci[2],
      fraction_below_point = mean(point_gain < 0),
      mean_gain_vs_fixed_eur = mean(fixed_gain),
      lower_95_fixed_eur = fixed_ci[1], upper_95_fixed_eur = fixed_ci[2],
      mean_abs_sensitivity_error = mean(abs(part$point_sensitivity -
                                              part$true_sensitivity)),
      mean_abs_specificity_error = mean(abs(part$point_specificity -
                                              part$true_specificity)),
      row.names = NULL
    )
  }))
  small <- by_budget$budget[which.min(by_budget$reference_per_class)]
  large <- by_budget$budget[which.max(by_budget$reference_per_class)]
  difference <- versus_point[, large] - versus_point[, small]
  ci <- interval(difference)
  contrast <- data.frame(
    contrast = paste0(large, "_minus_", small),
    mean_gain_difference_eur = mean(difference),
    lower_95_eur = ci[1], upper_95_eur = ci[2], row.names = NULL
  )
  list(by_budget = by_budget, contrast = contrast)
}
