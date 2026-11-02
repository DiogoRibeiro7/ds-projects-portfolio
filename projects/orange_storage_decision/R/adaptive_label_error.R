# Prospective label-error evaluation for the adaptive inspection policy.
# Source model.R, sampling.R, simulation.R, adaptive_policy.R,
# adaptive_evaluation.R and adaptive_stress.R first. Base R only.

#' Named, externally specified inspection-grade accuracy regimes.
default_adaptive_label_regimes <- function() {
  data.frame(regime = c("perfect", "moderate", "noisy"),
             sensitivity = c(1, 0.90, 0.75),
             specificity = c(1, 0.90, 0.80), stringsAsFactors = FALSE)
}

#' Validate an explicit label-error design with a perfect-label control.
validate_adaptive_label_regimes <- function(regimes) {
  if (!is.data.frame(regimes) ||
      !setequal(names(regimes), c("regime", "sensitivity", "specificity")) ||
      nrow(regimes) < 2L || !is.character(regimes$regime) ||
      anyNA(regimes) || any(!nzchar(trimws(regimes$regime))) ||
      anyDuplicated(regimes$regime) ||
      !is.numeric(regimes$sensitivity) || !is.numeric(regimes$specificity) ||
      any(!is.finite(regimes$sensitivity)) ||
      any(!is.finite(regimes$specificity))) {
    stop("regimes need unique labels and finite sensitivity/specificity",
         call. = FALSE)
  }
  for (i in seq_len(nrow(regimes))) {
    invisible(inspection_label_probability(0.5, regimes$sensitivity[i],
                                          regimes$specificity[i]))
  }
  perfect <- regimes[regimes$regime == "perfect", , drop = FALSE]
  if (nrow(perfect) != 1L || perfect$sensitivity != 1 ||
      perfect$specificity != 1 ||
      anyDuplicated(regimes[c("sensitivity", "specificity")])) {
    stop("regimes need a perfect (1, 1) control and distinct rate pairs",
         call. = FALSE)
  }
  invisible(TRUE)
}

#' Report fallible labels from independent true-state and grading draws.
#'
#' Within each cohort/day inspection, true quality is Bernoulli(S(day));
#' conditional on truth, the independent grader draw uses Se and Sp. The
#' same two uniform matrices can be reused across accuracy regimes for paired
#' comparisons, while the policy receives only aggregate reported counts.
simulate_noisy_adaptive_counts <- function(truth, schedule, state_uniforms,
                                           grading_uniforms, sensitivity,
                                           specificity) {
  invisible(inspection_label_probability(0.5, sensitivity, specificity))
  counts <- stress_inspection_counts(truth, schedule, state_uniforms)
  if (!is.matrix(grading_uniforms) || !is.numeric(grading_uniforms) ||
      !identical(dim(grading_uniforms), dim(state_uniforms)) ||
      anyNA(grading_uniforms) || any(!is.finite(grading_uniforms)) ||
      any(grading_uniforms < 0 | grading_uniforms >= 1)) {
    stop("grading_uniforms must match state_uniforms in [0, 1)",
         call. = FALSE)
  }
  inventory_index <- match(schedule$cohort, truth$cohort)
  survival <- weibull_survival(schedule$day, truth$shape[inventory_index],
                               truth$scale[inventory_index])
  truly_acceptable <- state_uniforms < survival
  reported_acceptable <-
    (truly_acceptable & grading_uniforms < sensitivity) |
    (!truly_acceptable & grading_uniforms < (1 - specificity))
  counts$n_acceptable <- as.integer(rowSums(reported_acceptable))
  counts
}

#' Evaluate perfect and noisy labels on one latent lot, with common draws.
#'
#' All day-12 decisions are locked before future inspection uniforms and
#' independent fruit lifetimes are generated. The naive policy sees the exact
#' same labels but wrongly assumes perfect grading; neither sees latent truth.
simulate_adaptive_label_lot <- function(lot_id,
                                        regimes = default_adaptive_label_regimes(),
                                        eur_per_check = 0.02,
                                        checks_per_cohort = 8L) {
  assert_nonnegative_scalar(lot_id, "lot_id", integer = TRUE)
  if (lot_id < 1L) stop("lot_id must be positive", call. = FALSE)
  validate_adaptive_label_regimes(regimes)
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
  eligible <- truth$cohort[truth$initially_sellable]
  pre_schedule <- expand.grid(cohort = eligible, day = c(6L, 12L),
                              KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
  pre_state <- matrix(stats::runif(nrow(pre_schedule) * 6L),
                      nrow = nrow(pre_schedule), ncol = 6L)
  pre_grading <- matrix(stats::runif(nrow(pre_schedule) * 6L),
                        nrow = nrow(pre_schedule), ncol = 6L)

  locked <- lapply(seq_len(nrow(regimes)), function(i) {
    sensitivity <- regimes$sensitivity[i]
    specificity <- regimes$specificity[i]
    historical <- simulate_noisy_adaptive_counts(truth, pre_schedule,
                                                  pre_state, pre_grading,
                                                  sensitivity, specificity)
    calibrated_posterior <- scenario_posterior(
      scenarios, prior, historical, 12L, sensitivity, specificity
    )
    naive_posterior <- scenario_posterior(scenarios, prior, historical, 12L)
    make_plan <- function(posterior, se, sp) {
      plan_adaptive_inspection(
        scenarios, posterior, prices, storage_eur_per_kg_day = 0.001,
        decision_day = 12L, inspection_day = 16L,
        checks_per_cohort = checks_per_cohort, eur_per_check = eur_per_check,
        delivery_lag_days = 2L, sensitivity = se, specificity = sp
      )
    }
    list(calibrated = make_plan(calibrated_posterior, sensitivity, specificity),
         naive = make_plan(naive_posterior, 1, 1))
  })

  # No information from these draws can reach a day-12 plan.
  future_schedule <- data.frame(cohort = eligible, day = 16L,
                                stringsAsFactors = FALSE)
  future_state <- matrix(stats::runif(length(eligible) * checks_per_cohort),
                         nrow = length(eligible), ncol = checks_per_cohort)
  future_grading <- matrix(stats::runif(length(eligible) * checks_per_cohort),
                           nrow = length(eligible), ncol = checks_per_cohort)
  lifetimes <- simulate_fruit_lifetimes(truth)
  realised <- function(day) {
    realised_sale_value(day, truth, prices, lifetimes, 0.001, 12L, 2L)
  }

  do.call(rbind, lapply(seq_len(nrow(regimes)), function(i) {
    calibrated <- locked[[i]]$calibrated
    naive <- locked[[i]]$naive
    future <- simulate_noisy_adaptive_counts(
      truth, future_schedule, future_state, future_grading,
      regimes$sensitivity[i], regimes$specificity[i]
    )
    calibrated_followup <- sale_after_inspection(calibrated, future)$sale_day
    naive_followup <- sale_after_inspection(naive, future)$sale_day
    calibrated_day <- if (calibrated$action == "inspect") {
      calibrated_followup
    } else calibrated$fixed_day
    naive_day <- if (naive$action == "inspect") {
      naive_followup
    } else naive$fixed_day
    labour <- calibrated$inspection_labour_eur
    data.frame(
      lot_id = lot_id, regime = regimes$regime[i],
      sensitivity = regimes$sensitivity[i],
      specificity = regimes$specificity[i],
      generating_scenario = names(scenarios)[latent_index],
      calibrated_action = calibrated$action, naive_action = naive$action,
      calibrated_fixed_day = calibrated$fixed_day,
      naive_fixed_day = naive$fixed_day,
      calibrated_day = calibrated_day, naive_day = naive_day,
      calibrated_planned_gain_eur = max(0, calibrated$net_information_value_eur),
      naive_planned_gain_eur = max(0, naive$net_information_value_eur),
      inspection_labour_eur = labour,
      fixed_eur = realised(calibrated$fixed_day),
      immediate_eur = realised(12L),
      calibrated_eur = realised(calibrated_day) -
        if (calibrated$action == "inspect") labour else 0,
      naive_eur = realised(naive_day) -
        if (naive$action == "inspect") labour else 0,
      always_inspect_eur = realised(calibrated_followup) - labour,
      stringsAsFactors = FALSE
    )
  }))
}

#' Independent synthetic lots, paired across label-accuracy regimes.
run_adaptive_label_experiment <- function(n_lots = 200L, seed = 20261105L,
                                          regimes = default_adaptive_label_regimes(),
                                          eur_per_check = 0.02,
                                          checks_per_cohort = 8L) {
  assert_nonnegative_scalar(n_lots, "n_lots", integer = TRUE)
  assert_nonnegative_scalar(seed, "seed", integer = TRUE)
  validate_adaptive_label_regimes(regimes)
  if (n_lots < 2L) stop("at least two independent lots required", call. = FALSE)
  set.seed(seed)
  do.call(rbind, lapply(seq_len(n_lots), function(i) {
    simulate_adaptive_label_lot(i, regimes, eur_per_check, checks_per_cohort)
  }))
}

#' Require complete, one-row-per-regime records from each shared latent lot.
validate_adaptive_label_records <- function(records) {
  required <- c("lot_id", "regime", "sensitivity", "specificity",
                "generating_scenario", "calibrated_action", "naive_action",
                "calibrated_fixed_day", "naive_fixed_day", "calibrated_day",
                "naive_day", "calibrated_planned_gain_eur",
                "naive_planned_gain_eur", "inspection_labour_eur",
                "fixed_eur", "immediate_eur", "calibrated_eur",
                "naive_eur", "always_inspect_eur")
  numeric_fields <- setdiff(required, c("regime", "generating_scenario",
                                        "calibrated_action", "naive_action"))
  if (!is.data.frame(records) || !all(required %in% names(records)) ||
      nrow(records) < 4L || anyNA(records[required]) ||
      !all(vapply(records[c("regime", "generating_scenario",
                            "calibrated_action", "naive_action")],
                  is.character, logical(1))) ||
      any(!records$calibrated_action %in% c("inspect", "commit")) ||
      any(!records$naive_action %in% c("inspect", "commit")) ||
      !all(vapply(records[numeric_fields], function(x) {
        is.numeric(x) && all(is.finite(x))
      }, logical(1)))) {
    stop("records need finite paired returns and valid inspection actions",
         call. = FALSE)
  }
  if (anyDuplicated(records[c("lot_id", "regime")]) ||
      any(records$lot_id < 1 | records$lot_id != floor(records$lot_id)) ||
      any(records$sensitivity < 0 | records$sensitivity > 1) ||
      any(records$specificity < 0 | records$specificity > 1) ||
      any(records$sensitivity + records$specificity <= 1) ||
      any(records[c("calibrated_planned_gain_eur", "naive_planned_gain_eur",
                    "inspection_labour_eur")] < 0)) {
    stop("records need unique lot/regime rows and valid rates and costs",
         call. = FALSE)
  }
  labels <- sort(unique(records$regime))
  ids <- unique(records$lot_id)
  if (length(labels) < 2L || !("perfect" %in% labels) || length(ids) < 2L ||
      any(!vapply(split(records, records$lot_id), function(part) {
        setequal(part$regime, labels) &&
          length(unique(part$generating_scenario)) == 1L
      }, logical(1))) ||
      any(!vapply(split(records, records$regime), function(part) {
        length(unique(part$sensitivity)) == 1L &&
          length(unique(part$specificity)) == 1L &&
          (part$regime[1] != "perfect" ||
             (part$sensitivity[1] == 1 && part$specificity[1] == 1))
      }, logical(1)))) {
    stop("records must balance regimes over the same latent lots",
         call. = FALSE)
  }
  invisible(TRUE)
}

#' Joint whole-lot bootstrap preserves both policy and regime pairing.
#' The same indices bootstrap all regimes and their within-lot contrasts.
summarize_adaptive_label_experiment <- function(records, n_boot = 1000L) {
  validate_adaptive_label_records(records)
  assert_nonnegative_scalar(n_boot, "n_boot", integer = TRUE)
  if (n_boot < 1L) stop("n_boot must be positive", call. = FALSE)
  labels <- sort(unique(records$regime))
  ids <- sort(unique(records$lot_id))
  ordered <- lapply(labels, function(label) {
    part <- records[records$regime == label, , drop = FALSE]
    part[match(ids, part$lot_id), , drop = FALSE]
  })
  paired_matrix <- function(numerator, denominator) {
    matrix(vapply(ordered, function(part) part[[numerator]] - part[[denominator]],
                  numeric(length(ids))), nrow = length(ids),
           dimnames = list(NULL, labels))
  }
  versus_naive <- paired_matrix("calibrated_eur", "naive_eur")
  versus_fixed <- paired_matrix("calibrated_eur", "fixed_eur")
  boot_indices <- matrix(replicate(n_boot, sample.int(length(ids), length(ids),
                                                      replace = TRUE)),
                         nrow = length(ids), ncol = n_boot)
  interval <- function(values) {
    means <- vapply(seq_len(n_boot), function(b) {
      mean(values[boot_indices[, b]])
    }, numeric(1))
    unname(stats::quantile(means, c(0.025, 0.975)))
  }

  by_regime <- do.call(rbind, lapply(seq_along(labels), function(i) {
    part <- ordered[[i]]
    naive_gain <- versus_naive[, i]
    fixed_gain <- versus_fixed[, i]
    naive_ci <- interval(naive_gain)
    fixed_ci <- interval(fixed_gain)
    data.frame(
      regime = labels[i], sensitivity = part$sensitivity[1],
      specificity = part$specificity[1], n_lots = length(ids),
      calibrated_inspected_fraction = mean(part$calibrated_action == "inspect"),
      naive_inspected_fraction = mean(part$naive_action == "inspect"),
      mean_gain_vs_naive_eur = mean(naive_gain),
      lower_95_naive_eur = naive_ci[1], upper_95_naive_eur = naive_ci[2],
      fraction_below_naive = mean(naive_gain < 0),
      mean_gain_vs_fixed_eur = mean(fixed_gain),
      lower_95_fixed_eur = fixed_ci[1], upper_95_fixed_eur = fixed_ci[2],
      fraction_below_fixed = mean(fixed_gain < 0),
      row.names = NULL
    )
  }))
  contrasts <- do.call(rbind, lapply(labels[labels != "perfect"], function(label) {
    difference <- versus_fixed[, label] - versus_fixed[, "perfect"]
    ci <- interval(difference)
    data.frame(contrast = paste0(label, "_minus_perfect"),
               mean_inspection_premium_difference_eur = mean(difference),
               lower_95_eur = ci[1], upper_95_eur = ci[2], row.names = NULL)
  }))
  list(by_regime = by_regime, contrasts = contrasts)
}
