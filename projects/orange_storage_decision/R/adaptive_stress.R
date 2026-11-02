# Prospective shelf-life misspecification for the adaptive inspection policy.
# Source model.R, sampling.R, simulation.R, adaptive_policy.R and
# adaptive_evaluation.R first. Base R only.

#' Validate prespecified, named scale perturbations around an unchanged world.
validate_adaptive_shifts <- function(shifts) {
  labels <- names(shifts)
  if (!is.numeric(shifts) || length(shifts) < 2L ||
      anyNA(shifts) || any(!is.finite(shifts)) || any(shifts <= 0) ||
      is.null(labels) || anyNA(labels) ||
      any(!nzchar(trimws(labels))) || anyDuplicated(labels) ||
      !("unchanged" %in% labels) || shifts[["unchanged"]] != 1 ||
      anyDuplicated(unname(shifts))) {
    stop("shifts need distinct positive named multipliers and unchanged = 1",
         call. = FALSE)
  }
  invisible(TRUE)
}

#' Convert pre-drawn common uniforms into independent binomial counts.
#'
#' The same matrix may be reused for distinct truth scenarios, which couples
#' their synthetic inspections without exposing future labels to any policy.
#' Rows follow the supplied cohort/day inspection plan; columns are draws.
stress_inspection_counts <- function(truth, schedule, uniforms) {
  validate_cohorts(truth)
  if (!is.data.frame(schedule) ||
      !setequal(names(schedule), c("cohort", "day")) ||
      nrow(schedule) < 1L || !is.character(schedule$cohort) ||
      anyNA(schedule$cohort) || anyDuplicated(schedule) ||
      !is.numeric(schedule$day) || anyNA(schedule$day) ||
      any(!is.finite(schedule$day)) ||
      any(schedule$day <= 0 | schedule$day != floor(schedule$day)) ||
      !setequal(unique(schedule$cohort), truth$cohort[truth$initially_sellable])) {
    stop("schedule needs unique, valid cohort/day rows for eligible grades",
         call. = FALSE)
  }
  if (!is.matrix(uniforms) || !is.numeric(uniforms) ||
      nrow(uniforms) != nrow(schedule) || ncol(uniforms) < 1L ||
      anyNA(uniforms) || any(!is.finite(uniforms)) ||
      any(uniforms < 0 | uniforms >= 1)) {
    stop("uniforms must be a finite [0, 1) matrix with one row per inspection",
         call. = FALSE)
  }
  index <- match(schedule$cohort, truth$cohort)
  p <- weibull_survival(schedule$day, truth$shape[index], truth$scale[index])
  counts <- schedule
  counts$n_checked <- rep(as.integer(ncol(uniforms)), nrow(schedule))
  # R recycles each row probability down the corresponding matrix column.
  counts$n_acceptable <- as.integer(rowSums(uniforms < p))
  counts
}

#' Draw unit-scale fruit lifetimes independently of inspection labels.
#' Only the evaluation stage receives these latent potential outcomes.
draw_stress_unit_lifetimes <- function(truth, mean_fruit_kg = 0.2) {
  validate_cohorts(truth)
  assert_nonnegative_scalar(mean_fruit_kg, "mean_fruit_kg")
  if (mean_fruit_kg == 0) stop("mean_fruit_kg must be positive", call. = FALSE)
  eligible <- truth[truth$initially_sellable, , drop = FALSE]
  draws <- lapply(seq_len(nrow(eligible)), function(i) {
    n <- max(1L, as.integer(round(eligible$mass_kg[i] / mean_fruit_kg)))
    stats::rweibull(n, shape = eligible$shape[i], scale = 1)
  })
  names(draws) <- eligible$cohort
  draws
}

#' One paired lot under shorter, unchanged and longer *true* shelf lives.
#'
#' All policies keep the same three-hypothesis model, prior and known price
#' path. Each shifted world supplies only its already-observed day-6/12 counts
#' to the day-12 plan. Future counts and fruit outcomes are drawn afterward.
simulate_adaptive_stress_lot <- function(lot_id,
                                         shifts = c(shorter = 0.75,
                                                    unchanged = 1,
                                                    longer = 1.25),
                                         eur_per_check = 0.02,
                                         checks_per_cohort = 8L) {
  assert_nonnegative_scalar(lot_id, "lot_id", integer = TRUE)
  if (lot_id < 1L) stop("lot_id must be positive", call. = FALSE)
  validate_adaptive_shifts(shifts)
  assert_nonnegative_scalar(eur_per_check, "eur_per_check")
  assert_nonnegative_scalar(checks_per_cohort, "checks_per_cohort", integer = TRUE)
  if (checks_per_cohort < 1L) {
    stop("checks_per_cohort must be positive", call. = FALSE)
  }

  scenarios <- adaptive_scenario_library(850 * stats::runif(1, 0.25, 0.60))
  prior <- normalize_scenario_weights(scenarios,
                                      c(short = 0.3, typical = 0.4, long = 0.3))
  latent_index <- sample.int(length(scenarios), 1L, prob = prior)
  base_truth <- scenarios[[latent_index]]
  prices <- simulate_price_scenario(12L, 28L)
  eligible <- base_truth$cohort[base_truth$initially_sellable]
  pre_schedule <- expand.grid(cohort = eligible, day = c(6L, 12L),
                              KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
  pre_uniforms <- matrix(stats::runif(nrow(pre_schedule) * 6L),
                         nrow = nrow(pre_schedule), ncol = 6L)

  # Freeze every day-12 policy before generating a single day-16 observation
  # or any future fruit failure time. Only the predecision counts vary by world.
  locked <- lapply(seq_along(shifts), function(i) {
    truth <- base_truth
    truth$scale[truth$initially_sellable] <-
      base_truth$scale[base_truth$initially_sellable] * shifts[i]
    observations <- stress_inspection_counts(truth, pre_schedule, pre_uniforms)
    posterior <- scenario_posterior(scenarios, prior, observations, 12L)
    plan <- plan_adaptive_inspection(
      scenarios, posterior, prices, storage_eur_per_kg_day = 0.001,
      decision_day = 12L, inspection_day = 16L,
      checks_per_cohort = checks_per_cohort, eur_per_check = eur_per_check,
      delivery_lag_days = 2L
    )
    list(truth = truth, plan = plan)
  })

  future_schedule <- data.frame(cohort = eligible, day = 16L,
                                stringsAsFactors = FALSE)
  future_uniforms <- matrix(stats::runif(length(eligible) * checks_per_cohort),
                            nrow = length(eligible), ncol = checks_per_cohort)
  unit_lifetimes <- draw_stress_unit_lifetimes(base_truth)

  do.call(rbind, lapply(seq_along(shifts), function(i) {
    truth <- locked[[i]]$truth
    plan <- locked[[i]]$plan
    future_counts <- stress_inspection_counts(truth, future_schedule,
                                               future_uniforms)
    followup_day <- sale_after_inspection(plan, future_counts)$sale_day
    adaptive_day <- if (plan$action == "inspect") followup_day else plan$fixed_day
    # Every world uses the same unit Weibull fruit draws, multiplied by its
    # own true scale. Prices and holding costs remain identical across worlds.
    lifetimes <- lapply(eligible, function(group) {
      unit_lifetimes[[group]] * truth$scale[match(group, truth$cohort)]
    })
    names(lifetimes) <- eligible
    realised <- function(day) {
      realised_sale_value(day, truth, prices, lifetimes, 0.001, 12L, 2L)
    }
    # The true-parameter oracle is used for evaluation only, after all
    # operational sale dates are fixed. Its realised return is not a per-lot bound.
    oracle_day <- optimal_batch_sale(truth, prices, 0.001, 12L, 2L)$sale_day
    labour <- plan$inspection_labour_eur
    data.frame(
      lot_id = lot_id, shift = names(shifts)[i], scale_multiplier = unname(shifts[i]),
      generating_scenario = names(scenarios)[latent_index],
      action = plan$action, fixed_day = plan$fixed_day,
      adaptive_day = adaptive_day, always_inspect_day = followup_day,
      oracle_day = oracle_day,
      planned_gain_eur = max(0, plan$net_information_value_eur),
      inspection_labour_eur = labour,
      fixed_eur = realised(plan$fixed_day),
      immediate_eur = realised(12L),
      adaptive_eur = realised(adaptive_day) -
        if (plan$action == "inspect") labour else 0,
      always_inspect_eur = realised(followup_day) - labour,
      oracle_eur = realised(oracle_day), stringsAsFactors = FALSE
    )
  }))
}

#' Independent lots; scale regimes remain paired within each lot.
run_adaptive_stress_experiment <- function(n_lots = 200L, seed = 20261103L,
                                           shifts = c(shorter = 0.75,
                                                      unchanged = 1,
                                                      longer = 1.25),
                                           eur_per_check = 0.02,
                                           checks_per_cohort = 8L) {
  assert_nonnegative_scalar(n_lots, "n_lots", integer = TRUE)
  assert_nonnegative_scalar(seed, "seed", integer = TRUE)
  validate_adaptive_shifts(shifts)
  if (n_lots < 2L) stop("at least two independent lots required", call. = FALSE)
  set.seed(seed)
  do.call(rbind, lapply(seq_len(n_lots), function(i) {
    simulate_adaptive_stress_lot(i, shifts, eur_per_check, checks_per_cohort)
  }))
}

#' Validate one complete paired record per lot and true scale regime.
validate_adaptive_stress_records <- function(records) {
  required <- c("lot_id", "shift", "scale_multiplier", "generating_scenario",
                "action", "fixed_day", "adaptive_day", "always_inspect_day",
                "oracle_day", "planned_gain_eur", "inspection_labour_eur",
                "fixed_eur", "immediate_eur", "adaptive_eur",
                "always_inspect_eur", "oracle_eur")
  if (!is.data.frame(records) || !all(required %in% names(records)) ||
      nrow(records) < 4L || anyNA(records[required]) ||
      !is.character(records$shift) ||
      !is.character(records$generating_scenario) ||
      !is.character(records$action) ||
      any(!records$action %in% c("commit", "inspect")) ||
      any(!vapply(records[c("lot_id", "scale_multiplier", "fixed_day",
                            "adaptive_day", "always_inspect_day", "oracle_day",
                            "planned_gain_eur", "inspection_labour_eur",
                            "fixed_eur", "immediate_eur", "adaptive_eur",
                            "always_inspect_eur", "oracle_eur")],
                  function(x) is.numeric(x) && all(is.finite(x)), logical(1)))) {
    stop("stress records need finite fields and valid actions", call. = FALSE)
  }
  if (any(records$lot_id < 1 | records$lot_id != floor(records$lot_id)) ||
      any(records$scale_multiplier <= 0) ||
      any(records$planned_gain_eur < 0) ||
      any(records$inspection_labour_eur < 0) ||
      anyDuplicated(records[c("lot_id", "shift")])) {
    stop("stress records need unique lot/shift rows and nonnegative costs",
         call. = FALSE)
  }
  labels <- sort(unique(records$shift))
  if (!("unchanged" %in% labels) || length(labels) < 2L ||
      length(unique(records$lot_id)) < 2L ||
      any(!vapply(split(records, records$lot_id), function(part) {
        setequal(part$shift, labels) &&
          length(unique(part$generating_scenario)) == 1L
      }, logical(1))) ||
      any(!vapply(split(records, records$shift), function(part) {
        length(unique(part$scale_multiplier)) == 1L &&
          (part$shift[1] != "unchanged" || part$scale_multiplier[1] == 1)
      }, logical(1)))) {
    stop("stress records must be complete, balanced and share latent lots",
         call. = FALSE)
  }
  invisible(TRUE)
}

#' Paired lot-level bootstrap by regime and versus the unchanged-world gain.
#' The same resampled lot indices are used for *all* true scale regimes.
summarize_adaptive_stress <- function(records, n_boot = 1000L) {
  validate_adaptive_stress_records(records)
  assert_nonnegative_scalar(n_boot, "n_boot", integer = TRUE)
  if (n_boot < 1L) stop("n_boot must be positive", call. = FALSE)
  labels <- sort(unique(records$shift))
  ids <- sort(unique(records$lot_id))
  ordered <- lapply(labels, function(label) {
    part <- records[records$shift == label, , drop = FALSE]
    part[match(ids, part$lot_id), , drop = FALSE]
  })
  gain <- vapply(ordered, function(part) part$adaptive_eur - part$fixed_eur,
                 numeric(length(ids)))
  gain <- matrix(gain, nrow = length(ids), ncol = length(labels),
                 dimnames = list(NULL, labels))
  boot_indices <- replicate(n_boot, sample.int(length(ids), length(ids),
                                                replace = TRUE))
  boot_indices <- matrix(boot_indices, nrow = length(ids), ncol = n_boot)
  boot_mean <- function(x) vapply(seq_len(n_boot), function(b) {
    mean(x[boot_indices[, b]])
  }, numeric(1))

  by_shift <- do.call(rbind, lapply(seq_along(labels), function(i) {
    part <- ordered[[i]]
    values <- gain[, i]
    interval <- stats::quantile(boot_mean(values), c(0.025, 0.975))
    data.frame(shift = labels[i], scale_multiplier = part$scale_multiplier[1],
               n_lots = length(ids),
               inspected_fraction = mean(part$action == "inspect"),
               mean_planned_gain_eur = mean(part$planned_gain_eur),
               mean_realised_gain_eur = mean(values),
               lower_95_eur = unname(interval[1]),
               upper_95_eur = unname(interval[2]),
               fraction_below_fixed = mean(values < 0),
               mean_gain_vs_immediate_eur = mean(part$adaptive_eur - part$immediate_eur),
               mean_gap_to_oracle_eur = mean(part$oracle_eur - part$adaptive_eur),
               row.names = NULL)
  }))
  base <- gain[, "unchanged"]
  contrasts <- do.call(rbind, lapply(labels[labels != "unchanged"], function(label) {
    difference <- gain[, label] - base
    interval <- stats::quantile(boot_mean(difference), c(0.025, 0.975))
    data.frame(contrast = paste0(label, "_minus_unchanged"),
               mean_gain_difference_eur = mean(difference),
               lower_95_eur = unname(interval[1]),
               upper_95_eur = unname(interval[2]), row.names = NULL)
  }))
  list(by_shift = by_shift, contrasts = contrasts)
}
