#' Estimate stratum-specific commercial survival from inspection samples.
#'
#' Source R/model.R before loading this file. Inspections are independent
#' cross-sectional samples from each initially eligible cohort, not repeated
#' observations of a fixed panel of identified fruits.

#' Validate inspections and require enough follow-up to identify a curve.
#' @param inspections Data frame: cohort, day, n_checked, n_acceptable.
#' @param cohorts Initial inventory. Only initially saleable cohorts qualify.
validate_inspections <- function(inspections, cohorts) {
  validate_cohorts(cohorts)
  required <- c("cohort", "day", "n_checked", "n_acceptable")
  if (!is.data.frame(inspections) || !all(required %in% names(inspections)) ||
      nrow(inspections) == 0L || !is.character(inspections$cohort) ||
      anyNA(inspections$cohort) || anyDuplicated(inspections[c("cohort", "day")])) {
    stop("inspections must have unique cohort/day rows and required columns",
         call. = FALSE)
  }
  for (column in c("day", "n_checked", "n_acceptable")) {
    values <- inspections[[column]]
    if (!is.numeric(values) || anyNA(values) || any(!is.finite(values)) ||
        any(values != floor(values)) || any(values < 0)) {
      stop(sprintf("%s must contain nonnegative whole numbers", column),
           call. = FALSE)
    }
  }
  if (any(inspections$day == 0) || any(inspections$n_checked == 0) ||
      any(inspections$n_acceptable > inspections$n_checked)) {
    stop("inspections need positive days and sample sizes, with valid counts",
         call. = FALSE)
  }
  groups <- cohorts$cohort[cohorts$initially_sellable]
  if (!setequal(unique(inspections$cohort), groups)) {
    stop("inspections must cover exactly the saleable cohorts", call. = FALSE)
  }
  for (group in groups) {
    part <- inspections[inspections$cohort == group, , drop = FALSE]
    if (length(unique(part$day)) < 3L ||
        sum(part$n_acceptable) == 0 ||
        sum(part$n_checked - part$n_acceptable) == 0) {
      stop(sprintf("insufficient survival information for cohort %s", group),
           call. = FALSE)
    }
  }
  invisible(TRUE)
}

#' Maximum-likelihood Weibull fits from cross-sectional binomial counts.
#'
#' Each observation is Binomial(n_checked, S(day)). A cohort is assumed
#' commercially acceptable at harvest; its membership cannot be inferred
#' using future measurements. Missing-at-random sampling and independence
#' within each inspection are modelling assumptions, not established facts.
#'
#' @param inspections Valid aggregated independent inspections.
#' @param cohorts Initial inventory with maturity strata.
#' @return Data frame of cohort, shape, scale, negative_log_likelihood.
fit_weibull_by_cohort <- function(inspections, cohorts) {
  validate_inspections(inspections, cohorts)
  groups <- cohorts$cohort[cohorts$initially_sellable]
  fits <- lapply(groups, function(group) {
    observations <- inspections[inspections$cohort == group, , drop = FALSE]
    negative_log_likelihood <- function(log_parameters) {
      probability <- weibull_survival(
        observations$day, exp(log_parameters[1]), exp(log_parameters[2])
      )
      # Clip only for stable evaluation of extreme binomial log-likelihoods.
      probability <- pmin(1 - 1e-12, pmax(1e-12, probability))
      -sum(stats::dbinom(observations$n_acceptable,
                         size = observations$n_checked,
                         prob = probability, log = TRUE))
    }
    result <- stats::optim(
      par = log(c(2, stats::median(observations$day))),
      fn = negative_log_likelihood,
      method = "L-BFGS-B",
      lower = log(c(0.25, min(observations$day) / 100)),
      upper = log(c(10, max(observations$day) * 100))
    )
    if (result$convergence != 0 || !is.finite(result$value)) {
      stop(sprintf("Weibull likelihood did not converge for %s", group),
           call. = FALSE)
    }
    data.frame(cohort = group, shape = exp(result$par[1]),
               scale = exp(result$par[2]),
               negative_log_likelihood = result$value)
  })
  do.call(rbind, fits)
}

#' Replace prospective survival assumptions with fitted parameters.
#' @param cohorts Initial inventory (including ineligible groups).
#' @param fits Output of fit_weibull_by_cohort().
#' @return Validated inventory with the updated survival parameters.
with_fitted_survival <- function(cohorts, fits) {
  validate_cohorts(cohorts)
  if (!is.data.frame(fits) ||
      !all(c("cohort", "shape", "scale") %in% names(fits)) ||
      !is.character(fits$cohort) || anyNA(fits$cohort) ||
      anyDuplicated(fits$cohort)) {
    stop("fits must provide unique cohort, shape and scale values",
         call. = FALSE)
  }
  eligible <- which(cohorts$initially_sellable)
  if (!setequal(fits$cohort, cohorts$cohort[eligible])) {
    stop("fitted cohort labels do not match eligible inventory", call. = FALSE)
  }
  updated <- cohorts
  position <- match(updated$cohort[eligible], fits$cohort)
  updated$shape[eligible] <- fits$shape[position]
  updated$scale[eligible] <- fits$scale[position]
  validate_cohorts(updated)
  updated
}

#' Parametric bootstrap of the optimal whole-batch selling day.
#'
#' Counts are resampled from the fitted binomial inspection model and both
#' survival parameters and selling decisions are re-estimated. This measures
#' inspection-parameter uncertainty conditional on the Weibull model and on
#' the supplied price scenario; it does not model price or weather uncertainty.
#'
#' @param inspections Inspections observed no later than decision_day.
#' @param cohorts Inventory from which inspections were sampled.
#' @param prices Price scenarios for all initially saleable cohorts.
#' @param storage_eur_per_kg_day Per-kg daily holding cost.
#' @param decision_day Information cut-off and earliest sale day.
#' @param delivery_lag_days Days until quality is assessed at delivery.
#' @param n_boot Number of bootstrap attempts; caller controls the RNG seed.
#' @return List with decisions data frame, attempted, and failed counts.
bootstrap_sale_days <- function(inspections, cohorts, prices,
                                storage_eur_per_kg_day, decision_day,
                                delivery_lag_days = 0, n_boot = 100L) {
  assert_nonnegative_scalar(decision_day, "decision_day", integer = TRUE)
  assert_nonnegative_scalar(n_boot, "n_boot", integer = TRUE)
  if (n_boot < 1L || !is.numeric(inspections$day) ||
      anyNA(inspections$day) || any(inspections$day > decision_day)) {
    stop("bootstrap needs positive n_boot and no future inspections",
         call. = FALSE)
  }
  fits <- fit_weibull_by_cohort(inspections, cohorts)
  position <- match(inspections$cohort, fits$cohort)
  probability <- weibull_survival(inspections$day,
                                  fits$shape[position], fits$scale[position])

  results <- vector("list", n_boot)
  for (iteration in seq_len(n_boot)) {
    simulated <- inspections
    simulated$n_acceptable <- stats::rbinom(
      nrow(inspections), size = inspections$n_checked, prob = probability
    )
    # Report failed refits, rather than silently interpreting them as success.
    results[[iteration]] <- tryCatch({
      refitted <- fit_weibull_by_cohort(simulated, cohorts)
      selected <- optimal_batch_sale(
        with_fitted_survival(cohorts, refitted), prices,
        storage_eur_per_kg_day, decision_day, delivery_lag_days
      )
      data.frame(iteration = iteration, sale_day = selected$sale_day,
                 expected_net_eur = selected$expected_net_eur)
    }, error = function(error) NULL)
  }
  successful <- Filter(Negate(is.null), results)
  if (length(successful) == 0L) {
    stop("all bootstrap refits failed", call. = FALSE)
  }
  list(decisions = do.call(rbind, successful),
       attempted = n_boot, failed = n_boot - length(successful))
}
