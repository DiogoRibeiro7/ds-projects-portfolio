# Higher-order numerical reference for one-step adaptive grader uncertainty.
# Source model.R, sampling.R, calibration.R, adaptive_policy.R and
# adaptive_calibration_policy.R first. Uses base R only.

#' Gauss-Legendre nodes and weights on [0, 1] via the Jacobi eigenproblem.
#' For a smooth polynomial of degree <= 2*order-1 this rule is exact up to
#' floating-point rounding. Order is capped to bound memory and runtime.
unit_gauss_legendre_rule <- function(order = 64L) {
  assert_nonnegative_scalar(order, "order", integer = TRUE)
  if (order < 2L || order > 96L) {
    stop("Gauss-Legendre order must be between 2 and 96", call. = FALSE)
  }
  jacobi <- matrix(0, nrow = order, ncol = order)
  index <- seq_len(order - 1L)
  off_diagonal <- index / sqrt(4 * index^2 - 1)
  for (i in index) {
    jacobi[i, i + 1L] <- off_diagonal[i]
    jacobi[i + 1L, i] <- off_diagonal[i]
  }
  decomposition <- eigen(jacobi, symmetric = TRUE)
  arrangement <- base::order((decomposition$values + 1) / 2)
  data.frame(node = (decomposition$values[arrangement] + 1) / 2,
             weight = decomposition$vectors[1L, arrangement]^2)
}

#' Quadrature over the informative triangular grader-rate domain.
#'
#' Se=x, Sp=1-x+x*u, (x,u) in (0,1)^2, with Jacobian x. Integrate the
#' independent audit-updated Beta densities over this triangle and normalise.
#' Unlike a midpoint quantile grid, no discontinuous indicator is sampled.
#' The returned normalising constant estimates P(Se + Sp > 1 | audit) under
#' the *untruncated* independent Beta posterior.
gauss_grader_rate_reference <- function(calibration, decision_day,
                                        order = 64L) {
  counts <- inspection_calibration_counts(calibration, decision_day)
  rule <- unit_gauss_legendre_rule(order)
  positions <- expand.grid(x = rule$node, u = rule$node,
                           KEEP.OUT.ATTRS = FALSE)
  sensitivity <- positions$x
  specificity <- 1 - positions$x + positions$x * positions$u
  area_weights <- rep(rule$weight, times = order) *
    rep(rule$weight, each = order) * positions$x
  unnormalised <- area_weights *
    stats::dbeta(sensitivity, 1 + counts$true_positive,
                 1 + counts$false_negative) *
    stats::dbeta(specificity, 1 + counts$true_negative,
                 1 + counts$false_positive)
  normaliser <- sum(unnormalised)
  if (!is.finite(normaliser) || normaliser <= 0) {
    stop("numerical grader posterior has zero or invalid normalising mass",
         call. = FALSE)
  }
  nodes <- data.frame(sensitivity = sensitivity,
                      specificity = specificity,
                      weight = unnormalised / normaliser)
  list(nodes = nodes, orientation_probability = normaliser,
       order = order, reference = counts)
}

#' Marginalise finite quality-label outcomes over a joint grader/survival state.
#'
#' state_weights is n_rate x n_survival, normalised over all cells. Rates are
#' common to eligible cohorts within a latent state; their shared uncertainty
#' must not be integrated separately for each inspected cohort.
adaptive_joint_outcome_weights <- function(scenarios, state_weights,
                                           rate_nodes, inspection_day,
                                           checks_per_cohort) {
  validate_survival_scenarios(scenarios)
  assert_nonnegative_scalar(inspection_day, "inspection_day", integer = TRUE)
  assert_nonnegative_scalar(checks_per_cohort, "checks_per_cohort", integer = TRUE)
  n_rate <- nrow(rate_nodes)
  if (!is.data.frame(rate_nodes) ||
      !all(c("sensitivity", "specificity") %in% names(rate_nodes)) ||
      n_rate < 1L ||
      !all(vapply(rate_nodes[c("sensitivity", "specificity")],
                  function(x) is.numeric(x) && all(is.finite(x)), logical(1))) ||
      any(rate_nodes$sensitivity < 0 | rate_nodes$sensitivity > 1) ||
      any(rate_nodes$specificity < 0 | rate_nodes$specificity > 1) ||
      any(rate_nodes$sensitivity + rate_nodes$specificity <= 1) ||
      !is.matrix(state_weights) || !is.numeric(state_weights) ||
      !identical(dim(state_weights), c(n_rate, length(scenarios))) ||
      anyNA(state_weights) || any(!is.finite(state_weights)) ||
      any(state_weights < 0) || abs(sum(state_weights) - 1) > 1e-8 ||
      checks_per_cohort < 1L) {
    stop("outcome integration needs aligned normalised joint states",
         call. = FALSE)
  }
  groups <- scenarios[[1]]$cohort[scenarios[[1]]$initially_sellable]
  if ((checks_per_cohort + 1)^length(groups) > 5000L) {
    stop("outcome enumeration limited to 5000 joint counts", call. = FALSE)
  }
  counts <- as.matrix(expand.grid(
    rep(list(seq.int(0L, checks_per_cohort)), length(groups)),
    KEEP.OUT.ATTRS = FALSE
  ))
  colnames(counts) <- groups
  joint <- matrix(0, nrow(counts), length(scenarios),
                  dimnames = list(NULL, names(scenarios)))
  for (h in seq_along(scenarios)) {
    inventory <- scenarios[[h]]
    index <- match(groups, inventory$cohort)
    survival <- weibull_survival(rep(inspection_day, length(groups)),
                                 inventory$shape[index], inventory$scale[index])
    tables <- lapply(seq_along(groups), function(g) {
      q <- rate_nodes$sensitivity * survival[g] +
        (1 - rate_nodes$specificity) * (1 - survival[g])
      matrix(vapply(seq.int(0L, checks_per_cohort), function(k) {
        stats::dbinom(k, checks_per_cohort, q)
      }, numeric(n_rate)), nrow = n_rate)
    })
    for (o in seq_len(nrow(counts))) {
      likelihood <- rep(1, n_rate)
      for (g in seq_along(groups)) {
        likelihood <- likelihood * tables[[g]][, counts[o, g] + 1L]
      }
      joint[o, h] <- sum(state_weights[, h] * likelihood)
    }
  }
  list(counts = counts, weights = joint, probability = rowSums(joint))
}

#' High-order reference policy without enumerating thousands of explicit
#' scenario copies. The rate quadrature is performed before each quality-count
#' outcome's posterior sale optimisation. No future observations are passed.
plan_adaptive_gauss_reference <- function(
    scenarios, prior, observations, calibration, prices,
    storage_eur_per_kg_day, decision_day, inspection_day,
    order = 64L, checks_per_cohort = 8L,
    eur_per_check = 0.02, delivery_lag_days = 2L) {
  weights <- normalize_scenario_weights(scenarios, prior)
  validate_prices(prices, scenarios[[1]])
  for (value in list(decision_day, inspection_day, checks_per_cohort,
                     delivery_lag_days)) {
    assert_nonnegative_scalar(value, "decision/inspection/checks/lag", integer = TRUE)
  }
  assert_nonnegative_scalar(storage_eur_per_kg_day, "storage_eur_per_kg_day")
  assert_nonnegative_scalar(eur_per_check, "eur_per_check")
  days <- sort(unique(prices$day))
  if (inspection_day <= decision_day || !(decision_day %in% days) ||
      !(inspection_day %in% days) || checks_per_cohort < 1L) {
    stop("reference needs later inspection and available sale dates", call. = FALSE)
  }
  if (!is.null(observations)) {
    fields <- c("cohort", "day", "n_checked", "n_acceptable")
    if (!is.data.frame(observations) || !setequal(names(observations), fields)) {
      stop("observations must contain only aggregate inspection counts",
           call. = FALSE)
    }
    validate_inspection_counts(observations, scenarios[[1]])
    if (any(observations$day > decision_day)) {
      stop("future labels cannot update the reference posterior", call. = FALSE)
    }
  }

  reference <- gauss_grader_rate_reference(calibration, decision_day, order)
  nodes <- reference$nodes
  n_rate <- nrow(nodes)
  n_hypotheses <- length(scenarios)
  log_joint <- matrix(-Inf, n_rate, n_hypotheses)
  for (h in seq_along(scenarios)) {
    log_likelihood <- rep(0, n_rate)
    if (!is.null(observations)) {
      inventory <- scenarios[[h]]
      index <- match(observations$cohort, inventory$cohort)
      survival <- weibull_survival(observations$day,
                                   inventory$shape[index], inventory$scale[index])
      for (j in seq_len(nrow(observations))) {
        q <- nodes$sensitivity * survival[j] +
          (1 - nodes$specificity) * (1 - survival[j])
        log_likelihood <- log_likelihood + stats::dbinom(
          observations$n_acceptable[j], observations$n_checked[j], q,
          log = TRUE
        )
      }
    }
    log_joint[, h] <- log(weights[h]) + log(nodes$weight) + log_likelihood
  }
  offset <- max(log_joint)
  if (!is.finite(offset)) {
    stop("inspection counts impossible under all reference states", call. = FALSE)
  }
  joint_posterior <- exp(log_joint - offset)
  joint_posterior <- joint_posterior / sum(joint_posterior)
  marginal <- stats::setNames(colSums(joint_posterior), names(scenarios))

  days <- days[days >= decision_day]
  values <- do.call(rbind, lapply(scenarios, function(inventory) {
    expected_value_grid(inventory, prices, storage_eur_per_kg_day,
                        decision_day, delivery_lag_days)$expected_net_eur
  }))
  dimnames(values) <- list(names(scenarios), as.character(days))
  fixed_values <- as.numeric(marginal %*% values)
  fixed_index <- which.max(fixed_values)
  outcomes <- adaptive_joint_outcome_weights(
    scenarios, joint_posterior, nodes, inspection_day, checks_per_cohort
  )
  future_index <- which(days >= inspection_day)
  future_days <- days[future_index]
  future_values <- values[, future_index, drop = FALSE]
  outcome_values <- outcomes$weights %*% future_values
  chosen_index <- max.col(outcome_values, ties.method = "first")
  followup_days <- future_days[chosen_index]
  before_labour <- sum(outcome_values[cbind(seq_len(nrow(outcome_values)),
                                            chosen_index)])
  n_groups <- sum(scenarios[[1]]$initially_sellable)
  labour <- n_groups * checks_per_cohort * eur_per_check
  inspection_value <- before_labour - labour
  fixed_value <- fixed_values[fixed_index]
  list(
    action = if (inspection_value > fixed_value + 1e-9) "inspect" else "commit",
    fixed_day = days[fixed_index], fixed_expected_eur = fixed_value,
    inspection_expected_eur = inspection_value,
    net_information_value_eur = inspection_value - fixed_value,
    inspection_labour_eur = labour,
    decision_day = decision_day, inspection_day = inspection_day,
    checks_per_cohort = checks_per_cohort, order = order,
    scenarios = scenarios, posterior = marginal, state_posterior = joint_posterior,
    reference = reference, outcomes = outcomes, followup_days = followup_days,
    days = days, values = values, future_days = future_days,
    future_values = future_values
  )
}

#' Score a finite-grid contingent policy under the numerical reference model.
#'
#' The grader/survival posterior of the finite plan determines its action and
#' its sale day for every *possible* inspection count. The higher-order
#' posterior then values that frozen mapping, with no oracle access at choice
#' time. The resulting regret is conditional expected-value loss, NOT realised
#' lot profit or an empirical regret bound.
score_finite_adaptive_under_reference <- function(finite_result, reference_plan) {
  if (!is.list(finite_result) || is.null(finite_result$integrated) ||
      is.null(finite_result$reference_rate_grid) ||
      !is.list(reference_plan) || is.null(reference_plan$outcomes)) {
    stop("need finite and numerical reference plans", call. = FALSE)
  }
  finite <- finite_result$integrated
  nodes <- finite_result$reference_rate_grid$nodes
  n_rate <- nrow(nodes)
  n_hypotheses <- length(reference_plan$scenarios)
  if (length(finite$posterior) != n_rate * n_hypotheses ||
      finite$inspection_day != reference_plan$inspection_day ||
      finite$checks_per_cohort != reference_plan$checks_per_cohort ||
      !identical(finite$days, reference_plan$days)) {
    stop("finite and reference plans must describe the same decision", call. = FALSE)
  }
  finite_state <- matrix(unname(finite$posterior),
                         nrow = n_rate, ncol = n_hypotheses)
  finite_outcomes <- adaptive_joint_outcome_weights(
    reference_plan$scenarios, finite_state, nodes,
    finite$inspection_day, finite$checks_per_cohort
  )
  if (!identical(finite_outcomes$counts, reference_plan$outcomes$counts)) {
    stop("outcome enumerations must use identical joint count order",
         call. = FALSE)
  }
  finite_values <- finite_outcomes$weights %*% reference_plan$future_values
  finite_followup <- reference_plan$future_days[
    max.col(finite_values, ties.method = "first")
  ]
  reference_scores <- reference_plan$outcomes$weights %*%
    reference_plan$future_values
  if (finite$action == "inspect") {
    chosen <- match(finite_followup, reference_plan$future_days)
    evaluated <- sum(reference_scores[cbind(seq_len(length(chosen)), chosen)]) -
      reference_plan$inspection_labour_eur
  } else {
    fixed_index <- match(finite$fixed_day, reference_plan$days)
    if (is.na(fixed_index)) {
      stop("finite fixed sale day absent from reference prices", call. = FALSE)
    }
    evaluated <- sum(reference_plan$posterior *
                       reference_plan$values[, fixed_index])
  }
  optimal <- if (reference_plan$action == "inspect") {
    reference_plan$inspection_expected_eur
  } else reference_plan$fixed_expected_eur
  finite_marginal <- colSums(finite_state)
  data.frame(
    grid_size = as.integer(round(sqrt(n_rate +
                                     finite_result$reference_rate_grid$n_discarded))),
    finite_action = finite$action,
    reference_action = reference_plan$action,
    finite_fixed_day = finite$fixed_day,
    reference_fixed_day = reference_plan$fixed_day,
    scenario_posterior_total_variation = 0.5 * sum(abs(
      finite_marginal - unname(reference_plan$posterior))),
    conditional_day_disagreement_mass = sum(
      reference_plan$outcomes$probability *
        (finite_followup != reference_plan$followup_days)
    ),
    absolute_information_value_error_eur = abs(
      finite$net_information_value_eur -
        reference_plan$net_information_value_eur
    ),
    finite_expected_eur_under_reference = evaluated,
    reference_optimal_expected_eur = optimal,
    expected_numerical_regret_eur = optimal - evaluated,
    stringsAsFactors = FALSE
  )
}
