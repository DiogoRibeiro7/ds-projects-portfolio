# Reproduce the published empirical synthesis from one frozen SIMA source.
# Run from projects/orange_storage_decision; base R only. No synthetic
# decision-policy results or measured postharvest costs enter this test.
source("R/sima_market_prices.R")
source("R/sima_rolling_backtest.R")
source("R/sima_composition_audit.R")
source("R/sima_forward_quote_windows.R")
source("R/sima_quote_break_even.R")
source("R/sima_quote_feasibility_frontier.R")
source("R/sima_missing_quote_bounds.R")

#' Get exactly one identified horizon/scenario row.
#' @param rows A model summary containing the documented grid fields.
#' @param horizon Whole calendar weeks ahead.
#' @param retention Assumed weekly saleable mass fraction.
#' @param cost Assumed weekly EUR cost per initial kilogram.
#' @param cohort Optional missing-outcome denominator definition.
#' @return One unambiguous data-frame row, retaining source column types.
one_sima_scenario <- function(rows, horizon, retention, cost, cohort = NULL) {
  stopifnot(is.data.frame(rows), is.numeric(horizon), length(horizon) == 1L,
            is.finite(horizon), horizon == floor(horizon), horizon >= 1,
            is.numeric(retention), length(retention) == 1L,
            is.finite(retention), retention > 0, retention <= 1,
            is.numeric(cost), length(cost) == 1L, is.finite(cost), cost >= 0)
  selected <- rows[
    rows$horizon_weeks == horizon &
      abs(rows$weekly_retention - retention) < 1e-12 &
      abs(rows$weekly_cost_eur_per_initial_kg - cost) < 1e-12,
    , drop = FALSE
  ]
  if (!is.null(cohort)) {
    stopifnot(is.character(cohort), length(cohort) == 1L,
              !is.na(cohort), "cohort" %in% names(selected))
    selected <- selected[selected$cohort == cohort, , drop = FALSE]
  }
  stopifnot(nrow(selected) == 1L)
  selected
}

#' Assert one-decimal-point percentages printed in the evidence synthesis.
#' @param value A single finite fraction from the recomputed source data.
#' @param percentage Expected displayed percent, rounded to one decimal.
check_percent <- function(value, percentage) {
  stopifnot(is.numeric(value), length(value) == 1L,
            is.finite(value), value >= 0, value <= 1,
            is.numeric(percentage), length(percentage) == 1L,
            is.finite(percentage), percentage >= 0, percentage <= 100,
            identical(round(100 * value, 1), as.numeric(percentage)))
  invisible(TRUE)
}

source_path <- "data/raw/sima_oranges_wholesale_2025-01-01_2026-10-31.csv.gz"
stopifnot(file.exists(source_path), file.exists("EMPIRICAL_SYNTHESIS.md"))
raw <- read_sima_orange_prices(source_path, market_stage = "wholesale")
stopifnot(nrow(raw) == 3412L)

windows <- pair_sima_forward_quotes(sima_weekly_panels(raw))
forward <- summarize_sima_forward_windows(windows)
direct <- evaluate_sima_quote_break_even(windows)
frontiers <- evaluate_sima_quote_feasibility_frontier(windows)
missing <- bound_sima_missing_forward_quotes(windows)

# Every 2026 anchor remains in the source population at every horizon.
# Distinguish time censorship from the absence of an exact later quote.
expected_horizons <- 1:4
coverage <- forward$coverage[order(forward$coverage$horizon_weeks), ]
stopifnot(identical(as.integer(coverage$horizon_weeks), expected_horizons),
          nrow(windows) == 485L * 4L,
          all(coverage$observed_anchor_quotes == 485L),
          identical(as.integer(coverage$complete_horizon_eligible),
                    c(453L, 442L, 431L, 420L)),
          identical(as.integer(coverage$exact_endpoint_observed),
                    c(433L, 403L, 373L, 343L)),
          all(coverage$balanced_all_horizon_anchors == 343L),
          all(forward$balanced$observed_pairs == 343L),
          all(!windows$endpoint_observed | windows$horizon_eligible),
          all(is.na(windows$endpoint_eur_per_kg) ==
                !windows$endpoint_observed))

eligible <- missing$coverage[
  missing$coverage$cohort == "horizon_eligible", , drop = FALSE
]
mature <- missing$coverage[
  missing$coverage$cohort == "common_maturity", , drop = FALSE
]
eligible <- eligible[order(eligible$horizon_weeks), ]
mature <- mature[order(mature$horizon_weeks), ]
stopifnot(identical(as.integer(eligible$eligible_anchors),
                    c(453L, 442L, 431L, 420L)),
          identical(as.integer(eligible$missing_eligible_endpoints),
                    c(20L, 39L, 58L, 77L)),
          all(mature$eligible_anchors == 420L),
          identical(as.integer(mature$observed_endpoints),
                    c(401L, 382L, 363L, 343L)),
          nrow(missing$missing_eligible_windows) == 194L)

# Quotation-only descriptive results on the ex-post 343-anchor cohort.
# An unchanged quote is weak break-even with zero assumed friction, but is
# never a strictly positive economic advantage.
matched <- forward$balanced[order(forward$balanced$horizon_weeks), ]
stopifnot(all(abs(matched$median_signed_change_eur_per_kg) < 1e-12),
          all(abs(matched$mean_signed_change_eur_per_kg -
                    c(0.0036, 0.0073, 0.0101, 0.0117)) < 0.00005))
for (i in seq_along(expected_horizons)) {
  check_percent(matched$share_price_higher[i],
                c(11.4, 22.7, 30.9, 37.3)[i])
  check_percent(matched$share_price_lower[i],
                c(7.0, 13.4, 17.8, 22.2)[i])
  free <- one_sima_scenario(direct$balanced, i, 1, 0)
  stopifnot(free$observed_pairs == 343L,
            abs(free$fraction_strictly_better -
                  matched$share_price_higher[i]) < 1e-12)
}
# Audit every percentage in the synthesis scenario table, including the
# distinction between nonnegative and strictly positive quote arithmetic.
scenarios <- data.frame(
  horizon = rep(c(1L, 4L), times = 4L),
  retention = rep(c(1, 0.99, 0.98, 0.95), each = 2L),
  cost = rep(c(0, 0.005, 0.010, 0.020), each = 2L),
  weak = c(93.0, 77.8, 11.4, 29.7, 11.4, 12.5, 4.7, 0.0),
  strict = c(11.4, 37.3, 11.4, 29.7, 11.4, 12.5, 4.7, 0.0)
)
for (i in seq_len(nrow(scenarios))) {
  scenario <- scenarios[i, , drop = FALSE]
  row <- one_sima_scenario(
    direct$balanced, scenario$horizon, scenario$retention, scenario$cost
  )
  check_percent(row$fraction_break_even_or_better, scenario$weak)
  check_percent(row$fraction_strictly_better, scenario$strict)
}

# Negative ceilings and impossible retention requirements are raw algebraic
# properties of the recorded prices, not estimated actual physical costs.
cost_ceiling <- frontiers$cost_ceiling_balanced[
  frontiers$cost_ceiling_balanced$horizon_weeks == 1 &
    frontiers$cost_ceiling_balanced$assumed_weekly_retention == 0.99,
  , drop = FALSE
]
retention_floor <- frontiers$retention_floor_balanced[
  frontiers$retention_floor_balanced$horizon_weeks == 1 &
    frontiers$retention_floor_balanced$assumed_weekly_cost_eur_per_initial_kg == 0.005,
  , drop = FALSE
]
stopifnot(nrow(cost_ceiling) == 1L, nrow(retention_floor) == 1L,
          abs(cost_ceiling$median_raw_ceiling_eur_per_initial_kg_week +
                0.007) < 0.000005,
          abs(retention_floor$median_required_weekly_retention -
                1.0067) < 0.00005)

# No missing quote is imputed. The reported bounds are exact binary
# completions on time-eligible starts, not confidence intervals.
week1 <- one_sima_scenario(missing$bounds, 1, 1, 0, "horizon_eligible")
week4 <- one_sima_scenario(missing$bounds, 4, 1, 0, "horizon_eligible")
loss4 <- one_sima_scenario(missing$bounds, 4, 0.99, 0.005,
                           "horizon_eligible")
check_percent(week1$lower_break_even_fraction, 88.7)
check_percent(week1$upper_break_even_fraction, 93.2)
check_percent(week1$observed_only_break_even_fraction, 92.8)
check_percent(week1$observed_only_strict_improvement_fraction, 10.6)
check_percent(week1$lower_strict_improvement_fraction, 10.2)
check_percent(week1$upper_strict_improvement_fraction, 14.6)
check_percent(week4$lower_break_even_fraction, 63.6)
check_percent(week4$upper_break_even_fraction, 81.9)
check_percent(week4$lower_strict_improvement_fraction, 30.5)
check_percent(week4$upper_strict_improvement_fraction, 48.8)
check_percent(loss4$lower_break_even_fraction, 24.3)
check_percent(loss4$upper_break_even_fraction, 42.6)
stopifnot(week4$missing_eligible_endpoints == 77L,
          week4$eligible_anchors == 420L,
          abs((week4$upper_break_even_fraction -
                 week4$lower_break_even_fraction) - 77 / 420) < 1e-12)

cat("Orange empirical quotation closure and headline audit passed.\n")
