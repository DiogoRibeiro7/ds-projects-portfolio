# Offline proof checks for inverted, hypothetical SIMA quote thresholds.
source("R/sima_market_prices.R")
source("R/sima_rolling_backtest.R")
source("R/sima_composition_audit.R")
source("R/sima_forward_quote_windows.R")
source("R/sima_quote_break_even.R")
source("R/sima_quote_feasibility_frontier.R")

expect_error <- function(expression) {
  stopifnot(tryCatch({ force(expression); FALSE }, error = function(e) TRUE))
}

# Flat prices: without friction, waiting breaks even but has no strictly
# positive cost or fruit-loss capacity. The last source week is censored.
fixture <- data.frame(
  quote_date = as.Date("2026-01-05") + 7L * 0:11,
  product_descriptor = "Laranja*Newhall*II*Caixa*Algarve*EUR/kg",
  region = "Beira Litoral", market = "MAC", market_stage = "wholesale",
  price_unit = "EUR/kg", min_eur_per_kg = rep(0.9, 12),
  max_eur_per_kg = rep(1.1, 12), frequent_eur_per_kg = rep(1, 12),
  is_provisional = rep(NA, 12), stringsAsFactors = FALSE
)
windows <- pair_sima_forward_quotes(sima_weekly_panels(fixture),
                                     horizons = 1:3)
result <- evaluate_sima_quote_feasibility_frontier(
  windows, weekly_retention = c(1, 0.99),
  weekly_cost_eur_per_initial_kg = c(0, 0.01)
)
stopifnot(nrow(result$cost_ceiling_pairs) == 2L * (10L + 9L + 8L),
          nrow(result$retention_floor_pairs) == 2L * (10L + 9L + 8L),
          nrow(result$cost_ceiling_balanced) == 6L,
          nrow(result$retention_floor_balanced) == 6L,
          all(result$cost_ceiling_balanced$observed_pairs == 8L),
          all(result$retention_floor_balanced$observed_pairs == 8L))

perfect <- result$cost_ceiling_pairs[
  result$cost_ceiling_pairs$assumed_weekly_retention == 1, , drop = FALSE
]
decay <- result$cost_ceiling_pairs[
  result$cost_ceiling_pairs$assumed_weekly_retention == 0.99, , drop = FALSE
]
stopifnot(all(abs(perfect$raw_maximum_weekly_cost_eur_per_initial_kg) < 1e-12),
          all(perfect$feasible_at_zero_cost),
          !any(perfect$positive_cost_room),
          all(decay$raw_maximum_weekly_cost_eur_per_initial_kg < 0),
          !any(decay$feasible_at_zero_cost))

free <- result$retention_floor_pairs[
  result$retention_floor_pairs$assumed_weekly_cost_eur_per_initial_kg == 0,
  , drop = FALSE
]
costly <- result$retention_floor_pairs[
  result$retention_floor_pairs$assumed_weekly_cost_eur_per_initial_kg == 0.01,
  , drop = FALSE
]
stopifnot(all(abs(free$required_weekly_retention - 1) < 1e-12),
          all(free$feasible_at_full_retention),
          !any(free$positive_loss_room),
          all(costly$required_weekly_retention > 1),
          all(costly$raw_maximum_weekly_loss_fraction < 0),
          !any(costly$feasible_at_full_retention))

# Positive price movement admits a strictly positive cost ceiling and
# tolerable mass loss, both calculated from the same actual quoted pair.
rising <- fixture
rising$frequent_eur_per_kg <- 1 + 0.02 * 0:11
rising$min_eur_per_kg <- rising$frequent_eur_per_kg - 0.1
rising$max_eur_per_kg <- rising$frequent_eur_per_kg + 0.1
rising_windows <- pair_sima_forward_quotes(sima_weekly_panels(rising),
                                            horizons = 1:3)
up <- evaluate_sima_quote_feasibility_frontier(
  rising_windows, weekly_retention = 1,
  weekly_cost_eur_per_initial_kg = 0
)
up_cost <- up$cost_ceiling_pairs[
  up$cost_ceiling_pairs$horizon_weeks == 1L, , drop = FALSE
]
up_loss <- up$retention_floor_pairs[
  up$retention_floor_pairs$horizon_weeks == 1L, , drop = FALSE
]
stopifnot(all(abs(up_cost$raw_maximum_weekly_cost_eur_per_initial_kg -
                  0.02) < 1e-10),
          all(up_cost$positive_cost_room),
          all(up_loss$required_weekly_retention < 1),
          all(up_loss$positive_loss_room))

# Missing exact endpoint weeks stay unavailable rather than becoming a
# zero future quotation, a sale or a fabricated profitable observation.
missing <- fixture[
  fixture$quote_date != as.Date("2026-01-26"), , drop = FALSE
]
gap <- evaluate_sima_quote_feasibility_frontier(
  pair_sima_forward_quotes(sima_weekly_panels(missing), horizons = 1:3),
  weekly_retention = c(1, 0.99),
  weekly_cost_eur_per_initial_kg = c(0, 0.01)
)
stopifnot(nrow(gap$cost_ceiling_pairs) < nrow(result$cost_ceiling_pairs),
          all(is.finite(
            gap$cost_ceiling_pairs$raw_maximum_weekly_cost_eur_per_initial_kg
          )),
          all(is.finite(gap$retention_floor_pairs$required_weekly_retention)))

# Delegated grid validation and required date provenance.
expect_error(evaluate_sima_quote_feasibility_frontier(
  windows, weekly_retention = 0
))
expect_error(evaluate_sima_quote_feasibility_frontier(
  windows, weekly_cost_eur_per_initial_kg = -0.01
))
expect_error(evaluate_sima_quote_feasibility_frontier(
  windows, weekly_retention = c(1, 1)
))
expect_error(evaluate_sima_quote_feasibility_frontier(
  windows, tolerance = Inf
))
expect_error(evaluate_sima_quote_feasibility_frontier(
  windows[, names(windows) != "endpoint_quote_date"]
))

# Archive regression: exact coverage, common 343-anchor denominator and
# equivalence to the prior 16-scenario grid are all checked end to end.
raw <- read_sima_orange_prices(
  "data/raw/sima_oranges_wholesale_2025-01-01_2026-10-31.csv.gz",
  market_stage = "wholesale"
)
observed_windows <- pair_sima_forward_quotes(sima_weekly_panels(raw))
empirical <- evaluate_sima_quote_feasibility_frontier(observed_windows)
expected_pairs <- sum(c(433L, 403L, 373L, 343L))
stopifnot(nrow(empirical$cost_ceiling_pairs) == expected_pairs * 4L,
          nrow(empirical$retention_floor_pairs) == expected_pairs * 4L,
          nrow(empirical$cost_ceiling_balanced) == 16L,
          nrow(empirical$retention_floor_balanced) == 16L,
          all(empirical$cost_ceiling_balanced$observed_pairs == 343L),
          all(empirical$retention_floor_balanced$observed_pairs == 343L),
          all(empirical$cost_ceiling_pairs$anchor_eur_per_kg > 0),
          all(empirical$retention_floor_pairs$endpoint_eur_per_kg > 0),
          all(is.finite(empirical$cost_ceiling_pairs$
                          raw_maximum_weekly_cost_eur_per_initial_kg)),
          all(is.finite(empirical$retention_floor_pairs$
                          required_weekly_retention)))

# Substituting the exact inverted boundary into the direct accounting
# expression must yield equality, even when the raw threshold is negative
# or implies physically impossible retention above one.
cost_rows <- empirical$cost_ceiling_pairs
retention_rows <- empirical$retention_floor_pairs
stopifnot(
  max(abs(cost_rows$retained_saleable_share *
            cost_rows$endpoint_eur_per_kg -
            cost_rows$horizon_weeks *
            cost_rows$raw_maximum_weekly_cost_eur_per_initial_kg -
            cost_rows$anchor_eur_per_kg)) < 1e-10,
  max(abs(retention_rows$required_weekly_retention ^
            retention_rows$horizon_weeks *
            retention_rows$endpoint_eur_per_kg -
            retention_rows$horizon_weeks *
            retention_rows$assumed_weekly_cost_eur_per_initial_kg -
            retention_rows$anchor_eur_per_kg)) < 1e-10,
  inherits(cost_rows$anchor_week_start, "Date"),
  inherits(retention_rows$endpoint_quote_date, "Date")
)

# Increasing assumed fruit retention can only increase the tolerable cost.
# Increasing hypothetical cost can only increase the required retention.
for (pair_id in unique(cost_rows$observed_pair_id)) {
  costs_for_pair <- cost_rows[
    cost_rows$observed_pair_id == pair_id, , drop = FALSE
  ]
  costs_for_pair <- costs_for_pair[
    order(costs_for_pair$assumed_weekly_retention), , drop = FALSE
  ]
  requirements_for_pair <- retention_rows[
    retention_rows$observed_pair_id == pair_id, , drop = FALSE
  ]
  requirements_for_pair <- requirements_for_pair[
    order(requirements_for_pair$assumed_weekly_cost_eur_per_initial_kg),
    , drop = FALSE
  ]
  stopifnot(all(diff(
    costs_for_pair$raw_maximum_weekly_cost_eur_per_initial_kg
  ) >= -1e-12),
  all(diff(requirements_for_pair$required_weekly_retention) >= -1e-12))
}

# Every market-specific breakdown must partition the common comparison
# population at the same horizon and hypothetical setting.
for (h in 1:4) {
  for (rho in c(1, 0.99, 0.98, 0.95)) {
    x <- empirical$cost_ceiling_market_balanced[
      empirical$cost_ceiling_market_balanced$horizon_weeks == h &
        empirical$cost_ceiling_market_balanced$assumed_weekly_retention == rho,
      , drop = FALSE
    ]
    stopifnot(sum(x$observed_pairs) == 343L)
  }
  for (cost in c(0, 0.005, 0.01, 0.02)) {
    x <- empirical$retention_floor_market_balanced[
      empirical$retention_floor_market_balanced$horizon_weeks == h &
        empirical$retention_floor_market_balanced$
          assumed_weekly_cost_eur_per_initial_kg == cost,
      , drop = FALSE
    ]
    stopifnot(sum(x$observed_pairs) == 343L)
  }
}

# At rho = 1 and c = 0 the strictly-positive room is precisely the
# observed proportion of *higher* quotes, not the larger share unchanged.
ref <- summarize_sima_forward_windows(observed_windows)$balanced
cost_zero <- empirical$cost_ceiling_balanced[
  empirical$cost_ceiling_balanced$assumed_weekly_retention == 1,
  , drop = FALSE
]
retention_zero <- empirical$retention_floor_balanced[
  empirical$retention_floor_balanced$
    assumed_weekly_cost_eur_per_initial_kg == 0,
  , drop = FALSE
]
cost_zero <- cost_zero[order(cost_zero$horizon_weeks), , drop = FALSE]
retention_zero <- retention_zero[
  order(retention_zero$horizon_weeks), , drop = FALSE
]
stopifnot(all(abs(cost_zero$fraction_with_positive_cost_room -
                  ref$share_price_higher) < 1e-10),
          all(abs(retention_zero$fraction_with_positive_loss_room -
                    ref$share_price_higher) < 1e-10),
          all(abs(cost_zero$fraction_feasible_at_zero_cost -
                    (1 - ref$share_price_lower)) < 1e-10))
cat("SIMA continuous quote feasibility frontier tests passed.\n")
