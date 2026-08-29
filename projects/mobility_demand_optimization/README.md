# Mobility Demand & Fleet Optimization

A decision-science case study connecting probabilistic spatiotemporal demand forecasting to constrained fleet allocation.

## Research question

> Does a forecast that is statistically better also produce better operational decisions?

The project evaluates forecasting models through two linked objectives:

1. **predictive quality** — point accuracy, probabilistic calibration and tail behaviour;
2. **decision quality** — relocation cost, unmet demand, idle capacity and total realised operating cost.

The primary dataset is NYC Taxi & Limousine Commission Yellow Taxi trip records aggregated to a zone-by-time demand panel.

## Decision pipeline

```text
TLC trip records
    -> zone/time demand panel
    -> rolling-origin features
    -> probabilistic demand forecasts
    -> constrained fleet allocation
    -> realised demand
    -> operational regret and cost
```

## Frozen first-stage data design

The first empirical study is prospectively fixed before model fitting:

- source: official NYC TLC Yellow Taxi monthly parquet files;
- source months: January 2025 through May 2026;
- demand unit: pickup count per TLC zone and civil clock hour;
- analysis zones: top 30 geographic pickup zones selected using training data only;
- training: 2025-01-01 through 2025-12-31;
- validation/model selection: 2026-01-01 through 2026-02-28;
- untouched test: 2026-03-01 through 2026-05-31;
- forecast horizon: 24 hours;
- forecast origins: every 24 hours;
- fit window: expanding;
- primary seasonal lag: 168 hours.

This produces 92 complete daily test origins before any prospective DST-day exclusion.

TLC timestamps are treated as published local wall-clock times. DST transition days are explicitly flagged in the panel and excluded from the headline comparison rather than silently coercing ambiguous wall-clock times to UTC.

The ingestion layer records row-level rejection counts for missing pickup times, missing pickup zones, non-geographic/invalid zones and observations outside the frozen study window. Zone selection is performed using the training period only.

Build the local panel with:

```bash
cd projects/mobility_demand_optimization
python scripts/build_tlc_panel.py --download
```

The script writes both the hourly parquet panel and a JSON manifest containing the selected zones, panel dimensions, test-origin count and monthly QA summaries. Raw TLC trip files are not committed to the portfolio repository.

## Core comparisons

The project is designed around prospective comparisons rather than model accumulation.

### Forecasting

- seasonal-naive baseline;
- count model (Poisson or Negative Binomial where dispersion requires it);
- gradient-boosted point model;
- quantile or distributional forecast model.

Random train/test splitting is prohibited. Evaluation uses rolling-origin backtesting.

### Allocation policies

- historical/no-rebalancing baseline;
- deterministic allocation from point forecasts;
- uncertainty-aware allocation from predictive distributions or quantiles;
- oracle allocation using realised demand, used only as an upper-bound benchmark.

The oracle is never a deployable policy.

## Primary estimand

For zone `i` and decision time `t`, define realised demand `d_it`, supplied vehicles `s_it`, unmet-demand penalty `p_i`, idle-capacity penalty `h_i`, and relocation cost `c_ij`.

The realised operational loss is

```text
L_t = relocation_cost_t
    + sum_i p_i * max(d_it - s_it, 0)
    + sum_i h_i * max(s_it - d_it, 0).
```

The headline comparison is the decision regret of each feasible policy relative to the oracle:

```text
regret_t(policy) = L_t(policy) - L_t(oracle).
```

A model can therefore have lower RMSE and still be operationally worse.

## Forecast evaluation

Point forecasts:

- MAE;
- WAPE;
- MASE where a valid seasonal scale exists.

Probabilistic forecasts:

- pinball loss for reported quantiles;
- empirical interval coverage;
- interval width/sharpness;
- CRPS when a full predictive distribution is available.

All metrics are computed globally and by zone-demand stratum so that strong aggregate scores cannot hide poor service in lower-volume zones.

## Optimization model

The first allocation model is a linear/min-cost-flow formulation. Decision variables represent vehicles retained in or relocated between zones. Constraints include:

- fleet conservation;
- non-negative integer vehicle counts where required;
- relocation feasibility;
- optional relocation-budget or distance limits;
- available starting inventory by zone.

The uncertainty-aware policy will be implemented without future leakage, using only predictive information available at the decision timestamp.

## Leakage rules

- trip records are ordered by event time before feature construction;
- rolling features use strictly past observations;
- future realised demand is unavailable to all deployable policies;
- the oracle appears only in retrospective decision evaluation;
- hyperparameter tuning occurs inside the rolling evaluation design;
- zone selection uses training demand only.

## Planned notebooks

1. `01_demand_panel.ipynb` — TLC ingestion, QA and zone-time aggregation.
2. `02_forecasting.ipynb` — rolling-origin baselines, point and probabilistic forecasting.
3. `03_allocation.ipynb` — deterministic and uncertainty-aware fleet allocation.
4. `04_decision_evaluation.ipynb` — operational cost, service level, regret and sensitivity analysis.

Reusable logic belongs in `src/`, not inside notebooks.

## Status

**Active research / portfolio project.** The data contract and initial evaluation window are frozen before fitting the main models. No empirical performance claim should be made until the end-to-end backtest has been executed on real TLC data.
