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

The first empirical study was prospectively fixed before model fitting:

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

TLC timestamps are treated as published local wall-clock times. DST transition days are explicitly flagged in the panel and excluded from the headline forecast comparison rather than silently coercing ambiguous wall-clock times to UTC.

The ingestion layer records row-level rejection counts for missing pickup times, missing pickup zones, non-geographic/invalid zones and observations outside the frozen study window. Zone selection is performed using the training period only.

Build the local panel with:

```bash
cd projects/mobility_demand_optimization
python scripts/build_tlc_panel.py --download
```

The script writes both the hourly parquet panel and a JSON manifest containing the selected zones, panel dimensions, test-origin count and monthly QA summaries. Raw TLC trip files are not committed to the portfolio repository.

## Frozen baseline models

The first modelling comparison is deliberately simple and interpretable.

### Seasonal naive

For each zone and forecast target hour,

```text
y_hat[i,t] = y[i,t-168].
```

The 168-hour lag is longer than the 24-hour forecast horizon, so every source observation predates the forecast origin.

### Expanding Poisson hour-of-week model

For zone `i` and hour-of-week `h`, demand is modelled as

```text
D[i,t] ~ Poisson(lambda[i,h]).
```

At each forecast origin, `lambda[i,h]` is estimated from observations strictly before that origin. The Poisson maximum-likelihood estimate is therefore the historical mean for the corresponding zone/hour-of-week cell. No test or future-validation observation enters the fitted rate.

Run both baselines with:

```bash
python scripts/run_baselines.py
```

The script evaluates validation and untouched-test origins, writes row-level forecast parquet files and a compact JSON summary containing MAE and WAPE. Headline metrics exclude prospectively flagged DST-transition rows.

## Frozen empirical result

The first complete empirical study is frozen in `evidence/empirical_freeze_v1.json`, with interpretation in `evidence/FINDINGS_V1.md`.

The source workflow is GitHub Actions run `33307567684` at commit `650ec60c575e7197e8ef246d4dd709f0fe147362`. The corresponding workflow artifact has SHA-256 `c00d7e0b16efc7e70e922de013555b4bd726597cadfe3305d1e9513b4e93a205`.

### Predictive result

On the untouched March-May 2026 test window, the expanding Poisson hour-of-week model improves point forecasting over seasonal naive:

| Model | MAE | WAPE |
|---|---:|---:|
| Seasonal naive | 25.5907 | 19.5521% |
| Poisson hour-of-week | 20.2521 | 15.4732% |

Negative-Binomial dispersion is selected on validation only, with `alpha = 0.05`. Around the same Poisson mean forecast, it materially improves probabilistic calibration on test:

| Distribution | 80% interval coverage | Mean pinball loss |
|---|---:|---:|
| Poisson | 50.72% | 7.5799 |
| Negative Binomial | 86.96% | 6.6638 |

### Decision result

The better-calibrated Negative-Binomial predictive distribution does **not** produce the lower-cost allocation policy under the frozen decision rule. The uncertainty-aware policy allocates to the asymmetric service quantile

```text
q* = p / (p + h) = 5 / 6,
```

while the deterministic policy allocates against the Poisson mean.

The ranking survives every frozen decision sensitivity:

| Sensitivity | Poisson mean cost | NB service-quantile cost |
|---|---:|---:|
| Uniform relocation cost | 4,828.56 | 4,865.65 |
| TLC-zone spatial cost | 4,835.26 | 4,883.61 |
| Policy-dependent rolling fleet state | 5,430.15 | 5,963.18 |
| Rolling state + observed passenger-trip duration | 7,311.70 | 7,661.76 |

Under the strongest frozen state model, which keeps vehicles unavailable until their observed TLC dropoff-time bucket, service levels are 78.72% for Poisson mean and 77.82% for the Negative-Binomial service-quantile policy. No-rebalancing falls to 67.57%.

The main finding is therefore deliberately narrow:

> Better probabilistic calibration did not translate into lower downstream operational loss under the frozen loss function and service-quantile decision rule.

This does **not** imply that uncertainty is generally harmful or that every uncertainty-aware policy underperforms. It shows why predictive quality and decision quality must be evaluated separately.

## Core comparisons

The project is designed around prospective comparisons rather than model accumulation.

### Forecasting

- seasonal-naive baseline;
- expanding Poisson hour-of-week count baseline;
- validation-selected Negative-Binomial predictive dispersion.

Random train/test splitting is prohibited. Evaluation uses rolling-origin backtesting.

### Allocation policies

- historical/no-rebalancing baseline;
- deterministic allocation from point forecasts;
- uncertainty-aware allocation from predictive distributions or quantiles;
- oracle allocation using realised demand, used only as an upper-bound benchmark in the static decision experiments.

The oracle is never a deployable policy.

## Primary estimand

For zone `i` and decision time `t`, define realised demand `d_it`, supplied vehicles `s_it`, unmet-demand penalty `p_i`, idle-capacity penalty `h_i`, and relocation cost `c_ij`.

The realised operational loss is

```text
L_t = relocation_cost_t
    + sum_i p_i * max(d_it - s_it, 0)
    + sum_i h_i * max(s_it - d_it, 0).
```

For experiments with a retrospective oracle, decision regret is

```text
regret_t(policy) = L_t(policy) - L_t(oracle).
```

A model can therefore score better probabilistically and still be operationally worse.

## Forecast evaluation

Point forecasts:

- MAE;
- WAPE.

Probabilistic forecasts:

- pinball loss for reported quantiles;
- empirical interval coverage;
- interval width/sharpness.

## Optimization and fleet-state model

The allocation model is a linear fleet-flow formulation. Decision variables represent vehicles retained in or relocated between zones. Fleet conservation is enforced explicitly.

The empirical sensitivities progress from a static stylised system to a policy-dependent dynamic state:

1. uniform relocation cost;
2. spatially heterogeneous relocation cost from official TLC taxi-zone geometry;
3. endogenous rolling fleet state where each policy inherits its own previous decisions;
4. observed passenger-trip duration, with served vehicles held in transit until their TLC dropoff-time bucket.

The strongest frozen model conserves total fleet across available and in-transit vehicles.

## Leakage rules

- trip records are ordered by event time before feature construction;
- rolling features use strictly past observations;
- future realised demand is unavailable to deployable policies;
- the oracle appears only in retrospective decision evaluation;
- model selection occurs inside the rolling evaluation design;
- zone selection uses training demand only;
- every expanding Poisson fit uses observations strictly before its forecast origin;
- Negative-Binomial dispersion is selected on validation only before one-time test evaluation.

## Interpretation boundaries

The frozen evidence must be read with these limitations:

- TLC pickup counts are observed served trips, not latent total demand or historical unmet demand;
- the allocation experiment is a counterfactual decision model, not a causal estimate of real NYC dispatch savings;
- the dynamic state model conditions on trips whose pickup and dropoff are both inside the frozen top-30-zone region;
- spatial relocation costs use projected zone-centroid distance rather than road travel time;
- passenger-trip availability uses observed TLC dropoff-time buckets, with lags beyond six hours clipped into the final bucket;
- empty-vehicle relocation remains instantaneous within an hourly decision epoch;
- the headline decision result compares one mean policy with one fixed service-quantile policy and does not establish a universal ranking of deterministic versus stochastic optimisation.

## Next-generation work

The v1 empirical result is frozen. Further development should be treated as a second-generation dispatch study rather than continuing to modify the same headline experiment.

Natural extensions include:

- explicit empty-vehicle relocation travel time;
- citywide fleet flow with an external-region state;
- stochastic or distributionally robust optimisation that consumes the full predictive distribution rather than one marginal quantile;
- alternative operational loss ratios selected prospectively.

## Status

**Empirical v1 frozen.** The end-to-end TLC backtest has been executed on real data, the key predictive and operational findings are preserved with workflow provenance, and the headline policy ranking survived spatial cost, endogenous fleet-state, and observed passenger-trip-duration sensitivities.
