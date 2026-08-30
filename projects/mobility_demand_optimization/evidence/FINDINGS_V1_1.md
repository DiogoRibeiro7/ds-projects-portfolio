# Mobility Demand & Fleet Optimization — Findings v1.1

## Status

This document records the scientific-integrity correction to the first complete empirical study. `empirical_freeze_v1.json` remains an immutable historical record. For quantitative citation, use `empirical_freeze_v1_1.json`.

The correction changed two contracts only:

1. each monthly TLC Yellow Taxi parquet is restricted to its own nominal calendar month before aggregation;
2. the prospectively declared DST-transition exclusion is applied consistently to point, probabilistic, and decision headline summaries.

The forecasting models, Negative-Binomial dispersion grid, selected alpha, optimization models, fleet size, penalties, spatial costs, and state-transition mechanics are unchanged. DST hours remain in rolling state trajectories and are removed only from headline summaries.

## Data-integrity effect

Across 67,721,884 raw rows, 282 rows have pickup timestamps outside the nominal month of the parquet in which they were published. Because some of those rows were already excluded by the previous global-window or zone checks, the corrected panel has a net 243 fewer valid raw rows:

- v1 valid rows: 67,589,085;
- v1.1 valid rows: 67,588,842;
- v1.1 rejected rows: 133,042.

The dense panel remains 371,520 rows. The frozen top-30 zone set is unchanged.

## Predictive result

The point-forecast result is effectively unchanged.

| Model | v1.1 MAE | v1.1 WAPE |
|---|---:|---:|
| Seasonal naive, 168 h | 25.5907 | 19.5521% |
| Poisson hour-of-week | 20.2522 | 15.4733% |

The Poisson model still reduces WAPE by about 20.86% relative to seasonal naive.

Negative-Binomial dispersion selection also remains unchanged:

```text
alpha = 0.05
```

After applying the same prospective DST rule to the test probabilistic summary:

| Distribution | 80% interval coverage | Mean pinball loss |
|---|---:|---:|
| Poisson | 50.85% | 7.5531 |
| Negative Binomial | 87.05% | 6.6414 |

The Negative Binomial therefore retains about a 12.07% mean-pinball improvement over Poisson.

## Decision result

The operational ranking survives every corrected sensitivity.

| Sensitivity | Poisson mean cost | NB service-quantile cost | NB − Poisson |
|---|---:|---:|---:|
| Uniform relocation cost | 4,851.98 | 4,889.38 | +37.40 |
| TLC-zone spatial cost | 4,858.02 | 4,906.80 | +48.77 |
| Policy-dependent rolling state | 5,451.11 | 5,984.40 | +533.30 |
| Rolling state + observed trip duration | 7,345.64 | 7,695.19 | +349.55 |

Under the strongest corrected state model:

- Poisson service level: 78.63%;
- Negative-Binomial service-quantile service level: 77.73%;
- no-rebalancing service level: 67.51%.

The Poisson policy therefore retains an approximately 0.90 percentage-point service advantage over the Negative-Binomial service-quantile policy.

## Materiality relative to v1

The correction does not alter the substantive finding.

The largest relative policy-cost change is under 0.49%. Under the strongest duration-aware model:

```text
Poisson cost:  7311.70 -> 7345.64  (+0.46%)
NB cost:       7661.76 -> 7695.19  (+0.44%)
NB-Poisson gap: 350.05 -> 349.55
```

The gap changes by only about -0.51 cost units per headline hour.

The selected alpha, top-30 zone set, fleet size, and all policy rankings remain unchanged.

## Corrected headline finding

> Better probabilistic calibration did not translate into lower downstream operational loss under the frozen loss function and the fixed 5/6 service-quantile decision rule.

The result is not a claim that uncertainty-aware optimization is generally inferior. It is evidence that predictive calibration and decision quality are distinct objectives, and that a marginal predictive quantile is not automatically the right object for a constrained dynamic allocation problem.

## Stopping rule

This closes the v1 line of work. Further modelling should be treated as v2 rather than another post-hoc modification of the same headline experiment.

The scientifically strongest next step is a joint stochastic decision model that consumes the predictive distribution inside the constrained fleet optimization itself, with explicit prospective sensitivity grids and no test-driven tuning.
