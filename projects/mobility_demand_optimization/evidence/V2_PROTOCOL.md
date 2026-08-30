# Mobility v2 prospective protocol

## Scientific question

v1.1 established that a validation-selected Negative-Binomial predictive distribution was better calibrated than Poisson while a fixed marginal service-quantile policy produced higher realised operational loss than allocation to the Poisson mean.

v2 asks a narrower question:

> Does the Negative-Binomial predictive distribution create operational value when uncertainty is consumed by a joint stochastic optimisation problem rather than collapsed to one marginal quantile?

This protocol is frozen before any v2 confirmatory test-policy result is inspected.

## Primary estimand

For each decision hour `t`, let `L_t(P)` be realised operational loss under deployable policy `P`. The primary contrast is

```text
Delta_t = L_t(NB stochastic SAA) - L_t(Poisson mean deterministic).
```

The primary estimand is the mean headline-hour difference over the v2 test period:

```text
Delta = mean_t Delta_t.
```

Interpretation:

- `Delta < 0`: the stochastic NB policy improves realised operational cost;
- `Delta > 0`: the deterministic Poisson-mean policy remains cheaper;
- the sign and magnitude are reported regardless of result.

No minimum-effect threshold is used to redefine success after observing test outcomes.

## Frozen holdout

The confirmatory v2 test window is June 2026:

```text
[2026-06-01, 2026-07-01)
```

June is not used for model selection, scenario calibration, solver tuning, loss-ratio selection, zone selection, fleet-size selection, or sensitivity selection.

The v1.1 top-30 TLC zone set is frozen and reused unchanged. The v1.1 fleet size of `4532` is frozen and reused unchanged.

The decision frequency remains hourly. The prospective DST headline exclusion is retained; June 2026 contains no US DST transition day, so this rule should not remove confirmatory hours but remains part of the contract.

## Forecast contract

The point mean is the existing expanding Poisson hour-of-week forecast.

The stochastic distribution is Negative Binomial with

```text
Var(Y_i,t) = mu_i,t + alpha * mu_i,t^2,
```

with `alpha = 0.05` frozen from v1.1. It is not reselected using June.

The v2 primary stochastic policy therefore differs from the deterministic comparator in the decision rule, not in the conditional mean forecast.

## Scenario generation

Primary scenario generation is conditionally independent Negative-Binomial sampling across the 30 frozen zones given the forecast mean vector.

This is an explicit modelling approximation. Cross-zone dependence is not introduced in the primary analysis because adding a dependence model would create a second estimand change at the same time as replacing the decision rule.

Frozen primary settings:

- scenarios per decision hour: `128`;
- base random seed: `20260830`;
- deterministic per-hour seed derivation from the timestamp plus the base seed;
- all deployable policies at the same timestamp see the same observed state before optimisation;
- scenario draws are generated before solving and are not adapted to the realised June demand.

A scenario-count stability analysis using `64`, `128`, and `256` scenarios is secondary and must report all three values. The primary result remains `128`.

## Primary stochastic optimisation

For one decision hour with initial available fleet `a_i`, relocation variables `x_ij`, final supply `s_i`, scenarios `d_i^(k)`, unmet variables `u_i^(k)`, and idle variables `v_i^(k)`, solve the sample-average approximation

```text
min  sum_ij c_ij x_ij
   + (1/K) sum_k [sum_i p_i u_i^(k) + sum_i h_i v_i^(k)]
```

subject to

```text
s_i = a_i - sum_j x_ij + sum_j x_ji
u_i^(k) >= d_i^(k) - s_i
v_i^(k) >= s_i - d_i^(k)
x_ij, s_i, u_i^(k), v_i^(k) >= 0
```

and source-flow limits preventing any origin from relocating more fleet than is initially available there.

The primary penalties remain exactly those of v1.1:

```text
unmet penalty p = 5
idle penalty h = 1
```

The spatial relocation-cost matrix remains the v1.1 TLC-zone centroid-distance matrix, normalised to median off-diagonal cost `0.25`.

The optimisation remains continuous. No integer fleet constraint is introduced in v2 primary analysis.

## Comparator policies

Primary comparison:

1. `poisson_mean_deterministic`
2. `negative_binomial_stochastic_saa`

Context-only comparators, reported but not used to redefine the primary hypothesis:

3. `negative_binomial_service_quantile_v1` using `q = 5/6`
4. `no_rebalancing`

A retrospective oracle may be reported only in static diagnostics. It is not a deployable comparator and does not determine the primary conclusion.

## Fleet-state contract

The primary v2 state model inherits the strongest v1.1 state contract:

- each policy has its own endogenous rolling fleet state;
- served vehicles move according to realised within-region TLC destination/duration profiles;
- passenger vehicles remain unavailable until their observed dropoff-time arrival bucket;
- idle vehicles remain available locally;
- total regional capacity is conserved across available and in-transit mass;
- trip profiles condition on pickup and dropoff both lying in the frozen top-30 zone region;
- passenger arrival lags beyond six hourly buckets remain clipped into the final bucket.

Empty-vehicle relocation remains instantaneous within the decision epoch. That limitation is deliberately held fixed in the primary v2 study so the decision-rule contrast remains interpretable.

## Headline metrics

Primary:

- mean realised total operating cost per eligible hour;
- mean paired cost difference `NB stochastic SAA - Poisson mean`.

Secondary:

- service level;
- unmet-demand cost;
- idle-capacity cost;
- relocation cost;
- paired hourly win rate;
- median paired cost difference;
- 5%, 50%, and 95% quantiles of paired hourly cost difference.

Uncertainty for the mean paired difference is estimated with a moving-block bootstrap over hourly paired losses. The frozen block length is `24` hours and the bootstrap replication count is `4999`, using base seed `20260831`.

The bootstrap confidence interval is descriptive uncertainty for the realised test period. It is not used as a gate to hide an unfavourable point estimate.

## Secondary sensitivities

All secondary sensitivities are reported as complete grids, not selected after looking for a preferred outcome.

### Scenario count

```text
K in {64, 128, 256}
```

### Operational loss ratio

Holding idle penalty at `1`:

```text
unmet penalty in {2, 5, 10}
```

The primary value remains `5`.

### Relocation-cost scale

Multiply the frozen spatial cost matrix by

```text
{0.5, 1.0, 2.0}
```

The primary multiplier remains `1.0`.

These grids are sensitivity surfaces. No cell replaces the primary analysis.

## Explicitly deferred questions

The following are not part of the v2 primary experiment:

- correlated multivariate demand scenarios;
- copulas or residual-bootstrap dependence models;
- distributionally robust optimisation;
- explicit empty-vehicle relocation travel time;
- external-region/citywide fleet reservoir;
- alternative forecast models;
- integer fleet optimisation;
- retuning Negative-Binomial dispersion;
- changing the top-30 zone set.

Each would alter an additional scientific component and therefore belongs in a separate extension after the primary stochastic-decision result is known.

## Leakage and reproducibility gates

- June 2026 cannot affect any frozen parameter or primary design choice.
- Scenario seeds are deterministic and recorded in output metadata.
- Every hourly policy result is saved before aggregation.
- The exact v1.1 freeze used as the parent evidence state is recorded by path and commit.
- The empirical run records commit SHA, workflow run ID, artifact digest, scenario count, seeds, alpha, penalties, fleet size, zone set, relocation-cost normalisation, and test boundaries.
- Any implementation correction discovered after examining June outcomes must preserve the original evidence and produce a versioned correction rather than silently overwrite it.

## Primary decision rule

The primary scientific conclusion will answer only this question:

> On the preregistered June 2026 holdout, does joint sample-average optimisation under the frozen Negative-Binomial predictive distribution reduce realised operational loss relative to deterministic allocation against the same conditional mean?

The result will be reported whether positive, negative, or approximately neutral.
