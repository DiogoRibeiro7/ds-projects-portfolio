# Mobility Demand & Fleet Optimization — Empirical Findings v1

## Frozen evidence

This note freezes the first complete empirical result for the mobility decision-science project. It is tied to GitHub Actions workflow run `33307567684`, head commit `650ec60c575e7197e8ef246d4dd709f0fe147362`, and workflow artifact SHA-256 `c00d7e0b16efc7e70e922de013555b4bd726597cadfe3305d1e9513b4e93a205`.

The study uses official NYC TLC Yellow Taxi records from January 2025 through May 2026. The frozen panel contains 371,520 hourly zone observations across the top 30 pickup zones selected on 2025 training data only. Of 67,721,884 source rows, 132,799 were rejected by the ingestion contract, approximately 0.196%.

## Forecast result

The expanding Poisson hour-of-week baseline is a substantially stronger point forecast than the 168-hour seasonal naive baseline on the untouched March-May 2026 test window:

| Model | Test MAE | Test WAPE |
|---|---:|---:|
| Seasonal naive | 25.5907 | 19.5521% |
| Poisson hour-of-week | 20.2521 | 15.4732% |

The Poisson baseline reduces WAPE by approximately 20.9% relative to the seasonal naive benchmark.

## Uncertainty result

Negative-Binomial dispersion was selected on the January-February 2026 validation window only. The selected value is

\[
\alpha = 0.05,
\]

under

\[
\operatorname{Var}(Y)=\mu+\alpha\mu^2.
\]

On the untouched test window, the Negative-Binomial distribution is materially better calibrated than Poisson around the same mean forecast:

| Distribution | 80% interval coverage | Mean pinball loss |
|---|---:|---:|
| Poisson | 50.72% | 7.5799 |
| Negative Binomial | 86.96% | 6.6638 |

The Negative-Binomial model reduces mean pinball loss by approximately 12.1%. Its intervals are substantially wider, so this is improved calibration with reduced sharpness rather than a free improvement.

## Decision result

The operational question is deliberately different from the forecasting question. The fleet-allocation loss penalises relocation, unmet demand, and idle capacity. The uncertainty-aware policy uses the Negative-Binomial service quantile

\[
q^*=\frac{p}{p+h}=\frac{5}{6},
\]

while the deterministic policy allocates against the Poisson mean.

Across every frozen decision sensitivity, the better-calibrated Negative-Binomial distribution produces a higher realised operational cost than the Poisson-mean policy:

| Decision sensitivity | Poisson mean cost | NB service-quantile cost | NB minus Poisson |
|---|---:|---:|---:|
| Uniform relocation cost | 4,828.56 | 4,865.65 | +37.09 |
| TLC-zone spatial cost | 4,835.26 | 4,883.61 | +48.35 |
| Policy-dependent rolling state | 5,430.15 | 5,963.18 | +533.03 |
| Rolling state + observed passenger-trip duration | 7,311.70 | 7,661.76 | +350.05 |

The strongest state model keeps served vehicles unavailable until their observed TLC dropoff-time bucket. Under that model, service levels are

\[
78.72\%\quad\text{for Poisson mean}
\]

and

\[
77.82\%\quad\text{for the Negative-Binomial service-quantile policy}.
\]

The no-rebalancing policy falls to 67.57% service, showing that fleet geography compounds materially once state is propagated rather than reset.

## Main finding

The result is not that probabilistic calibration is unimportant. The Negative-Binomial distribution is clearly better calibrated than Poisson.

The result is narrower and more useful:

> Better probabilistic calibration did not translate into a better operational allocation policy under the frozen loss function and the specific quantile decision rule tested here.

Equivalently,

\[
\boxed{
\text{better predictive distribution}
\not\Rightarrow
\text{lower downstream decision loss}
}
\]

for this experiment.

This is precisely why predictive and decision metrics are evaluated separately.

## Interpretation boundaries

The following limitations remain part of the frozen result and must travel with any external description of it:

- TLC pickup counts represent observed served trips, not latent total taxi demand or historical unmet demand.
- The fleet-allocation study is a counterfactual decision model, not a causal estimate of realised NYC dispatch savings.
- The dynamic state model is conditional on trips whose pickup and dropoff both lie inside the frozen top-30-zone analysis region.
- Spatial relocation costs use projected TLC-zone centroid distance, normalised to the earlier cost scale; they are not road travel times.
- Passenger-trip availability uses observed pickup/dropoff timestamps, with arrival lags beyond six hourly buckets clipped into the final bucket.
- Empty-vehicle relocation is still instantaneous within an hourly decision epoch.
- The result compares one deterministic mean policy with one fixed asymmetric quantile policy. It does not establish that every uncertainty-aware decision rule must underperform.

## Stopping rule

The v1 result is considered empirically frozen because the policy ranking survives changes to:

1. predictive dispersion and calibration;
2. spatially heterogeneous relocation costs from official TLC geometry;
3. endogenous policy-dependent fleet state;
4. observed passenger-trip duration and in-transit vehicle mass.

Further work should be framed as a second-generation dispatch model rather than another post-hoc robustness check on the same headline claim. The natural next research extension is explicit empty-vehicle relocation time or a direct stochastic optimisation policy that consumes the full predictive distribution rather than a single marginal service quantile.
