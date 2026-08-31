# Mobility v2 primary reporting template

Status: prospectively frozen before the June 2026 TLC holdout is available.

This file defines how the first successful preregistered v2 confirmatory run is to be reported. It must be completed from the `mobility-v2-primary` artifact produced by the one-shot workflow on `main`. Do not alter the structure or replace the primary result after seeing the outcome.

## Provenance

- Protocol PR: #561
- Implementation PR: #565
- Corrective provenance PRs: #567, #569, #571, #573, #575, #577, #580, #582, #584, #586, #587
- Confirmatory workflow run: `<RUN_ID>`
- Confirmatory commit on `main`: `<COMMIT_SHA>`
- Artifact ID: `<ARTIFACT_ID>`
- Artifact digest: `<ARTIFACT_DIGEST>`
- Relocation-matrix SHA-256: `bf3ebdf7eaa8391c4a5c4554fbb39d0a098f5d4fc31af429cd39f7b4b17bb8b4`
- Holdout: `[2026-06-01T00:00:00, 2026-07-01T00:00:00)`
- Headline hours: `720`
- Frozen zones: `30`
- Fleet size: `4532`
- NB2 alpha: `0.05`
- Primary scenario count: `128`
- Scenario dependence: conditionally independent across zones
- Unmet / idle penalties: `5 / 1`
- Bootstrap: 24-hour moving blocks, 4,999 replications, seed `20260831`

## Primary estimand

\[
\Delta
=
\frac{1}{720}
\sum_t
\left[
L_t(\text{NB stochastic SAA})
-
L_t(\text{Poisson mean deterministic})
\right].
\]

Report exactly:

- mean paired hourly cost difference: `<DELTA>`
- median paired hourly cost difference: `<MEDIAN_DELTA>`
- paired hourly NB win rate: `<NB_WIN_RATE>`
- 5th / 50th / 95th percentiles: `<Q05>`, `<Q50>`, `<Q95>`
- moving-block bootstrap mean: `<BOOT_MEAN>`
- bootstrap standard error: `<BOOT_SE>`
- 95% bootstrap interval: `[<BOOT_LOWER>, <BOOT_UPPER>]`

Do not replace the preregistered primary estimand with a sensitivity result, subgroup, alternative penalty, alternative scenario count, or alternative relocation multiplier.

## Policy summaries

Report the frozen policy summaries exactly as produced for:

1. `negative_binomial_stochastic_saa`
2. `poisson_mean_deterministic`
3. `negative_binomial_service_quantile_v1`
4. `no_rebalancing`

For each policy report:

- mean total cost
- mean relocation cost
- mean unmet-demand cost
- mean idle-capacity cost
- mean service level

## Interpretation rule

Choose exactly one interpretation branch from the observed primary estimate and interval. Do not strengthen the wording beyond the frozen model and data scope.

### If \(\Delta < 0\)

> Under the preregistered June 2026 holdout and the frozen stylised fleet, loss, scenario-generation and rolling-state model, consuming the Negative-Binomial predictive distribution through joint stochastic sample-average optimisation reduced realised operational cost relative to deterministic allocation against the same conditional mean.

If the bootstrap interval includes zero, append:

> The estimated direction favoured stochastic allocation, but the paired moving-block bootstrap interval included zero, so the magnitude is uncertain at the preregistered operational scale.

### If \(\Delta > 0\)

> Under the preregistered June 2026 holdout and the frozen stylised fleet, loss, scenario-generation and rolling-state model, the better-calibrated Negative-Binomial predictive distribution did not translate into lower realised operational cost when consumed through the preregistered joint stochastic SAA objective; deterministic Poisson-mean allocation remained cheaper on average.

If the bootstrap interval includes zero, append:

> The estimated direction favoured deterministic allocation, but the paired moving-block bootstrap interval included zero, so the magnitude is uncertain at the preregistered operational scale.

### If the estimate is operationally near zero

Use this branch only when the point estimate is small relative to the policy-level mean costs and the bootstrap interval spans practically small positive and negative values:

> Under the preregistered June 2026 holdout, the realised operational value of consuming Negative-Binomial uncertainty through the frozen stochastic SAA policy was approximately indistinguishable from deterministic Poisson-mean allocation at the scale of this experiment.

Do not invent a post-hoc equivalence threshold. Describe the scale numerically instead.

## Required scope limitations

Every final interpretation must preserve all of the following:

- Demand is observed NYC TLC Yellow Taxi pickup count, not latent unmet passenger demand.
- The fleet is a stylised counterfactual continuous-capacity system, not observed persistent empty-taxi inventory.
- The 30-zone region is frozen from v1.1.
- Passenger-trip destination and duration dynamics are conditioned on realised trips remaining within the frozen region.
- Scenario vectors are joint only through simultaneous optimisation; demand draws are conditionally independent across zones.
- Empty-vehicle relocation travel time is instantaneous within the decision epoch.
- The result does not establish that uncertainty is generally useful or useless in operations.
- The result does not establish superiority of Negative Binomial or Poisson forecasting outside this frozen forecasting, scenario, loss and state model.

## Relationship to v1.1

Always report v1.1 and v2 as different decision mappings of the same broader predictive comparison:

- v1.1: NB2 improved probabilistic calibration but the fixed marginal `q = 5/6` policy had higher realised operational cost than deterministic Poisson-mean allocation.
- v2: tests whether the NB2 distribution creates operational value when consumed directly by joint stochastic sample-average optimisation rather than collapsed to one marginal quantile.

Do not describe v2 as a repair of v1.1 or use the v2 result to erase the v1.1 result.

## Freeze sequence after the first successful run

Before running any secondary sensitivity:

1. verify the panel manifest and source-month QA;
2. verify 720 June hours and 21,600 forecast rows;
3. verify exact frozen 30-zone set;
4. verify `alpha_reselected == false` in the JSON artifact;
5. verify 720 deterministic timestamp seeds;
6. recompute the primary paired mean from `hourly_policy_results.csv`;
7. verify the bootstrap configuration is 24 hours / 4,999 / seed 20260831;
8. verify matrix SHA-256 is the frozen v1.1 artifact digest;
9. create `empirical_freeze_v2_primary.json`;
10. create `FINDINGS_V2_PRIMARY.md` using this template;
11. only then run the preregistered secondary sensitivity grid.
