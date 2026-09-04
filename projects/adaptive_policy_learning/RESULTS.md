# Adaptive Policy Learning Results

## Study 1: terminal model-fit failure

The preregistered primary study (Study 1) terminated at reward-model fitting and did not reach off-policy evaluation. Its frozen SAGA L2 logistic reward model failed to converge under both prospectively allowed optimizer budgets:

| Protocol | Maximum iterations | Outcome |
|---|---:|---|
| v0.7 | 200 | failed to converge |
| v0.8 amendment | 1000 | failed to converge |

No Study 1 evaluation-period outcome, Random-reference outcome, OPE estimate, or promotion decision was observed. That negative result remains part of the scientific record.

## Study 2: corrected primary OPE

Study 2 was prospectively specified after Study 1 terminated. It preserved the official source, 70/30 temporal split, feature/preprocessing contract, target policies, OPE estimators, clipping sensitivities, moving-block bootstrap, and promotion rule. The only methodological redesign was the reward-model optimizer: deterministic `newton-cholesky` for the same L2 logistic objective.

The training-only qualification succeeded on all 2,882,936 BTS training rows with 73 features, `n_iter=6`, zero optimizer warnings, and coefficient SHA256:

`7438db24286013c628dc7f74e2dd7f4913cdd05297c717fff78a214c9afb5684`

The first Study 2 OPE execution then stopped on an implementation-only pandas datetime-resolution mismatch after the BTS evaluation clicks had been loaded in memory but before any empirical summary was emitted. The correction normalized timestamps explicitly to nanoseconds and moved the frozen boundary validation before outcome loading. No model, estimator, split, policy, clipping, bootstrap, or promotion rule changed.

The single authorized corrected rerun completed successfully.

## Frozen sample

- BTS training rows: `2,882,936`
- BTS evaluation rows: `1,235,545`
- Random-reference rows: `137,634`
- action count: `80`
- uniform target probability: `1/80 = 0.0125`
- evaluation window: `2019-11-28 16:55:17.867529+00:00` to `2019-11-30 23:59:59.920907+00:00`

Observed BTS evaluation value:

`0.0048561566`

Independently logged Random-reference value:

`0.0034802447`

## Uniform-random benchmark

The uniform-random target gives a useful calibration check because an independently logged Random-policy reference exists for the same evaluation window.

| Estimator | OPE estimate | Absolute error vs Random reference | Relative error |
|---|---:|---:|---:|
| IPS | 0.0031984081 | 0.0002818366 | 8.10% |
| SNIPS | 0.0031968143 | 0.0002834304 | 8.14% |
| DM | 0.0040789821 | 0.0005987374 | 17.20% |
| DR | 0.0032145520 | 0.0002656927 | 7.63% |

For the unclipped uniform target, the importance-weight ESS fraction was `0.0242607` (2.43%). Clipping increased ESS materially, reaching 26.92% at cap 5, 19.54% at cap 10, and 13.93% at cap 20, while shifting the value estimates.

## Challenger estimates

| Estimator | Estimate |
|---|---:|
| IPS | 0.0085745875 |
| SNIPS | 0.0084071780 |
| DM | 0.0063351039 |
| DR | 0.0086129872 |

The primary DR-minus-BTS point difference was:

`0.0037568306`

The frozen paired 24-hour moving-block bootstrap used 1,999 replications with seed `20260831`. Its 95% interval was:

`[0.0015186077, 0.0100335434]`

The lower endpoint is strictly positive, so the first promotion condition passed.

## Overlap and promotion decision

The challenger overlap diagnostics were poor under the frozen unclipped primary analysis:

- ESS: `2,444.69` effective observations out of `1,235,545` evaluation rows;
- ESS fraction: `0.0019786` (about 0.20%);
- frozen minimum ESS fraction: `0.10` (10%);
- maximum importance weight: `3,835.11`;
- 99th percentile importance weight: `22.51`.

The frozen promotion rule required both:

1. a strictly positive lower 95% bootstrap bound for DR challenger minus BTS;
2. challenger ESS fraction at least 10%.

Condition 1 passed. Condition 2 failed by a very large margin. Therefore:

**Promotion decision: `do_not_promote`.**

This is the intended conservative behavior. The challenger has attractive point estimates and a positive bootstrap interval, but those estimates rely on extremely weak logging-policy overlap. The project therefore refuses deployment rather than treating extrapolation as evidence.

Clipping illustrates the bias-variance/overlap trade-off but does not override the preregistered primary decision. Challenger ESS fraction rises to 6.36% at cap 5, 4.40% at cap 10, and 3.32% at cap 20, still below the 10% primary threshold.

## Scientific interpretation

The corrected Study 2 result answers the project question more sharply than a simple leaderboard comparison would:

- OPE estimators can be checked against an independently logged Random-policy reference, and IPS/SNIPS/DR were within roughly 7.6% to 8.1% relative error for that benchmark in this evaluation window;
- the learned challenger appears better than observed BTS under all four reported point estimators;
- however, the challenger policy is too far from the logging policy for the frozen support criterion to authorize deployment;
- the conservative LCB-plus-ESS rule therefore prevents promotion despite a positive estimated effect.

The correct conclusion is not that the challenger is bad. It is that the available logged data do not provide enough support for a deployment claim under the preregistered safety rule.

## Reproducibility provenance

Corrected Study 2 primary OPE:

- workflow run: `33876357702`;
- workflow run number: `9`;
- run head: `498d34b28266a8887e7259d975eda664e82784ac`;
- workflow conclusion: `success`;
- artifact: `adaptive-policy-learning-study2-primary-ope`;
- artifact ID: `9938429768`;
- artifact digest: `sha256:776da9a894690628e95db46920fa9af055f57c154df74e601b4dcb6c9359ad4e`;
- result protocol version: `1.2-study2-primary-ope-execution-erratum`;
- source archive SHA256: `e8ec18196582a5937381a1776382ca940689b90a18d2dcd1fb635be6df614d78`.

The machine-readable terminal record is [`protocol/study2_primary_ope_terminal_status_v1_2.json`](protocol/study2_primary_ope_terminal_status_v1_2.json).

## Future work

Any attempt to change the challenger policy, overlap threshold, clipping rule, reward model, features, split, or estimator after observing these outcomes must be treated as a new prospective study. Study 2 is complete and should not be tuned post hoc.
