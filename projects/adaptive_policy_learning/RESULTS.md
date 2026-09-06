# Adaptive Policy Learning Results

## Study 1: terminal model-fit failure

The preregistered primary study (Study 1) terminated at reward-model fitting and did not reach off-policy evaluation. Its frozen SAGA L2 logistic reward model failed to converge under both prospectively allowed optimizer budgets (`max_iter=200` and `max_iter=1000`). No Study 1 evaluation-period outcome, Random-reference outcome, OPE estimate, or promotion decision was observed.

## Study 2: corrected primary OPE

Study 2 prospectively changed only the reward-model solver to deterministic `newton-cholesky` while preserving the source, 70/30 temporal split, feature/preprocessing contract, target policies, OPE estimators, clipping sensitivities, moving-block bootstrap, and promotion rule.

The corrected Study 2 run completed successfully. Challenger DR was `0.0086129872` versus observed BTS `0.0048561566`; DR-minus-BTS was `0.0037568306` with paired 95% bootstrap interval `[0.0015186077, 0.0100335434]`. However, unclipped challenger ESS fraction was only `0.0019786`, far below the frozen 10% threshold. Study 2 therefore returned **`do_not_promote`** because overlap was inadequate despite a positive estimated effect.

## Study 3: terminal deterministic primary OPE

Study 3 was specified prospectively on the distinct `women` campaign. Source selection, exact temporal split, reward model, target policies, OPE estimators, clipping sensitivities, 24-hour moving-block bootstrap, and promotion rule were frozen before evaluation outcomes were opened.

After two implementation-only execution failures, a training-only reproducibility diagnostic fit the exact frozen model four times and reproduced the same coefficient SHA on every fit. The final v2.12 execution therefore pinned a single-thread numerical environment and required an immediate training-only preflight before OPE. That preflight reproduced:

- coefficient SHA256 `8e8ba7827c80c256e1c980007053fdcbb22d2ac8793673df3fe6669fafd3802c`;
- `65` features;
- `n_iter=6`;
- zero optimizer warnings.

The terminal OPE then completed successfully.

### Frozen sample

- BTS training rows: `1,811,697`
- BTS evaluation rows: `776,442`
- Random-reference rows: `85,990`
- action count: `46`
- uniform target probability: `1/46 = 0.0217391304`
- evaluation window: `2019-11-28 15:23:48.989271+00:00` to `2019-11-30 23:59:59.862467+00:00`

Observed BTS evaluation value: `0.0062400025`.

Independently logged Random-reference value: `0.0049424352`.

### Uniform-random benchmark

| Estimator | OPE estimate | Absolute error vs Random reference | Relative error |
|---|---:|---:|---:|
| IPS | 0.0042885840 | 0.0006538512 | 13.23% |
| SNIPS | 0.0046135233 | 0.0003289118 | 6.65% |
| DM | 0.0049584995 | 0.0000160643 | 0.33% |
| DR | 0.0046289572 | 0.0003134780 | 6.34% |

The unclipped uniform-target ESS fraction was `0.0354091` (3.54%). Clipping increased ESS to 28.02% at cap 5, 20.17% at cap 10, and 14.62% at cap 20.

### Challenger estimates

| Estimator | Estimate |
|---|---:|
| IPS | 0.0042627450 |
| SNIPS | 0.0038909502 |
| DM | 0.0126170082 |
| DR | 0.0044247405 |

The primary DR-minus-BTS point difference was:

`-0.0018152619`

The frozen paired 24-hour moving-block bootstrap used 1,999 replications with seed `20260831`. Its 95% interval was:

`[-0.0049919409, -0.0009922200]`

The entire interval is below zero, so the first promotion condition fails.

### Overlap and promotion decision

The challenger overlap diagnostics are extremely poor under the frozen unclipped primary analysis:

- ESS: `176.11` effective observations out of `776,442` evaluation rows;
- ESS fraction: `0.0002268` (about 0.023%);
- frozen minimum ESS fraction: `0.10` (10%);
- maximum importance weight: `25,776.40`;
- 99th percentile importance weight: `20.18`.

The frozen promotion rule required both a strictly positive lower 95% bootstrap bound for challenger DR minus BTS and challenger ESS fraction of at least 10%.

Study 3 fails both conditions. Therefore:

**Promotion decision: `do_not_promote`.**

The large gap between DM (`0.0126170`) and DR (`0.0044247`), together with the extreme unclipped weights, is consistent with severe support failure. Clipping changes the sensitivity estimates and increases ESS, but those analyses are secondary and do not replace the preregistered unclipped decision rule.

### Scientific interpretation

Study 3 provides a stronger negative deployment conclusion than Study 2. In Study 2 the challenger looked beneficial but could not be supported because overlap was too weak. In Study 3 the primary DR estimate is below observed BTS, its paired confidence interval is entirely negative, and overlap is even weaker.

The correct conclusion is therefore not merely “insufficient evidence to promote.” Under the frozen Study 3 design, the logged data provide evidence against promoting this challenger, while also showing that extrapolation risk is extreme. No post-outcome change to the challenger, model, clipping rule, ESS threshold, estimator, split, or bootstrap is authorized within Study 3.

## Reproducibility provenance

Terminal Study 3 primary OPE:

- workflow run: `34058975285`;
- workflow run number: `4`;
- run head: `c828fb0cc81b3d47dd2e9f3a0ceb67f42b201e65`;
- workflow conclusion: `success`;
- artifact: `adaptive-policy-learning-study3-primary-ope`;
- artifact ID: `9996901283`;
- artifact digest: `sha256:27fdfc20ef0b2f7cc8797282b07b2833dadd2ec726302e464615b473ecd23f3b`;
- result protocol version: `2.12-study3-deterministic-primary-ope-execution`;
- design hash: `abca33cc2c237b3d8b0f030d6f7c8e1cdd372a003eb64a4386fba9caf3dd9887`;
- source archive SHA256: `e8ec18196582a5937381a1776382ca940689b90a18d2dcd1fb635be6df614d78`.

The machine-readable terminal record is [`protocol/study3_primary_ope_terminal_status_v2_12.json`](protocol/study3_primary_ope_terminal_status_v2_12.json).

## Future work

Studies 1–3 are complete. Any attempt to alter the challenger, reward model, features, source, split, overlap threshold, clipping rule, estimator, bootstrap, or decision rule after observing Study 3 outcomes must be a genuinely new prospective study rather than a continuation or repair of Study 3.
