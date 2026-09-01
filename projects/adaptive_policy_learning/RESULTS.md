# Primary Study Results

## Terminal outcome

The preregistered primary adaptive-policy study terminated at **reward-model fitting** and did not reach off-policy evaluation.

The frozen reward model was an L2-penalized logistic regression using scikit-learn `LogisticRegression` with `solver="saga"`, `C=1.0`, `tol=0.0001`, `random_state=20260831`, the frozen BTS training partition, and the preregistered feature/preprocessing contract.

Two prospective optimizer budgets were attempted:

| Protocol | Maximum iterations | Outcome |
|---|---:|---|
| v0.7 | 200 | failed to converge |
| v0.8 amendment | 1000 | failed to converge |

The v0.8 amendment changed only the numerical optimization budget. It was committed after the v0.7 training-only failure and before any BTS evaluation-period or Random-reference outcome was loaded.

## What was successfully validated

Before the terminal model-fit failure, the project established and re-verified:

- the official ZOZO Open Bandit Dataset archive SHA256 and archive structure;
- the archived action catalog of 80 actions, IDs 0 through 79;
- the BTS raw-position-1 sample size and exact chronological 70/30 split;
- 2,882,936 BTS training rows and 1,235,545 BTS evaluation rows;
- the exact evaluation interval from `2019-11-28 16:55:17.867529+00:00` through `2019-11-30 23:59:59.920907+00:00`;
- the exact matched Random-policy reference window containing 137,634 raw-position-1 rows and all 80 actions;
- unit tests, Ruff, and strict mypy gates for the frozen empirical implementation.

## What was not observed

Because model fitting failed before evaluation data were opened, the primary study did **not** observe or compute:

- BTS evaluation-period reward summaries;
- Random-reference reward summaries;
- IPS estimates;
- SNIPS estimates;
- direct-method estimates;
- doubly-robust estimates;
- clipping sensitivities;
- challenger effective sample size or overlap diagnostics;
- the paired 24-hour moving-block bootstrap interval;
- any challenger-versus-BTS value difference;
- a promotion or `do_not_promote` estimate.

Accordingly, `promotion_authorized=false` means the study never established the prerequisites for promotion. It must not be interpreted as an empirical estimate that the challenger is worse than BTS.

## Scientific interpretation

The empirical conclusion is narrow:

> Under the preregistered reward-model specification and the only permitted pre-evaluation optimizer-budget amendment, the SAGA logistic fit did not converge on the full BTS training partition.

This result does **not** answer the original OPE reliability question. It does not validate or invalidate IPS, SNIPS, DM, or DR on this dataset, because those estimators were never evaluated in the primary study.

The protocol deliberately stops here rather than increasing `max_iter` again, changing solvers, changing the reward model, reducing the sample, altering preprocessing, or otherwise modifying the design after repeated training failures.

## Reproducibility provenance

The terminal v0.8 run is recorded in:

- workflow run: `33491157874`;
- workflow job: `99802721963`;
- run head: `d412e86d4ab0637761046e283bf9d5448cc7d5e2`;
- empirical artifact: `adaptive-policy-learning-primary-ope`;
- artifact ID: `9794819452`;
- artifact digest: `sha256:65c4f7d458fa8de7af3d81f610b5cbc268bb0969b87bc075f36bc74a9e24a159`.

The machine-readable terminal state is [`protocol/primary_empirical_terminal_status_v0_8.json`](protocol/primary_empirical_terminal_status_v0_8.json).

## Future work

Any attempt to change the reward model, solver, numerical tolerance, preprocessing, training sample, or optimization strategy should be treated as a **new prospectively specified study**, not as another amendment to this primary analysis.
