# Adaptive Policy Learning

A reproducible contextual-bandit project for evaluating and selecting decision policies from logged recommendation data.

## Current empirical status

**Study 2 status: `success`. Promotion decision: `do_not_promote`.**

Study 1 remains a valid historical negative result: its preregistered SAGA logistic reward model failed to converge under the allowed `max_iter=200` and `max_iter=1000` budgets, so Study 1 terminated before off-policy evaluation.

Study 2 was then specified prospectively as a new study. It preserved the source, temporal split, features, target policies, OPE estimators, bootstrap, clipping sensitivities, and promotion rule, while changing only the reward-model solver to deterministic `newton-cholesky`. The training qualification reproduced the frozen 73-feature model in 6 iterations with zero warnings on all 2,882,936 BTS training rows.

The corrected Study 2 primary OPE completed successfully on 1,235,545 BTS evaluation rows. The challenger produced a doubly robust value estimate of `0.0086129872`, compared with observed BTS value `0.0048561566`, for a DR-minus-BTS point difference of `0.0037568306`. The preregistered 95% paired moving-block-bootstrap interval was `[0.0015186077, 0.0100335434]`.

The challenger was nevertheless **not promoted** because its unclipped importance-weight effective-sample-size fraction was only `0.0019786` (about 0.20%), far below the frozen minimum of `0.10` (10%). The result therefore demonstrates the intended safety property of the decision rule: a positive point estimate and positive confidence interval are not sufficient when overlap is inadequate.

See [`RESULTS.md`](RESULTS.md) for the complete empirical record and [`protocol/study2_primary_ope_terminal_status_v1_2.json`](protocol/study2_primary_ope_terminal_status_v1_2.json) for machine-readable provenance.

## Research question

> How reliable are off-policy estimators for real logged recommendation data, and can a transparent lower-confidence-bound rule prevent unsupported promotion of a learned challenger policy?

The empirical studies use the ZOZO Research Open Bandit Dataset. Bernoulli Thompson Sampling feedback is the logging-policy dataset. The uniform Random policy provides an independently logged on-policy reference over the same evaluation window.

The project deliberately focuses on **policy-value estimation and deployment decisions**, not on maximizing an opaque leaderboard score.

## Primary estimators

For logged observations \((x_i,a_i,r_i,p_i)\), target policy \(\pi_e\), and reward model \(\hat q\):

\[
w_i = \frac{\pi_e(a_i\mid x_i)}{p_i}.
\]

The implemented estimators are:

\[
\widehat V_{IPS}=\frac{1}{n}\sum_i w_i r_i,
\]

\[
\widehat V_{SNIPS}=\frac{\sum_i w_i r_i}{\sum_i w_i},
\]

\[
\widehat V_{DM}=\frac{1}{n}\sum_i\sum_a\pi_e(a\mid x_i)\hat q(x_i,a),
\]

and

\[
\widehat V_{DR}=\frac{1}{n}\sum_i\left[\sum_a\pi_e(a\mid x_i)\hat q(x_i,a)+w_i(r_i-\hat q(x_i,a_i))\right].
\]

Study 2 produced empirical IPS, SNIPS, DM, and DR estimates for both the uniform-random benchmark and the challenger. The independently logged Random reference provides an external calibration check for the uniform-random target.

## Promotion rule

The challenger is promoted only when both frozen conditions hold:

1. the lower endpoint of the 95% moving-block-bootstrap interval for
   \(V_{DR}(\pi_{challenger})-V(\pi_{BTS})\) is strictly positive;
2. the challenger importance-weight effective-sample-size fraction is at least `0.10`.

For Study 2, condition 1 passed but condition 2 failed:

- bootstrap lower endpoint: `0.0015186077 > 0`;
- challenger ESS fraction: `0.0019786 < 0.10`.

Therefore the frozen rule returns **`do_not_promote`**.

## Scientific boundaries

- Clicks measure engagement, not revenue or long-term welfare.
- The independently logged Random CTR is an on-policy reference estimate with sampling uncertainty, not an exact population truth.
- The task is one-step contextual policy evaluation and selection, not reinforcement learning.
- Study 1's SAGA failures remain part of the record and were not rewritten after Study 2 succeeded.
- Study 2 was prospectively specified before its evaluation outcomes were opened.
- The timestamp-resolution erratum changed only implementation-level datetime normalization and boundary validation; it did not change any model, estimator, split, policy, bootstrap setting, or promotion threshold.
- The challenger's very low unclipped ESS and extreme maximum importance weight mean its attractive unclipped point estimates must not be treated as deployment evidence.
- Any post-result redesign must be treated as a new prospective study rather than tuning Study 2 after observing outcomes.

The protocol chain begins at [`protocol/design_lock.json`](protocol/design_lock.json), continues through [`protocol/study2_design_lock_v1_0.json`](protocol/study2_design_lock_v1_0.json), and ends with the corrected Study 2 result record in [`protocol/study2_primary_ope_terminal_status_v1_2.json`](protocol/study2_primary_ope_terminal_status_v1_2.json).
