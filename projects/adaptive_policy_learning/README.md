# Adaptive Policy Learning

A reproducible contextual-bandit project for evaluating and selecting decision policies from logged recommendation data.

## Current empirical status

**Study 3 status: `success`. Promotion decision: `do_not_promote`.**

Study 1 remains a valid historical negative result: its preregistered SAGA logistic reward model failed to converge under the allowed `max_iter=200` and `max_iter=1000` budgets, so Study 1 terminated before off-policy evaluation.

Study 2 then preserved the source, temporal split, features, target policies, OPE estimators, bootstrap, clipping sensitivities, and promotion rule while changing only the reward-model solver to deterministic `newton-cholesky`. Its challenger had a positive DR-minus-BTS interval but failed the frozen 10% ESS rule, so Study 2 correctly returned `do_not_promote` because overlap was inadequate.

Study 3 was prospectively specified on the distinct `women` campaign before outcomes were opened. The deterministic final execution reproduced the frozen 65-feature reward model (`n_iter=6`, zero warnings, coefficient SHA256 `8e8ba7827c80c256e1c980007053fdcbb22d2ac8793673df3fe6669fafd3802c`) and then completed primary OPE on 776,442 BTS evaluation rows plus 85,990 independently logged Random-reference rows.

Observed BTS value was `0.0062400025`. The challenger DR value was `0.0044247405`, giving a DR-minus-BTS point difference of `-0.0018152619`. The frozen paired 24-hour moving-block bootstrap gave a 95% interval of `[-0.0049919409, -0.0009922200]`, entirely below zero. Challenger unclipped ESS fraction was only `0.0002268` (0.023%), versus the frozen minimum of 10%.

Therefore Study 3 fails **both** frozen promotion conditions and the terminal decision is **`do_not_promote`**. The correct interpretation is not to tune the study after seeing outcomes; any redesigned challenger or estimator must be a new prospective study.

See [`RESULTS.md`](RESULTS.md) for the full empirical record and [`protocol/study3_primary_ope_terminal_status_v2_12.json`](protocol/study3_primary_ope_terminal_status_v2_12.json) for machine-readable provenance.

## Research question

> How reliable are off-policy estimators for real logged recommendation data, and can a transparent lower-confidence-bound rule prevent unsupported promotion of a learned challenger policy?

The empirical studies use the ZOZO Research Open Bandit Dataset. Bernoulli Thompson Sampling feedback is the logging-policy dataset. The uniform Random policy provides an independently logged on-policy reference over the same evaluation window.

The project deliberately focuses on **policy-value estimation and deployment decisions**, not on maximizing an opaque leaderboard score.

## Primary estimators

For logged observations \((x_i,a_i,r_i,p_i)\), target policy \(\pi_e\), and reward model \(\hat q\):

\[
w_i = \frac{\pi_e(a_i\mid x_i)}{p_i}.
\]

The implemented estimators are IPS, SNIPS, DM, and DR, with DR as the primary estimator for the challenger decision.

## Promotion rule

The challenger is promoted only when both frozen conditions hold:

1. the lower endpoint of the 95% moving-block-bootstrap interval for \(V_{DR}(\pi_{challenger})-V(\pi_{BTS})\) is strictly positive;
2. the challenger importance-weight effective-sample-size fraction is at least `0.10`.

For Study 3 both conditions fail:

- bootstrap lower endpoint: `-0.0049919409 < 0`;
- challenger ESS fraction: `0.0002268 < 0.10`.

Therefore the frozen rule returns **`do_not_promote`**.

## Scientific boundaries

- Clicks measure engagement, not revenue or long-term welfare.
- The independently logged Random CTR is an on-policy reference estimate with sampling uncertainty, not an exact population truth.
- The task is one-step contextual policy evaluation and selection, not reinforcement learning.
- Study 1's SAGA failures remain part of the record and were not rewritten after later studies succeeded.
- Study 2 and Study 3 were prospectively specified before their respective evaluation outcomes were opened.
- Study 3's timestamp and numerical-execution corrections were implementation-level provenance fixes only; they did not alter the frozen scientific design.
- Study 3 shows severe challenger overlap failure and a negative primary DR-minus-BTS interval; neither sensitivity clipping nor post-outcome tuning overrides the preregistered primary decision.
- Any post-result redesign must be treated as a new prospective study rather than tuning Study 3 after observing outcomes.
