# Adaptive Policy Learning

A reproducible contextual-bandit project for evaluating and selecting decision policies from logged recommendation data.

## Final primary-study status

**Terminal status: `model_fit_failure`. The preregistered primary empirical study did not reach off-policy evaluation.**

The frozen L2 logistic reward model using scikit-learn `LogisticRegression(solver="saga", C=1.0, tol=0.0001)` failed to converge on the BTS training partition under the original `max_iter=200` budget. A single pre-evaluation numerical-optimization amendment increased only the iteration budget to `max_iter=1000`; the same training-only fit again failed to converge.

The protocol therefore terminated before loading BTS evaluation-period outcomes or Random-reference outcomes. No IPS, SNIPS, direct-method, or doubly-robust estimate was computed, and no challenger promotion decision was made. Promotion is not authorized.

This is **not evidence that OPE is unreliable** and it is not evidence for or against the challenger policy. It is a negative result about the preregistered reward-model fitting specification under the allowed optimization budgets.

See [`RESULTS.md`](RESULTS.md) for the final study record and [`protocol/primary_empirical_terminal_status_v0_8.json`](protocol/primary_empirical_terminal_status_v0_8.json) for machine-readable provenance.

## Research question

> How reliable are off-policy estimators for real logged recommendation data, and can a transparent lower-confidence-bound rule prevent unsupported promotion of a learned challenger policy?

The primary empirical study uses the ZOZO Research Open Bandit Dataset. Bernoulli Thompson Sampling feedback is the logging-policy dataset. The uniform Random policy provides an independently logged on-policy reference over the same evaluation window.

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

These estimators remain implemented and tested, but the preregistered primary study did not produce empirical values because the frozen reward-model fit did not converge.

## Promotion rule

Had the primary analysis reached evaluation, the challenger would have been promoted only when both conditions held:

1. the lower endpoint of the preregistered 95% moving-block-bootstrap interval for
   \(V_{DR}(\pi_{challenger})-V(\pi_{BTS})\) was strictly positive;
2. the challenger importance-weight effective-sample-size fraction was at least 0.10.

Because the reward model did not converge, this gate was never evaluated. The terminal decision is therefore **no promotion authorization**, not an estimated `do_not_promote` result.

## Scientific boundaries

- Clicks measure engagement, not revenue or long-term welfare.
- The independently logged Random CTR would be an on-policy reference estimate with sampling uncertainty, not an exact population truth.
- The primary task is one-step contextual policy evaluation and selection, not reinforcement learning.
- The v0.8 optimizer amendment was made before any evaluation-period or Random-reference outcome was loaded.
- No further optimizer-budget increase is authorized for the primary study.
- Any future redesign must be treated as a new study with a new prospective protocol rather than a continuation of this primary result.

The full prospective contract begins at [`protocol/design_lock.json`](protocol/design_lock.json), with the terminal outcome recorded in [`protocol/primary_empirical_terminal_status_v0_8.json`](protocol/primary_empirical_terminal_status_v0_8.json).
