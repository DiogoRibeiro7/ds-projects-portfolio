# Adaptive Policy Learning

A reproducible contextual-bandit project for evaluating and selecting decision policies from logged recommendation data.

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

## Promotion rule

The challenger is promoted only when both conditions hold:

1. the lower endpoint of the preregistered 95% moving-block-bootstrap interval for
   \(V_{DR}(\pi_{challenger})-V(\pi_{BTS})\) is strictly positive;
2. the challenger importance-weight effective-sample-size fraction is at least 0.10.

Otherwise the decision is `do_not_promote`.

## Scientific boundaries

- Clicks measure engagement, not revenue or long-term welfare.
- The independently logged Random CTR is an on-policy reference estimate with sampling uncertainty, not an exact population truth.
- The primary task is one-step contextual policy evaluation and selection, not reinforcement learning.
- No model, split, epsilon, estimator, or promotion threshold may be changed after the evaluation result is observed.

The full prospective contract is in [`protocol/design_lock.json`](protocol/design_lock.json).
