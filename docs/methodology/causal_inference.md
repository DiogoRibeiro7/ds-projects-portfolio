# Causal Inference

This guide outlines causal approaches used in the portfolio and how to choose
them for real product questions.

## 1. When to use causal inference

Use causal methods when you need to estimate impact (not just correlation), and
randomized experiments are unavailable or incomplete.

Typical questions:
- Did a pricing change reduce churn?
- What is the impact of a marketing campaign on revenue?
- Does a feature rollout improve retention for a subgroup?

## 2. Core methods

### Difference-in-differences (DiD)

Best for policy or campaign changes with clear pre/post windows and control
groups.

Assumptions:
- Parallel trends between treatment and control before the intervention.
- No other interventions affecting one group uniquely.

### Regression discontinuity (RD)

Best when treatment is assigned by a threshold (e.g., score cutoffs).

Assumptions:
- Units cannot precisely manipulate the running variable.
- Potential outcomes are smooth around the cutoff.

### Instrumental variables (IV)

Best when a natural “instrument” affects treatment assignment but not the
outcome directly.

Assumptions:
- Instrument relevance (affects treatment).
- Exclusion restriction (no direct path to outcome).

## 3. Diagnostics and robustness

- Plot pre-trends and placebo tests for DiD.
- Check density around the RD cutoff.
- Test instrument strength (first-stage F-stat).
- Run sensitivity analyses on bandwidths and covariates.

## 4. Reporting checklist

- [ ] Clearly state identification strategy and assumptions
- [ ] Show pre-trend diagnostics or balance checks
- [ ] Report effect size + confidence interval
- [ ] Include robustness and sensitivity checks

## Related example

- `projects/causal_inference/campaign_diff_in_diff/README.md`
