# Experimental Design

Use this guide to plan experiments that are statistically valid, operationally
safe, and aligned with business decisions.

## 1. Frame the decision

- Define the decision you want to make (ship, rollback, iterate).
- Identify the primary metric that will drive that decision.
- List guardrail metrics that must not regress (latency, churn, revenue).

## 2. Write hypotheses

- Null hypothesis (H0): no change in the primary metric.
- Alternative (H1): meaningful improvement or degradation.
- Specify directionality (two-sided by default; one-sided only with strong
  product rationale).

## 3. Choose the metric and unit

- Use a stable, well-understood metric (conversion, retention, ARPU).
- Define the unit of analysis: user, session, account, or order.
- Confirm deduping rules and attribution window.

## 4. Plan power and sample size

- Pick significance level (alpha) and desired power (1 - beta).
- Define the minimum detectable effect (MDE) that is worth acting on.
- Use the utilities in `src/statistics/core.py` to compute required sample
  sizes for binary metrics.

## 5. Randomization and allocation

- Randomize at the unit of analysis (user-level is common).
- Use stratification if you have strong known confounders.
- Keep allocation simple (50/50) unless rollout constraints require otherwise.

## 6. Experiment duration

- Ensure at least one full business cycle (weekly seasonality).
- Avoid stopping early unless sequential rules are pre-registered.
- Document the minimum run time in the analysis plan.

## 7. Data quality checks

- Monitor sample ratio mismatch (SRM) daily.
- Validate ingestion completeness and event schema stability.
- Use the utilities in `src/data_quality/` to automate checks.

## 8. Analysis plan

- Pre-register: metric, test, transformation, outlier rules.
- Decide handling for missing data and invalid events.
- Use the A/B utilities in `src/statistics/core.py` and
  `src/statistics/robust.py`.

## 9. Sequential considerations

- If you must peek early, use sequential testing.
- Never re-run significance tests daily without correction.
- Document stopping boundaries and alpha spending rules.

## Quick checklist

- [ ] Clear decision and primary metric
- [ ] Guardrails and data quality monitors
- [ ] Power/MDE calculation
- [ ] Randomization and allocation plan
- [ ] Analysis plan and stopping rules

## Related examples

- `examples/run_demo.py` for a minimal A/B test walkthrough.
- `docs/ROBUSTNESS.md` for trimming and robust estimators.
