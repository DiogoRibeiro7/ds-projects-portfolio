# Advanced Testing

This tutorial covers advanced experimentation techniques that improve power,
reduce variance, and prevent false positives.

## 1. Variance reduction with CUPED

Use pre-experiment covariates to reduce variance in the primary metric.

```python
import pandas as pd
from src.statistics.core import apply_cuped

# Example: baseline metric before the experiment
df = pd.read_csv("experiment.csv")
df["metric_cuped"] = apply_cuped(
    df,
    metric_col="metric",
    covariate_col="baseline_metric",
)
```

Tips:
- Pick covariates correlated with the outcome.
- Keep covariates fixed prior to treatment exposure.
- Re-run power calculations after CUPED (lower variance => smaller MDE).

## 2. Sequential testing

If you must peek early, use a sequential design and pre-registered stopping
rules.

Recommended approach:
- Define boundaries before launch.
- Use alpha spending (e.g., O’Brien–Fleming style).
- Log each interim analysis.

## 3. Multiple variants and traffic splits

When testing more than two variants:
- Use ANOVA or Kruskal–Wallis for global tests.
- Apply Holm or Benjamini–Hochberg for pairwise comparisons.
- Keep traffic splits balanced unless there is a strong product constraint.

## 4. Guardrails and safety checks

- Always monitor guardrail metrics (latency, churn, errors).
- Add SRM checks to validate randomization.
- Use `src/data_quality/` utilities for automated checks.

## 5. Sample size sensitivity

Run sensitivity analysis around MDE and baseline rate:

```python
from src.statistics.core import calculate_sample_size

baseline = 0.10
mde = 0.01
required_n = calculate_sample_size(baseline_rate=baseline, mde=mde)
print(required_n)
```

## 6. Reporting checklist

- [ ] Pre-registered test design
- [ ] CUPED/variance reduction documented
- [ ] Sequential plan documented (if used)
- [ ] Multiple-comparison adjustments applied
- [ ] Guardrails monitored and summarized

## Related references

- `docs/methodology/experimental_design.md`
- `docs/methodology/statistical_tests.md`
- `docs/ROBUSTNESS.md`
