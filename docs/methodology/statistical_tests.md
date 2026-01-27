# Statistical Tests

Use this guide to select the right test, validate assumptions, and report
results clearly.

## 1. Choose the test

### Binary outcomes (conversion, retention)
- **Two-proportion z-test**: default for large samples.
- **Fisher’s exact test**: small samples or sparse counts.

### Continuous outcomes (revenue, duration)
- **t-test**: approximately normal, similar variances.
- **Welch’s t-test**: default when variances differ.
- **Mann–Whitney U**: non-normal distributions, ordinal data.

### Multiple groups or variants
- **ANOVA**: normal + equal variances.
- **Kruskal–Wallis**: non-parametric alternative.
- Follow with post-hoc pairwise testing (Holm or Benjamini–Hochberg).

### Paired data (before/after, matched users)
- **Paired t-test**: normal differences.
- **Wilcoxon signed-rank**: non-normal differences.

## 2. Validate assumptions

- **Normality**: Q–Q plot, Shapiro test (avoid over-reliance at large N).
- **Variance equality**: Levene or Bartlett tests.
- **Independence**: verify experimental unit and sampling strategy.
- **Distribution shift**: check for outliers and heavy tails.

## 3. Effect sizes and intervals

- Always report **effect size** (absolute and relative lift).
- Include **confidence intervals** for interpretability.
- Use **Cohen’s d** (continuous) or **risk difference / ratio** (binary).

## 4. Multiple comparisons

- Pre-register the primary metric.
- Use correction methods for secondary metrics:
  - Holm (strong control of FWER)
  - Benjamini–Hochberg (controls FDR)

## 5. Sequential testing

- If you look at results during the test, use sequential methods.
- Do not re-run plain t-tests daily.
- Document alpha spending and stopping boundaries.

## 6. Practical checklist

- [ ] Test choice matches metric and distribution
- [ ] Assumptions checked (or non-parametric used)
- [ ] Effect size + interval reported
- [ ] Multiple comparisons handled
- [ ] SRM and data quality checks passed

## Related code

- Binary and continuous testing: `src/statistics/core.py`
- Robust alternatives: `src/statistics/robust.py`
