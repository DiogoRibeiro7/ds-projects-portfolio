# Usage Guide

This guide shows how to call real modules from the repository with minimal,
copy-pasteable examples.

## Non-parametric Testing

```python
import numpy as np
from statistical_methods.advanced_statistical_tests import NonParametricTests

baseline = np.random.normal(loc=0.0, scale=1.0, size=200)
variant = np.random.normal(loc=0.3, scale=1.2, size=200)

result = NonParametricTests.mann_whitney_u(
    baseline,
    variant,
    alternative="two-sided",
    confidence_level=0.95,
)

print(f"U statistic: {result.statistic:.2f}")
print(f"p-value: {result.p_value:.4f}")
print(f"Median lift CI: {result.confidence_interval}")
```

`result` is a `TestResult` dataclass with statistic, p-value, effect size, and
bootstrap confidence intervals.

## Power Analysis Simulations

```python
from statistical_methods.power_analysis_simulations import PowerAnalysisSimulator

simulator = PowerAnalysisSimulator(n_simulations=2000, n_jobs=1)
summary = simulator.simulate_ab_test_power(
    baseline_rate=0.12,
    effect_sizes=[0.01, 0.02, 0.03],
    sample_sizes=[5000, 10000],
    alpha=0.05,
    test_type="proportion",
)

print(summary[["effect_size", "sample_size", "power"]])
```

The returned DataFrame shows the observed power for each effect size/sample size
pair so you can choose an appropriate design.

## Experiment Health Checks

```python
import numpy as np
import pandas as pd
from src.statistics.core import ExperimentAnalyzer

df = pd.DataFrame({
    "group": ["control"] * 500 + ["variant"] * 520,
    "converted": np.random.binomial(1, p=0.12, size=1020),
})

analyzer = ExperimentAnalyzer(alpha=0.05, power=0.8)
srm_report = analyzer.check_srm(df, group_col="group")

print(f"Chi-square p-value: {srm_report['p_value']:.4f}")
print(f"Expected ratio: {srm_report['expected_ratio']}")
print(f"Observed ratio: {srm_report['observed_ratio']}")
```

`ExperimentAnalyzer` provides additional helpers for power curves, CUPED, CUPAC,
and related experiment diagnostics.

## Next Steps

- Run `python scripts/analyze_experiment.py --help` for CLI-driven reporting.
- Browse `docs/api/index.md` for the curated API overview.
- Check `docs/contributor/development.md` for testing, linting, and validation
  workflows.
