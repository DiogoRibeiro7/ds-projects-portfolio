# Robust Training Modes

The portfolio includes optional robustness toggles in `ExperimentAnalyzer` for
experiments where outliers or heavy-tailed metrics could distort the default
summary.

## Options

| Mode | Description | Trade-offs |
| --- | --- | --- |
| `trim_fraction` | Drops the top and bottom quantiles in each group before computing rates or summaries. | Reduces leverage from extreme observations but lowers effective sample size. Avoid values above `0.1` unless the experiment has enough traffic. |
| `robust=True` with `huber_delta` | Applies Huber-style clipping to continuous metric residuals. | Stabilizes mean differences under heavy tails but can bias estimates when true effects sit in the clipped region. |

## Usage

```python
from src.statistics.core import ExperimentAnalyzer

analyzer = ExperimentAnalyzer(alpha=0.05)
robust_report = analyzer.analyze_conversion(
    df,
    conversion_col="converted",
    group_col="group",
    robust=True,
    trim_fraction=0.05,
)
```

For continuous metrics via `run_comprehensive_analysis`, pass the same flags:

```python
summary = analyzer.run_comprehensive_analysis(
    df,
    metrics=["ltv"],
    robust=True,
    trim_fraction=0.05,
    huber_delta=1.5,
)
```

## When To Use

- Heavy-tailed KPIs, such as revenue or lifetime value.
- Log or data pipeline glitches that create corrupted spikes.
- Sensitivity analysis where robust and default summaries are compared side by
  side.

Keep `robust=False` for official reporting unless the robust method is part of
the pre-registered analysis plan. Differences between robust and non-robust
summaries should trigger a data quality review.
