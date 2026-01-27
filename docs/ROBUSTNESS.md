# Robust Training Modes

This repo ships optional robustness toggles inside ExperimentAnalyzer and related utilities.

## Options

| Mode | Description | Trade-offs |
| --- | --- | --- |
| 	rim_fraction | Drops the top/bottom quantiles in each group before computing rates. | Removes leverage of extreme observations but slightly reduces effective sample size. Avoid >0.1 unless you can afford the data loss. |
| obust=True + huber_delta | Applies a Huber-style shrinkage to continuous metrics, clipping residuals to ±delta. | Stabilizes mean differences under heavy tails, but introduces bias when true effects live in the clipped region. |

## Usage

`python
analyzer = ExperimentAnalyzer(alpha=0.05)
robust_report = analyzer.analyze_conversion(
    df,
    conversion_col="converted",
    group_col="group",
    robust=True,
    trim_fraction=0.05,
)
`

For continuous metrics via un_comprehensive_analysis, pass the same flags:

`python
summary = analyzer.run_comprehensive_analysis(
    df,
    metrics=["ltv"],
    robust=True,
    trim_fraction=0.05,
    huber_delta=1.5,
)
`

## When to use

- **Heavy-tailed KPIs** (e.g., revenue): Use Huber smoothing with a modest delta.
- **Log data glitches/outliers**: Trim 1–5% per tail to protect against corrupted spikes.
- **Regulatory reporting**: Keep obust=False to avoid biasing official numbers.

Always compare robust vs non-robust summaries; differences highlight data quality issues.
