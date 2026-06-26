# Notebook Summary

## Headline

- Public grouped-data window: `2007-2024`
- Most frequent BIC winner: `lognormal`
- Median Gamma decile error: `0.2013`
- Median Lognormal decile error: `0.1439`

## Model wins

- `lognormal`: 18 yearly BIC wins

## Sensitivity scenarios

- `baseline`: lognormal=18
- `body_only_1000_to_5000`: lognormal=18
- `drop_open_top`: lognormal=18
- `include_minimum_wage_bin`: lognormal=18

## Tail-only winners

- `xmin=2500.0`: pareto_tail=18
- `xmin=3750.0`: pareto_tail=18

## Interpretation

- The current grouped-data evidence supports Lognormal as the strongest full-distribution benchmark in this notebook.
- Gamma remains useful as an interpretable comparison model, especially for the body of the distribution.
- The sensitivity and tail sections should be checked alongside the headline winner before making a strong distributional claim.
