# First A/B Test

This walkthrough shows the minimal steps to run an A/B test using the
portfolio utilities.

## 1. Run the built-in demo

```bash
python examples/run_demo.py
```

You should see:
- Control and treatment conversion rates
- Absolute lift
- P-value

## 2. Understand the output

- **Conversion rate**: metric per group (users, sessions, etc.).
- **Lift**: treatment minus control (absolute or relative).
- **P-value**: how likely the observed lift is under the null hypothesis.

If the p-value is below your alpha (often 0.05), the result is statistically
significant. Always check guardrails before shipping.

## 3. Try your own data

```python
import pandas as pd
from src.statistics.core import ExperimentAnalyzer

df = pd.read_csv("my_experiment.csv")
# Expected columns: group (control/treatment), outcome (0/1), and optional user_id

analyzer = ExperimentAnalyzer()
results = analyzer.analyze_conversion(
    df,
    group_col="group",
    metric_col="outcome",
)

print(results)
```

## 4. Common pitfalls

- **Mismatched units**: use the same unit of analysis across groups.
- **Multiple peeks**: only use sequential testing if pre-registered.
- **SRM**: large allocation imbalances can invalidate results.
- **Outliers**: apply robust options (see `docs/ROBUSTNESS.md`).

## 5. Next steps

- Add minimum detectable effect planning via `calculate_sample_size`.
- Compare alternative metrics and sensitivity checks.
- Turn the notebook into a shareable report.
