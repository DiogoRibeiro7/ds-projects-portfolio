# Standardize Evaluation Metrics Across Projects

Consistent evaluation metrics make it easier to compare models, share results, and reuse analysis across projects. This guide defines standard metric categories, naming conventions, and documentation expectations for the portfolio.

## Why standardize evaluation metrics?

- Improves cross-project comparability and reporting.
- Prevents ambiguous or inconsistent metric names.
- Makes it easier to onboard new analysts and collaborators.
- Enables reusable dashboards, experiment summaries, and evaluation pipelines.

## Core metric categories

### 1. Predictive performance metrics
These metrics describe how well a model predicts outcomes.

- Classification: `accuracy`, `precision`, `recall`, `f1`, `auc_roc`, `pr_auc`, `log_loss`
- Regression: `rmse`, `mae`, `mape`, `r2`, `explained_variance`
- Ranking / recommender: `precision_at_k`, `recall_at_k`, `ndcg_at_k`, `map_at_k`, `mrr`
- Calibration: `calibration_error`, `brier_score`

### 2. Business and operational metrics
These metrics connect model evaluation to business value and health.

- Business value: `revenue_per_user`, `ltv`, `cac`, `roi`, `profit_margin`
- Retention and churn: `retention_30d`, `retention_90d`, `churn_rate`
- Engagement: `active_users`, `session_duration`, `feature_adoption`
- Cost and efficiency: `cost_per_acquisition`, `cost_per_prediction`, `latency_ms`

### 3. Experiment and guardrail metrics
Use a small set of consistent primary and guardrail metrics across projects.

- Primary experiment metric: project-specific decision KPI
- Guardrails: `error_rate`, `data_drift_score`, `model_drift_score`, `cpu_utilization`, `memory_usage`
- Quality checks: `sample_ratio_mismatch`, `data_freshness_delay`, `schema_validation_failures`

## Naming conventions

- Use `snake_case` for all metric identifiers.
- Prefer clear, descriptive names rather than abbreviations.
- For thresholded or ranked metrics, include the suffix and value:
  - `precision_at_10`
  - `ndcg_at_5`
  - `recall_at_20`
- For time-windowed metrics, include the window in the name:
  - `retention_30d`
  - `revenue_7d`
  - `active_users_28d`

## Definition requirements

Every metric should be documented with the following fields:

- **Name**: canonical metric identifier
- **Description**: what it measures and why it matters
- **Type**: predictive, business, experiment, or operational
- **Unit**: percent, count, dollars, seconds, etc.
- **Direction**: higher is better, lower is better, or neutral
- **Baseline / target**: reference value or business threshold
- **Data source**: dataset, event stream, or model output field
- **Calculation details**: formula, aggregation window, and filtering rules

## Recommended standard metric set

Use this sample set as a baseline for most machine learning projects.

### Classification

- `accuracy`
- `precision`
- `recall`
- `f1`
- `auc_roc`
- `pr_auc`
- `log_loss`
- `calibration_error`

### Regression

- `rmse`
- `mae`
- `mape`
- `r2`
- `explained_variance`

### Ranking / recommendation

- `precision_at_k`
- `recall_at_k`
- `ndcg_at_k`
- `map_at_k`
- `mrr`

### Business value

- `revenue_per_user`
- `cac`
- `ltv`
- `churn_rate`
- `retention_30d`
- `roi`

## Implementation guidance

- Document metric definitions in a shared location such as `docs/methodology/business_metrics.md` or the new `evaluation_metrics_standardization.md` page.
- Use the experiment README template in `docs/methodology/experiment_readme_template.md` to capture metric context in each experiment.
- Keep dashboards aligned with the portfolio dashboard baseline metrics guidance in `docs/modules/dashboard.md`.
- When adding new projects, map project-specific metric names to the standard metric vocabulary.

## Example documentation block

```markdown
- **metric:** `auc_roc`
  - **description:** Area under the receiver operating characteristic curve.
  - **type:** predictive performance
  - **unit:** probability
  - **direction:** higher is better
  - **baseline:** 0.70
  - **source:** scored test dataset
```

## Next step

Use this guide to audit existing project metric pages and migrate project-specific README sections to the standard metric vocabulary.
