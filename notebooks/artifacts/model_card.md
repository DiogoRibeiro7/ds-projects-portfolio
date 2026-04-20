# Model Card — healthcare_readmission_and_cost (v1.0.0)
_Generated: 2026-04-20T12:52:58+00:00_

## Intended Use
```json
{
  "primary": "Forecast expected annual healthcare cost and flag patients at elevated readmission risk for targeted nurse-led intervention.",
  "out_of_scope": [
    "Individual clinical diagnosis or treatment decisions without clinician review.",
    "Populations substantially different from the training distribution.",
    "Insurance pricing without legal / compliance sign-off."
  ]
}
```

## Data
```json
{
  "source": "synthetic generator",
  "n_rows": 6000,
  "n_features": 10,
  "target_regression": "healthcare_cost",
  "target_classification": "readmission",
  "train_test_split": {
    "test_size": 0.2,
    "stratify": "readmission",
    "random_state": 42
  }
}
```

## Models
```json
{
  "regression": {
    "family": "elastic_net",
    "best_params": {
      "model__alpha": 0.003618723330959626,
      "model__l1_ratio": 0.5200680211778108,
      "model__max_iter": 18417
    },
    "tuned_test_metrics": {
      "mae": 850.7129783689861,
      "rmse": 1069.6810370157536,
      "r2": 0.748035995057494
    },
    "rmse_95pct_ci": [
      1024.2748301175247,
      1116.9068368966925
    ]
  },
  "classification": {
    "family": "logistic",
    "best_params": {
      "model__C": 0.006517070593249451,
      "model__solver": "lbfgs"
    },
    "tuned_test_metrics": {
      "roc_auc": 0.760436559468337,
      "pr_auc": 0.4443928470209642,
      "brier": 0.12347386398804928
    },
    "roc_auc_95pct_ci": [
      0.7233581837384212,
      0.7972367180845418
    ],
    "operational_threshold": 0.18,
    "threshold_rationale": "Cost-optimal at C_FP=1.0, C_FN=6.0"
  }
}
```

## Fairness
```json
{
  "groups_audited": [
    "gender",
    "smoker",
    "region",
    "plan_tier"
  ],
  "max_tpr_gap": {
    "gender": 0.04460534549030115,
    "smoker": 0.33393267405853566,
    "region": 0.1895424836601307,
    "plan_tier": 0.11227902532250367
  },
  "max_fpr_gap": {
    "gender": 0.021661481737527355,
    "smoker": 0.2716920675915504,
    "region": 0.04864914202263598,
    "plan_tier": 0.07591161001094282
  },
  "max_brier_gap": {
    "gender": 0.021531934900709107,
    "smoker": 0.05430867455699981,
    "region": 0.034445766414868795,
    "plan_tier": 0.019004096508120655
  }
}
```

## Drift
```json
{
  "psi_summary": [
    {
      "feature": "cholesterol",
      "psi": 0.01328365750970608,
      "drift_level": "stable"
    },
    {
      "feature": "age",
      "psi": 0.012964247316630493,
      "drift_level": "stable"
    },
    {
      "feature": "bmi",
      "psi": 0.007513902282816381,
      "drift_level": "stable"
    },
    {
      "feature": "blood_pressure",
      "psi": 0.00644999775403889,
      "drift_level": "stable"
    },
    {
      "feature": "chronic_conditions",
      "psi": 0.003993997636827064,
      "drift_level": "stable"
    },
    {
      "feature": "visits_last_year",
      "psi": 0.003824138671126612,
      "drift_level": "stable"
    }
  ],
  "temporal_holdout_metrics": {
    "regression_rmse": 1055.517829712567,
    "regression_r2": 0.7539712198987771,
    "classification_roc_auc": 0.7592817504063978,
    "classification_pr_auc": 0.44049177688619595
  }
}
```

## Causal Uplift
```json
{
  "qini_coefficient": -0.02320532983836543,
  "simulated_treatment_effect": "Heterogeneous risk reduction, stronger for smokers / multi-chronic patients."
}
```

## Survival
```json
{
  "horizon_days": 365,
  "events_observed": 824,
  "censored": 5176
}
```

## Limitations
```json
[
  "Dataset may be synthetic; production deployment requires retraining on real clinical data.",
  "Causal uplift is simulated; real RCT data required before deploying targeting policy.",
  "No fairness post-processing applied; disparity gaps reported but not mitigated."
]
```

## Governance
```json
{
  "retrain_triggers": [
    "ROC-AUC drop > 0.03 on temporal holdout",
    "any subgroup error rate worsens > 20% relative",
    "any feature PSI > 0.2 for two consecutive weeks",
    "decision-curve net benefit at operational threshold falls below treat-all baseline"
  ]
}
```
