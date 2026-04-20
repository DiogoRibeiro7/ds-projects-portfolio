# Notebook Catalog

Quick index of notebooks under `notebooks/`, with suggested usage and expected runtime on a standard laptop CPU.

| Notebook | Domain | Level | Primary Use Case | Est. Runtime |
| --- | --- | --- | --- | --- |
| `healthcare_analysis.ipynb` | Life Science / Healthcare | Advanced | End-to-end healthcare modeling, explainability, calibration, drift/fairness checks | 8-15 min |
| `life_science_clinical_response_safety.ipynb` | Life Science | Core | Dual-endpoint benchmarking, calibration, subgroup consistency checks, and benefit-risk dose policy optimization | 4-7 min |
| `life_science_survival_censoring_analysis.ipynb` | Life Science | Advanced | Right-censored survival analysis with KM/RMST, log-rank, Cox PH, and discrete-time hazard trajectory modeling | 6-10 min |
| `insurance_data_science.ipynb` | Insurance | Advanced | Pricing/risk segmentation workflow extended with governance benchmarking, calibration economics, PSI drift, and subgroup stability monitoring | 8-14 min |
| `actuarial_pricing_and_reserving.ipynb` | Insurance | Core | Integrated actuarial pricing (frequency/severity) and reserving (chain-ladder + bootstrap) governance workflow | 5-9 min |
| `insurance_fraud_triage_optimization.ipynb` | Insurance | Core | Fraud triage benchmarking, calibration, capacity queue simulation, and economics-driven SIU policy optimization | 4-7 min |
| `insurance_reserving_triangle_chainladder.ipynb` | Insurance | Advanced | Chain-ladder reserving with triangle diagnostics, bootstrap uncertainty, risk-margin view, and governance metrics | 4-7 min |
| `finance_credit_risk_stress_testing.ipynb` | Finance | Core | Credit risk PD benchmarking, cost-aware thresholding, and multi-scenario stress expected-loss analysis | 4-7 min |
| `finance_market_risk_var_backtesting.ipynb` | Finance | Advanced | Multi-model VaR estimation with Kupiec/Christoffersen backtests, regime diagnostics, and stressed capital interpretation | 4-7 min |

## Notes

- Most notebooks are self-contained and use synthetic data.
- For `healthcare_analysis.ipynb`, prefer the pinned environment from the repository root `README.md` (`requirements-notebook-healthcare-shap.txt`).
- To execute a notebook non-interactively:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/<notebook_name>.ipynb
```
