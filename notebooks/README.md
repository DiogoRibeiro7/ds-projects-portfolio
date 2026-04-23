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
| `causal_inference_policy_evaluation.ipynb` | Causal Inference | Advanced | Panel-data policy evaluation with known ground truth: Naive, OLS, PSM/IPTW, DiD, event study, Synthetic Control, IV (2SLS); estimators compared against a simulated true ATT | 2-4 min |
| `llm_rag_evaluation.ipynb` | LLM / MLOps | Advanced | RAG evaluation pipeline: retrieval metrics (Hit@k, MRR, nDCG), simulated generation with known hallucination rate, LLM-as-judge with seeded biases, human calibration (Cohen's κ, Pearson), and cost vs quality Pareto | 2-4 min |
| `genai_rag_pipeline.ipynb` | Portfolio / AI Engineering | Advanced | Deep RAG pipeline with corpus scaling, hybrid retrieval (BM25 + dense hashing), reranking, retrieval ablations (Hit@k/MRR/nDCG), cost-latency tradeoff simulation, and quality-gate artifact generation | 6-9 min |
| `genai_service_delivery.ipynb` | Portfolio / AI Engineering | Advanced | Delivery pipeline with quality-gated release logic, production-grade FastAPI artifact generation, CI/CD hardening (matrix + security scan + container build), Kubernetes/service manifests, and release recommendation reporting | 5-8 min |
| `genai_evalops_observability.ipynb` | Portfolio / AI Engineering | Advanced | Continuous RAG EvalOps pipeline with online/offline quality metrics, drift detection, alerting logic, SQL/NoSQL metric persistence, and CI release-gate integration patterns | 5-8 min |
| `genai_api_release_engineering.ipynb` | Portfolio / AI Engineering | Advanced | API release-engineering workflow with contract-compatibility checks, load and error-budget simulation, canary promote/rollback policy, and dual CI/CD definitions for GitHub Actions and Azure DevOps | 4-7 min |
| `genai_dataops_vector_platform.ipynb` | Portfolio / AI Engineering | Advanced | DataOps and vector-platform pipeline for multi-source ingestion, SQL/document/vector indexing, retrieval consistency benchmarking, and API/testing/cloud integration bundle generation | 4-7 min |
| `maritime_illegal_fishing_detection.ipynb` | Geospatial / Conservation | Advanced | AIS vessel-track anomaly detection for MPA enforcement: simulated 60-vessel fleet with bad-actor routing, rule-based + supervised (HistGBC) + ten-detector unsupervised benchmark (IF / LOF / OC-SVM / ECOD / COPOD / MLP-AE / VAE / LSTM-AE / Transformer-AE / TCN-AE), precision@K patrol queue, and geospatial map of flagged tracks | 6-10 min |
| `fisheries_stock_assessment_bayesian.ipynb` | Fisheries Science / Bayesian | Advanced | Full stock-assessment + management pipeline: Schaefer dynamics, hand-rolled Metropolis-Hastings MCMC state-space, R-hat/ESS/PPC/retrospective/prior-sensitivity diagnostics, Kobe plot, HCR simulation testing, SST-dependent r climate extension, Schaefer-vs-Fox AIC comparison, decision-theoretic utility, full MSE with annual re-estimation | 4-6 min |
| `demographics_lee_carter_mortality_forecasting.ipynb` | Demographics / Actuarial | Advanced | Lee-Carter mortality forecasting with MCMC, life expectancy and annuity valuation, longevity SCR; plus extensions for parameter bootstrap, Renshaw-Haberman cohort, CBD, Li-Lee multi-population, COVID shock, and Solvency II QRT template | 5-7 min |
| `urban_demographics_service_planning.ipynb` | Demographics / Urban Planning | Advanced | Multi-district cohort-component population projection with stochastic Monte Carlo, service-demand translation (schools/healthcare/long-term care), capacity-gap analysis, and investment prioritisation for municipal capital planning | 3-5 min |

## Notes

- Most notebooks are self-contained and use synthetic data.
- For `healthcare_analysis.ipynb`, prefer the pinned environment from the repository root `README.md` (`requirements-notebook-healthcare-shap.txt`).
- To execute a notebook non-interactively:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/<notebook_name>.ipynb
```
