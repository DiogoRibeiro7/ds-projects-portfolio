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
| `genai_rag_pipeline.ipynb` | GenAI / AI Engineering | Advanced | Production RAG pipeline reusing `src/genai/` primitives: 40-doc synthetic KB, recursive chunking, sentence-transformers embeddings, FAISS dense + BM25 + reciprocal-rank-fusion hybrid + cross-encoder reranker, HyDE and multi-query transforms, versioned prompt registry, Pydantic structured output with citation parsing, ingress PII/injection and egress hallucination guardrails, per-request tracing + cost/latency accounting. Emits `artifacts/genai/predictions.jsonl` for the eval notebook. Offline-reproducible via `FakeLLMClient`; `USE_LIVE_API` flag swaps in Anthropic or OpenAI | 1-2 min |
| `llm_rag_evaluation.ipynb` | GenAI / AI Engineering | Advanced | RAGAS-style RAG evaluation that consumes the pipeline's `predictions.jsonl`: retrieval metrics (Hit@k / MRR@10 / nDCG@5 / context-recall) with bootstrap 95% CIs, LLM-as-judge generation metrics (faithfulness, answer-relevance, context-precision) via versioned judge prompts, judge calibration vs structural oracle (Cohen's κ, Pearson r, Spearman ρ) with degenerate-case guard and synthetic-noise methodology demo, difficulty-stratified slicing, cost-quality Pareto, release-gate decision emitted as `eval_report.json` | 1 min |
| `genai_service_delivery.ipynb` | GenAI / AI Engineering | Advanced | Production delivery layer wrapping the same `RAGPipeline` behind a FastAPI service. Refuses to boot in production mode unless `eval_report.json` passed the gate. Pydantic schemas, bearer-token + tenant auth, per-tenant token-bucket rate limiting, retry with backoff + circuit breaker, append-only PII-redacted JSONL audit log, per-tenant P50/P95 latency + cost metrics, full TestClient e2e suite (401 / 403 / 200 / PII / injection / rate-limit), pre-deployment canary run with SLO gate (p95 / error / grounded), API-contract backward-compatibility check, K8s Deployment + HPA + PDB, GitHub Actions workflow with eval gate, and a consolidated `release_decision.json` (eval + canary + contract) | 1 min |
| `genai_dataops_vector_platform.ipynb` | GenAI / AI Engineering | Advanced | Operational sibling to the RAG pipeline. Multi-tenant 60-doc corpus with seeded near-duplicates; ingestion telemetry; persistent FAISS (save/reload/verify bit-identical results); `IndexFlatIP` vs `IndexIVFFlat` vs `IndexHNSWFlat` benchmark (build time / query latency / size / recall@5 vs Flat); union-find near-duplicate detection with tunable threshold; versioned incremental ingestion (v1 → v2 delta with parent manifest); metadata filtering + tenant-isolation audit; four policy-driven data-quality gates at ingestion; `vector_platform_manifest.json` with promote / block recommendation | 1 min |
| `maritime_illegal_fishing_detection.ipynb` | Geospatial / Conservation | Advanced | AIS vessel-track anomaly detection for MPA enforcement: simulated 60-vessel fleet with bad-actor routing, rule-based + supervised (HistGBC) + ten-detector unsupervised benchmark (IF / LOF / OC-SVM / ECOD / COPOD / MLP-AE / VAE / LSTM-AE / Transformer-AE / TCN-AE), precision@K patrol queue, and geospatial map of flagged tracks | 6-10 min |
| `fisheries_stock_assessment_bayesian.ipynb` | Fisheries Science / Bayesian | Advanced | Full stock-assessment + management pipeline: Schaefer dynamics, hand-rolled Metropolis-Hastings MCMC state-space, R-hat/ESS/PPC/retrospective/prior-sensitivity diagnostics, Kobe plot, HCR simulation testing, SST-dependent r climate extension, Schaefer-vs-Fox AIC comparison, decision-theoretic utility, full MSE with annual re-estimation | 4-6 min |
| `demographics_lee_carter_mortality_forecasting.ipynb` | Demographics / Actuarial | Advanced | Lee-Carter mortality forecasting with MCMC, life expectancy and annuity valuation, longevity SCR; plus extensions for parameter bootstrap, Renshaw-Haberman cohort, CBD, Li-Lee multi-population, COVID shock, and Solvency II QRT template | 5-7 min |
| `urban_demographics_service_planning.ipynb` | Demographics / Urban Planning | Advanced | Multi-district cohort-component population projection with stochastic Monte Carlo, service-demand translation (schools/healthcare/long-term care), capacity-gap analysis, and investment prioritisation for municipal capital planning | 3-5 min |

## GenAI stack — how the four notebooks compose

The four `genai_*` / `llm_rag_*` notebooks share a single contract (`Prediction` → `EvalReport` → release gate) and run in sequence:

```text
genai_dataops_vector_platform  →  genai_rag_pipeline  →  llm_rag_evaluation  →  genai_service_delivery
            (optional)              predictions.jsonl     eval_report.json      release_decision.json
```

All four import the same primitives from [`src/genai/`](../src/genai/): `Document` / `Chunk` / `Prediction` schemas, `SentenceTransformerEmbedder`, `DenseRetriever` / `BM25Retriever` / `HybridRetriever` / `CrossEncoderReranker`, `InputGuardrail` / `OutputGuardrail`, `RAGPipeline`, `Tracer` / `CostTracker`, and the RAGAS-style metrics in `src/genai/evals.py`. Toggling `USE_LIVE_API = True` at the top of each notebook swaps the deterministic `FakeLLMClient` for real Anthropic or OpenAI calls without any other changes.

Running the full end-to-end takes ~2 minutes on a CPU-only laptop; CI / offline runs are reproducible because the fake LLM's outputs are deterministic on input.

## Notes

- Most notebooks are self-contained and use synthetic data.
- For `healthcare_analysis.ipynb`, prefer the pinned environment from the repository root `README.md` (`requirements-notebook-healthcare-shap.txt`).
- To execute a notebook non-interactively:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/<notebook_name>.ipynb
```
