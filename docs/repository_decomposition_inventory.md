# Repository Decomposition Inventory

## Executive Summary

This repository should remain a data science projects portfolio, not be reduced to a tiny link hub. The revised decomposition goal is to keep portfolio-facing projects, notebooks, project docs, and reproducibility assets in this repository while copy-exporting reusable or unrelated engineering components into focused repositories.

The rule for future phases is:

- `KEEP`: stays active in this repository.
- `EXPORT_COPY`: copy to another repository, validate there, and keep the original here for now.
- `MOVE_LATER`: eligible for removal only after a validated destination exists and the user explicitly approves removal.
- `ARCHIVE`: historical/reference material that can stay in this repo but should not be promoted on the active portfolio surface.
- `DELETE_GENERATED`: generated/local/runtime artifacts that can be removed only after explicit cleanup approval.

No project/source path should be deleted simply because it was copied elsewhere.

## Repository Topology

| Path | Role | Portfolio disposition |
|---|---|---|
| `README.md` | Portfolio entry point | `KEEP`; refresh later to show maintained project groups. |
| `docs/` | Portfolio docs, API docs, methodology notes, contributor docs | `KEEP`; prune/rewrite later only with approval. |
| `projects/` | Data science projects and project collections | `KEEP`; strongest projects may also be `EXPORT_COPY`. |
| `notebooks/` | Top-level featured/lab notebooks | `KEEP`; GenAI notebooks may also be `EXPORT_COPY`. |
| `examples/` | Runnable demos and EDA examples | `KEEP`; examples tied to reusable exports may be copied. |
| `src/` | Reusable code plus domain packages and infrastructure demos | Mixed: some `EXPORT_COPY`, some `KEEP`, some `ARCHIVE`. |
| `tests/` | Root test suite covering reusable code, projects, integration, notebooks | `KEEP`; copy relevant slices with exported packages. |
| `scripts/` | Repo automation and analysis scripts | Mixed; copy only maintained scripts into destination repos. |
| `tools/` | Lightweight repo tools | `KEEP` or `ARCHIVE`; no separate repo. |
| `deployment/` | Generic ML/API deployment examples | `ARCHIVE` or `EXPORT_COPY` only when tied to a service repo. |
| `archive/` | Historical/internal/runtime legacy material | `ARCHIVE`; keep out of active portfolio presentation. |
| `artifacts/` | Generated model/vector/runtime artifacts | `DELETE_GENERATED` candidate, not source. |
| `.github/workflows/` | Monorepo CI and project-specific CI | `KEEP` until replacements are approved; copy relevant CI to exports. |
| root package/dependency files | Monorepo dev/package setup | `KEEP` for now; simplify only after exports and repo policy are agreed. |

## Logical Components

### Portfolio Projects To Keep Active

| Component | Purpose | Entry points | Tests/docs | Notes |
|---|---|---|---|---|
| `projects/portugal_gdp_bayesian_revision` | Portugal GDP/population revision analysis | `scripts/run_analysis.py`, package `pt_gdp_bayes`, notebook | project tests, README | Strong standalone project; `KEEP` and `EXPORT_COPY`. |
| `projects/porto_lisbon_uhi_exposure` | Porto/Lisbon urban heat island exposure | `scripts/run_analysis.py`, package `uhi_exposure`, notebook | project tests, README | Strong standalone project; `KEEP` and `EXPORT_COPY`. |
| `projects/city_wage_cost_global` | City wage/cost comparisons | notebooks | README, methodology | Strong notebook project; `KEEP` and optional `EXPORT_COPY`. |
| `projects/pt_salary_gamma_distribution` | Portugal salary distribution modelling | `scripts/run_full_analysis.py`, package, notebook | project tests, README | Strong standalone project; `KEEP` and `EXPORT_COPY`. |
| `projects/churn_prediction` | Telco churn analytics notebooks | notebooks | README | Keep as portfolio customer analytics. |
| `projects/advanced_customer_segmentation` | Segmentation pipeline and dashboard | `pipeline/*`, `dashboard/app.py` | local tests, README | Keep; may be copied into a customer analytics repo later. |
| `projects/customer_segmentation` | Basic segmentation demo | `pipeline.py`, `dashboard.py` | README | Keep as simple portfolio demo or merge later with customer analytics after approval. |
| `projects/ab_testing` | A/B testing notebooks and core utilities | notebooks, `core.py` | root tests indirectly | Keep as portfolio project collection; core utilities may be copied into experimentation package. |
| `projects/statistical_methods` | Statistical testing and Bayesian methods | Python modules | project-local test | Keep; selected APIs may be copied into experimentation package. |
| `projects/time_series` | Time-series notebooks and demos | notebooks and `train.py` | READMEs | Keep as project collection. |
| `projects/deep_learning`, `projects/nlp`, `projects/machine_learning`, `projects/causal_inference` | Educational/project collections | notebooks/scripts | READMEs where present | Keep as portfolio/labs content. |
| `projects/oecd_obesity_analysis_notebook`, `projects/excessive_drinking_map_reproduction`, `projects/portuguese_emigration_crises_labour_law`, `projects/portugal_gdp_income_distribution` | Domain notebooks | notebooks | requirements/README varies | Keep; may be organized in portfolio/labs sections. |

### Top-Level Notebooks To Keep Active

Top-level notebooks are a major portfolio surface and should stay unless explicitly archived later. They cover healthcare, finance, insurance, GenAI, forecasting, operations research, public health, life sciences, NLP, fairness, graph/geospatial, and specialized methods.

GenAI notebooks (`genai_rag_pipeline`, `llm_rag_evaluation`, `genai_service_delivery`, `genai_dataops_vector_platform`) are coherent enough for an `EXPORT_COPY` to a focused `genai-rag-engineering` repository, but originals should remain here until explicit removal approval.

### Reusable Libraries / Export Candidates

| Component | Purpose | Dependencies | Disposition |
|---|---|---|---|
| `src/statistics` | A/B testing and statistical primitives | NumPy, SciPy, statsmodels, sklearn | `EXPORT_COPY` to `experimentation-toolkit`; keep original here for project notebooks/tests. |
| `src/data_processing` | Cleaning/validation primitives and performance helpers | pandas, NumPy, optional optimization deps | Copy experiment-relevant subset to toolkit; keep original here. |
| `src/genai` | RAG/LLM primitives: chunking, retrieval, prompts, guardrails, evals, telemetry, pipeline | Pydantic, NumPy, optional FAISS/sentence-transformers/providers | `EXPORT_COPY` to `genai-rag-engineering`; keep original notebooks and package here. |
| `src/feature_engineering` | Generic and insurance/customer feature engineering | pandas, sklearn | Keep here; optional copy into customer analytics if that repo is created. |
| `src/modern_bank_churn` | Churn/customer analytics code | pandas, sklearn and ML deps | Keep here; optional copy into customer analytics. |
| `src/time_series` | Time-series utilities | NumPy/pandas/sklearn | Keep here; optional copy if a focused time-series repo is created. |
| `src/utils` | Logging, caching, observability, constants | mixed optional deps | Keep here; copy only required utility slices with exports. |

### Infrastructure / Archive Candidates

| Component | Purpose | Disposition |
|---|---|---|
| `src/api` and `deployment/model_server` | Generic ML API/model serving | `ARCHIVE` or copy only if a maintained service repo is created. |
| `src/cloud`, `src/security`, `src/privacy`, `src/compliance`, `src/automl`, `src/scalability` | Platform/infrastructure demos | Keep in repo for now but classify as archive/reference, not active project core. |
| `projects/performance_optimization` | Broad optimization tooling and benchmark output | `ARCHIVE` unless curated later. |
| `archive/internal`, `archive/legacy`, `archive/quality-reports`, `archive/runtime` | Historical/internal/runtime content | `ARCHIVE`; should not be promoted as active portfolio work. |

## Cross-Boundary Imports

Important import assumptions found:

- `src.*` imports appear in notebooks, docs, examples, scripts, and tests. This works from repository-root execution but needs rewrite for standalone exports.
- `projects.advanced_customer_segmentation.*` imports appear in that project's own tests and README; these are self-imports through the monorepo namespace.
- `projects/feature_engineering/utils.py` imports `src.feature_engineering.utils`, making it a wrapper/demo around root source.
- `projects/streamlit_apps/ab_test_calculator.py` imports `src.statistics.core`.
- GenAI notebooks import `src.genai`.
- Root tests import `src.statistics`, `src.data_processing`, `src.api`, `src.cloud`, and `src.utils`.
- Docs reference many `src.*` modules, including stale names such as `src.statistics.experimental_design` and `src.visualization` while the source directory is spelled `src/vizualization`.

These imports are acceptable inside the current portfolio repo for now. Any exported copy must replace them with real package-local namespaces.

## Packaging Problems

- `pyproject.toml` and `setup.py` both define package metadata for `ds-portfolio`.
- The console script `ds-portfolio = ds_portfolio.cli:main` points to a package not visible in the current tree.
- `setuptools.find_packages(where = ["src"])` installs packages under names such as `statistics`, `genai`, and `data_processing`, not as `src.statistics`; most repository code imports `src.*`.
- Several important `src/` directories lack `__init__.py` and are not installed packages under the current build config.
- Dependency files are broad and combine notebook, platform, docs, tests, ML, and optional infrastructure stacks.
- Root package metadata still references `data-science-portfolio` URLs while the current repository is `ds-projects-portfolio`.

Do not fix these during inventory. Exports should receive their own focused `pyproject.toml`; this repo can be simplified later after agreement.

## CI Problems

- `.github/workflows/ci.yml` is a monorepo fast CI but excludes many directories in Ruff/mypy config and installs only part of the stack.
- `.github/workflows/advanced_customer_segmentation.yml` is project-specific and should stay until a customer analytics export exists.
- `.github/workflows/notebook-tests.yml` contains stale paths and a large-notebook check ending in `|| true`, so large notebook failures can be ignored.
- `.github/workflows/dependency-update.yml` references dependency update patterns not aligned with the current dependency files.
- `.github/workflows/release.yml` targets the invalid root package release flow.
- `.github/workflows/todo.yml` is workflow automation for monorepo TODOs, not core project validation.
- CodeQL/dependency review should be recreated per destination repo rather than copied wholesale.

## Large Files and Artifacts

Tracked files larger than 5 MB:

| Path | Size | Classification | Disposition |
|---|---:|---|---|
| `artifacts/medical_imaging/model.pt` | 42.71 MB | generated model artifact | `DELETE_GENERATED` candidate after approval; externalize/regenerate if needed. |
| `projects/excessive_drinking_map_reproduction/notebooks/reproduce_excessive_drinking_map.ipynb` | 14.48 MB | notebook with likely embedded output | Keep for now; strip outputs only with approval. |

Other generated artifacts:

- `artifacts/genai/*`: predictions, audit logs, vector indexes, deployment snippets.
- `artifacts/recsys/*.pt`: generated recommender model artifacts.
- coverage, benchmark, cache, and local virtual environment directories may exist locally but should not be propagated.

## Candidate Copy-First Exports

| Destination | Copy source | Original remains here? | Rationale |
|---|---|---:|---|
| `experimentation-toolkit` | selected `src/statistics`, selected `src/data_processing`, selected `projects/ab_testing`, selected `projects/statistical_methods`, selected examples/tests | Yes | Focused reusable library, but portfolio notebooks still depend on originals. |
| `genai-rag-engineering` | `src/genai`, GenAI notebooks, focused service/demo/tests | Yes | Coherent RAG engineering package and notebooks. |
| `customer-analytics` | selected churn, segmentation, uplift, feature engineering material | Yes | Optional future repo; not required before keeping this portfolio intact. |
| individual flagship repos | selected project folders | Yes | Useful GitHub-pinnable project snapshots while retaining portfolio source. |

## Risks

- Deleting originals after export would break the agreed portfolio model unless explicitly approved.
- Root import assumptions mean exported copies require namespace rewrites.
- Overbroad dependency files make the root repo expensive to install; simplification should be a later, approved cleanup.
- Generated model/vector artifacts inflate clone size; cleanup is useful but should be explicit and limited.
- Some docs overstate or stale-reference implementation details; documentation cleanup should prioritize active portfolio presentation.

## Recommended Next Phase

Build a non-destructive migration matrix using the revised decisions:

- `KEEP`
- `EXPORT_COPY`
- `MOVE_LATER`
- `ARCHIVE`
- `DELETE_GENERATED`
- `REVIEW`

Then create export snapshots under a `migration/exports/` workspace without deleting or moving originals.
