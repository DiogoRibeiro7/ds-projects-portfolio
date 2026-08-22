# Final Disposition Report

## Scope

This report resolves the remaining monorepo disposition phase under the revised
non-destructive portfolio policy:

- keep data science projects and portfolio files in this repository;
- copy-export focused packages and standalone snapshots where useful;
- do not delete or move originals;
- treat generated artifacts as cleanup candidates only after explicit approval.

## Resolution Summary

There are no remaining `REVIEW` rows in `migration/migration_matrix.csv`.

Resolved destination families:

| Destination | Status |
|---|---|
| `ds-projects-portfolio` | Active portfolio repository for projects, notebooks, examples, docs, root tests, scripts, and selected source modules. |
| `experimentation-toolkit` | Copy-first export created and validated. |
| `genai-rag-engineering` | Copy-first export created and validated. |
| `standalone-project-copy` | Four flagship project snapshots created and validated. |
| `customer-analytics` | Copy-first export created and validated. |
| `archive / reference` | Historical, broad infrastructure, and stale tooling are documented as retained but not promoted. |
| `generated cleanup` | Generated artifacts are identified but not removed. |

## Active Portfolio Content

The following remain active in `ds-projects-portfolio`:

- `projects/`
- `notebooks/`
- `examples/`
- `docs/`
- root tests and test tooling
- root scripts and lightweight tools
- root packaging/dependency files until a separate package-policy cleanup is approved
- shared source modules still used by the portfolio

This preserves the user's revised intent that the repository remain a data science projects
and portfolio repository rather than being reduced to a small hub.

## Copy-First Exports Completed

Created and validated:

- `migration/exports/experimentation-toolkit`
- `migration/exports/genai-rag-engineering`
- `migration/exports/portugal-gdp-bayesian-revision`
- `migration/exports/porto-lisbon-uhi-exposure`
- `migration/exports/city-wage-cost-global`
- `migration/exports/pt-salary-gamma-distribution`
- `migration/exports/customer-analytics`

The source files for all exports remain in the portfolio repository.

## Remaining Projects And Collections

Remaining project collections are explicitly resolved:

- customer analytics material remains active here and also has a copy-first export.
- A/B testing and statistical-methods notebooks remain active portfolio material; selected reusable APIs were copied into `experimentation-toolkit`.
- deep learning, NLP, time series, machine learning, causal inference, health/public-health, Portugal economics, and tutorial content remain portfolio/labs-style material inside this repository.
- `projects/performance_optimization` is retained as archive/reference unless a later curated tooling repo is explicitly approved.

No `ds-labs` export was created in this phase because the revised policy keeps portfolio
labs/projects here.

## Source And Infrastructure Disposition

Resolved source-module groups:

- `src/statistics`, selected `src/data_processing`, and related examples: copied to `experimentation-toolkit`; originals stay.
- `src/genai`: copied to `genai-rag-engineering`; original stays.
- `src/modern_bank_churn`: copied into `customer-analytics`; original stays.
- generic API/cloud/security/privacy/compliance/AutoML/scalability/deployment material: retained as archive/reference, not promoted.
- data quality, preprocessing, profiling, feature engineering, time series, utilities, models, and web assets: kept with the portfolio unless a later explicit export is approved.

## Workflow Resolution

The two previously unresolved workflow rows are now resolved as `KEEP`:

- `.github/workflows/dependency-update.yml`
- `.github/workflows/release.yml`

They are known stale root automation, but disabling or rewriting CI is a separate cleanup
task. They are not export candidates and were not deleted.

## Generated Cleanup Candidates

The matrix identifies generated/local artifacts as `DELETE_GENERATED` candidates, including:

- executed duplicate notebook outputs;
- runtime archives;
- GenAI vector/prediction/audit artifacts;
- large model artifacts;
- recommender/uplift/insurance generated artifacts;
- local caches and virtual environments.

No generated artifact was removed in this phase. Removal requires explicit cleanup approval.

## Zero Unresolved Active Paths

The migration matrix has no `REVIEW` decisions remaining. Every meaningful path is now one
of:

- `KEEP`
- `EXPORT_COPY`
- `ARCHIVE`
- `DELETE_GENERATED`

`DELETE_GENERATED` does not mean delete now; it means eligible for a future explicit cleanup
task.

## Manual Work Still Open

- Create actual standalone GitHub repositories from the validated export snapshots if desired.
- Decide whether to repair, disable, or replace stale root workflows.
- Decide whether generated artifacts should be removed or externalized.
- Refresh the root README/catalog to highlight the validated projects and exports.
- Decide whether any remaining labs-style collections deserve a future curated export.
