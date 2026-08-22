# Migration Matrix

This matrix is non-destructive. The repository remains the portfolio plus active data science projects. Exporting means copying into another repository or `migration/exports/` snapshot; it does not mean deleting the original.

## Decision Definitions

| Decision | Meaning |
|---|---|
| `KEEP` | Stays active in this repository. |
| `EXPORT_COPY` | Copy to a destination repo/snapshot, validate there, and keep original here for now. |
| `MOVE_LATER` | May be removed from this repository only after a validated destination exists and the user explicitly approves removal. |
| `ARCHIVE` | Keep historical/reference material out of the active portfolio surface; do not promote or export by default. |
| `DELETE_GENERATED` | Generated/local/runtime artifact eligible for cleanup only with explicit approval. |
| `REVIEW` | Needs closer inspection before copy/export or cleanup. |

## Destination Families

| Destination | Role |
|---|---|
| `ds-projects-portfolio` | Current repository: portfolio, active data science projects, top-level notebooks, docs, examples, and root tests. |
| `experimentation-toolkit` | Optional focused package copied from experimentation/statistics primitives. |
| `genai-rag-engineering` | Optional focused package copied from RAG/LLM engineering code and notebooks. |
| `customer-analytics` | Optional focused copy of churn, segmentation, uplift, and customer feature engineering. |
| `standalone-project-copy` | Optional copy of a strong project into its own GitHub repository while leaving the source here. |
| `archive / reference` | Historical/internal/infrastructure material retained in this repo but not promoted. |
| `generated cleanup` | Generated artifacts and local outputs that can be removed only after explicit approval. |

## Core Policy

- Keep active portfolio data science projects in this repository.
- Use copy-first exports for reusable libraries or projects that benefit from a separate repo.
- Do not delete originals because an export exists.
- Do not delete project/source files without explicit approval.
- Generated/local junk can be cleaned only as a narrow, explicit cleanup task.

The full machine-readable table is in `migration/migration_matrix.csv`.

## Highest-Value Copy-First Exports

1. `experimentation-toolkit`: copy selected `src/statistics`, selected `src/data_processing`, selected A/B testing/statistical-methods examples, and focused tests.
2. `genai-rag-engineering`: copy `src/genai` and the four RAG notebooks into a standalone package with offline fake-LLM CI.
3. `standalone-project-copy` snapshots for the strongest projects:
   - `projects/portugal_gdp_bayesian_revision`
   - `projects/porto_lisbon_uhi_exposure`
   - `projects/city_wage_cost_global`
   - `projects/pt_salary_gamma_distribution`
4. Optional later `customer-analytics`: copy churn, segmentation, uplift, and selected feature-engineering material.

## Drain Resolution

All active rows in `migration/migration_matrix.csv` now have a concrete non-`REVIEW`
disposition. The stale dependency-update and release workflows are classified as `KEEP`
root automation because changing or disabling CI is a separate cleanup decision, not part of
copy-first extraction. Generated artifacts remain `DELETE_GENERATED` candidates only; they
must not be removed without explicit cleanup approval.

## What Stays Here

- `projects/`
- `notebooks/`
- `examples/`
- `docs/`
- root tests and scripts needed by this portfolio
- original source modules until project usage is deliberately refactored
- archive/reference material unless a cleanup task is approved

## What Should Not Be Promoted

- Generic platform/deployment frameworks not tied to an active project.
- Internal archive/runtime artifacts.
- Generated model binaries, vector indexes, predictions, audit logs, caches, and local virtual environments.
